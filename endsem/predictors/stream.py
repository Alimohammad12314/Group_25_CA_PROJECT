"""Run the predictors over a workload: offline accuracy (Deliverable 1) + a prediction stream.

Predictions never depend on timing, only on routing history, so they are computed once per
(workload, predictor config) and replayed by every timeline run (any link, cache size, policy).

Information each prediction may use (no look-ahead leak):

  target (s, l) at distance d >= 1  issued when layer l-d of step s finishes routing: knows the
                                    current token's experts at layers <= l-d and all earlier tokens
  target (s, l) at d = 0            issued when layer l of step s-1 finished: earlier tokens only
                                    (also used when l - d < 0, i.e. the target is in the next token)

Combining (see calibration.py): each predictor's ranking -> calibrated probability per sequence,
mixed with the chooser's weights ('mean') or pooled in log-odds ('logodds'), then
P(e) = 1 - prod_b (1 - q_b(e)) is the chance that at least one sequence of the batch routes to e.

Stream layout
  P[s, l, d, e]  float16  predicted need of expert e (batch union), also the eviction score
  M[s, l, d]     int16    prefetch budget before the link-time cap
"""
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .budget import AdaptiveBudget
from .calibration import RankCalibrator, competition_rank
from .chooser import PREDICTORS, Chooser
from .cross_layer import CrossLayerPredictor
from .popularity import PopularityPredictor
from .temporal import TemporalPredictor

EPS = 1e-6


@dataclass
class PredictorConfig:
    name: str = "full"
    predictors: tuple = ("temporal", "cross", "pop")
    chooser: str = "adaptive"          # adaptive | equal
    gamma: float = 2.0
    combine: str = "mean"              # mean | logodds
    budget: str = "adaptive"           # adaptive | topk | topk+<x> (fixed extra x per sequence) | oracle
    tau: float = 0.75
    D: int = 4                          # deepest lookahead distance predicted
    counter_bits: int = 4
    w_init: int = 4
    w_max: int = 16
    temporal_adaptive: bool = True
    pop_window: int = 64
    off_threshold: float = 0.40

    def key(self):
        d = asdict(self)
        d.pop("name")
        d["predictors"] = "+".join(self.predictors)
        return "_".join(f"{k}={v}" for k, v in sorted(d.items()))


@dataclass
class Stream:
    config: PredictorConfig
    P: np.ndarray
    M: np.ndarray
    metrics: pd.DataFrame               # one row per (layer, distance)
    predictor_recall: pd.DataFrame      # per (layer, distance, predictor) top-k recall and weight
    seconds_per_layer_event: float      # wall time of one predict+update, this Python implementation
    state_bits: dict = field(default_factory=dict)
    temporal_window: np.ndarray = None  # final adaptive W per layer


def _topk_recall(ranks, member, k):
    """Mean over sequences of the share of routed experts among the predictor's top-k ranks."""
    return float(((ranks < k) & member).sum() / member.sum())


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


class _Runner:
    def __init__(self, wl, cfg: PredictorConfig):
        self.wl, self.cfg = wl, cfg
        S, B, L, k, N, D = wl.S, wl.B, wl.L, wl.top_k, wl.num_experts, cfg.D
        self.S, self.B, self.L, self.k, self.N, self.D = S, B, L, k, N, D
        self.on = [p in cfg.predictors for p in PREDICTORS]
        self.temporal = TemporalPredictor(L, N, k, B, w_init=cfg.w_init, w_max=cfg.w_max,
                                          adaptive=cfg.temporal_adaptive)
        self.cross = CrossLayerPredictor(L, N, D, bits=cfg.counter_bits)
        self.pop = PopularityPredictor(L, N, window=cfg.pop_window)
        self.chooser = Chooser(L, D, cfg.predictors, mode=cfg.chooser, gamma=cfg.gamma)
        self.budget = AdaptiveBudget(L, D, N, tau=cfg.tau, off_threshold=cfg.off_threshold)
        self.cal_t = RankCalibrator(L, N, k)
        self.cal_p = RankCalibrator(L, N, k)
        self.cal_c = RankCalibrator(L * (D + 1), N, k)
        self.P = np.zeros((S, L, D + 1, N), np.float16)
        self.M = np.zeros((S, L, D + 1), np.int16)
        self.rank_t = np.zeros((L, B, N), np.int64)   # ranks for the next target of each layer
        self.rank_p = np.zeros((L, B, N), np.int64)
        self.pending = {}
        z = lambda: np.zeros((L, D + 1))
        self.acc = dict(n=z(), overlap=z(), recall=z(), precision=z(), fullcov=z(), budget=z(),
                        union=z(), khat=z(), delta=z(), extra=z())
        self.pacc = np.zeros((L, D + 1, 3))
        self.pn = np.zeros((L, D + 1, 3))
        self.prior_logit = float(_logit(np.array(k / N)))

    # --- combining -------------------------------------------------------------------------
    def _combine(self, l, d, rank_c):
        w = self.chooser.weights(l, d)
        probs = [self.cal_t.prob(l, self.rank_t[l]) if self.on[0] else None,
                 self.cal_c.prob(l * (self.D + 1) + d, rank_c) if rank_c is not None else None,
                 self.cal_p.prob(l, self.rank_p[l]) if self.on[2] else None]
        if self.cfg.combine == "logodds":
            n = sum(1 for i, p in enumerate(probs) if p is not None and w[i] > 0)
            z = np.full((self.B, self.N), self.prior_logit)
            for i, p in enumerate(probs):
                if p is not None and w[i] > 0:
                    z += n * w[i] * (_logit(p) - self.prior_logit)
            q = 1.0 / (1.0 + np.exp(-z))
        else:
            q = np.zeros((self.B, self.N))
            for i, p in enumerate(probs):
                if p is not None and w[i] > 0:
                    q += w[i] * p
            if not q.any():
                q[:] = self.k / self.N
        q = np.clip(q, 0.0, 1.0 - EPS)
        return 1.0 - np.exp(np.log1p(-q).sum(axis=0))

    def _predict(self, s, l, d, rank_c):
        cfg = self.cfg
        P = self._combine(l, d, rank_c)
        entry = dict(rank_c=rank_c)
        if cfg.budget == "oracle":
            score = np.zeros(self.N)
            score[np.unique(self.wl.A[s, :, l, :])] = 1.0
            M = int(score.sum())
        elif cfg.budget.startswith("topk"):
            # Fixed top-(k+x) per sequence ("topk", "topk+2", ...), union over the batch.
            x = min(self.k + (int(cfg.budget[5:]) if "+" in cfg.budget else 0), self.N)
            member = np.zeros(self.N, bool)
            q_rank = self._seq_scores(l, d, rank_c)
            member[np.unique(np.argsort(-q_rank, axis=1, kind="stable")[:, :x])] = True
            score = np.where(member, 0.5 + 0.5 * P, 0.5 * P)
            M = int(member.sum())
        else:
            score = P
            M, k_hat = self.budget.size(l, d, float(P.sum()))
            entry["k_hat"] = k_hat
        entry["score"], entry["M"] = score, M
        self.P[s, l, d] = score
        self.M[s, l, d] = M
        return entry

    def _seq_scores(self, l, d, rank_c):
        """Per-sequence combined probability (for the fixed top-x baselines)."""
        w = self.chooser.weights(l, d)
        q = np.full((self.B, self.N), 1e-9) + 1e-12 * self.pop.scores(l)
        if self.on[0]:
            q += w[0] * self.cal_t.prob(l, self.rank_t[l])
        if rank_c is not None:
            q += w[1] * self.cal_c.prob(l * (self.D + 1) + d, rank_c)
        if self.on[2]:
            q += w[2] * self.cal_p.prob(l, self.rank_p[l])
        return q

    # --- evaluation when the router reveals (s, l) -----------------------------------------
    def _evaluate(self, s, l, A, U):
        k, N, D = self.k, self.N, self.D
        member = np.zeros((self.B, N + 1), bool)
        member[np.arange(self.B)[:, None], np.where(A < 0, N, A)] = True
        member = member[:, :N]
        rec_t = _topk_recall(self.rank_t[l], member, k) if self.on[0] else None
        rec_p = _topk_recall(self.rank_p[l], member, k) if self.on[2] else None
        if self.on[0]:
            self.cal_t.update(l, self.rank_t[l], member)
        if self.on[2]:
            self.cal_p.update(l, self.rank_p[l], member)
        nU = len(U)
        a = self.acc
        for d in range(D + 1):
            if d > 0 and l - d < 0:
                continue
            entry = self.pending.pop((l, d))
            rec_c = None
            if entry["rank_c"] is not None:
                rec_c = _topk_recall(entry["rank_c"], member, k)
                self.cal_c.update(l * (D + 1) + d, entry["rank_c"], member)
            recalls = (rec_t, rec_c, rec_p)
            self.chooser.update(l, d, recalls)
            for i, r in enumerate(recalls):
                if r is not None:
                    self.pacc[l, d, i] += r
                    self.pn[l, d, i] += 1
            score, M = entry["score"], entry["M"]
            pos = np.empty(N, np.int64)
            pos[np.argsort(-score, kind="stable")] = np.arange(N)
            pu = pos[U]
            in_U, in_M = int((pu < nU).sum()), int((pu < M).sum())
            a["n"][l, d] += 1
            a["overlap"][l, d] += in_U / nU
            a["recall"][l, d] += in_M / nU
            a["precision"][l, d] += in_M / M
            a["fullcov"][l, d] += float(in_M == nU)
            a["budget"][l, d] += M
            a["union"][l, d] += nU
            if "k_hat" in entry:
                delta = self.budget.update(l, d, pu, entry["k_hat"])
                a["khat"][l, d] += entry["k_hat"]
                a["delta"][l, d] += delta
                a["extra"][l, d] += M - entry["k_hat"]
        if rec_t is not None:
            self.temporal.feedback(l, rec_t)

    def _refresh_d0(self, s, l):
        """Temporal and popularity ranks for the next token at layer l, and its d = 0 prediction."""
        if self.on[0]:
            self.rank_t[l] = competition_rank(self.temporal.scores(l))
        if self.on[2]:
            self.rank_p[l] = competition_rank(np.broadcast_to(self.pop.scores(l), (self.B, self.N)))
        self.pending[(l, 0)] = self._predict(s, l, 0, None)

    def run(self):
        wl, L, D = self.wl, self.L, self.D
        group = -1
        t_work, n_work = 0.0, 0
        for s in range(self.S):
            if wl.group_start[s]:
                group += 1
                self.temporal.reset(wl.warm[group])
                self.pop.warm(wl.warm[group])
                self.pending = {}
                for l in range(L):
                    self._refresh_d0(s, l)
            last_of_group = (s + 1 == self.S) or wl.group_start[s + 1]
            for l in range(L):
                t0 = time.perf_counter()
                A = wl.A[s, :, l, :]
                U = np.unique(A)
                U = U[U >= 0]
                self._evaluate(s, l, A, U)
                if self.on[1]:
                    for d in range(1, D + 1):
                        if l - d >= 0:
                            self.cross.update(l - d, d, wl.A[s, :, l - d, :], A)
                self.temporal.push(l, A)
                self.pop.push(l, A)
                for d in range(1, D + 1):
                    lt = l + d
                    if lt < L:
                        rc = competition_rank(self.cross.scores(l, d, A)) if self.on[1] else None
                        self.pending[(lt, d)] = self._predict(s, lt, d, rc)
                if not last_of_group:
                    self._refresh_d0(s + 1, l)
                t_work += time.perf_counter() - t0
                n_work += 1
        for d in range(1, D + 1):   # targets whose issue layer would be negative reuse d = 0
            self.P[:, :d, d] = self.P[:, :d, 0]
            self.M[:, :d, d] = self.M[:, :d, 0]
        return t_work / max(n_work, 1)

    def tables(self):
        a, rows = self.acc, []
        for l in range(self.L):
            for d in range(self.D + 1):
                n = a["n"][l, d]
                if n == 0:
                    continue
                row = dict(layer=l, distance=d, n=int(n))
                for key in ("overlap", "recall", "precision", "fullcov", "budget", "union"):
                    row[key] = a[key][l, d] / n
                if self.cfg.budget == "adaptive":
                    row["k_hat"] = a["khat"][l, d] / n
                    row["delta_star"] = a["delta"][l, d] / n
                    row["extra_budget"] = a["extra"][l, d] / n
                rows.append(row)
        prow = []
        for l in range(self.L):
            for d in range(self.D + 1):
                w = self.chooser.weights(l, d)
                for i, name in enumerate(PREDICTORS):
                    if self.pn[l, d, i]:
                        prow.append(dict(layer=l, distance=d, predictor=name,
                                         recall=self.pacc[l, d, i] / self.pn[l, d, i], weight=w[i]))
        return pd.DataFrame(rows), pd.DataFrame(prow)


def build_stream(wl, cfg: PredictorConfig) -> Stream:
    runner = _Runner(wl, cfg)
    sec = runner.run()
    metrics, prec = runner.tables()
    L, D, N = runner.L, cfg.D, runner.N
    bits = dict(temporal=runner.temporal.state_bits() if runner.on[0] else 0,
                cross=runner.cross.state_bits() if runner.on[1] else 0,
                pop=runner.pop.state_bits() if runner.on[2] else 0,
                calibration=(2 * L + L * (D + 1)) * N * 2 * 16,
                chooser=L * (D + 1) * 3 * 16,
                budget=L * (D + 1) * 64 * 8)
    return Stream(cfg, runner.P, runner.M, metrics, prec, sec, bits, runner.temporal.W.copy())


def summarize(metrics: pd.DataFrame, distance=None):
    """Average over layers (weighted by count) for one distance, or the whole table."""
    m = metrics if distance is None else metrics[metrics.distance == distance]
    w = m["n"]
    return {key: float((m[key] * w).sum() / w.sum()) for key in
            ("overlap", "recall", "precision", "fullcov", "budget", "union")}
