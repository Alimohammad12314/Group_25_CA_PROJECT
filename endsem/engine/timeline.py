"""Expert-level decode timeline: compute stream + DRAM->HBM link + HBM expert cache + prefetcher.

Per decode step, per MoE layer l (global index g = step * L + l):

  1. non-expert work (attention, router, dense/shared FFN): t_nonexpert, slowed by HBM contention
     while the link is copying (SSD-offloading model: compute sees HBM BW - link BW)
  2. router result: the batch needs union U of experts. Each is a hit (resident and arrived), late
     (prefetch still in flight; promoted to demand priority) or a miss (demand fetch, evicting a victim)
  3. prefetcher: predicts layers l+1 .. l+d (crossing into the next token) and queues transfers
  4. expert execution: strict = wait for all of U, then run (the Q2 on-demand model);
     overlap = run arrived experts first while missing ones load (APEX's correctness-preserving mode)
  5. bookkeeping: recency/frequency, unpin, evict-after-use when there is no retention, drop stale
     prefetches whose target layer has passed

With cache ratio 0, no prefetch and strict execution this reproduces Q2's on-demand step latency
(the validation gate, experiments/e0_validation_gate.py).
"""
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .costs import CostModel
from .energy import energy_breakdown
from .hbm import ExpertCache
from .link import DEMAND, SPEC, Link, Transfer


@dataclass
class SimConfig:
    gpu: str = "A100"
    link: str = "C2C-900"
    cache_ratio: float = 0.5         # share of all routed experts HBM can hold; 0 = no retention
    eviction: str = "lru"            # lru | lfu | need | belady
    prefetch: bool = False
    lookahead: str = "fixed"         # fixed | adaptive
    lookahead_cap: int = 1           # fixed: the distance; adaptive: the largest distance
    link_mode: str = "fifo"          # fifo | prio (demand preempts at chunk boundaries)
    chunk_MB: float = 16.0
    link_cap: bool = False           # cap prefetches to what the link can deliver in the window
    stale_cancel: bool = False
    miss_overlap: bool = False
    early_bonus: float = 0.0
    early_layers: int = 4
    need_horizon: int = 0            # 'need' eviction, rate mode: tokens ahead the score looks (0 = next only)
    need_mode: str = "rate"          # 'need' eviction: rate (usage rate since load) | recency (LRU blend)
    need_halflife: float = 4.0       # recency mode: steps after which an unused expert's reuse odds halve
    contention: bool = True
    value_gate: bool = False         # issue a prefetch only if its expected saving beats its expected cost
    pred_latency_us: float = 0.0     # time to run the predictor once per layer
    pred_on_critical_path: bool = False


class Timeline:
    def __init__(self, wl, needed, stream, cfg: SimConfig):
        self.wl, self.needed, self.stream, self.cfg = wl, needed, stream, cfg
        self.S, self.B, self.L, self.N = wl.S, wl.B, wl.L, wl.num_experts
        self.cost = CostModel(cfg.gpu, wl.model, cfg.link, wl.B)
        c = self.cost
        self.retention = cfg.cache_ratio > 0
        total = self.N * self.L
        max_union = max(len(u[0]) for row in needed for u in row)
        if self.retention:
            cap = int(round(cfg.cache_ratio * total))
            if cap < max_union + 1:      # one layer's experts must fit; prefetched slots can be reclaimed
                raise ValueError(f"cache of {cap} experts cannot hold one layer of {max_union}")
        else:
            cap = total
        self.cache = ExpertCache(cap, self.L, self.N)
        self.link = Link(c.link_bw, cfg.chunk_MB * 1e6, cfg.link_mode)
        self.f = c.contention_factor if cfg.contention else 1.0
        self.t_pred = cfg.pred_latency_us * 1e-6
        self.D = stream.P.shape[2] - 1 if stream is not None else 0
        self.t_layer_est = c.t_nonexpert_layer + c.t_exec(1) * c.k
        self.link_util = 0.0            # EMA of link busy fraction per layer (for the value gate)
        self._util_mark = (0.0, 0.0)
        self.issued = defaultdict(list)
        self._pcur_key = None
        self._nxt = self._next_use_table() if cfg.eviction == "belady" else None
        self.m = defaultdict(float)
        self.layer_stall = np.zeros(self.L)
        self.layer_hits = np.zeros(self.L)
        self.layer_needs = np.zeros(self.L)

    # --- compute with HBM contention -------------------------------------------------------
    def _compute(self, t, work):
        if work <= 0:
            return t
        if self.f == 1.0:
            return t + work
        # Bring the link up to now first: its clock can lag behind compute while holding queued work
        # it would already have moved, and that backlog must not be counted as still ahead of us.
        self.link.run_until(t)
        tb = self.link.busy_until(t)
        if tb <= t:
            return t + work
        span = tb - t
        if work * self.f <= span:
            return t + work * self.f
        return tb + (work - span / self.f)

    # --- eviction scores -------------------------------------------------------------------
    def _next_use_table(self):
        S, L, N = self.S, self.L, self.N
        big = np.iinfo(np.int32).max // (L + 1)
        nxt = np.full((S + 1, L, N), big, np.int32)
        for s in range(S - 1, -1, -1):
            nxt[s] = nxt[s + 1]
            for l in range(L):
                nxt[s, l, self.needed[s][l][0]] = s
        return nxt

    def _pcur(self, s, j):
        """Predicted need of every (layer, expert) at its next possible use, seen from (s, j)."""
        if self._pcur_key == (s, j):
            return self._pcur_val
        L, D = self.L, self.D
        rows = np.arange(L)
        ahead = rows > j
        d_idx = np.where(ahead, np.minimum(rows - j, D), 0)
        s_idx = np.where(ahead, s, min(s + 1, self.S - 1))
        P = self.stream.P[s_idx, rows, d_idx].astype(np.float32)
        if s + 1 >= self.S:
            P[~ahead] = 0.0
        dist = np.where(ahead, rows - j, L - j + rows)
        P *= (0.5 ** (dist / L))[:, None]
        P[: self.cfg.early_layers] += self.cfg.early_bonus
        self._pcur_key, self._pcur_val = (s, j), P
        return P

    def _scores(self, slots, s, j, g):
        c = self.cache
        pol = self.cfg.eviction
        if pol == "lru":
            return np.maximum(c.last_use[slots], c.load_g[slots]).astype(float)
        if pol == "lfu":
            return c.freq[slots] * 1e9 + np.maximum(c.last_use[slots], c.load_g[slots])
        keys = c.key[slots]
        lay, exp = keys // self.N, keys % self.N
        if pol == "need":
            p = self._pcur(s, j)[lay, exp]
            H = self.cfg.need_horizon
            if self.cfg.need_mode in ("recency", "both"):
                # P(reuse) blends the prediction with recency: with p = 0 this is exactly LRU order,
                # and a predicted expert is protected on top of that.
                since = (g - np.maximum(c.last_use[slots], c.load_g[slots])) / self.L
                rho = 0.5 ** (since / self.cfg.need_halflife)
                p = 1.0 - (1.0 - np.clip(p, 0.0, 1.0)) * (1.0 - rho)
                if self.cfg.need_mode == "both" and H > 0:
                    age = np.maximum((g - c.load_g[slots]) / self.L, 0.0)
                    u = np.clip(c.freq[slots] / (age + H), 0.0, 1.0)
                    p = 1.0 - (1.0 - p) * (1.0 - u) ** H
            elif H > 0:
                # Retention outlives one token: P(used within H tokens) from the next-token prediction
                # and the slot's own usage rate per step since it was loaded.
                # smoothed rate: an expert used once right after loading must not look certain (u = 1)
                age = np.maximum((g - c.load_g[slots]) / self.L, 0.0)
                u = np.clip(c.freq[slots] / (age + H), 0.0, 1.0)
                p = 1.0 - (1.0 - np.clip(p, 0.0, 1.0)) * (1.0 - u) ** H
            return p + 1e-6 * c.freq[slots]
        # belady: farther next use = lower score
        step = np.where(lay > j, self._nxt[s, lay, exp], self._nxt[min(s + 1, self.S), lay, exp])
        return -(step.astype(np.int64) * self.L + lay).astype(float)

    def _cand_score(self, l, e, s, j, g, target_g):
        if self.cfg.eviction == "need":
            return float(self._pcur(s, j)[l, e])
        if self.cfg.eviction == "belady":
            step = self._nxt[s, l, e] if l > j else self._nxt[min(s + 1, self.S), l, e]
            return -float(int(step) * self.L + l)
        return None

    def _evict(self, slot, g):
        c = self.cache
        tr = c.transfer[slot]
        if tr is not None and tr.finish is None:
            self.link.cancel(tr)
            self.m["cancelled_bytes"] += tr.done
        if c.prefetched[slot] and not c.used[slot]:
            self.m["prefetch_wasted"] += 1
        c.remove(slot)

    def _make_room(self, n, s, j, g, t, mode, cand_scores=None):
        """Free n slots. Returns how many were freed (prefetch mode may refuse some)."""
        c = self.cache
        free = len(c.free)
        if free >= n:
            return n
        need = n - free
        c.refresh(t)
        idx = c.evictable(g, t)
        if mode == "demand" and len(idx) < need:
            extra = c.fallback(g)
            idx = np.concatenate([idx, extra[~np.isin(extra, idx)]])
        if len(idx) == 0:
            return free
        sc = self._scores(idx, s, j, g)
        k = min(need, len(idx))
        if k < len(idx):          # only the k lowest scores are needed: partial sort
            part = np.argpartition(sc, k - 1)[:k]
            sel = part[np.argsort(sc[part], kind="stable")]
        else:
            sel = np.argsort(sc, kind="stable")
        order, sc_sorted = idx[sel], sc[sel]
        if mode == "prefetch" and cand_scores is not None:
            # Evict a victim only if it is predicted to be needed less than the candidate.
            cs = sorted(cand_scores, reverse=True)
            take = 0
            while take < min(need, len(order)) and sc_sorted[take] < cs[take]:
                take += 1
        else:
            take = min(need, len(order))
        for slot in order[:take]:
            self._evict(int(slot), g)
        return free + take

    # --- prefetching -----------------------------------------------------------------------
    def _lookahead(self, s, j):
        cfg = self.cfg
        if cfg.lookahead == "fixed":
            return cfg.lookahead_cap
        jt = j + 1
        st, d = (s, 1) if jt < self.L else (s + 1, 0)
        jt %= self.L
        if st >= self.S:
            return 1
        P, M = self.stream.P[st, jt, d], int(self.stream.M[st, jt, d])
        order = np.argsort(-P.astype(np.float32), kind="stable")[:M]
        miss = sum(1 for e in order if (jt * self.N + int(e)) not in self.cache.index)
        load = miss * self.cost.t_load_expert + self.link.pending_bytes / self.link.bw
        return int(np.clip(math.ceil(load / max(self.t_layer_est, 1e-12)), 1, cfg.lookahead_cap))

    def _issue(self, s, j, g, t_issue):
        cfg, c, cost = self.cfg, self.cache, self.cost
        depth = self._lookahead(s, j)
        self.m["lookahead_sum"] += depth
        for d in range(1, depth + 1):
            jt = j + d
            if jt < self.L:
                st, dd = s, min(d, self.D)
            else:
                st, jt, dd = s + 1, jt - self.L, 0
            if st >= self.S:
                break
            gt = st * self.L + jt
            P = self.stream.P[st, jt, dd].astype(np.float32)
            M = int(self.stream.M[st, jt, dd])
            order = np.argsort(-P, kind="stable")[:M]
            new = []
            for e in order:
                key = jt * self.N + int(e)
                slot = c.index.get(key)
                if slot is not None:
                    c.pin[slot] = max(c.pin[slot], gt)
                else:
                    new.append(int(e))
            if cfg.value_gate and new:
                # Expected value of one speculative transfer, in units of one expert's load time:
                #   saving if needed  ~ 1          (the miss it avoids)
                #   cost if not       ~ (1 - 1/f)  (compute slowed by HBM contention during the copy)
                #                     + rho        (chance the busy link delays a later demand miss)
                # => prefetch only experts with predicted need p >= c / (1 + c).
                c_waste = (1.0 - 1.0 / self.f) + self.link_util
                theta = c_waste / (1.0 + c_waste)
                new = [e for e in new if P[e] >= theta]
            if cfg.link_cap and new:
                window = (gt - g) * self.t_layer_est - self.link.pending_bytes / self.link.bw
                cap = math.ceil(max(window, 0.0) / cost.t_load_expert)
                new = new[:cap]
            if not new:
                continue
            cand = None
            if cfg.eviction in ("need", "belady") and self.retention:
                cand = [self._cand_score(jt, e, s, j, g, gt) for e in new]
            room = self._make_room(len(new), s, j, g, t_issue, "prefetch", cand)
            if cand is not None and room < len(new):          # keep the most needed candidates
                keep = np.argsort(cand)[::-1][:room]
                new = [new[i] for i in sorted(keep)]
            for e in new[:room]:
                key = jt * self.N + e
                tr = Transfer(key, cost.expert_bytes, SPEC, t_issue, gt)
                self.link.enqueue(tr)
                slot = c.insert(key, tr, True, gt, g)
                self.issued[gt].append((slot, tr, key))
                self.m["prefetch_issued"] += 1

    # --- main loop -------------------------------------------------------------------------
    def run(self):
        cfg, cost, c, link = self.cfg, self.cost, self.cache, self.link
        L, N = self.L, self.N
        t = 0.0
        step_times = np.zeros(self.S)
        m = self.m
        for s in range(self.S):
            t_step = t
            for j in range(L):
                g = s * L + j
                pred_on = cfg.prefetch and cfg.pred_on_critical_path
                t = self._compute(t, cost.t_nonexpert_layer + (self.t_pred if pred_on else 0.0))
                t_r = t
                link.run_until(t_r)
                experts, counts = self.needed[s][j]
                slots, missing = [], []
                for i, e in enumerate(experts):
                    key = j * N + int(e)
                    slot = c.index.get(key)
                    if slot is not None:
                        tr = c.transfer[slot]
                        if tr is None or (tr.finish is not None and tr.finish <= t_r):
                            m["hits"] += 1
                            self.layer_hits[j] += 1
                            if c.prefetched[slot] and not c.used[slot]:
                                m["prefetch_timely"] += 1
                        else:
                            m["late"] += 1
                            if c.prefetched[slot] and not c.used[slot]:
                                m["prefetch_late"] += 1
                            link.promote(tr)
                        c.pin[slot] = max(c.pin[slot], g)
                    else:
                        missing.append((i, key))
                    slots.append(slot)
                if missing:
                    # victim scores cannot change within a layer, so pick all victims at once
                    m["misses"] += len(missing)
                    if self._make_room(len(missing), s, j, g, t_r, "demand") < len(missing):
                        raise RuntimeError("no slot for a demand miss")
                    for i, key in missing:
                        tr = Transfer(key, cost.expert_bytes, DEMAND, t_r, g)
                        link.enqueue(tr)
                        slots[i] = c.insert(key, tr, False, g, g)
                m["needs"] += len(experts)
                self.layer_needs[j] += len(experts)

                if cfg.prefetch:
                    self._issue(s, j, g, t_r + (0.0 if cfg.pred_on_critical_path else self.t_pred))

                ready = []
                for slot in slots:
                    tr = c.transfer[slot]
                    ready.append(t_r if tr is None else max(t_r, link.finish_time(tr)))
                if cfg.miss_overlap:
                    # run experts in arrival order; stall = time compute sat idle waiting for weights
                    tc, stall = t_r, 0.0
                    for r, n in sorted(zip(ready, counts)):
                        if r > tc:
                            stall += r - tc
                            tc = r
                        tc = self._compute(tc, cost.t_exec(int(n)))
                else:
                    r = max(ready) if ready else t_r
                    tc = self._compute(r, sum(cost.t_exec(int(n)) for n in counts))
                    stall = r - t_r
                m["stall"] += stall
                self.layer_stall[j] += stall
                m["exec_experts"] += len(experts)
                m["exec_tokens"] += int(counts.sum())
                layer_compute = cost.t_nonexpert_layer + sum(cost.t_exec(int(n)) for n in counts)
                self.t_layer_est += 0.1 * (layer_compute - self.t_layer_est)
                t = tc
                busy0, t0 = self._util_mark
                if t > t0:
                    util = min(1.0, (link.busy_time - busy0) / (t - t0))
                    self.link_util += 0.1 * (util - self.link_util)
                self._util_mark = (link.busy_time, t)

                for slot in slots:
                    if c.prefetched[slot] and not c.used[slot]:
                        m["prefetch_used"] += 1
                        m["useful_prefetch_bytes"] += cost.expert_bytes
                    c.used[slot] = True
                    c.last_use[slot] = g
                    c.freq[slot] += 1
                    tr = c.transfer[slot]
                    if tr is not None and tr.finish is not None:
                        c.ready_at[slot] = tr.finish
                        c.inflight.discard(slot)
                self._end_layer(s, j, g, experts)
            step_times[s] = t - t_step
        return self._results(t, step_times)

    def _end_layer(self, s, j, g, experts):
        c, cfg = self.cache, self.cfg
        issued = self.issued.pop(g, [])
        if cfg.stale_cancel:
            needed_keys = set(int(j * self.N + e) for e in experts)
            for slot, tr, key in issued:
                if key in needed_keys or tr.cancelled or tr.finish is not None or tr.done > 0:
                    continue
                if c.index.get(key) == slot and c.pin[slot] <= g:
                    self.m["stale_cancelled"] += 1
                    self._evict(slot, g)
        if not self.retention:
            # evict after use: everything whose last target has passed
            for slot in np.nonzero((c.state == 1) & (c.pin <= g))[0]:
                self._evict(int(slot), g)

    def _results(self, t_total, step_times):
        m, cost, cfg = self.m, self.cost, self.cfg
        S, B = self.S, self.B
        for slot in list(self.cache.index.values()):     # prefetches never used by the end
            if self.cache.prefetched[slot] and not self.cache.used[slot]:
                m["prefetch_wasted"] += 1
        bytes_moved = self.link.bytes_moved
        useful = (m["misses"] * cost.expert_bytes + m["useful_prefetch_bytes"])
        res = dict(model=self.wl.model, batch=B, steps=S, **{k: v for k, v in asdict(cfg).items()},
                   cache_slots=self.cache.cap if self.retention else 0,
                   token_latency_ms=1e3 * t_total / S,
                   tokens_per_s=S * B / t_total,
                   stall_ms_per_token=1e3 * m["stall"] / S,
                   stall_fraction=m["stall"] / t_total,
                   hit_rate=m["hits"] / max(m["needs"], 1),
                   late_rate=m["late"] / max(m["needs"], 1),
                   miss_rate=m["misses"] / max(m["needs"], 1),
                   prefetch_issued_per_token=m["prefetch_issued"] / S,
                   prefetch_accuracy=m["prefetch_used"] / max(m["prefetch_issued"], 1),
                   prefetch_wasted_frac=m["prefetch_wasted"] / max(m["prefetch_issued"], 1),
                   prefetch_timely_frac=m["prefetch_timely"] / max(m["prefetch_timely"] + m["prefetch_late"], 1),
                   stale_cancelled_per_token=m["stale_cancelled"] / S,
                   GB_per_token=bytes_moved / S / 1e9,
                   useful_GB_per_token=min(useful, bytes_moved) / S / 1e9,
                   wasted_GB_per_token=max(bytes_moved - useful, 0.0) / S / 1e9,
                   link_utilization=self.link.busy_time / t_total,
                   mean_lookahead=m["lookahead_sum"] / max(S * self.L, 1) if cfg.prefetch else 0.0,
                   p95_token_latency_ms=1e3 * float(np.percentile(step_times, 95)))
        res.update(energy_breakdown(cost, S, B, t_total, m["stall"], bytes_moved,
                                    m["exec_experts"], m["exec_tokens"]))
        res["layer_stall_ms"] = (1e3 * self.layer_stall / S).tolist()
        res["layer_hit_rate"] = (self.layer_hits / np.maximum(self.layer_needs, 1)).tolist()
        return res


def simulate(wl, needed, stream, cfg: SimConfig):
    return Timeline(wl, needed, stream, cfg).run()
