"""Hidden-state predictor (measured on the H100) vs the ID-only predictor, on the same tokens.

The public traces hold expert IDs only, which caps next-layer accuracy near 46 % for Mixtral even for a lookup
table fitted to the test data. On the H100 the real model runs, so the runtime can also apply layer (l+d)'s own
router to the hidden state at layer l (training-free; GPU.md Phase B''). It exports, per decode step s, sequence
b, target layer l and distance d, the router probabilities q[s, b, l, d, :] it predicted, plus the routing A the
model actually chose:

    gpu/results/hs_pred_b{B}_ctx{ctx}.npz   q: float16 [S, B, L, D+1, N]   A: int16 [S, B, L, k]
                                            doc_ids: [B * groups], tokens_per_request, offset (= ctx)

This script turns q into an engine prediction stream (batch union P = 1 - prod_b (1 - q_b), budget M = expected
number of needed experts), scores it exactly like e2 (overlap, recall, precision, full coverage per distance) next
to the ID-only stream on the same workload, and runs the timeline policies with both.

    python -m experiments.e7_hidden_state [--glob 'gpu/results/hs_pred_*.npz'] [--gpu H100] [--calibration ...]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from engine.timeline import SimConfig, simulate
from predictors.stream import Stream, build_stream, summarize
from traces.workload import Workload

from .common import FULL_SIM, PRED, ROOT, out_dir

LINKS = ["PCIe4", "PCIe5", "C2C-900"]
POLICIES = [
    ("on-demand", None, 0.0, dict()),
    ("LRU + overlap", None, 0.25, dict(eviction="lru", miss_overlap=True)),
    ("predict-evict only", "pred", 0.25, dict(eviction="need", early_bonus=0.05, need_horizon=4, miss_overlap=True)),
    ("full design", "pred", 0.25, FULL_SIM),
    ("oracle", "oracle", 0.25, FULL_SIM),
]


def workload_from(A, tokens_per_request, model="mixtral", num_experts=8):
    S, B, L, k = A.shape
    gs = np.zeros(S, bool)
    gs[::tokens_per_request] = True
    return Workload(model=model, source="live:H100", num_experts=num_experts, top_k=k, A=A.astype(np.int16),
                    group_start=gs, warm=np.full((1, B, 1, L, k), -1, np.int16), tokens_per_request=tokens_per_request)


def stream_from_q(q, k, name="hidden_state"):
    """Per-sequence router probabilities -> engine stream (batch union, expected-need budget)."""
    q = np.clip(q.astype(np.float32), 0.0, 1.0)
    P = 1.0 - np.prod(1.0 - q, axis=1)                                  # [S, L, D+1, N]
    N = P.shape[-1]
    M = np.clip(np.ceil(P.sum(-1)), k, N).astype(np.int16)
    cfg = PRED["full"].__class__(name=name, D=P.shape[2] - 1)
    return Stream(cfg, P.astype(np.float16), M, pd.DataFrame(), pd.DataFrame(), 0.0, {}, None)


def score(stream, wl):
    """Same definitions as predictors/stream.py: overlap = share of the batch's experts U in the top-|U| scores."""
    S, L, D1, N = stream.P.shape
    rows = []
    needed = wl.needed()
    for d in range(D1):
        acc = dict(overlap=0.0, recall=0.0, precision=0.0, fullcov=0.0, budget=0.0, union=0.0)
        n = 0
        for s in range(1 if d == 0 else 0, S):
            for l in range(L):
                if d > 0 and l - d < 0:
                    continue
                U = needed[s][l][0]
                sc = stream.P[s, l, d].astype(np.float32)
                M = int(stream.M[s, l, d])
                pos = np.empty(N, np.int64)
                pos[np.argsort(-sc, kind="stable")] = np.arange(N)
                pu = pos[U]
                nU, inU, inM = len(U), int((pu < len(U)).sum()), int((pu < M).sum())
                acc["overlap"] += inU / nU
                acc["recall"] += inM / nU
                acc["precision"] += inM / M
                acc["fullcov"] += float(inM == nU)
                acc["budget"] += M
                acc["union"] += nU
                n += 1
        rows.append(dict(distance=d, n=n, **{k_: v / max(n, 1) for k_, v in acc.items()}))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(ROOT, "gpu", "results", "hs_pred_*.npz"))
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--calibration", default="")
    args = ap.parse_args(argv)
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no hidden-state prediction files match {args.glob} (run GPU.md Phase B'' first)")
    out = out_dir("e7_hidden_state")
    acc_rows, sim_rows = [], []
    for f in files:
        z = np.load(f)
        A, q = z["A"], z["q"]
        ctx = int(z["offset"]) if "offset" in z else 1024
        tpr = int(z["tokens_per_request"]) if "tokens_per_request" in z else A.shape[0]
        wl = workload_from(A, tpr)
        B = wl.B
        hs = stream_from_q(q, wl.top_k)
        streams = {"hidden-state (H100)": hs, "ID-only (ours)": build_stream(wl, PRED["full"])}
        streams["oracle"] = build_stream(wl, PRED["oracle"])
        for name in ("hidden-state (H100)", "ID-only (ours)"):
            t = score(streams[name], wl) if name.startswith("hidden") else \
                pd.DataFrame([dict(distance=d, **summarize(streams[name].metrics, d)) for d in range(5)])
            t.insert(0, "predictor", name)
            t.insert(0, "batch", B)
            t.insert(1, "context", ctx)
            acc_rows.append(t)
        needed = wl.needed()
        for link in LINKS:
            for pol, pred, ratio, kw in POLICIES:
                for sname in (["hidden-state (H100)", "ID-only (ours)"] if pred == "pred" else [None]):
                    st = streams["oracle"] if pred == "oracle" else (streams[sname] if sname else None)
                    cfg = SimConfig(gpu=args.gpu, link=link, cache_ratio=ratio, context=ctx,
                                    calibration=args.calibration, **kw)
                    r = simulate(wl, needed, st, cfg)
                    r = {k_: v for k_, v in r.items() if not isinstance(v, list)}
                    r.update(policy=pol if sname is None else f"{pol} [{sname}]", context=ctx)
                    sim_rows.append(r)
    acc = pd.concat(acc_rows, ignore_index=True)
    acc.to_csv(os.path.join(out, "accuracy.csv"), index=False)
    sim = pd.DataFrame(sim_rows)
    base = sim[sim.policy == "on-demand"].set_index(["batch", "link", "context"]).token_latency_ms
    sim = sim.join(base.rename("ondemand_ms"), on=["batch", "link", "context"])
    sim["speedup"] = sim.ondemand_ms / sim.token_latency_ms
    sim.to_csv(os.path.join(out, "policies.csv"), index=False)
    pd.set_option("display.width", 220)
    print("overlap by distance (share of the batch's experts named in advance)")
    print(acc.pivot_table(index=["batch", "context", "predictor"], columns="distance", values="overlap").round(3))
    print("\nprecision / recall with the budget, distance 1")
    print(acc[acc.distance == 1][["batch", "context", "predictor", "recall", "precision", "budget", "union"]]
          .round(3).to_string(index=False))
    print("\nspeedup over on-demand (25 % cache)")
    print(sim.pivot_table(index=["batch", "context", "link"], columns="policy", values="speedup").round(2))


if __name__ == "__main__":
    main()
