"""Ablation ladder: what each borrowed idea contributes (Deliverables 2 and 3).

Rungs (experiments/common.py LADDER), each adding one component to the one before:
   1 on-demand, no cache (the Q2 baseline)          6 + adaptive budget (delta, link cap, off-switch)
   2 + HBM cache, LRU                               7 + multi-layer lookahead + demand preemption
   3 + temporal predictor (top-k, 1 layer ahead)    8 + predicted-need eviction + miss overlap
   4 + cross-layer + popularity (equal weights)     9 + expected-value gate = full design
   5 + chooser                                     10 oracle predictor (upper bound)
                                                   11 oracle + Belady eviction
plus reference rows: LRU and LFU with and without miss overlap (no prediction), prediction used only
to choose evictions (no prefetch), and the full design / oracle with no retention (evict after use),
which isolates prefetching from caching.

    python -m experiments.e3_ablation [--models ...] [--gpu A100]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import BATCHES, EXTRA, LADDER, LINKS, available_models, out_dir, run_many, sim_kwargs

RATIOS = [0.25, 0.5]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--procs", type=int, default=28)
    args = ap.parse_args(argv)
    models = args.models or available_models()
    out = out_dir("e3_ablation")
    jobs = []
    for m in models:
        for b in BATCHES:
            for link in LINKS:
                for ratio in RATIOS:
                    for rung, pred, over, retention in LADDER + EXTRA:
                        r = ratio if retention else 0.0
                        tag = dict(rung=rung, ratio_setting=ratio)
                        jobs.append((tag, m, b, pred, sim_kwargs(args.gpu, link, r, over), None))
    df = run_many(jobs, args.procs)
    df.drop(columns=["layer_stall_ms", "layer_hit_rate"], errors="ignore").to_csv(
        os.path.join(out, "ablation.csv"), index=False)
    if "error" in df and df["error"].notna().any():
        print("FAILED RUNS:\n", df[df["error"].notna()][["model", "batch", "rung", "error"]].to_string())
    report(df, out)


def report(df, out):
    df = df[df.get("error").isna()] if "error" in df else df
    order = [r for r, *_ in LADDER + EXTRA]
    pd.set_option("display.width", 220)
    for m in df.model.unique():
        for ratio in sorted(df.ratio_setting.unique()):
            sub = df[(df.model == m) & (df.ratio_setting == ratio)]
            t = sub.pivot_table(index="rung", columns=["batch", "link"], values="token_latency_ms").reindex(order)
            base = t.loc["1 on-demand, no cache"]
            print(f"\n{m}, cache ratio {ratio}: speedup over rung 1 (token latency)")
            print((base / t).round(2).to_string())
    # one compact plot per model: speedup by rung, one line per link, batch 1, ratio 0.25
    for m in df.model.unique():
        fig, axes = plt.subplots(1, len(BATCHES), figsize=(16, 4.5), sharey=False)
        for ax, b in zip(axes, BATCHES):
            sub = df[(df.model == m) & (df.batch == b) & (df.ratio_setting == 0.25)]
            for link in LINKS:
                s = sub[sub.link == link].set_index("rung").reindex([r for r, *_ in LADDER])
                base = s["token_latency_ms"].iloc[0]
                ax.plot(range(1, len(s) + 1), base / s["token_latency_ms"], marker="o", label=link)
            ax.set_xticks(range(1, len(LADDER) + 1))
            ax.set_xlabel("ladder rung")
            ax.set_ylabel("speedup over on-demand")
            ax.set_title(f"{m}, batch {b}, cache 25%")
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"ladder_{m}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
