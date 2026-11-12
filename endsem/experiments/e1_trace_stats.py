"""Plan step 2: the Q4 statistics, rerun on the real traces (and the calibrated Grok-1 stand-in).

Per model and MoE layer: lag-d temporal reuse and cross-layer hit rate at distance 1, 2, 4, each
next to its shuffled baseline, plus popularity entropy and top-20 % share. Observed - shuffled is
the part of the signal a predictor can exploit beyond frequency.

    python -m experiments.e1_trace_stats
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from traces.stats import all_stats

from .common import available_models, get_trace, out_dir


def main():
    out = out_dir("e1_trace_stats")
    rows = []
    for m in available_models():
        tr = get_trace(m)
        print(tr.describe(), flush=True)
        rows += all_stats(tr, max_requests=120, max_tokens=512)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_layer.csv"), index=False)
    cols = ["random", "entropy", "top20_share", "reuse_lag1", "reuse_lag1_shuffled", "reuse_lag4",
            "cross_d1", "cross_d1_shuffled", "cross_d2", "cross_d4"]
    summary = df.groupby("model")[cols].mean()
    summary.to_csv(os.path.join(out, "summary.csv"))
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())

    models = list(df.model.unique())
    fig, axes = plt.subplots(2, len(models), figsize=(4.5 * len(models), 7), squeeze=False)
    for i, m in enumerate(models):
        s = df[df.model == m]
        ax = axes[0, i]
        ax.plot(s.layer, s.reuse_lag1, label="next token, same layer", marker=".")
        ax.plot(s.layer, s.reuse_lag1_shuffled, "--", label="shuffled")
        ax.axhline(s.random.iloc[0], color="grey", lw=0.8, label="random k/N")
        ax.set_title(f"{m}: temporal reuse")
        ax = axes[1, i]
        for d in (1, 2, 4):
            ax.plot(s.layer, s[f"cross_d{d}"], label=f"layer l -> l+{d}", marker=".")
        ax.plot(s.layer, s.cross_d1_shuffled, "--", label="shuffled (d=1)")
        ax.axhline(s.random.iloc[0], color="grey", lw=0.8)
        ax.set_title(f"{m}: cross-layer hit rate")
        for ax in axes[:, i]:
            ax.set_xlabel("MoE layer")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "predictability_per_layer.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
