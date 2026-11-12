"""Deliverable 1: offline accuracy of every predictor on the real traces.

For each model and batch size, every predictor configuration is run over the decode workload and
scored per layer and per lookahead distance d (d = 0: predicted at the previous token's end).

  overlap      top-|U| predicted vs the |U| experts actually needed (budget-independent; at batch 1
               this is the usual top-k overlap accuracy)
  recall       share of needed experts inside the prefetch budget M
  full cover   share of (token, layer) where every needed expert was inside M (APEX: 87.5 % overlap
               can still stall every token)
  precision    share of the M prefetched experts that were needed (1 - offline waste)
  batch union  the above are all computed on the batch union, ExpertFlow's batch-level metric

Also: the counter-width ablation (2/4/8 bits), the adaptive delta per layer (APEX Figure 10 check),
per-predictor recall and chooser weights per layer, predictor state size and runtime.

    python -m experiments.e2_predictor_accuracy [--models mixtral ...]
"""
import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from predictors.stream import summarize

from .common import BATCHES, PRED, available_models, get_stream, out_dir, prebuild_streams

CONFIGS = ["prev_token", "temporal", "cross", "popularity", "equal_topk", "chooser_topk", "full", "oracle"]
BITS = [2, 4, 8]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--procs", type=int, default=28)
    args = ap.parse_args(argv)
    models = args.models or available_models()
    out = out_dir("e2_predictor_accuracy")

    bit_cfgs = [replace(PRED["equal_topk"], name=f"bits{b}", counter_bits=b) for b in BITS]
    jobs = [(m, b, PRED[c]) for m in models for b in BATCHES for c in CONFIGS]
    jobs += [(m, 1, c) for m in models for c in bit_cfgs]
    prebuild_streams(jobs, args.procs)

    summary, layers, preds, bits_rows, cost_rows = [], [], [], [], []
    for m in models:
        for b in BATCHES:
            for c in CONFIGS:
                st = get_stream(m, b, PRED[c])
                for d in sorted(st.metrics.distance.unique()):
                    summary.append(dict(model=m, batch=b, predictor=c, distance=d, **summarize(st.metrics, d)))
                lm = st.metrics.copy()
                lm.insert(0, "predictor", c)
                lm.insert(0, "batch", b)
                lm.insert(0, "model", m)
                layers.append(lm)
                if c == "full":
                    pr = st.predictor_recall.copy()
                    pr.insert(0, "batch", b)
                    pr.insert(0, "model", m)
                    preds.append(pr)
                    cost_rows.append(dict(model=m, batch=b, us_per_layer_event_python=st.seconds_per_layer_event * 1e6,
                                          **{f"{k}_KiB": v / 8 / 1024 for k, v in st.state_bits.items()},
                                          total_KiB=sum(st.state_bits.values()) / 8 / 1024))
        for c in bit_cfgs:
            st = get_stream(m, 1, c)
            for d in (1, 2, 4):
                bits_rows.append(dict(model=m, counter_bits=c.counter_bits, distance=d, **summarize(st.metrics, d)))

    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(out, "summary.csv"), index=False)
    pd.concat(layers).to_csv(os.path.join(out, "per_layer.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(out, "per_predictor_per_layer.csv"), index=False)
    pd.DataFrame(bits_rows).to_csv(os.path.join(out, "counter_bits.csv"), index=False)
    pd.DataFrame(cost_rows).to_csv(os.path.join(out, "predictor_cost.csv"), index=False)

    pd.set_option("display.width", 200)
    d1 = S[S.distance == 1].pivot_table(index=["model", "batch"], columns="predictor", values="overlap")
    print("overlap at d = 1 (one layer ahead)\n", d1[CONFIGS].round(3).to_string())
    full = S[(S.predictor == "full")].pivot_table(index=["model", "batch"], columns="distance",
                                                  values=["recall", "fullcov", "precision", "budget", "union"])
    print("\nfull predictor with adaptive budget, by distance\n", full.round(3).to_string())
    print("\ncounter bits (batch 1)\n", pd.DataFrame(bits_rows).pivot_table(
        index=["model", "distance"], columns="counter_bits", values="overlap").round(3).to_string())
    print("\npredictor cost\n", pd.DataFrame(cost_rows).round(1).to_string(index=False))
    plots(S, pd.concat(layers), pd.concat(preds), out)


def plots(S, L, PR, out):
    for m in S.model.unique():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, b in zip(axes, BATCHES):
            sub = L[(L.model == m) & (L.batch == b) & (L.distance == 1)]
            for c in ["prev_token", "cross", "popularity", "full", "oracle"]:
                s = sub[sub.predictor == c]
                ax.plot(s.layer, s.overlap, label=c, marker=".")
            ax.set_title(f"{m}, batch {b}: overlap one layer ahead")
            ax.set_xlabel("MoE layer")
            ax.set_ylabel("overlap (top-|U|)")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"overlap_per_layer_{m}.png"), dpi=130)
        plt.close(fig)

        sub = L[(L.model == m) & (L.predictor == "full") & (L.distance == 1)]
        fig, ax = plt.subplots(figsize=(6, 4))
        for b in BATCHES:
            s = sub[sub.batch == b]
            ax.plot(s.layer, s.extra_budget, label=f"batch {b}", marker=".")
        ax.set_xlabel("MoE layer")
        ax.set_ylabel("extra budget delta (experts)")
        ax.set_title(f"{m}: adaptive delta per layer (APEX Fig. 10 check)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"delta_per_layer_{m}.png"), dpi=130)
        plt.close(fig)

        sub = PR[(PR.model == m) & (PR.batch == 1) & (PR.distance == 1)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for p in sub.predictor.unique():
            s = sub[sub.predictor == p]
            axes[0].plot(s.layer, s.recall, label=p, marker=".")
            axes[1].plot(s.layer, s.weight, label=p, marker=".")
        axes[0].set_title(f"{m}, batch 1: per-predictor top-k recall (d = 1)")
        axes[1].set_title("chooser weight at the end of the run")
        for ax in axes:
            ax.set_xlabel("MoE layer")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"chooser_{m}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
