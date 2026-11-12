"""Sensitivity to context length (Deliverable: "multiple batch sizes and context lengths").

Context (KV-cache length) changes two things in the timeline, both taken from LLMSimulator:
  * non-expert time and energy per layer: attention reads the whole KV cache every decode step
    (experiments/ctx_base_runs.py reruns Q2's base pass at 256 ... 4096 tokens)
  * HBM left for the expert cache: the KV cache of every sequence occupies HBM ("physical" cache size)
Routing comes from the same real requests at every context (the traces are not long enough to replay
4096-token positions), so the sweep isolates the memory-system effect of context.

  model    all four          batch   1, 4, 16          context  256, 512, 1024, 2048, 4096
  link     PCIe 4.0, NVLink-C2C 900                    cache    25 % of experts, and the physical share
  policy   on-demand, LRU, LRU + overlap, predict-evict only, full design, oracle

    python -m experiments.ctx_base_runs      # once: LLMSimulator base pass at every context
    python -m experiments.e6_context [--models ...]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from engine.costs import CostModel

from .common import BATCHES, FULL_SIM, available_models, out_dir, run_many, sim_kwargs

CONTEXTS = [256, 512, 1024, 2048, 4096]
LINKS = ["PCIe4", "C2C-900"]
POLICIES = {
    "LRU": (None, dict(eviction="lru")),
    "LRU + overlap": (None, dict(eviction="lru", miss_overlap=True)),
    "predict-evict only": ("full", dict(eviction="need", early_bonus=0.05, need_horizon=4, miss_overlap=True)),
    "full design": ("full", FULL_SIM),
    "oracle": ("oracle", FULL_SIM),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--procs", type=int, default=26)
    args = ap.parse_args(argv)
    models = args.models or available_models()
    out = out_dir("e6_context")
    jobs = []
    for m in models:
        for b in BATCHES:
            for ctx in CONTEXTS:
                phys = round(CostModel(args.gpu, m, "C2C-900", b, context=ctx).physical_cache_ratio(), 4)
                for link in LINKS:
                    tag = dict(context=ctx, policy="on-demand", cache="0%", cache_ratio_used=0.0)
                    jobs.append((tag, m, b, None, sim_kwargs(args.gpu, link, 0.0, dict(context=ctx)), None))
                    for ratio, label in ((0.25, "25%"), (phys, "physical")):
                        for name, (pred, over) in POLICIES.items():
                            tag = dict(context=ctx, policy=name, cache=label, cache_ratio_used=ratio)
                            jobs.append((tag, m, b, pred,
                                         sim_kwargs(args.gpu, link, ratio, dict(over, context=ctx)), None))
    print(f"{len(jobs)} timeline runs", flush=True)
    df = run_many(jobs, args.procs)
    df = df.drop(columns=["layer_stall_ms", "layer_hit_rate"], errors="ignore")
    if "error" in df and df["error"].notna().any():
        print("skipped runs (cache smaller than one layer, or other errors):\n",
              df[df["error"].notna()][["model", "batch", "context", "cache", "policy", "error"]]
              .drop_duplicates(["model", "batch", "context", "cache"]).to_string())
    df.to_csv(os.path.join(out, "context.csv"), index=False)
    report(df, out)


def report(df, out):
    if "error" in df:
        df = df[df["error"].isna()].copy()
    base = df[df.policy == "on-demand"].set_index(["model", "batch", "link", "context"]).token_latency_ms
    df = df.join(base.rename("ondemand_ms"), on=["model", "batch", "link", "context"])
    df["speedup"] = df.ondemand_ms / df.token_latency_ms
    lru = df[df.policy == "LRU + overlap"].set_index(["model", "batch", "link", "context", "cache"]).token_latency_ms
    df = df.join(lru.rename("lru_overlap_ms"), on=["model", "batch", "link", "context", "cache"])
    df["gain_over_lru_overlap"] = df.lru_overlap_ms / df.token_latency_ms
    df.to_csv(os.path.join(out, "context_with_speedup.csv"), index=False)
    pd.set_option("display.width", 250)
    for cache in ("25%", "physical"):
        sub = df[(df.cache == cache) & (df.policy == "full design")]
        print(f"\nfull design, cache {cache}: token latency ms by context")
        print(sub.pivot_table(index=["model", "batch", "link"], columns="context", values="token_latency_ms")
              .round(2).to_string())
        print(f"\nfull design vs LRU + overlap, cache {cache} (>1 = full design faster)")
        print(sub.pivot_table(index=["model", "batch", "link"], columns="context", values="gain_over_lru_overlap")
              .round(3).to_string())
    phys = df[(df.cache == "physical") & (df.policy == "full design")]
    print("\nphysical cache share (experts that fit in HBM after weights + KV cache)")
    print(phys.pivot_table(index=["model", "batch"], columns="context", values="cache_ratio_used")
          .round(3).to_string())

    for cache in ("25%", "physical"):
        models = list(df.model.unique())
        fig, axes = plt.subplots(len(LINKS), len(models), figsize=(4.6 * len(models), 7.5), squeeze=False)
        for col, m in enumerate(models):
            for row, link in enumerate(LINKS):
                ax = axes[row, col]
                for b in BATCHES:
                    for pol, ls in (("LRU + overlap", "--"), ("full design", "-")):
                        s = df[(df.model == m) & (df.batch == b) & (df.link == link) & (df.cache == cache)
                               & (df.policy == pol)].sort_values("context")
                        if len(s):
                            ax.plot(s.context, s.token_latency_ms / b, ls, marker="o", ms=3,
                                    label=f"b{b} {pol}")
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_title(f"{m}, {link}, cache {cache}", fontsize=9)
                ax.set_xlabel("context (tokens)")
                ax.set_ylabel("ms per generated token")
                ax.grid(alpha=0.3, which="both")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"latency_vs_context_{cache.rstrip('%')}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
