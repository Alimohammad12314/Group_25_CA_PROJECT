"""Deliverables 3 and 4: latency, stall, traffic and energy across the plan's sweep.

  model        Mixtral, Grok-1 (synthetic), Llama-4 Maverick, DeepSeek-R1
  batch        1, 4, 16
  link         PCIe 4.0, PCIe 5.0, NVLink-C2C 450, 900
  HBM cache    0 % (no retention), 25 %, 50 %, 75 %, and the physical share that fits in 80 GB
  policy       on-demand, LRU, LRU + miss overlap, LFU + miss overlap, prediction for eviction only,
               full design, oracle predictor, oracle + Belady
plus the coverage target tau (0.60 / 0.75 / 0.90) and lookahead cap (1 / 2 / 4) at 25 % cache.

    python -m experiments.e4_sweeps [--models ...] [--gpu A100]
"""
import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from engine.costs import CostModel

from .common import (BATCHES, CACHE_RATIOS, FULL_SIM, LINKS, PRED, available_models, out_dir, run_many,
                     sim_kwargs)

POLICIES = {
    "LRU": (None, dict(eviction="lru")),
    "LRU + overlap": (None, dict(eviction="lru", miss_overlap=True)),
    "LFU + overlap": (None, dict(eviction="lfu", miss_overlap=True)),
    "predict-evict only": ("full", dict(eviction="need", early_bonus=0.05, need_horizon=4, miss_overlap=True)),
    "full design": ("full", FULL_SIM),
    "oracle": ("oracle", FULL_SIM),
    "oracle + Belady": ("oracle", dict(FULL_SIM, eviction="belady")),
}
NO_RETENTION = {
    "on-demand": (None, dict()),
    "full design": ("full", FULL_SIM),
    "oracle": ("oracle", FULL_SIM),
}


def physical_ratio(model, batch, gpu):
    return CostModel(gpu, model, "C2C-900", batch).physical_cache_ratio()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--procs", type=int, default=30)
    args = ap.parse_args(argv)
    models = args.models or available_models()
    out = out_dir("e4_sweeps")
    jobs = []
    for m in models:
        for b in BATCHES:
            ratios = [(r, f"{int(100 * r)}%") for r in CACHE_RATIOS if r > 0]
            ratios.append((round(physical_ratio(m, b, args.gpu), 4), "physical"))
            for link in LINKS:
                for name, (pred, over) in NO_RETENTION.items():
                    jobs.append((dict(policy=name, cache="0%", sweep="main"), m, b, pred,
                                 sim_kwargs(args.gpu, link, 0.0, over), None))
                for ratio, label in ratios:
                    for name, (pred, over) in POLICIES.items():
                        if label == "physical" and name in ("LFU + overlap", "oracle + Belady"):
                            continue
                        jobs.append((dict(policy=name, cache=label, sweep="main"), m, b, pred,
                                     sim_kwargs(args.gpu, link, ratio, over), None))
                for tau in (0.6, 0.75, 0.9):
                    cfg = replace(PRED["full"], name=f"full_tau{tau}", tau=tau)
                    jobs.append((dict(policy="full design", cache="25%", sweep="tau", tau=tau), m, b, None,
                                 sim_kwargs(args.gpu, link, 0.25, FULL_SIM), cfg))
                for cap in (1, 2, 4):
                    jobs.append((dict(policy="full design", cache="25%", sweep="lookahead", cap=cap), m, b,
                                 "full", sim_kwargs(args.gpu, link, 0.25, dict(FULL_SIM, lookahead_cap=cap)), None))
    print(f"{len(jobs)} timeline runs", flush=True)
    df = run_many(jobs, args.procs)
    df.drop(columns=["layer_stall_ms", "layer_hit_rate"], errors="ignore").to_csv(
        os.path.join(out, "sweeps.csv"), index=False)
    per_layer = df[(df.sweep == "main") & (df.policy.isin(["LRU", "full design", "oracle"]))][
        ["model", "batch", "link", "cache", "policy", "layer_stall_ms", "layer_hit_rate"]]
    per_layer.to_json(os.path.join(out, "per_layer.json"), orient="records")
    if "error" in df and df["error"].notna().any():
        print("FAILED RUNS:\n", df[df["error"].notna()][["model", "batch", "policy", "error"]].to_string())
    report(df, out)


def report(df, out):
    if "error" in df:
        df = df[df["error"].isna()]
    main = df[df.sweep == "main"]
    pd.set_option("display.width", 250)
    base = main[main.policy == "on-demand"].set_index(["model", "batch", "link"])["token_latency_ms"]
    main = main.join(base.rename("ondemand_ms"), on=["model", "batch", "link"])
    main["speedup"] = main["ondemand_ms"] / main["token_latency_ms"]
    main.to_csv(os.path.join(out, "sweeps_with_speedup.csv"), index=False)
    for m in main.model.unique():
        t = main[main.model == m].pivot_table(index=["cache", "policy"], columns=["batch", "link"], values="speedup")
        print(f"\n{m}: speedup over on-demand with no cache\n", t.round(2).to_string())
    tau = df[df.sweep == "tau"].pivot_table(index=["model", "batch", "tau"], columns="link",
                                            values=["token_latency_ms", "wasted_GB_per_token", "energy_J_per_token"])
    print("\ncoverage target tau (25 % cache)\n", tau.round(3).to_string())
    cap = df[df.sweep == "lookahead"].pivot_table(index=["model", "batch", "cap"], columns="link",
                                                  values="token_latency_ms")
    print("\nlookahead cap (25 % cache), token latency ms\n", cap.round(2).to_string())

    # Plot: speedup vs cache size for each model, batch 1 and 16, C2C-900 and PCIe4
    for m in main.model.unique():
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for row, link in enumerate(["C2C-900", "PCIe4"]):
            for col, b in enumerate(BATCHES):
                ax = axes[row, col]
                s = main[(main.model == m) & (main.batch == b) & (main.link == link) & (main.cache != "physical")]
                for pol in ["LRU", "LRU + overlap", "predict-evict only", "full design", "oracle", "oracle + Belady"]:
                    p = s[(s.policy == pol) | ((s.cache == "0%") & (s.policy == {"LRU": "on-demand"}.get(pol, pol)))]
                    p = p.assign(c=p.cache.str.rstrip("%").astype(float)).sort_values("c")
                    ax.plot(p.c, p.speedup, marker="o", label=pol)
                ax.set_title(f"{m}, batch {b}, {link}")
                ax.set_xlabel("HBM cache (% of all experts)")
                ax.set_ylabel("speedup over on-demand")
                ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"speedup_vs_cache_{m}.png"), dpi=130)
        plt.close(fig)

    # Energy breakdown at 25 % cache, batch 1, C2C-900
    sub = main[(main.cache.isin(["0%", "25%"])) & (main.batch == 1) & (main.link == "C2C-900")]
    parts = ["transfer_J_per_token", "dynamic_J_per_token", "background_J_per_token"]
    fig, axes = plt.subplots(1, len(sub.model.unique()), figsize=(4.5 * len(sub.model.unique()), 4), squeeze=False)
    for ax, m in zip(axes[0], sub.model.unique()):
        s = sub[sub.model == m].set_index("policy")
        order = [p for p in ["on-demand", "LRU", "LRU + overlap", "predict-evict only", "full design", "oracle"]
                 if p in s.index]
        s = s.loc[order]
        s = s[~s.index.duplicated(keep="last")]
        bottom = None
        for part in parts:
            ax.bar(s.index, s[part], bottom=bottom, label=part.replace("_J_per_token", ""))
            bottom = s[part] if bottom is None else bottom + s[part]
        ax.set_title(f"{m}, batch 1, C2C-900, 25 % cache")
        ax.set_ylabel("J per token")
        ax.tick_params(axis="x", rotation=60)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "energy_breakdown.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
