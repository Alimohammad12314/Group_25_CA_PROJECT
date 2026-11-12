"""Prediction vs prefetching trade-offs (your question: does predicting or prefetching cost anything?).

  A. Predictor overhead. The predictor runs once per layer. Its time is swept from 0 to 1 ms, either
     off the critical path (a host thread: it only delays when prefetches are issued) or on it (it
     delays the layer). The measured cost of this Python implementation and an estimate for a
     compiled one (from the operation count) are marked.
  B. Prefetch aggressiveness. Fixed budgets top-k, top-(k+1), top-(k+2), top-(k+4) against the adaptive
     budget at tau 0.6 / 0.75 / 0.9, with and without the expected-value gate: latency vs traffic vs
     energy (APEX: static top-(k+8) used the most energy of all policies).
  C. Off-switch: the adaptive budget with and without STEP's 40 % off-switch.
  D. Predictor state size (bits) per model, against ST-MoE's 256-entry x 8-candidate x 10-bit table.

    python -m experiments.e5_tradeoffs [--models ...]
"""
import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.costs import MODELS

from .common import FULL_SIM, NO_GATE_SIM, PRED, available_models, get_stream, out_dir, run_many, sim_kwargs

LAT_US = [0, 2, 10, 50, 200, 1000]
LINKS_T = ["PCIe4", "C2C-450", "C2C-900"]
RATIO = 0.25


def compiled_estimate_us(model, batch):
    """Operation count of one predict+update per layer, at 1 ns per simple op (one CPU core).

    temporal: W*k votes per sequence; cross: k rows of N counters per distance (D = 4) plus k*k
    updates; popularity: N; calibration: N log2 N sort per predictor per sequence; budget: 64-entry sort.
    """
    m = MODELS[model]
    N, k, B, D, W = m["N"], m["k"], batch, 4, 16
    ops = B * (W * k + D * k * N + 3 * N * np.log2(N) + D * k * k) + N + 64 * 6
    return ops * 1e-3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--procs", type=int, default=30)
    args = ap.parse_args(argv)
    models = args.models or available_models()
    out = out_dir("e5_tradeoffs")
    jobs = []
    for m in models:
        for b in (1, 16):
            for link in LINKS_T:
                for on_path in (False, True):
                    for us in LAT_US:
                        jobs.append((dict(part="A", on_path=on_path, pred_us=us), m, b, "full",
                                     sim_kwargs(args.gpu, link, RATIO, dict(FULL_SIM, pred_latency_us=us,
                                                                            pred_on_critical_path=on_path)), None))
        for b in (1, 4):
            for link in LINKS_T:
                budgets = [("top-k", "chooser_topk"), ("top-(k+2)", "chooser_topk+2")]
                for label, name in budgets:
                    for gate in (False, True):
                        sim = FULL_SIM if gate else NO_GATE_SIM
                        jobs.append((dict(part="B", budget=label, gate=gate), m, b, name,
                                     sim_kwargs(args.gpu, link, RATIO, sim), None))
                for x in (1, 4):
                    cfg = replace(PRED["chooser_topk"], name=f"chooser_topk+{x}", budget=f"topk+{x}")
                    for gate in (False, True):
                        sim = FULL_SIM if gate else NO_GATE_SIM
                        jobs.append((dict(part="B", budget=f"top-(k+{x})", gate=gate), m, b, None,
                                     sim_kwargs(args.gpu, link, RATIO, sim), cfg))
                for tau in (0.6, 0.75, 0.9):
                    cfg = replace(PRED["full"], name=f"full_tau{tau}", tau=tau)
                    for gate in (False, True):
                        sim = FULL_SIM if gate else NO_GATE_SIM
                        jobs.append((dict(part="B", budget=f"adaptive tau={tau}", gate=gate), m, b, None,
                                     sim_kwargs(args.gpu, link, RATIO, sim), cfg))
                cfg = replace(PRED["full"], name="full_nooff", off_threshold=0.0)
                for gate in (False, True):
                    sim = FULL_SIM if gate else NO_GATE_SIM
                    jobs.append((dict(part="C", budget="adaptive, no off-switch", gate=gate), m, b, None,
                                 sim_kwargs(args.gpu, link, RATIO, sim), cfg))
    print(f"{len(jobs)} timeline runs", flush=True)
    df = run_many(jobs, args.procs)
    df = df.drop(columns=["layer_stall_ms", "layer_hit_rate"], errors="ignore")
    df.to_csv(os.path.join(out, "tradeoffs.csv"), index=False)

    cost = []
    for m in models:
        for b in (1, 16):
            st = get_stream(m, b, PRED["full"])
            spec = MODELS[m]
            st_mo = 256 * 8 * 10
            cost.append(dict(model=m, batch=b, python_us=st.seconds_per_layer_event * 1e6,
                             compiled_estimate_us=compiled_estimate_us(m, b),
                             nonexpert_us_per_layer=None,
                             state_KiB=sum(st.state_bits.values()) / 8 / 1024,
                             cross_table_KiB=st.state_bits["cross"] / 8 / 1024,
                             st_moe_reference_KiB=st_mo / 8 / 1024, N=spec["N"], layers=spec["moe_layers"]))
    C = pd.DataFrame(cost)
    from engine.costs import CostModel
    C["nonexpert_us_per_layer"] = [CostModel(args.gpu, r.model, "C2C-900", r.batch).t_nonexpert_layer * 1e6
                                   for r in C.itertuples()]
    C.to_csv(os.path.join(out, "predictor_cost.csv"), index=False)
    report(df, C, out)


def report(df, C, out):
    if "error" in df:
        bad = df[df["error"].notna()]
        if len(bad):
            print("FAILED RUNS:\n", bad[["model", "batch", "error"]].to_string())
        df = df[df["error"].isna()]
    pd.set_option("display.width", 250)
    A = df[df.part == "A"]
    t = A.pivot_table(index=["model", "batch", "link", "on_path"], columns="pred_us", values="token_latency_ms")
    print("A. token latency (ms) vs predictor time per layer (us)\n", (t.div(t[0], axis=0)).round(3).to_string())
    print("\npredictor cost\n", C.round(2).to_string(index=False))
    B = df[df.part.isin(["B", "C"])]
    t = B.pivot_table(index=["model", "batch", "link", "budget"], columns="gate",
                      values=["token_latency_ms", "GB_per_token", "energy_J_per_token", "prefetch_accuracy"])
    print("\nB/C. budget vs latency, traffic, energy\n", t.round(3).to_string())

    for m in df.model.unique():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, link in zip(axes, LINKS_T):
            s = B[(B.model == m) & (B.batch == 1) & (B.link == link)]
            for gate, mk in ((False, "o"), (True, "s")):
                g = s[s.gate == gate]
                ax.scatter(g.GB_per_token, g.token_latency_ms, marker=mk, label=f"value gate={gate}")
                for r in g.itertuples():
                    ax.annotate(r.budget.replace("adaptive ", ""), (r.GB_per_token, r.token_latency_ms), fontsize=7)
            ax.set_xlabel("GB moved per token")
            ax.set_ylabel("token latency (ms)")
            ax.set_title(f"{m}, batch 1, {link}, cache 25%")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"budget_tradeoff_{m}.png"), dpi=130)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for (link, on_path), g in A[(A.model == m) & (A.batch == 1)].groupby(["link", "on_path"]):
            g = g.sort_values("pred_us")
            ax.plot(np.maximum(g.pred_us, 0.5), g.token_latency_ms / g.token_latency_ms.iloc[0], marker="o",
                    label=f"{link}, {'on' if on_path else 'off'} critical path")
        cm = C[(C.model == m) & (C.batch == 1)].iloc[0]
        ax.axvline(cm.python_us, color="red", ls="--", lw=0.8, label="this Python implementation")
        ax.axvline(max(cm.compiled_estimate_us, 0.5), color="green", ls="--", lw=0.8, label="compiled estimate")
        ax.set_xscale("log")
        ax.set_xlabel("predictor time per layer (us)")
        ax.set_ylabel("token latency / latency at 0 us")
        ax.set_title(f"{m}, batch 1, cache 25%")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"predictor_overhead_{m}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
