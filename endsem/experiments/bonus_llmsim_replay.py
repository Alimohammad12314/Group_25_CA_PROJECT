"""Bonus: real expert-routing traffic inside LLMSimulator, and the prefetcher evaluated on it.

LLMSimulator normally routes every token with a dice roll (uniform, independent). The SSD-offloading
copy is patched with `simulation.routing.model: trace` (src/scheduler/sequence.cpp), which replays a
recorded routing trace for any top-k. This script

  1. takes a decode workload of real requests from the Hugging Face traces (traces/workload.py)
     and writes it in the simulator's trace format
  2. runs the SSD-offloading simulator (A100, single GPU, all routed experts in host DRAM) for
     128 decode steps with real routing and, for reference, with its default uniform routing:
       base pass  (experts in HBM)        -> step time without offloading, expert execution
       copy pass  per link                -> routed-expert bytes moved over the link per step
     and logs every routing decision the simulator made
  3. checks that the simulator's own routing log equals the input trace (replay is exact)
  4. compares the traffic real vs uniform routing creates (unique experts, GB per step, on-demand
     step latency = base step + bytes / link bandwidth, as in Q2)
  5. turns the simulator's routing log back into a workload and runs the predictor, prefetcher and
     HBM policies on exactly that traffic (on-demand, LRU, LRU + overlap, predict-evict only, full
     design, oracle), for both real and uniform routing; also checks that the timeline's on-demand
     step latency tracks the simulator's step by step

    python -m experiments.bonus_llmsim_replay [--models ...] [--procs 28]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from engine.costs import CostModel, E_LINK_J_PER_BIT, MODELS as GEOM
from engine.timeline import SimConfig, simulate
from predictors.stream import build_stream, summarize
from traces.workload import Workload, make_workload

from .common import FULL_SIM, PRED, ROOT, get_trace, out_dir

REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "results-ca", "scripts"))
import q2_offload as q2                                          # noqa: E402
from simrun import SSD_BASE_CFG, SSDSIM, load_yaml, run_many    # noqa: E402

MODELS = ["mixtral", "llama4_maverick", "deepseekR1"]           # real traces only (Grok-1 has none)
BATCHES = [1, 4, 16]
STEPS = 128
GPU = "A100"
LINKS = {"PCIe4": 32.0, "C2C-900": 900.0}
CACHE = 0.25
RUNS = os.path.join(ROOT, "data", "bonus_runs")
POLICIES = [
    ("on-demand", None, 0.0, dict()),
    ("LRU", None, CACHE, dict(eviction="lru")),
    ("LRU + overlap", None, CACHE, dict(eviction="lru", miss_overlap=True)),
    ("predict-evict only", "full", CACHE, dict(eviction="need", early_bonus=0.05, need_horizon=4,
                                                miss_overlap=True)),
    ("full design", "full", CACHE, FULL_SIM),
    ("oracle", "oracle", CACHE, FULL_SIM),
]


def write_trace(wl, path):
    S, B, L, k = wl.A.shape
    with open(path, "w") as fh:
        fh.write(f"{B} {S} {L} {k}\n")
        a = wl.A.transpose(1, 0, 2, 3).reshape(-1, k)          # (seq, token, layer) order
        np.savetxt(fh, a, fmt="%d")


def sim_over(model, batch, mode, link_gbps, routing, trace_file):
    over = q2.ssd_over(GPU, model, batch, mode, link_gbps)
    over["simulation"].update(output_len=STEPS + 1, iter=STEPS)
    r = {"model": routing, "seed": 777}
    if routing == "trace":
        r["trace_file"] = trace_file
    if mode == "base":
        r["log_file"] = "routing_log.csv"
    over["simulation"]["routing"] = r
    over["_meta"].update(routing=routing)
    return over


def read_log(path, B, L, k):
    """Simulator routing log -> A[S, B, L, k] (sequences and tokens numbered by first appearance)."""
    df = pd.read_csv(path, header=None, skiprows=1)
    seqs = {s: i for i, s in enumerate(pd.unique(df[2]))}
    start = df.groupby(2)[3].min()
    S = int((df[3] - df[2].map(start)).max()) + 1
    A = np.full((S, B, L, k), -1, np.int16)
    t = (df[3] - df[2].map(start)).values
    b = df[2].map(seqs).values
    A[t, b, df[1].values] = df.iloc[:, 4:4 + k].values
    return A


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--procs", type=int, default=28)
    args = ap.parse_args(argv)
    out = out_dir("bonus_llmsim")
    os.makedirs(RUNS, exist_ok=True)
    base_cfg = load_yaml(SSD_BASE_CFG)

    wls, jobs = {}, []
    for m in args.models:
        for b in BATCHES:
            wl = make_workload(get_trace(m), batch=b, tokens=STEPS, groups=1)
            wls[(m, b)] = wl
            tf = os.path.join(RUNS, f"trace_{m}_b{b}.txt")
            write_trace(wl, tf)
            for routing in ("trace", "uniform"):
                name = f"{m}_b{b}_{routing}"
                jobs.append((SSDSIM, base_cfg, sim_over(m, b, "base", 0.0, routing, tf),
                             os.path.join(RUNS, name + "_base")))
                for ln, bw in LINKS.items():
                    jobs.append((SSDSIM, base_cfg, sim_over(m, b, "copy", bw, routing, tf),
                                 os.path.join(RUNS, f"{name}_{ln}_copy")))
    print(f"{len(jobs)} LLMSimulator runs ({STEPS} decode steps each)", flush=True)
    results = run_many(jobs, args.procs)

    per_step = {}
    for over, df, log in results:
        meta = over["_meta"]
        key = (meta["model"], meta["batch"], meta["routing"])
        if df is None:
            raise RuntimeError(f"simulator failed for {meta}: {log[-600:]}")
        df = df[df["type"] == "t2t"].reset_index(drop=True) if "type" in df else df
        if meta["mode"] == "base":
            per_step.setdefault(key, {})["base_ms"] = ((df.non_MoE_decoders + df.MoE_decoders) / 1e6).values
            per_step[key]["exec_ms"] = (df.expert_ffn / 1e6).values
        else:
            geo = GEOM[meta["model"]]
            router_bytes = geo["hidden"] * geo["N"] * 2 * geo["moe_layers"]
            byts = df.moe_offload.values * 1e-9 / (E_LINK_J_PER_BIT * 8) - router_bytes
            ln = [k for k, v in LINKS.items() if v == meta["link_GBps"]][0]
            per_step.setdefault(key, {})[f"bytes_{ln}"] = byts

    traffic, policy_rows, check_rows, pred_rows = [], [], [], []
    for (m, b), wl in wls.items():
        geo = GEOM[m]
        eb = 3 * geo["hidden"] * geo["d_expert"] * 2
        for routing in ("trace", "uniform"):
            name = f"{m}_b{b}_{routing}"
            A = read_log(os.path.join(RUNS, name + "_base", "routing_log.csv"), b, wl.L, wl.top_k)
            if routing == "trace":
                same = np.mean(np.sort(A[:wl.S], -1) == np.sort(wl.A, -1))
                check_rows.append(dict(model=m, batch=b, steps_logged=A.shape[0], replay_match=same))
                print(f"{m} b={b}: simulator replayed {same * 100:.2f} % of routing decisions exactly", flush=True)
                sim_wl = wl
            else:
                sim_wl = Workload(model=m, source="LLMSimulator:uniform", num_experts=wl.num_experts,
                                  top_k=wl.top_k, A=A[:STEPS], group_start=wl.group_start[:STEPS],
                                  warm=np.full((1, b, 1, wl.L, wl.top_k), -1, np.int16),
                                  tokens_per_request=STEPS)
            ps = per_step[(m, b, routing)]
            n = min(len(ps["base_ms"]), STEPS)
            uniq = np.array([[len(np.unique(sim_wl.A[s, :, l])) for l in range(wl.L)] for s in range(n)])
            for ln, bw in LINKS.items():
                load_ms = ps[f"bytes_{ln}"][:n] / (bw * 1e9) * 1e3
                od = ps["base_ms"][:n] + load_ms
                traffic.append(dict(model=m, batch=b, routing=routing, link=ln,
                                    unique_experts_per_layer=uniq.mean(),
                                    sim_GB_per_step=ps[f"bytes_{ln}"][:n].mean() / 1e9,
                                    trace_GB_per_step=uniq.sum(1).mean() * eb / 1e9,
                                    sim_base_step_ms=ps["base_ms"][:n].mean(),
                                    sim_exec_ms=ps["exec_ms"][:n].mean(),
                                    sim_ondemand_step_ms=od.mean(), sim_ondemand_p95_ms=np.percentile(od, 95)))

            needed = sim_wl.needed()
            streams = {p: build_stream(sim_wl, PRED[p]) for p in ("full", "oracle")}
            s1 = summarize(streams["full"].metrics, distance=1)
            pred_rows.append(dict(model=m, batch=b, routing=routing, **{k: float(v) for k, v in s1.items()
                                                                      if np.isscalar(v)}))
            for ln in LINKS:
                for pol, pred, ratio, kw in POLICIES:
                    cfg = SimConfig(gpu=GPU, link=ln, cache_ratio=ratio, **kw)
                    try:
                        r = simulate(sim_wl, needed, streams.get(pred), cfg)
                    except ValueError as exc:          # cache smaller than one layer's experts
                        print(f"  skip {m} b={b} {pol}: {exc}")
                        continue
                    r = {k: v for k, v in r.items() if not isinstance(v, list)}
                    r.update(routing=routing, policy=pol)
                    policy_rows.append(r)
                    if pol == "on-demand":
                        # timeline (constant Q2 costs) vs simulator, step by step
                        cost = CostModel(GPU, m, ln, b)
                        tl_step = np.array([cost.t_nonexpert_layer * wl.L * 1e3
                                            + sum(cost.t_exec(int(c)) * 1e3 + cost.t_load_expert * 1e3
                                                  for l in range(wl.L) for c in needed[s][l][1])
                                            for s in range(n)])
                        sim_step = ps["base_ms"][:n] + ps[f"bytes_{ln}"][:n] / (LINKS[ln] * 1e9) * 1e3
                        check_rows.append(dict(model=m, batch=b, routing=routing, link=ln,
                                               timeline_ms=r["token_latency_ms"], sim_ms=sim_step.mean(),
                                               mean_rel_err=float(np.mean(np.abs(tl_step / sim_step - 1))),
                                               max_rel_err=float(np.max(np.abs(tl_step / sim_step - 1))),
                                               step_corr=float(np.corrcoef(tl_step, sim_step)[0, 1])
                                               if np.std(sim_step) > 0 else np.nan))

    pd.DataFrame(traffic).to_csv(os.path.join(out, "traffic.csv"), index=False)
    pol = pd.DataFrame(policy_rows)
    pol.to_csv(os.path.join(out, "policies.csv"), index=False)
    pd.DataFrame(check_rows).to_csv(os.path.join(out, "checks.csv"), index=False)
    pd.DataFrame(pred_rows).to_csv(os.path.join(out, "predictor.csv"), index=False)
    report(out)


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.set_option("display.width", 220)
    tr = pd.read_csv(os.path.join(out, "traffic.csv"))
    pol = pd.read_csv(os.path.join(out, "policies.csv"))
    chk = pd.read_csv(os.path.join(out, "checks.csv"))
    print("\nreplay and timeline checks\n", chk.round(4).to_string())
    print("\ntraffic inside LLMSimulator: real (trace) vs uniform routing\n",
          tr.pivot_table(index=["model", "batch", "link"], columns="routing",
                         values=["unique_experts_per_layer", "sim_GB_per_step", "sim_ondemand_step_ms"])
          .round(3).to_string())
    base = pol[pol.policy == "on-demand"].set_index(["model", "batch", "link", "routing"]).token_latency_ms
    pol = pol.join(base.rename("od"), on=["model", "batch", "link", "routing"])
    pol["speedup"] = pol.od / pol.token_latency_ms
    pol.to_csv(os.path.join(out, "policies.csv"), index=False)
    print("\nspeedup over on-demand on LLMSimulator traffic (25 % cache)\n",
          pol.pivot_table(index=["model", "batch", "link"], columns=["routing", "policy"], values="speedup")
          .round(2).to_string())
    models = list(pol.model.unique())
    fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 8), squeeze=False)
    order = [p for p, *_ in POLICIES]
    for col, m in enumerate(models):
        for row, ln in enumerate(LINKS):
            ax = axes[row, col]
            s = pol[(pol.model == m) & (pol.link == ln) & (pol.batch == 16)]
            x = np.arange(len(order))
            for i, routing in enumerate(("trace", "uniform")):
                v = s[s.routing == routing].set_index("policy").reindex(order).speedup
                ax.bar(x + (i - 0.5) * 0.4, v, 0.4, label="real trace" if routing == "trace" else "uniform (default)")
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{m}, batch 16, {ln}")
            ax.set_ylabel("speedup over on-demand")
            ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "policies_real_vs_uniform.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
