"""Phase E: real GPU vs the engine -> results/compare.csv, results/compare_summary.csv, plots

For every row of real_runs.csv the engine is run on the same workload (same documents, batch, ctx,
cache, configuration, prediction stream):

  sim_trace_default   trace routing, default costs (LLMSimulator / Q2, H100 + PCIe5 64 GB/s):
                      exactly what the report's simulations do
  sim_default         the GPU run's own (live) routing, default costs
  sim_cal             live routing, calibrated costs (gpu/results/calibration_h100.json, warm GPU)
  sim_cal_idle        live routing, calibrated costs measured right after an 8 ms idle (the GPU and
                      CPU clock down while the link is busy, so this is closer to an offloaded run)

Live routing makes the timing-independent policies (C0, C1, C1b, C2) an exact test: their hit rate
and GB per token must equal the GPU's, or the engine's cache logic (or the runtime) has a bug.

Energy: the GPU's NVML counter vs the engine's GPU-side share (GPU+HBM background, dynamic energy,
and the HBM-write half of the transfer energy; the engine's host DRAM terms are left out).

    python -m gpu.compare [--procs 32]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from engine import costs as C
from engine.timeline import SimConfig, simulate
from experiments.common import PRED

from .common import CALIBRATION, DATA, RESULTS, gpu_workload, live_workload, load_live, sim_overrides, stream_for

REAL = os.path.join(RESULTS, "real_runs.csv")
CAL_IDLE = os.path.join(DATA, "calibration_h100_after_idle.json")
METRICS = ["token_latency_ms", "stall_ms_per_token", "hit_rate", "GB_per_token", "gpu_energy_J_per_token"]
EXACT = ("C0", "C1", "C1b", "C2")


def write_idle_profile():
    with open(CALIBRATION) as f:
        cal = json.load(f)
    cal["t_exec_s"] = cal["t_exec_after_idle_s"]
    cal["t_nonexpert_layer_s"] = cal["t_nonexpert_layer_after_idle_s"]
    os.makedirs(DATA, exist_ok=True)
    with open(CAL_IDLE, "w") as f:
        json.dump(cal, f)


def gpu_side_energy(res, cost, p_idle=None):
    """Engine energy per token restricted to the GPU (what NVML measures). p_idle replaces the
    engine's 70 W GPU+HBM background with a measured idle power."""
    p_gpu = p_idle or cost.background_power(offloaded=False)
    p_all = cost.background_power(offloaded=True)          # the engine's background (70 + 57 W)
    hbm_share = C.E_HBM_READ_J_PER_BIT / C.E_LINK_J_PER_BIT       # 4.2 of the 13.4 pJ/bit
    return (res["background_J_per_token"] * p_gpu / p_all + res["dynamic_J_per_token"]
            + res["transfer_J_per_token"] * hbm_share)


def _sim(args):
    row, profile = args
    b, ctx, g, T = int(row["batch"]), int(row["ctx"]), int(row["groups"]), int(row["tokens"])
    wl = gpu_workload(b, ctx, groups=g, tokens=T)
    live = load_live(b, ctx, g)
    pn, kw = sim_overrides(row["config"], float(row["cache_ratio"]))
    var = json.loads(row["variant"]) if isinstance(row["variant"], str) and row["variant"] else {}
    if "lookahead_cap" in var:
        kw["lookahead_cap"] = var["lookahead_cap"]
    from dataclasses import replace
    pc = replace(PRED["full"], tau=var["tau"], name=f"full_tau{var['tau']}") if "tau" in var else None
    use_live = profile != "sim_trace_default"
    wl_run = live_workload(wl, live) if use_live else wl
    stream = None
    if pn == "full":
        stream = stream_for(wl, "full", f"ctx{ctx}", pred_cfg=pc)       # the stream the GPU replayed
    elif pn == "oracle":
        stream = stream_for(wl_run, "oracle", f"live_ctx{ctx}" if use_live else f"ctx{ctx}")
    elif pn == "hidden":
        # C5: replay the hidden-state predictions recorded on the H100 (Phase B'', same documents)
        from experiments.e7_hidden_state import stream_from_q
        z = np.load(os.path.join(RESULTS, f"hs_pred_b{b}_ctx{ctx}.npz"))
        stream = stream_from_q(z["q"], wl.top_k)
    cal = {"sim_cal": CALIBRATION, "sim_cal_idle": CAL_IDLE}.get(profile)
    cfg = SimConfig(gpu="H100", link="PCIe5", calibration=cal, context=ctx, **kw)
    res = simulate(wl_run, wl_run.needed(), stream, cfg)
    cost = C.CostModel("H100", "mixtral", "PCIe5", b, context=ctx, calibration=cal)
    out = {m: res[m] for m in METRICS if m in res}
    p_idle = None
    if cal:                                   # calibrated profiles: the GPU's measured idle power
        with open(cal) as f:
            p_idle = json.load(f).get("power_W", {}).get("idle")
    out["gpu_energy_J_per_token"] = gpu_side_energy(res, cost, p_idle)
    out["energy_J_per_token_total"] = res["energy_J_per_token"]
    for m in ("late_rate", "miss_rate", "prefetch_issued_per_token", "prefetch_accuracy", "link_utilization",
              "wasted_GB_per_token", "p95_token_latency_ms"):
        out[m] = res[m]
    return row["run_id"], profile, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    real = pd.read_csv(REAL)
    write_idle_profile()
    profiles = ["sim_trace_default", "sim_default", "sim_cal", "sim_cal_idle"]
    jobs = [(r, p) for _, r in real.iterrows() for p in profiles]
    sims = {}
    with ProcessPoolExecutor(args.procs) as ex:
        for rid, prof, out in ex.map(_sim, jobs, chunksize=1):
            sims[(rid, prof)] = out
    rows = []
    for _, r in real.iterrows():
        base = {k: r[k] for k in ("run_id", "batch", "ctx", "cache_ratio", "config", "spec_item", "variant")}
        for m in METRICS + ["late_rate", "miss_rate", "prefetch_issued_per_token", "prefetch_accuracy",
                            "link_utilization", "wasted_GB_per_token", "p95_token_latency_ms"]:
            base[f"real_{m}"] = r.get(m, np.nan)
        for p in profiles:
            for m, v in sims[(r["run_id"], p)].items():
                base[f"{p}_{m}"] = v
            for m in METRICS:
                rv = r.get(m, np.nan)
                base[f"{p}_err_{m}"] = abs(sims[(r["run_id"], p)][m] - rv) / abs(rv) if rv else np.nan
        base["exact_check"] = ""
        if r["config"] in EXACT:
            sd = sims[(r["run_id"], "sim_default")]
            ok = abs(sd["hit_rate"] - r["hit_rate"]) < 1e-9 and abs(sd["GB_per_token"] - r["GB_per_token"]) < 1e-6
            base["exact_check"] = "match" if ok else "MISMATCH"
        rows.append(base)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "compare.csv"), index=False)

    # summary: median / max relative error per profile and metric, by config group
    summ = []
    df["group"] = np.where(df.config.isin(EXACT), "C0-C2 (timing-independent)", "C3/C4/C5/O (prefetch)")
    for grp, sub in list(df.groupby("group")) + [("all", df)]:
        for p in profiles:
            for m in METRICS:
                e = sub[f"{p}_err_{m}"].dropna()
                if len(e):
                    summ.append(dict(group=grp, profile=p, metric=m, n=len(e), median_rel_err=e.median(),
                                     mean_rel_err=e.mean(), max_rel_err=e.max()))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RESULTS, "compare_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(summ[summ.metric.isin(["token_latency_ms", "hit_rate", "GB_per_token"])].round(4).to_string(index=False))
    bad = df[df.exact_check == "MISMATCH"]
    print(f"\nexact check (C0-C2 hit rate and GB/token, live routing): {len(df[df.exact_check == 'match'])} match, "
          f"{len(bad)} mismatch")
    if len(bad):
        print(bad[["run_id", "real_hit_rate", "sim_default_hit_rate", "real_GB_per_token",
                   "sim_default_GB_per_token"]].to_string(index=False))
    plots(df)


def plots(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(RESULTS, "plots")
    os.makedirs(out, exist_ok=True)
    order = ["C0", "C1", "C1b", "C2", "C3", "C4", "C5", "O"]
    main = df[(df.ctx == 1024) & (df.variant.isna() | (df.variant == ""))]
    for b in sorted(main.batch.unique()):
        sub = main[(main.batch == b) & (main.cache_ratio.isin([0.0, 0.25]))].set_index("config").reindex(order).dropna(
            subset=["real_token_latency_ms"])
        fig, ax = plt.subplots(figsize=(8, 3.6))
        x = np.arange(len(sub))
        w = 0.2
        for i, (col, lab) in enumerate([("real_token_latency_ms", "real H100"),
                                        ("sim_default_token_latency_ms", "sim, default costs"),
                                        ("sim_cal_token_latency_ms", "sim, calibrated"),
                                        ("sim_cal_idle_token_latency_ms", "sim, calibrated after idle")]):
            ax.bar(x + (i - 1.5) * w, sub[col], w, label=lab)
        ax.set_xticks(x, sub.index)
        ax.set_ylabel("ms per token")
        ax.set_title(f"Mixtral, batch {b}, ctx 1024, cache 25 % (C0: none), PCIe 5.0")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"latency_real_vs_sim_b{b}.png"), dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    base = df[(df.variant.isna() | (df.variant == "")) & (df.cache_ratio == 0.25)]
    for ax, key in zip(axes, ["batch", "ctx"]):
        sub = base[base.ctx == 1024] if key == "batch" else base
        for p, lab in [("sim_default", "default"), ("sim_cal", "calibrated"), ("sim_cal_idle", "cal. after idle")]:
            g = sub.groupby(key)[f"{p}_err_token_latency_ms"].median()
            ax.plot(g.index, 100 * g.values, marker="o", label=lab)
        ax.set_xlabel(key)
        ax.set_ylabel("median |sim - real| / real, latency (%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "latency_error_vs_batch_ctx.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
