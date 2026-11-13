"""Phase C: every configuration x batch x context x cache on the real GPU -> results/real_runs.csv

Stages (GPU.md section 7, in priority order):
  minimum   batch {1, 16} x ctx 1024 x cache {25 %, 50 %} x all 7 configurations
  context   batch {1, 16} x ctx {256, 512, 1536} x cache 25 % x {C1, C2, C4, O}
  batch4    batch 4 at ctx 1024, all configurations, cache {25 %, 50 %}
  aggr      C4 with lookahead cap {1, 2, 4} and tau {0.6, 0.75, 0.9}, batch 16, ctx 1024, cache 25 %

Every finished run is appended to the CSV at once, and runs already in the CSV are skipped, so an
interrupted job resumes where it stopped. The live routing of the first run of each (batch, ctx) is
saved (data/gpu/) and is the "true experts" of the oracle and of the engine comparison.

    gpurun -g 1 gpu/numa_run.sh python -m gpu.run_matrix --stages minimum context batch4 aggr
"""
import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from engine.timeline import SimConfig
from experiments.common import PRED

from .common import (CALIBRATION, CONFIGS, MODEL_DIR, RESULTS, doc_tokens, groups_for, gpu_workload,
                     live_workload, load_live, save_live, sim_overrides, stream_for)
from .energy import GPUEnergy, HostEnergy

OUT = os.path.join(RESULTS, "real_runs.csv")
ALL = list(CONFIGS)


def plan(stages):
    jobs = []        # (batch, ctx, config, cache_ratio, variant dict)
    if "minimum" in stages:
        for b in (1, 16):
            for c in ALL:
                for r in ((0.0,) if c == "C0" else (0.25, 0.5)):
                    jobs.append((b, 1024, c, r, {}))
    if "context" in stages:
        for b in (1, 16):
            for ctx in (256, 512, 1536):
                for c in ("C1", "C2", "C4", "O"):
                    jobs.append((b, ctx, c, 0.25, {}))
    if "batch4" in stages:
        for c in ALL:
            for r in ((0.0,) if c == "C0" else (0.25, 0.5)):
                jobs.append((4, 1024, c, r, {}))
    if "aggr" in stages:
        for cap in (1, 2, 4):
            for tau in (0.6, 0.75, 0.9):
                if cap == 4 and tau == 0.75:
                    continue                      # = C4 of the minimum stage
                jobs.append((16, 1024, "C4", 0.25, dict(lookahead_cap=cap, tau=tau)))
    return jobs


def ordered(stages):
    """Stages in priority order; within a stage, by workload with C1 first (it records the routing
    reference the oracle needs)."""
    order = {c: i for i, c in enumerate(["C1"] + [c for c in ALL if c != "C1"])}
    out = []
    for st in ("minimum", "context", "batch4", "aggr"):
        if st in stages:
            out += sorted(plan([st]), key=lambda j: (j[0], j[1], order[j[2]], j[3]))
    return out


def run_id(b, ctx, c, r, var, groups, tokens):
    v = "_".join(f"{k}{v}" for k, v in sorted(var.items()))
    return f"b{b}_ctx{ctx}_{c}_r{r}_g{groups}_t{tokens}" + (f"_{v}" if v else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["minimum"])
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--groups", type=int, default=None, help="override request groups per run")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--slab", type=int, default=128, help="HBM slots (>= 50 %% of 256 experts)")
    ap.add_argument("--dry", action="store_true", help="only list the runs and prebuild streams")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    jobs = ordered(args.stages)
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["run_id"])
    todo = []
    for b, ctx, c, r, var in jobs:
        g = args.groups or groups_for(b)
        rid = run_id(b, ctx, c, r, var, g, args.tokens)
        if rid not in done:
            todo.append((rid, b, ctx, c, r, var, g))
    print(f"{len(jobs)} runs planned, {len(todo)} to do", flush=True)
    ids, pred_tokens = doc_tokens()
    # prebuild the trace streams on the CPU (timing-independent)
    for rid, b, ctx, c, r, var, g in todo:
        wl = gpu_workload(b, ctx, groups=g, tokens=args.tokens)
        pn, _ = sim_overrides(c, r)
        if pn == "full":
            pc = replace(PRED["full"], tau=var["tau"], name=f"full_tau{var['tau']}") if "tau" in var else None
            stream_for(wl, pn, f"ctx{ctx}", pred_cfg=pc)
    if args.dry:
        for t in todo:
            print(" ", t[0])
        return

    from .runtime import Runtime
    cal = CALIBRATION if os.path.exists(CALIBRATION) else None
    rt = Runtime(args.model_dir, slab_slots=args.slab)
    energy, host_energy = GPUEnergy(), HostEnergy()
    # warm-up: kernels, allocator, cuBLAS handles (discarded)
    wl0 = gpu_workload(1, 1024, groups=1, tokens=12)
    _, kw = sim_overrides("C1", 0.25)
    rt.run(wl0, wl0.needed(), None, SimConfig(gpu="H100", link="PCIe5", calibration=cal, context=1024, **kw),
           ids, 1024, progress=False)

    t_start = time.time()
    for n, (rid, b, ctx, c, r, var, g) in enumerate(todo):
        wl = gpu_workload(b, ctx, groups=g, tokens=args.tokens)
        needed = wl.needed()
        pn, kw = sim_overrides(c, r)
        if "lookahead_cap" in var:
            kw["lookahead_cap"] = var["lookahead_cap"]
        ref = load_live(b, ctx, g) if args.tokens == 128 else None
        stream = None
        if pn == "full":
            pc = replace(PRED["full"], tau=var["tau"], name=f"full_tau{var['tau']}") if "tau" in var else None
            stream = stream_for(wl, pn, f"ctx{ctx}", pred_cfg=pc)
        elif pn == "hidden":
            # C5: the hidden-state predictor runs live inside the decode and fills the stream as it goes
            from .hs_predictor import HSPredictor, LiveStream, chosen_variant
            stream = LiveStream(wl.S, rt.L, rt.N, rt.k)
            rt.hs = HSPredictor(rt, chosen_variant(), live=stream)
        elif pn == "oracle":
            if ref is None:
                raise RuntimeError(f"{rid}: oracle needs the live routing of an earlier run of this workload")
            lwl = live_workload(wl, ref)
            stream = stream_for(lwl, "oracle", f"live_ctx{ctx}")
            needed = lwl.needed()
        cfg = SimConfig(gpu="H100", link="PCIe5", calibration=cal, context=ctx, **kw)
        print(f"[{n + 1}/{len(todo)}] {rid}  ({(time.time() - t_start) / 60:.0f} min so far)", flush=True)
        h0 = host_energy.read()
        try:
            res, live, p = rt.run(wl, needed, stream, cfg, ids, ctx, energy=energy)
        finally:
            if rt.hs is not None:
                res_hs_s = rt.hs.seconds
                rt.hs.close()
                rt.hs = None
        h1 = host_energy.read()
        if ref is None and args.tokens == 128:
            save_live(b, ctx, g, live)
            ref = live
        # agreement of our routing with the trace and with the reference run
        trace_A = wl.A
        res.update(
            run_id=rid, ctx=ctx, link="PCIe5", cache_ratio=r, config=c, spec_item=CONFIGS[c][0],
            groups=g, tokens=args.tokens, variant=json.dumps(var) if var else "",
            tau=var.get("tau", 0.75 if pn == "full" else np.nan),
            routing_vs_trace=float((np.sort(live, -1) == np.sort(trace_A, -1)).all(-1).mean()),
            routing_vs_reference=float((live == ref).all(-1).mean()) if ref is not None else np.nan,
            host_energy_J_per_token=(h1 - h0) / (wl.S * wl.B) if host_energy.available else np.nan,
            calibration_used=bool(cal), spin_wait=rt.link.spin, chunk_MB=16, inflight_depth=rt.link.depth,
            wall_s=time.time() - t_start)
        if pn == "hidden":
            res["hs_variant"] = chosen_variant()
            res["hs_predictor_ms_per_token"] = 1e3 * res_hs_s / wl.S
        T = wl.tokens_per_request
        tp = np.stack([pred_tokens[wl.requests[s // T], ctx + s % T] for s in range(wl.S)])
        res["token_agreement_vs_trace"] = float((p["decode"] == tp).mean())
        res.pop("layer_hit_rate", None)
        row = pd.DataFrame([res])
        os.makedirs(RESULTS, exist_ok=True)
        if os.path.exists(args.out):        # rewrite with the union of columns (C5 adds predictor columns)
            row = pd.concat([pd.read_csv(args.out), row], ignore_index=True)
        row.to_csv(args.out, index=False)
        print(f"    -> {res['token_latency_ms']:.1f} ms/token, stall {res['stall_ms_per_token']:.1f}, "
              f"hit {res['hit_rate']:.3f}, {res['GB_per_token']:.2f} GB/token, "
              f"{res.get('gpu_energy_J_per_token', float('nan')):.2f} J/token, "
              f"routing vs ref {res['routing_vs_reference']:.4f}", flush=True)
    rt.close()
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
