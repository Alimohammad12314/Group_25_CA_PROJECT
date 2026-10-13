#!/usr/bin/env python3
"""Part 2/3 sweep for the SSD-offloading LLMSimulator (single device, experts off-chip).
Per (model, batch): baseline (offload off) once; per link gen: compute pass + copy pass.
Output: results/part2_3/<timestamp>/{runs/, runs_long.csv, summary.csv, sweep.log, meta.json}
"""
import argparse, copy, datetime, glob, json, os, platform, shutil, subprocess, sys, time
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
CFG = os.path.join(BUILD, "config.yaml")
DEFAULT = os.path.join(ROOT, "config_default.yaml")
BW_GBPS = {4: 450.0, 5: 900.0}      # link bandwidth per generation (hardware_config.h)
BACKGROUND_W = 70 + 57              # W per device (figure_4_and_5.py: GPU+HBM, CPU memory)
NDEV = 1
FC_COLS = ["qkvgen", "q_down_proj", "kv_down_proj", "kr_proj", "q_up_proj", "qr_proj",
           "kv_up_proj", "tr_k_up_proj", "v_up_proj", "o_proj", "ffn", "shared_expert_ffn"]
LOGF = None


def log(msg):
    line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n"); LOGF.flush()


def make_cfg(base, model, batch, gen, offload, collect, input_len):
    c = copy.deepcopy(base)
    c["model"]["model_name"] = model
    s = c["system"]
    s["gpu_gen"] = "H100"
    s["num_node"] = 1
    s["num_device"] = NDEV
    s["processor_type"] = "GPU"                      # Logic-PIM off
    s["distribution"]["expert_tensor_degree"] = 1    # no expert sharding
    s["distribution"]["none_expert_tensor_degree"] = 1
    mla = model.startswith("deepseek")
    s["optimization"]["compressed_kv"] = mla
    s["optimization"]["use_absorb"] = mla
    o = s["offload"]
    o["offload_expert_to_cpu_memory"] = offload
    o["offload_expert_to_ssd"] = False
    o["collects_gpu_compute"] = collect
    o["offload_bandwidth"] = gen
    o["ssd_energy_scale"] = 1.0
    c["serving"]["max_batch_size"] = batch
    sim = c["simulation"]
    sim["input_len"] = input_len
    sim["output_len"] = 2          # first decode step after prefill (as in figure_4_and_5.py)
    sim["iter"] = 2
    sim["precision_byte"] = 2      # FP16
    return c


def run_one(cfg, rdir):
    os.makedirs(rdir, exist_ok=True)
    with open(CFG, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    shutil.copy(CFG, os.path.join(rdir, "config.yaml"))
    odir = os.path.join(BUILD, cfg["log"]["output_directory"])
    os.makedirs(odir, exist_ok=True)
    for f in glob.glob(os.path.join(odir, "*.csv")):
        os.remove(f)
    t0 = time.time()
    p = subprocess.run([os.path.join(BUILD, "run")], cwd=BUILD, capture_output=True, text=True)
    wall = round(time.time() - t0, 2)
    with open(os.path.join(rdir, "run.log"), "w") as f:
        f.write(p.stdout + p.stderr)
    csvs = glob.glob(os.path.join(odir, "*.csv"))
    if p.returncode != 0 or len(csvs) != 1:
        return dict(exit_code=p.returncode, wall_s=wall, n_csv=len(csvs)), {}
    shutil.copy(csvs[0], os.path.join(rdir, "output.csv"))
    df = pd.read_csv(csvs[0])
    row = {k: v for k, v in df.iloc[0].items() if pd.api.types.is_number(v)}  # drops '->' markers
    return dict(exit_code=0, wall_s=wall, n_csv=1, n_rows=len(df)), row


def g(r, c):
    v = r.get(c, 0.0)
    return 0.0 if pd.isna(v) else float(v)


def div(a, b):
    return a / b if b else float("nan")


def energy_j(r, extra_off_nj=0.0):
    off = (g(r, "fc_dram") + g(r, "ffn_dram") + g(r, "shared_moe_hbm") +
           g(r, "attn_dram") + g(r, "moe_hbm") + extra_off_nj) / 1e9
    on = (g(r, "fc_comp") + g(r, "ffn_comp") + g(r, "shared_moe_comp") +
          g(r, "attn_comp") + g(r, "moe_comp")) / 1e9
    return off, on


def summarize(runs):
    out = []
    for (model, bs), grp in runs.groupby(["model", "batch"]):
        base = grp[grp["mode"] == "baseline"]
        if base.empty:
            continue
        b = base.iloc[0]
        fc_b = sum(g(b, x) for x in FC_COLS)
        base_ms = (fc_b + g(b, "atten_gen") + g(b, "expert_ffn") + g(b, "communication")) / 1e6
        off_b, on_b = energy_j(b)
        bg_b = NDEV * BACKGROUND_W * base_ms / 1e3
        for gen in sorted(grp["nvlink_gen"].dropna().unique()):
            c = grp[(grp["mode"] == "compute") & (grp["nvlink_gen"] == gen)]
            k = grp[(grp["mode"] == "copy") & (grp["nvlink_gen"] == gen)]
            if c.empty or k.empty:
                continue
            c, k, gen = c.iloc[0], k.iloc[0], int(gen)
            load_ms = g(k, "expert_ffn") / 1e6
            exec_c_ms = g(c, "expert_ffn") / 1e6
            exec_b_ms = g(b, "expert_ffn") / 1e6
            fc_c = sum(g(c, x) for x in FC_COLS)
            # framework rule (figure_4_and_5.py): compute-bound only for bs=1024 over gen 5
            if bs == 1024 and gen == 5:
                step_ms = (fc_c + g(c, "atten_gen") + g(c, "expert_ffn") + g(c, "communication")) / 1e6
            else:
                step_ms = (g(k, "expert_ffn") + g(c, "non_MoE_decoders")) / 1e6
            off_o, on_o = energy_j(c, extra_off_nj=g(k, "moe_offload"))
            bg_o = NDEV * BACKGROUND_W * step_ms / 1e3
            tot_b, tot_o = off_b + on_b + bg_b, off_o + on_o + bg_o
            out.append(dict(
                model=model, batch=int(bs), nvlink_gen=gen, link_GBps=BW_GBPS[gen],
                exec_nooffload_ms=exec_b_ms, exec_offloadmode_ms=exec_c_ms, load_ms=load_ms,
                load_over_exec_nooffload=div(load_ms, exec_b_ms),
                load_over_exec_offloadmode=div(load_ms, exec_c_ms),
                bound_by="load" if load_ms > exec_c_ms else "compute",
                implied_GB_loaded=g(k, "expert_ffn") * BW_GBPS[gen] / 1e9,
                base_step_ms=base_ms, framework_offload_step_ms=step_ms,
                slowdown_framework=div(step_ms, base_ms),
                serial_ondemand_ms=load_ms + base_ms,     # my construction: load, then compute
                serial_over_framework=div(load_ms + base_ms, step_ms),
                energy_base_J_tok=tot_b / bs, energy_offload_J_tok=tot_o / bs,
                energy_ratio=div(tot_o, tot_b),
                base_background_J=bg_b, offload_background_J=bg_o,
                offload_moe_offchip_J=(g(c, "moe_hbm") + g(k, "moe_offload")) / 1e9))
    return out


def git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mixtral"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--nvlink", nargs="+", type=int, default=[5, 4])
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for p in (DEFAULT, os.path.join(BUILD, "run")):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{a.tag}" if a.tag else "")
    out = os.path.join(ROOT, "results", "part2_3", stamp)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    LOGF = open(os.path.join(out, "sweep.log"), "w")
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(argv=sys.argv, started=stamp, host=platform.node(),
                       python=platform.python_version(), pandas=pd.__version__,
                       repo_rev=git_rev(ROOT), simulator_rev=git_rev(os.path.join(ROOT, "src")),
                       args=vars(a)), f, indent=2)
    with open(DEFAULT) as f:
        base_cfg = yaml.safe_load(f)

    records = []
    plan = []
    for m in a.models:
        for bs in a.batches:
            plan.append((m, bs, "baseline", None, False, True))
            for gen in a.nvlink:
                plan.append((m, bs, "compute", gen, True, True))
                plan.append((m, bs, "copy", gen, True, False))
    log(f"{len(plan)} runs -> {out}")
    try:
        for i, (m, bs, mode, gen, offload, collect) in enumerate(plan, 1):
            rid = f"{m}_bs{bs}_{mode}" + (f"_nv{gen}" if gen else "")
            cfg = make_cfg(base_cfg, m, bs, gen if gen else 5, offload, collect, a.input_len)
            meta, row = run_one(cfg, os.path.join(out, "runs", rid))
            log(f"[{i}/{len(plan)}] {rid} exit={meta['exit_code']} wall={meta['wall_s']}s "
                f"expert_ffn_ms={g(row, 'expert_ffn') / 1e6:.3f}")
            records.append(dict(run_id=rid, model=m, batch=bs, mode=mode, offload=offload,
                                collects_gpu_compute=collect, nvlink_gen=gen,
                                link_GBps=BW_GBPS.get(gen), input_len=a.input_len, **meta, **row))
    finally:
        shutil.copy(DEFAULT, CFG)        # restore a clean build/config.yaml
        pd.DataFrame(records).to_csv(os.path.join(out, "runs_long.csv"), index=False)

    runs = pd.DataFrame(records)
    summ = pd.DataFrame(summarize(runs[runs["exit_code"] == 0]))
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    if summ.empty:
        log("no summary rows (check runs/*/run.log)")
    else:
        cols = ["model", "batch", "nvlink_gen", "exec_nooffload_ms", "exec_offloadmode_ms",
                "load_ms", "load_over_exec_nooffload", "bound_by", "implied_GB_loaded",
                "framework_offload_step_ms", "serial_ondemand_ms", "energy_ratio"]
        print(summ[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    log(f"done: {out}")


if __name__ == "__main__":
    main()
