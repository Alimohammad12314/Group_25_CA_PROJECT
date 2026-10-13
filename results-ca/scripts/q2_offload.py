"""Q2: single GPU, no expert sharding, no Logic-PIM, all routed-expert weights in off-chip DRAM.

Part A (ablation, H100, Mixtral): walk from the Duplex configuration to the single-GPU
offloaded system:  Duplex(4 dev) -> GPU(4 dev) -> GPU(1 dev, experts in HBM) -> GPU(1 dev,
experts in off-chip DRAM).  Steps 1-3 use the original LLMSimulator (Ramulator off, so both
simulators use the same analytic roofline timing); step 3 is cross-checked against the
SSD-offloading simulator.

Part B (main sweep, SSD-offloading simulator):
  GPU   : A100-80GB (reference in the assignment), H100-80GB
  model : mixtral, grok1, llama4_maverick, deepseekR1
  link  : PCIe4 x16 32 GB/s, PCIe5 x16 64 GB/s, NVLink-C2C 450 GB/s, NVLink-C2C 900 GB/s
  batch : 1, 4, 16, 64, 256         context 1024, first decode step (output_len 2, iter 2)
For every point three simulations are run (see q2 README):
  base    offload off                        -> expert execution time, step latency, HBM energy
  compute offload on, collects_gpu_compute   -> expert execution with HBM bandwidth reduced by link
  copy    offload on, copy pass              -> routed-expert bytes moved over the link + energy
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from simrun import LLMSIM, LLMSIM_BASE_CFG, RESULTS, SSD_BASE_CFG, SSDSIM, load_yaml, run_many  # noqa: E402

OUT = os.path.join(RESULTS, "q2_offload_load_vs_exec")
GPUS = ["A100", "H100"]
MODELS = ["mixtral", "grok1", "llama4_maverick", "deepseekR1"]
LINKS = {"PCIe4-x16": 32.0, "PCIe5-x16": 64.0, "C2C-450": 450.0, "C2C-900": 900.0}
BATCHES = [1, 4, 16, 64, 256]
CTX = 1024
# model geometry (model_config.h): hidden, expert intermediate, routed experts, top-k,
# shared experts, number of MoE layers
GEOM = {
    "mixtral":         dict(h=4096, de=14336, N=8,   k=2, shared=0, L=32),
    "grok1":           dict(h=6144, de=32768, N=8,   k=2, shared=0, L=64),
    "llama4_maverick": dict(h=5120, de=8192,  N=128, k=1, shared=1, L=24),
    "deepseekR1":      dict(h=7168, de=2048,  N=256, k=8, shared=1, L=57),
}
OFFLOAD_PJ_PER_BIT = 9.2 + 4.2          # DDR5 read + HBM write (module_graph.cpp)
P_BG_GPU_W = 70.0                       # GPU+HBM background power (figure_4_and_5.py)
P_BG_DRAM_W = 57.0                      # 256 GB host DRAM background power (figure_4_and_5.py)
DYN_COLS = ["fc_dram", "fc_comp", "attn_dram", "attn_comp", "ffn_dram", "ffn_comp",
            "shared_moe_hbm", "shared_moe_comp", "moe_hbm", "moe_comp"]


def expert_bytes(m):
    g = GEOM[m]
    return 3 * g["h"] * g["de"] * 2      # gate, up, down projections, FP16


def ssd_over(gpu, model, batch, mode, link_gbps):
    mla = model.startswith("deepseek")
    return {
        "model": {"model_name": model},
        "system": {"gpu_gen": gpu, "num_node": 1, "num_device": 1, "processor_type": "GPU",
                   "offload": {"collects_gpu_compute": mode != "copy",
                               "offload_expert_to_cpu_memory": mode != "base",
                               "offload_expert_to_ssd": False, "ssd_energy_scale": 1.0,
                               "offload_bandwidth": 5, "offload_bandwidth_GBps": float(link_gbps)},
                   "distribution": {"expert_tensor_degree": 1, "none_expert_tensor_degree": 1},
                   "optimization": {"compressed_kv": mla, "use_absorb": mla,
                                    "prefill_mode": False, "decode_mode": True}},
        "serving": {"max_batch_size": batch, "max_process_token": 0},
        "simulation": {"data": "synthesis", "input_len": CTX, "output_len": 2,
                       "precision_byte": 2, "iter": 2, "routing": {"model": "uniform"}},
        "_meta": {"gpu": gpu, "model": model, "batch": batch, "mode": mode, "link_GBps": link_gbps},
    }


def orig_over(sysname, ndev, ptype, batch):
    return {
        "model": {"model_name": "mixtral"},
        "system": {"gpu_gen": "H100", "nvlink_gen": 4, "infiniband_gen": 400, "num_node": 1,
                   "num_device": ndev, "processor_type": ptype,
                   "distribution": {"expert_tensor_degree": 1, "none_expert_tensor_degree": ndev},
                   "optimization": {"parallel_execution": False, "use_ramulator": False,
                                    "hetero_subbatch": False, "disagg_system": False,
                                    "use_low_unit_moe_only": False, "compressed_kv": False,
                                    "use_absorb": False, "use_flash_mla": False,
                                    "use_flash_attention": True, "reuse_kv_cache": False,
                                    "prefill_mode": False, "decode_mode": True}},
        "serving": {"max_batch_size": batch, "max_process_token": 0},
        "simulation": {"data": "synthesis", "input_len": CTX, "output_len": 2, "precision_byte": 2,
                       "skewness": 0.0, "iter": 2, "exit_out_of_memory": False, "mem_cap_limit": False},
        "log": {"print_log": False, "export_gantt": False},
        "_meta": {"system": sysname, "batch": batch},
    }


def g(r, c):
    v = r.get(c, 0.0)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(v) else v


def main():
    ssd_base = load_yaml(SSD_BASE_CFG)
    orig_base = load_yaml(LLMSIM_BASE_CFG)
    jobs = []
    for gpu in GPUS:
        for m in MODELS:
            for b in BATCHES:
                jobs.append((SSDSIM, ssd_base, ssd_over(gpu, m, b, "base", 0.0),
                             os.path.join(OUT, "runs", f"{gpu}_{m}_b{b}_base")))
                for ln, bw in LINKS.items():
                    for mode in ("compute", "copy"):
                        jobs.append((SSDSIM, ssd_base, ssd_over(gpu, m, b, mode, bw),
                                     os.path.join(OUT, "runs", f"{gpu}_{m}_b{b}_{ln}_{mode}")))
    abl = [("Duplex (4 dev, Logic-PIM)", 4, "GPU+LOGIC"), ("GPU (4 dev, EP)", 4, "GPU"),
           ("GPU (1 dev, experts in HBM)", 1, "GPU")]
    for b in (1, 16, 64):
        for name, nd, pt in abl:
            jobs.append((LLMSIM, orig_base, orig_over(name, nd, pt, b),
                         os.path.join(OUT, "runs", f"ablation_orig_{pt}_D{nd}_b{b}")))
    if "--summarize-only" in sys.argv:          # reuse archived runs, recompute summary
        results = []
        for _, _, over, arch in jobs:
            f = os.path.join(arch, "output.csv")
            results.append((over, pd.read_csv(f) if os.path.exists(f) else None, "missing " + f))
    else:
        results = run_many(jobs)

    long_rows, abl_rows = [], []
    for over, df, log in results:
        meta = over["_meta"]
        if df is None:
            print("FAILED", meta, log[-500:])
            continue
        r = df.iloc[-1].to_dict()
        if "system" in meta:      # original simulator (ablation)
            abl_rows.append(dict(**meta, step_ms=g(r, "latency") / 1e6,
                                 expert_ffn_ms=g(r, "expert_ffn") / 1e6,
                                 non_expert_ms=(g(r, "latency") - g(r, "expert_ffn")) / 1e6))
            continue
        long_rows.append(dict(**meta, **{k: g(r, k) for k in r if k not in ("type", "##Latency", "##Energy")}))
    runs = pd.DataFrame(long_rows)
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    runs.to_csv(os.path.join(OUT, "data", "q2_runs_long.csv"), index=False)

    rows = []
    idx = runs.set_index(["gpu", "model", "batch", "mode", "link_GBps"])
    for gpu in GPUS:
        for m in MODELS:
            geo = GEOM[m]
            Se = expert_bytes(m)
            for b in BATCHES:
                base = idx.loc[(gpu, m, b, "base", 0.0)]
                base_step = (g(base, "non_MoE_decoders") + g(base, "MoE_decoders")) / 1e6
                exec_ms = g(base, "expert_ffn") / 1e6
                dyn_base = sum(g(base, c) for c in DYN_COLS) / 1e9
                for ln, bw in LINKS.items():
                    comp = idx.loc[(gpu, m, b, "compute", bw)]
                    copy = idx.loc[(gpu, m, b, "copy", bw)]
                    off_J = g(copy, "moe_offload") / 1e9
                    bytes_counted = g(copy, "moe_offload") * 1e-9 / (OFFLOAD_PJ_PER_BIT * 1e-12 * 8)
                    # the copy pass also counts the router (gate) weights, h x N x 2 B per layer;
                    # they are tiny and would stay resident in HBM, so they are excluded here
                    router_bytes = geo["h"] * geo["N"] * 2 * geo["L"]
                    bytes_moved = bytes_counted - router_bytes
                    off_J *= bytes_moved / bytes_counted
                    load_ms = bytes_moved / (bw * 1e9) * 1e3
                    uniq = bytes_moved / Se / geo["L"]
                    exp_uniq = geo["N"] * (1 - (1 - geo["k"] / geo["N"]) ** b)
                    step_contended = (g(comp, "non_MoE_decoders") + g(comp, "MoE_decoders")) / 1e6
                    on_demand = base_step + load_ms
                    overlap_bound = max(load_ms, step_contended)
                    e_base = dyn_base + P_BG_GPU_W * base_step / 1e3
                    e_od = dyn_base + off_J + (P_BG_GPU_W + P_BG_DRAM_W) * on_demand / 1e3
                    rows.append(dict(
                        gpu=gpu, model=m, batch=b, link=ln, link_GBps=bw,
                        N_experts=geo["N"], top_k=geo["k"], moe_layers=geo["L"], expert_MB=Se / 1e6,
                        all_experts_GB=Se * geo["N"] * geo["L"] / 1e9,
                        tokens_per_active_expert=b * geo["k"] / min(exp_uniq, geo["N"]),
                        unique_experts_per_layer_sim=uniq, unique_experts_per_layer_uniform=exp_uniq,
                        offchip_GB_per_step=bytes_moved / 1e9,
                        offchip_GB_per_token=bytes_moved / 1e9 / b,
                        load_ms=load_ms, copy_pass_expert_ffn_ms=g(copy, "expert_ffn") / 1e6,
                        exec_ms=exec_ms, exec_contended_ms=g(comp, "expert_ffn") / 1e6,
                        load_per_layer_ms=load_ms / geo["L"], exec_per_layer_ms=exec_ms / geo["L"],
                        load_one_expert_ms=Se / (bw * 1e9) * 1e3,
                        load_over_exec=load_ms / exec_ms,
                        base_step_ms=base_step, non_expert_ms=base_step - exec_ms,
                        ondemand_step_ms=on_demand, overlap_bound_step_ms=overlap_bound,
                        exposed_stall_ms=load_ms, stall_fraction=load_ms / on_demand,
                        slowdown_ondemand=on_demand / base_step,
                        ondemand_ms_per_token=on_demand / b,
                        energy_base_J_per_token=e_base / b, energy_ondemand_J_per_token=e_od / b,
                        energy_offload_transfer_J=off_J,
                        energy_background_stall_J=(P_BG_GPU_W + P_BG_DRAM_W) * load_ms / 1e3,
                        energy_dynamic_base_J=dyn_base))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "data", "q2_summary.csv"), index=False)
    abl = pd.DataFrame(abl_rows)
    # cross-check: SSD simulator single-GPU baseline vs original simulator, H100 Mixtral
    for b in (1, 16, 64):
        base = idx.loc[("H100", "mixtral", b, "base", 0.0)]
        abl = pd.concat([abl, pd.DataFrame([dict(
            system="GPU (1 dev, experts in HBM) [SSD sim]", batch=b,
            step_ms=(g(base, "non_MoE_decoders") + g(base, "MoE_decoders")) / 1e6,
            expert_ffn_ms=g(base, "expert_ffn") / 1e6)])])
        for ln, bw in LINKS.items():
            s = summ[(summ.gpu == "H100") & (summ.model == "mixtral") & (summ.batch == b) & (summ.link == ln)].iloc[0]
            abl = pd.concat([abl, pd.DataFrame([dict(
                system=f"GPU (1 dev, experts off-chip, {ln}, on-demand)", batch=b,
                step_ms=s.ondemand_step_ms, expert_ffn_ms=s.exec_ms, load_ms=s.load_ms)])])
    abl.to_csv(os.path.join(OUT, "data", "q2_ablation_mixtral_H100.csv"), index=False)
    pd.set_option("display.width", 250)
    print(abl.round(3).to_string())
    print(summ[summ.gpu == "A100"][["model", "batch", "link", "load_ms", "exec_ms", "load_over_exec",
                                     "unique_experts_per_layer_sim", "unique_experts_per_layer_uniform",
                                     "stall_fraction"]].round(3).to_string())


if __name__ == "__main__":
    main()
