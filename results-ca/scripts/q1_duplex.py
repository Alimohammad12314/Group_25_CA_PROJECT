"""Q1: reproduce the Duplex evaluation (MICRO'24) with the original LLMSimulator.

Systems (paper Fig. 11 / 14):
  GPU          4x H100, expert parallelism                 processor_type GPU
  2xGPU        8x H100                                     processor_type GPU
  Bank-PIM     4x H100 + bank-level PIM (Fig. 14)          processor_type GPU+PIM
  Duplex       4x (xPU + Logic-PIM), one unit at a time    processor_type GPU+LOGIC
  Duplex+PE    + expert/attention co-processing            parallel_execution on
  Duplex+PE+ET + tensor-parallel experts (MoE only)        expert_tensor_degree 4
Models: mixtral, glam, grok1 (MoE) and llama3_70B (dense, Fig. 14 only).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from simrun import LLMSIM, LLMSIM_BASE_CFG, RESULTS, load_yaml, run_many  # noqa: E402

OUT = os.path.join(RESULTS, "q1_duplex_reproduction")
SYSTEMS = {
    "GPU":          dict(num_device=4, processor_type="GPU",       et=1, pe=False),
    "2xGPU":        dict(num_device=8, processor_type="GPU",       et=1, pe=False),
    "Bank-PIM":     dict(num_device=4, processor_type="GPU+PIM",   et=1, pe=False),
    "Duplex":       dict(num_device=4, processor_type="GPU+LOGIC", et=1, pe=False),
    "Duplex+PE":    dict(num_device=4, processor_type="GPU+LOGIC", et=1, pe=True),
    "Duplex+PE+ET": dict(num_device=4, processor_type="GPU+LOGIC", et=4, pe=True),
}
MODELS = ["mixtral", "glam", "grok1", "llama3_70B"]
BATCHES = [32, 64, 128]
CTX = [256, 1024, 4096]
FC = ["qkvgen", "o_proj", "ffn"]


def overrides(model, sysname, batch, ctx):
    s = SYSTEMS[sysname]
    return {
        "model": {"model_name": model},
        "system": {
            "gpu_gen": "H100", "nvlink_gen": 4, "infiniband_gen": 400, "num_node": 1,
            "num_device": s["num_device"], "processor_type": s["processor_type"],
            "distribution": {"expert_tensor_degree": s["et"], "none_expert_tensor_degree": 4},
            "optimization": {"parallel_execution": s["pe"], "use_ramulator": True,
                             "hetero_subbatch": False, "disagg_system": False,
                             "use_low_unit_moe_only": False, "compressed_kv": False,
                             "use_absorb": False, "use_flash_mla": False,
                             "use_flash_attention": True, "reuse_kv_cache": False,
                             "prefill_mode": False, "decode_mode": True},
        },
        "serving": {"max_batch_size": batch, "max_process_token": 0},
        "simulation": {"data": "synthesis", "input_len": ctx, "output_len": 4,
                       "precision_byte": 2, "skewness": 0.0, "iter": 3,
                       "exit_out_of_memory": False, "mem_cap_limit": False},
        "log": {"print_log": False, "export_gantt": False},
        "_meta": {"model": model, "system": sysname, "batch": batch, "ctx": ctx},
    }


def main():
    base = load_yaml(LLMSIM_BASE_CFG)
    jobs = []
    for m in MODELS:
        for sysname in SYSTEMS:
            if m == "llama3_70B" and sysname == "Duplex+PE+ET":
                continue  # no experts to tensor-parallelise
            for b in BATCHES:
                for c in CTX:
                    o = overrides(m, sysname, b, c)
                    arch = os.path.join(OUT, "runs", f"{m}_{sysname}_b{b}_ctx{c}")
                    jobs.append((LLMSIM, base, o, arch))
    rows = []
    for over, df, log in run_many(jobs):
        meta = over["_meta"]
        if df is None:
            print("FAILED", meta, log[-400:])
            continue
        d = df.mean(numeric_only=True)          # average over the simulated decode steps
        lat_ms = d["latency"] / 1e6
        rows.append(dict(**meta, step_latency_ms=lat_ms,
                         throughput_tok_s=meta["batch"] / (lat_ms / 1e3),
                         expert_ffn_ms=d["expert_ffn"] / 1e6,
                         attention_ms=(d["atten_gen"] + d["atten_sum"]) / 1e6,
                         fc_ms=sum(d[c] for c in FC) / 1e6,
                         communication_ms=d["communication"] / 1e6,
                         other_ms=(d["rope"] + d["layernorm"] + d["residual"]) / 1e6,
                         energy_total_J=d["total_energy"] / 1e9,
                         energy_per_token_mJ=d["total_energy"] / 1e6 / meta["batch"],
                         e_fc_dram_J=d["fc_dram"] / 1e9, e_fc_comp_J=d["fc_comp"] / 1e9,
                         e_attn_dram_J=d["attn_dram"] / 1e9, e_attn_comp_J=d["attn_comp"] / 1e9,
                         e_moe_dram_J=d["moe_dram"] / 1e9, e_moe_comp_J=d["moe_comp"] / 1e9,
                         n_steps=len(df)))
    res = pd.DataFrame(rows)
    gpu = res[res.system == "GPU"].set_index(["model", "batch", "ctx"])
    key = list(zip(res.model, res.batch, res.ctx))
    res["speedup_vs_GPU"] = [r.throughput_tok_s / gpu.loc[k, "throughput_tok_s"] for k, r in zip(key, res.itertuples())]
    res["energy_vs_GPU"] = [r.energy_per_token_mJ / gpu.loc[k, "energy_per_token_mJ"] for k, r in zip(key, res.itertuples())]
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    res.sort_values(["model", "batch", "ctx", "system"]).to_csv(os.path.join(OUT, "data", "q1_results.csv"), index=False)
    print(res.groupby(["model", "system"])[["speedup_vs_GPU", "energy_vs_GPU"]].agg(["mean", "max", "min"]).round(2))


if __name__ == "__main__":
    main()
