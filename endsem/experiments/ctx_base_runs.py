"""Context-length costs: rerun Q2's "base" pass of the SSD-offloading LLMSimulator at other contexts.

Q2 ran every point at context 1024. Context changes three things the timeline needs:
  non-expert time   attention reads the whole KV cache every decode step (grows with context x batch)
  non-expert energy HBM reads of that KV cache
  free HBM          the KV cache takes HBM away from the expert cache (CostModel.physical_cache_ratio)
Expert execution time and expert bytes do not depend on context, so only the base pass is rerun.

    python -m experiments.ctx_base_runs          # ~2 min on 28 cores, writes results/context_base/

Uses the same simulator binary, base config and overrides as results-ca/scripts/q2_offload.py;
context 1024 is rerun as a check that it matches the archived Q2 numbers.
"""
import os
import sys

import pandas as pd

from .common import RESULTS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "results-ca", "scripts"))
import q2_offload as q2                                                     # noqa: E402
from simrun import SSD_BASE_CFG, SSDSIM, load_yaml, run_many               # noqa: E402

GPUS = ["A100", "H100"]
MODELS = ["mixtral", "grok1", "llama4_maverick", "deepseekR1"]
BATCHES = [1, 4, 16]
CONTEXTS = [256, 512, 1024, 2048, 4096]
NE_COLS = ["fc_dram", "fc_comp", "attn_dram", "attn_comp", "ffn_dram", "ffn_comp",
           "shared_moe_hbm", "shared_moe_comp"]


def main():
    out = os.path.join(RESULTS, "context_base")
    os.makedirs(out, exist_ok=True)
    base_cfg = load_yaml(SSD_BASE_CFG)
    jobs = []
    for gpu in GPUS:
        for m in MODELS:
            for b in BATCHES:
                for ctx in CONTEXTS:
                    over = q2.ssd_over(gpu, m, b, "base", 0.0)
                    over["simulation"]["input_len"] = ctx
                    over["_meta"]["context"] = ctx
                    jobs.append((SSDSIM, base_cfg, over, None))
    print(f"{len(jobs)} LLMSimulator base runs", flush=True)
    rows = []
    for over, df, log in run_many(jobs):
        meta = over["_meta"]
        if df is None:
            print("FAILED", meta, log[-400:])
            continue
        r = df.iloc[-1].to_dict()
        step = (q2.g(r, "non_MoE_decoders") + q2.g(r, "MoE_decoders")) / 1e6
        exec_ms = q2.g(r, "expert_ffn") / 1e6
        rows.append(dict(gpu=meta["gpu"], model=meta["model"], batch=meta["batch"], context=meta["context"],
                         base_step_ms=step, exec_ms=exec_ms, non_expert_ms=step - exec_ms,
                         attention_ms=(q2.g(r, "atten_sum") + q2.g(r, "atten_gen")) / 1e6,
                         nonexpert_dyn_nJ=sum(q2.g(r, c) for c in NE_COLS),
                         moe_comp_nJ=q2.g(r, "moe_comp")))
    t = pd.DataFrame(rows).sort_values(["gpu", "model", "batch", "context"])
    t.to_csv(os.path.join(out, "context_base.csv"), index=False)

    # check: context 1024 must reproduce the archived Q2 base pass
    q2s = pd.read_csv(os.path.join(REPO, "results-ca", "q2_offload_load_vs_exec", "data", "q2_summary.csv"))
    q2s = q2s.drop_duplicates(["gpu", "model", "batch"]).set_index(["gpu", "model", "batch"])
    chk = t[t.context == 1024].set_index(["gpu", "model", "batch"]).join(q2s[["non_expert_ms"]], rsuffix="_q2")
    err = (chk.non_expert_ms / chk.non_expert_ms_q2 - 1).abs().max()
    print(f"context 1024 vs archived Q2 non-expert time: max relative difference {err:.2e}")
    pd.set_option("display.width", 200)
    print(t.pivot_table(index=["gpu", "model", "batch"], columns="context", values="non_expert_ms").round(3))


if __name__ == "__main__":
    main()
