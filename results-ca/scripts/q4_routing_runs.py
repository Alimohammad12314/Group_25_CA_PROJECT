"""Q4 step 1: log every routing decision the simulator makes.

SSD-offloading simulator (patched sequence.cpp), single H100, experts in HBM (routing does
not depend on where weights live), decode mode, batch 16, 128 decode steps, context 1024.
Routing scenarios (config.yaml -> simulation.routing):
  uniform     : simulator default, i.i.d. uniform top-k per token per layer
  zipf        : i.i.d. draws from a per-layer Zipf(s=1.0) popularity (frequency skew only)
  correlated  : zipf_s=0.8 base + p_temporal=0.5 (reuse previous token's expert at the same
                layer) + p_spatial=0.6 (follow a fixed layer-to-layer expert mapping).
                The parameters are ASSUMED (ST-MoE-style locality), not measured.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from simrun import RESULTS, SSD_BASE_CFG, SSDSIM, load_yaml, run_many  # noqa: E402

OUT = os.path.join(RESULTS, "q4_expert_correlation", "runs")
MODELS = ["mixtral", "grok1", "llama4_maverick", "deepseekR1"]
SCEN = {
    "uniform":    {"model": "uniform"},
    "zipf":       {"model": "zipf", "zipf_s": 1.0, "seed": 2026},
    "correlated": {"model": "correlated", "zipf_s": 0.8, "p_temporal": 0.5, "p_spatial": 0.6, "seed": 2026},
}
BATCH, STEPS = 16, 128


def main():
    base = load_yaml(SSD_BASE_CFG)
    jobs = []
    models = [a for a in sys.argv[1:] if a in MODELS] or MODELS
    for m in models:
        for sc, r in SCEN.items():
            mla = m.startswith("deepseek")
            o = {"model": {"model_name": m},
                 "system": {"gpu_gen": "H100", "num_device": 1, "processor_type": "GPU",
                            "offload": {"offload_expert_to_cpu_memory": False, "collects_gpu_compute": True},
                            "distribution": {"expert_tensor_degree": 1, "none_expert_tensor_degree": 1},
                            "optimization": {"compressed_kv": mla, "use_absorb": mla}},
                 "serving": {"max_batch_size": BATCH},
                 "simulation": {"input_len": 1024, "output_len": STEPS + 1, "iter": STEPS,
                                "precision_byte": 2, "routing": dict(r, log_file="routing_log.csv")},
                 "_meta": {"model": m, "scenario": sc}}
            jobs.append((SSDSIM, base, o, os.path.join(OUT, f"{m}_{sc}")))
    for over, df, log in run_many(jobs, procs=12):
        print(over["_meta"], "ok" if df is not None else "FAILED " + log[-300:])


if __name__ == "__main__":
    main()
