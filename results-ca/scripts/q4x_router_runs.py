"""Q4-extended step 1: routing decisions from a mechanistic router inside the simulator.

routing.model = router (patched sequence.cpp): every token carries a hidden vector that drifts
from token to token (token_similarity, rho) and is changed a little by every layer through the
residual stream (layer_update, alpha); each MoE layer's router scores all experts as W_l h + b_l
and keeps the top-k -- the same gating rule a real MoE layer uses. Router weights are random
(no trained model), so rho and alpha are ASSUMED knobs; the experiment shows how the expert
patterns depend on them.

Runs (SSD-offloading simulator, single H100, decode, batch 16, 128 steps, context 1024):
  sweep : mixtral, deepseekR1  x  rho in RHOS  x  alpha in ALPHAS
  ref   : all four models at rho=0.9, alpha=0.5, with router_bias 0 and 0.5 (popular experts)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from simrun import RESULTS, SSD_BASE_CFG, SSDSIM, load_yaml, run_many  # noqa: E402

OUT = os.path.join(RESULTS, "q4_extended_router_logic", "runs")
MODELS = ["mixtral", "grok1", "llama4_maverick", "deepseekR1"]
SWEEP_MODELS = ["mixtral", "deepseekR1"]
RHOS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
ALPHAS = [0.05, 0.25, 0.5, 1.0, 2.0]
REF_RHO, REF_ALPHA = 0.9, 0.5
BATCH, STEPS = 16, 128


def run_name(model, rho, alpha, bias):
    return f"{model}_rho{rho:g}_alpha{alpha:g}_bias{bias:g}"


def job(base, model, rho, alpha, bias):
    mla = model.startswith("deepseek")
    o = {"model": {"model_name": model},
         "system": {"gpu_gen": "H100", "num_device": 1, "processor_type": "GPU",
                    "offload": {"offload_expert_to_cpu_memory": False, "collects_gpu_compute": True},
                    "distribution": {"expert_tensor_degree": 1, "none_expert_tensor_degree": 1},
                    "optimization": {"compressed_kv": mla, "use_absorb": mla}},
         "serving": {"max_batch_size": BATCH},
         "simulation": {"input_len": 1024, "output_len": STEPS + 1, "iter": STEPS, "precision_byte": 2,
                        "routing": {"model": "router", "router_dim": 64, "token_similarity": rho,
                                    "layer_update": alpha, "router_bias": bias, "seed": 2026,
                                    "log_file": "routing_log.csv"}},
         "_meta": {"model": model, "rho": rho, "alpha": alpha, "bias": bias}}
    return (SSDSIM, base, o, os.path.join(OUT, run_name(model, rho, alpha, bias)))


def all_points():
    pts = {(m, r, a, 0.0) for m in SWEEP_MODELS for r in RHOS for a in ALPHAS}
    pts |= {(m, REF_RHO, REF_ALPHA, b) for m in MODELS for b in (0.0, 0.5)}
    return sorted(pts)


def main():
    base = load_yaml(SSD_BASE_CFG)
    pts = [p for p in all_points() if not os.path.exists(
        os.path.join(OUT, run_name(*p), "routing_log.csv"))]
    print(f"{len(pts)} runs to do")
    for over, df, log in run_many([job(base, *p) for p in pts], procs=28):
        print(over["_meta"], "ok" if df is not None else "FAILED " + log[-300:])


if __name__ == "__main__":
    main()
