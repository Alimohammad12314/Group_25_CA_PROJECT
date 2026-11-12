"""Validation gate: with no cache, no prefetch and strict on-demand loading, the timeline must
reproduce Q2's on-demand step latency and energy (LLMSimulator / SSD-offloading runs).

Routing is LLMSimulator's own (uniform, independent per token), as in Q2. Nothing else in the
project is trusted until every point here is within tolerance.

Two checks:
  batch 1    every token routes to exactly k experts per layer in both runs, so the expert sets
             have the same size and latency and energy must match exactly (tolerance 0.1 %)
  batch > 1  Q2 ran a single decode step, so its unique-expert count is one random draw
             (Mixtral batch 4: 5.06 per layer, against the exact expectation 8 (1 - 0.75^4) = 5.47).
             The gate compares cost per unique expert instead: Q2's non-expert time plus Q2's
             load + execution time rescaled by (our unique experts / Q2's unique experts).

    python -m experiments.e0_validation_gate
"""
import os
import sys

import pandas as pd

from engine.costs import LINKS, MODELS, CostModel
from engine.timeline import SimConfig, simulate
from traces.workload import make_uniform_workload

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e0_validation")
TOL = 0.02     # 2 %


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for gpu in ("A100", "H100"):
        for model, spec in MODELS.items():
            for batch in (1, 4, 16):
                steps = 16 if spec["N"] > 8 else 32
                wl = make_uniform_workload(model, spec["N"], spec["k"], spec["moe_layers"], batch, steps, seed=1)
                needed = wl.needed()
                for link in LINKS:
                    cfg = SimConfig(gpu=gpu, link=link, cache_ratio=0.0, prefetch=False)
                    r = simulate(wl, needed, None, cfg)
                    q2 = CostModel(gpu, model, link, batch)
                    ref_ms = q2.q2("ondemand_step_ms")
                    ref_e = q2.q2("energy_ondemand_J_per_token")
                    uniq = r["GB_per_token"] * 1e9 / q2.expert_bytes / q2.L
                    q2_uniq = q2.q2("unique_experts_per_layer_sim")
                    scale = uniq / q2_uniq
                    norm_ms = q2.q2("non_expert_ms") + scale * (q2.q2("load_ms") + q2.q2("exec_ms"))
                    rows.append(dict(gpu=gpu, model=model, batch=batch, link=link,
                                     sim_ms=r["token_latency_ms"], q2_ms=ref_ms,
                                     err_ms=r["token_latency_ms"] / ref_ms - 1,
                                     q2_ms_per_unique=norm_ms,
                                     err_ms_per_unique=r["token_latency_ms"] / norm_ms - 1,
                                     sim_J=r["energy_J_per_token"], q2_J=ref_e,
                                     err_J=r["energy_J_per_token"] / ref_e - 1,
                                     unique_per_layer=uniq, q2_unique_per_layer=q2_uniq))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "validation_gate.csv"), index=False)
    b1 = df[(df.gpu == "A100") & (df.model == "mixtral") & (df.batch == 1)]
    print(b1[["link", "sim_ms", "q2_ms", "err_ms", "sim_J", "q2_J", "err_J"]].round(4).to_string(index=False))
    one, many = df[df.batch == 1], df[df.batch > 1]
    e1 = max(one.err_ms.abs().max(), one.err_J.abs().max())
    e2 = many.err_ms_per_unique.abs().max()
    print(f"\nbatch 1, {len(one)} points: worst latency/energy error {100 * e1:.3f} %")
    print(f"batch 4 and 16, {len(many)} points: worst error per unique expert {100 * e2:.2f} % "
          f"(raw error vs Q2's single-step draw up to {100 * many.err_ms.abs().max():.1f} %)")
    ok = e1 <= 0.001 and e2 <= TOL
    print("GATE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
