# Project 2: The MoE Memory Wall — mid-evaluation results

Every experiment here was run in simulation, with **LLMSimulator** (Duplex, modified Ramulator 2.0) and the **SSD-offloading** copy of LLMSimulator. No real model, model weights or external data were used.

| Mid-eval item | Folder | One-line result |
|---|---|---|
| 1. Reproduce Duplex | [`q1_duplex_reproduction/`](q1_duplex_reproduction/README.md) | 207 runs, 4 models × 6 systems. Ordering GPU < 2xGPU < Duplex ≤ PE ≤ PE+ET reproduced; Mixtral Duplex 2.0–2.9× at B=32–64, gain shrinking at B=128; Fig. 14 points within 0.05–0.13×. |
| 2. Single GPU, no sharding, no Logic-PIM, experts in off-chip DRAM | [`q2_offload_load_vs_exec/`](q2_offload_load_vs_exec/README.md) | 360 runs: A100/H100 × 4 models × 5 batches × 4 links. Loading exceeds execution in 160 of 160 configurations, by 2.2× (A100, C2C-900) to 105× (H100, PCIe4). |
| 3. MoE flow and why loading is the bottleneck | [`q3_moe_flow_bottleneck/`](q3_moe_flow_bottleneck/README.md) | load/exec ≈ HBM BW ÷ link BW, because decode experts have Op/B = tokens/expert ≪ ridge. Break-even needs 347–30919 tokens per expert. |
| 4. Temporal and spatial correlation of expert accesses | [`q4_expert_correlation/`](q4_expert_correlation/README.md) | The simulator's routing is i.i.d. (uniform or Zipf), so measured correlation is zero; Zipf adds only frequency skew. A parameterised correlated mode is added for the predictor work. |

`simulator_patches/` documents the source changes to both simulators, `scripts/` holds every script used, and each `qN_*/runs/` folder keeps the exact `config.yaml`, simulator log and raw CSV of every run.

## Reproduce everything
```bash
cd results-ca        # from the repository root
python3 scripts/q1_duplex.py        && python3 scripts/plot_q1.py        # Q1
python3 scripts/q2_offload.py       && python3 scripts/plot_q2_q3.py     # Q2 + Q3
python3 scripts/q4_routing_runs.py  && python3 scripts/q4_analysis.py    # Q4
```
`scripts/simrun.py` runs each configuration in its own temporary directory (both binaries read `./config.yaml`), 28 at a time. Base configs are in `configs/`. Figures (`qN_*/plots/*.png`) are not committed: regenerate them with the `plot_*` / `q4_analysis.py` scripts, which only read the committed `data/` and `runs/` (no simulation needed). `scripts/legacy/` holds the earlier part-2/3 sweep scripts, superseded by `q2_offload.py`.

## Units and conventions
- Simulator CSV latency is in **ns** and energy in **nJ**. All tables here are in ms and J.
- "Step" means one decode iteration: every sequence in the batch produces one token.
- FP16 weights everywhere (`precision_byte: 2`).
- HBM capacity is not enforced by the simulator, so "all experts in HBM" is an idealised reference.


