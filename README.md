# CA Project 2: The MoE Memory Wall — Predicting and Prefetching Experts

**Group 25**

| Members |
|---|
| Palak Mishra |
| Mohammad Ali |
| Shivansh Bhatnagar |

Course project (Computer Architecture, IIIT Delhi). We study expert-weight movement in Mixture-of-Experts LLM inference **entirely in simulation**, using:

- **LLMSimulator** (scale-snu), the simulator of the MICRO 2024 *Duplex* paper, with a modified Ramulator 2.0; and
- **SSD-offloading** (scale-snu), an extension of LLMSimulator that models MoE expert weights offloaded to CPU memory or SSD.

This repository is a fork of SSD-offloading. It adds our experiment scripts, results and simulator patches.

| Mid-evaluation item | Where |
|---|---|
| Q1 Reproduce the Duplex configuration | [`results-ca/q1_duplex_reproduction/`](results-ca/q1_duplex_reproduction/README.md) |
| Q2 Single GPU, no expert sharding, no Logic-PIM, experts in off-chip DRAM: expert loading vs execution time | [`results-ca/q2_offload_load_vs_exec/`](results-ca/q2_offload_load_vs_exec/README.md) |
| Q3 MoE execution flow and why expert loading is the bottleneck | [`results-ca/q3_moe_flow_bottleneck/`](results-ca/q3_moe_flow_bottleneck/README.md) |
| Q4 Temporal and spatial correlation of expert accesses | [`results-ca/q4_expert_correlation/`](results-ca/q4_expert_correlation/README.md) |

The overview of all results is in [`results-ca/README.md`](results-ca/README.md).

## Repository layout
```
.
├── setup.sh                     # one-time setup: submodules + all patches + build both simulators
├── LLMSimulator/                # submodule: SSD-offloading's LLMSimulator copy (GPU-only + offloading) -> Q2-Q4
├── external/LLMSimulator/       # submodule: original LLMSimulator (Duplex, Logic-PIM)                  -> Q1
├── patch/ssd_offloading.patch   # upstream SSD-offloading patch (applied by setup.sh / patches.sh)
├── patches.sh                   # upstream patch script (kept for reference; setup.sh supersedes it)
└── results-ca/
    ├── README.md                # summary of all mid-evaluation results
    ├── q1_.. q4_../             # per question: README (method + results), data/, runs/ (exact configs + raw output)
    ├── scripts/                 # every script used (simulation sweeps, analysis, plots)
    ├── configs/                 # base configs used for all runs
    ├── simulator_patches/       # our changes to both simulators (+ README explaining each change)
    └── requirements.txt
```

## Setup on a new machine

### Prerequisites
Tested on Ubuntu 20.04 with g++ 11.4, cmake 3.16, clang 10 and Python 3.8 (numpy 1.24, pandas 2.0, scipy 1.10, matplotlib 3.7, PyYAML 5.3). No GPU is needed.

```bash
sudo apt install build-essential g++-11 cmake clang git python3-pip   # if not already installed
```

### 1. Clone (with submodules)
```bash
git clone --recursive https://github.com/Alimohammad12314/ca-project.git
cd ca-project
```
If you cloned without `--recursive`, `setup.sh` fetches the submodules for you.

### 2. Patch and build both simulators
```bash
./setup.sh          # about 5-10 min; set JOBS=8 to limit parallel compile jobs
```
`setup.sh` is safe to re-run. In order, it:
1. fetches both submodules (and their nested Ramulator 2.0 submodule);
2. applies the upstream patches: `ramulator2_pim.patch` (both simulators) and `patch/ssd_offloading.patch`;
3. applies our patches from `results-ca/simulator_patches/` (see its [README](results-ca/simulator_patches/README.md)): PCIe link bandwidth option, A100 support, routing models and routing log, MoE-layer counter fix, expert-RNG bias fix, config caching, and the `llama3_70B` / `glam` presets;
4. builds `LLMSimulator/build/run` and `external/LLMSimulator/build/run`.

### 3. Python packages
```bash
pip install -r results-ca/requirements.txt
```

### 4. Regenerate figures (fast, no simulation)
Figures are not committed (binary files). Rebuild them from the committed data:
```bash
cd results-ca
python3 scripts/plot_q1.py        # Q1 figures
python3 scripts/plot_q2_q3.py     # Q2 + Q3 figures
python3 scripts/q4_analysis.py    # Q4 statistics + figures (reads runs/*/routing_log.csv)
```

### 5. Re-run the simulations (optional)
```bash
cd results-ca
python3 scripts/q1_duplex.py          # 207 runs,  ~3 min on 28 cores
python3 scripts/q2_offload.py         # 369 runs,  ~5 min
python3 scripts/q4_routing_runs.py    # 12 runs,   ~5 min
```
Each script writes `data/` and `runs/` in its question folder. To use simulator builds elsewhere, set `LLMSIM_ROOT` (original) or `SSDSIM_ROOT` (SSD-offloading copy) to the simulator directory that contains `build/run`.

## Simulation conventions
- Simulator CSV output: latency in ns, energy in nJ. Our tables use ms and J.
- Decode mode: one decode step generates one token per sequence in the batch. FP16 weights.
- HBM capacity is not enforced by the simulator.

---

## Upstream: SSD-offloading
This modeling tool builds upon the [simulator](https://github.com/scale-snu/LLMSimulator.git) introduced in the MICRO 2024 paper "Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching." It breaks down the energy consumption and token generation latency of LLM inference per operation (e.g., QKV generation, attention, output projection). For Mixture-of-Experts (MoE) models, the breakdown can be split by whether the gated expert weights are offloaded to CPU memory or SSD.

Key features:
- The CPU-memory and SSD read bandwidths are modeled, together with the resulting reduction in device-memory bandwidth available for GPU compute when offloaded weights are read concurrently with the GPU compute.
- The memory/storage access energy, the GPU compute energy and the latency are calculated by the tool.
- In the experimental workflow, the resulting latency is used to calculate the background energy due to the memory's static power and the GPU's idle power.

Upstream experiment workflow (after `setup.sh`): `cd LLMSimulator && python figure_4_and_5.py` and `python figure_6.py`.

Upstream contact: Kwanhee Kyung (kwanhee.kyung@scale.snu.ac.kr), Sungmin Yun (sungmin.yun@snu.ac.kr).
