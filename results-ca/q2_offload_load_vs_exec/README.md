# Q2 — Single GPU, no expert sharding, no Logic-PIM, all experts in off-chip DRAM

**Goal (mid-eval item 2):** disable multi-GPU expert sharding and Logic-PIM, add an off-chip DRAM that holds all expert weights, and report **expert loading time** and **expert execution time**.

## At a glance: expert loading time vs expert execution time

In every MoE layer the router picks k experts for each token. With all experts stored in off-chip DRAM, each picked expert must go through two steps before the layer can finish:

```
 off-chip DRAM ──(1) LOAD: copy the expert's weights over the link──► GPU HBM ──(2) EXECUTE: GPU runs the expert FFN──► output
               (PCIe 32/64 GB/s or NVLink-C2C 450/900 GB/s)          (HBM 2039 GB/s on A100)
```

| | **Expert loading time** | **Expert execution time** |
|---|---|---|
| What it is | Time to copy the selected experts' weights from off-chip DRAM into GPU HBM | Time for the GPU to run the selected experts (gate, up and down matrix multiplies plus SiLU) once their weights are in HBM |
| Limited by | link bandwidth (32–900 GB/s) | HBM bandwidth (2039 GB/s on A100). During decode an expert sees only a few tokens, so the GPU spends its time reading weights, not computing |
| Formula | bytes of selected experts ÷ link bandwidth | ≈ bytes of selected experts ÷ HBM bandwidth (simulator roofline: max(compute, memory)) |
| Simulator run used | **copy pass** (offload on, `collects_gpu_compute: off`) | **base pass** (offload off, experts already in HBM) |
| CSV column | `moe_offload` energy → bytes → ÷ link bandwidth (copy-pass `expert_ffn` agrees within 0.3%) | `expert_ffn` (ns → ms) |

Both times are reported **per decode step**: one step generates one token for every sequence in the batch and runs through all MoE layers. Per-layer and per-expert values are given too.

### Worked example: Mixtral on A100, batch 1, PCIe 4.0 link
- One Mixtral expert = 3 matrices × 4096 × 14336 × 2 bytes = **352 MB**.
- Batch 1 → 1 token → the router picks **k = 2 experts per layer**, in each of **32 layers**.

| | One expert | One layer (2 experts) | Whole decode step (32 layers) |
|---|---|---|---|
| **Load** over PCIe4 (32 GB/s) | 352 MB ÷ 32 GB/s = **11.0 ms** | 22.0 ms | **704.6 ms** |
| **Execute** from HBM (2039 GB/s) | 352 MB ÷ 2039 GB/s = **0.173 ms** | 0.35 ms | **11.07 ms** |
| **Load ÷ Execute** | 2039 ÷ 32 = **63.7×** | 63.7× | **63.7×** |

The hand values match the simulator: 0.173 ms per expert executed and 704.6 ms per step loaded. Both operations move the **same bytes** (the expert's weights); only the pipe differs. So **loading ÷ execution ≈ HBM bandwidth ÷ link bandwidth**.

### Comparison for all four models (A100, batch 1, one decode step)

| Model | Experts moved per step | Execution time | Loading time, PCIe4 (32 GB/s) | Load ÷ Exec | Loading time, C2C-900 (900 GB/s) | Load ÷ Exec |
|---|---|---|---|---|---|---|
| Mixtral | 2 × 32 layers × 352 MB = 22.5 GB | 11.07 ms | 704.6 ms | **63.7×** | 25.1 ms | **2.26×** |
| Grok-1 | 2 × 64 × 1208 MB = 154.6 GB | 75.86 ms | 4832 ms | **63.7×** | 171.8 ms | **2.26×** |
| Llama-4 Maverick | 1 × 24 × 252 MB = 6.0 GB | 2.98 ms | 188.7 ms | **63.4×** | 6.7 ms | **2.25×** |
| DeepSeek-R1 | 8 × 57 × 88 MB = 40.2 GB | 19.82 ms | 1255 ms | **63.3×** | 44.6 ms | **2.25×** |

The ratio stays the same at larger batch: Mixtral at batch 64 loads all 8 experts per layer (90 GB) in 2819 ms vs 44.6 ms execution, **63.1×** on PCIe4 and **2.25×** on C2C-900. On H100 (3352 GB/s HBM) the ratios are larger: up to 105× (PCIe4) and 3.7× (C2C-900).

**Conclusion:** expert loading is slower than expert execution in **all 160 configurations** (2 GPUs × 4 models × 5 batch sizes × 4 links). Even the fastest link (900 GB/s) is 2.3× (A100) to 3.7× (H100) slower than HBM. If experts are fetched only when the router asks for them, the GPU sits idle for most of the step (§4.4). Sections 2–4 give the full method and all results; Q3 explains why the ratio equals the bandwidth ratio.

## 1. What was changed relative to Q1

| Q1 Duplex configuration | Q2 configuration | Config key |
|---|---|---|
| 4 devices, expert parallelism | **1 device** | `num_device: 1`, `expert_tensor_degree: 1`, `none_expert_tensor_degree: 1` |
| xPU + Logic-PIM | **GPU only** | `processor_type: GPU`. In the SSD-offloading copy the `LOGIC` / `GPU+LOGIC` / `GPU+PIM` branches are commented out in `eval/test.cpp`, so Logic-PIM cannot be enabled. |
| experts in HBM | **routed experts in off-chip DRAM** (host DDR5) | `offload.offload_expert_to_cpu_memory: on` |
| H100 | **A100-80GB** (reference in the assignment) and H100-80GB | `gpu_gen` |

Simulator: the `LLMSimulator/` submodule, a copy of LLMSimulator extended by the SSD-offloading authors for expert offloading. Shared experts (Llama-4, DeepSeek), attention, router and embedding weights stay in HBM; only the **routed** experts live off-chip.

### Off-chip DRAM link
The off-chip DRAM is modeled as a bandwidth-limited link feeding the GPU's HBM. Four link types were swept:

| Link | Bandwidth | Typical system |
|---|---|---|
| PCIe 4.0 ×16 | 32 GB/s | A100 PCIe / DGX A100 host DRAM |
| PCIe 5.0 ×16 | 64 GB/s | H100 PCIe host DRAM |
| NVLink-C2C 450 | 450 GB/s | Grace-Hopper, one direction (simulator's `offload_bandwidth: 4`) |
| NVLink-C2C 900 | 900 GB/s | Grace-Hopper, both directions (simulator's `offload_bandwidth: 5`) |

### Simulator changes (`../simulator_patches/ssd_offloading_llmsim.patch`)
1. `hardware_config.h`: a new optional key `offload.offload_bandwidth_GBps` sets the link bandwidth directly. The stock code only knows 450 and 900 GB/s, so this was needed to model PCIe.
2. `device.cpp`: `gpu_gen: A100` crashed because no DRAM config file was selected for A100. A100 now reuses the HBM3 file. In this GPU-only build that file only initializes the unused Ramulator interface (`use_ramulator` is off). All A100 timing comes from the A100 preset: 312 TFLOPS FP16, 2.039 TB/s HBM, 80 GB.
3. `sequence.cpp`: routing models and a routing log (used in Q4). The default stays the original i.i.d. uniform routing.
4. `scheduler.cpp`: **bug fix** in the stock uniform expert draw. `uniform_int(0, 256) % N` over-weighted expert 0 (2× for N=256). It is now `uniform_int(0, N−1)`, as in the original LLMSimulator. All Q2 results here were produced after the fix.
5. `linear_impl.cpp`, `module_graph.cpp`: `config.yaml` is parsed once per run instead of in every layer. Results are bit-identical and runs are about 13× faster.

## 2. How the simulator computes time (so the numbers can be checked by hand)
Every linear layer (`src/hardware/linear_impl.cpp`) with input `m×k` and weight `k×n` is timed as a roofline:

```
compute  = 2·m·k·n / peak_FLOPS
memory   = (m·k + k·n + m·n)·2 B / HBM_bandwidth
latency  = max(compute, memory)
```

With offloading enabled, the HBM bandwidth available to compute is reduced: `HBM_bandwidth = base − link_bandwidth`, because the copy also writes into HBM. The stock framework separates "loading" from "executing" by running the **same workload three times**:

| Pass | Config | What `expert_ffn` contains |
|---|---|---|
| **base** | offload off | expert FFN execution with experts already in HBM (full HBM bandwidth) |
| **compute** | offload on, `collects_gpu_compute: on` | expert execution with HBM bandwidth reduced by the concurrent copy |
| **copy** | offload on, `collects_gpu_compute: off` | for every linear layer: `latency = k·n·2 B / link_bandwidth`, the time to move its weights over the link |

The copy pass also accumulates `moe_offload` energy = bytes × 8 × (9.2 pJ/bit DDR5 read + 4.2 pJ/bit HBM write).

All runs: context 1024, first decode step after prefill (`output_len 2`, `iter 2`, as in the authors' `figure_4_and_5.py`), FP16, batch ∈ {1, 4, 16, 64, 256}, decode mode. Total: 2 GPUs × 4 models × 5 batches × (1 base + 4 links × 2 passes) = 360 SSD-simulator runs, plus 9 ablation runs on the original simulator.

## 3. Definitions used in this project

| Quantity | Definition | Source |
|---|---|---|
| **Off-chip bytes per step** | `moe_offload` energy ÷ (13.4 pJ/bit × 8), minus the router weights (h·N·2 B per layer, which stay in HBM) | copy pass |
| **Unique experts per layer** | off-chip bytes ÷ (expert size × MoE layers) | copy pass |
| **Expert loading time** | off-chip bytes ÷ link bandwidth, i.e. time to bring every expert the router selected in this step from DRAM into HBM | copy pass |
| **Expert execution time** | `expert_ffn` of the base pass (experts resident, full HBM bandwidth) | base pass |
| Expert execution (contended) | `expert_ffn` of the compute pass (HBM shared with a concurrent copy) | compute pass |
| Step latency, all in HBM | `non_MoE_decoders + MoE_decoders` of the base pass (ideal, capacity ignored) | base pass |
| **On-demand step latency** | base step + loading time. Experts are fetched only after the router picks them, and compute waits for them (no overlap). | derived |
| **Exposed stall** | on-demand step − base step = loading time | derived |
| Overlap bound | max(loading time, contended step), the best any prefetcher could do with this link | derived |
| **Energy per token** | [dynamic HBM+compute energy (base) + off-chip transfer energy + P_bg × latency] ÷ batch, with P_bg = 70 W (GPU+HBM) + 57 W (256 GB host DRAM) for offloaded runs and 70 W for all-in-HBM. Background constants are taken from `figure_4_and_5.py`. | derived |

Expert size = 3 · hidden · d_expert · 2 B (gate, up and down projections, FP16): Mixtral 352 MB, Grok-1 1208 MB, Llama-4 Maverick 252 MB, DeepSeek-R1 88 MB.

**Check of the simulator against the formula.** Mixtral, batch 1, PCIe4 has 2 experts × 32 layers × 352.3 MB = 22.55 GB. At 32 GB/s that is 704.6 ms; the simulator's copy-pass `expert_ffn` is 704.7 ms (the extra 0.07 ms is the 2 MB of router weights). The copy-pass `expert_ffn` agrees with bytes ÷ bandwidth to within 0.3% everywhere. The small excess at large batch comes from the SiLU·mul activation and scatter/gather ops inside `expert_ffn`, which are charged at HBM bandwidth even in the copy pass. This also explains why "loaded GB" read from copy-pass time slightly exceeded 90 GB at batch ≥ 64 in the earlier sweep.

## 4. Results

`data/q2_summary.csv` has one row per (GPU, model, batch, link) with every quantity above. `data/q2_runs_long.csv` holds the raw simulator outputs, and `runs/` has the per-run config, log and CSV.

### 4.1 Loading vs execution per decode step, A100, batch 1 (`plots/q2_fig1_load_vs_exec_A100.png`)

| Model | experts × size | GB moved / step | Execute (ms) | Load PCIe4 | Load PCIe5 | Load C2C-450 | Load C2C-900 |
|---|---|---|---|---|---|---|---|
| Mixtral | 2 × 32 layers × 352 MB | 22.5 | 11.07 | 704.6 | 352.3 | 50.1 | 25.1 |
| Grok-1 | 2 × 64 × 1208 MB | 154.6 | 75.86 | 4832 | 2416 | 343.6 | 171.8 |
| Llama-4 Maverick | 1 × 24 × 252 MB | 6.0 | 2.98 | 188.7 | 94.4 | 13.4 | 6.7 |
| DeepSeek-R1 | 8 × 57 × 88 MB | 40.2 | 19.82 | 1255 | 627.6 | 89.3 | 44.6 |

Per layer (Mixtral, A100, B=1): executing the two selected experts takes **0.35 ms**. Loading them takes **22.0 ms** (PCIe4), **11.0 ms** (PCIe5), **1.57 ms** (C2C-450) or **0.78 ms** (C2C-900).

### 4.2 Ratio loading ÷ execution (`plots/q2_fig2_load_exec_ratio_*.png`)

| GPU | PCIe4 | PCIe5 | C2C-450 | C2C-900 |
|---|---|---|---|---|
| A100 | 61–64× | 31–32× | 4.4–4.5× | 2.2–2.3× |
| H100 | 101–105× | 51–52× | 7.2–7.4× | 3.6–3.7× |

The ratio is essentially **HBM bandwidth ÷ link bandwidth** (A100: 2039/32 = 63.7; H100: 3352/900 = 3.72). It is the same for all four models and nearly independent of batch size (see Q3 for why). **Loading is more expensive than execution in every configuration simulated (160 of 160).**

### 4.3 How many experts must be loaded (`plots/q2_fig3_unique_experts_and_traffic.png`)
The unique experts per layer measured in the simulator match the uniform-routing expectation N·(1 − (1 − k/N)^B) almost exactly:

| Model | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| Mixtral (N=8, k=2) | 2.00 | 5.06 (5.47) | 7.84 (7.92) | 8 | 8 |
| Llama-4 Maverick (128, 1) | 1.00 | 3.88 (3.95) | 15.0 (15.1) | 49.7 (50.5) | 110.3 (110.8) |
| DeepSeek-R1 (256, 8) | 8.00 | 30.7 (30.5) | 101.4 (102.0) | 223.4 (222.4) | 255.9 (255.9) |

Values in parentheses are the formula. Q2 simulates a single decode step, so the measured values carry sampling noise (e.g. Mixtral B=4: 5.06 vs 5.47). The 128-step routing logs of Q4 match the formula closely (Mixtral B=16: 7.926 vs 7.920 unique experts per layer per step). Consequence: with 8 experts (Mixtral, Grok-1), **every** expert of every layer is loaded at every step from batch ≈ 16 on, which is 90 GB per step for Mixtral and 618 GB for Grok-1. Fine-grained models (128 or 256 experts) need only a subset at small batch.

### 4.4 Effect on the decode step (`plots/q2_fig4_step_breakdown_A100_b*.png`)
Exposed stall as a fraction of the on-demand step (A100):

| Model, batch | PCIe4 | PCIe5 | C2C-450 | C2C-900 | slowdown vs all-in-HBM (C2C-900) |
|---|---|---|---|---|---|
| Mixtral, 1 | 98.3% | 96.6% | 80.1% | 66.8% | 3.0× |
| Mixtral, 64 | 98.2% | 96.5% | 79.9% | 66.5% | 3.0× |
| Grok-1, 1 | 98.3% | 96.7% | 80.8% | 67.8% | 3.1× |
| Maverick, 1 | 92.7% | 86.3% | 47.3% | 31.0% | 1.45× |
| DeepSeek-R1, 1 | 97.3% | 94.8% | 72.0% | 56.3% | 2.3× |

### 4.5 Ablation: from the Duplex configuration to the offloaded single GPU (`plots/q2_fig5_ablation_duplex_to_offload.png`)
Mixtral, H100, context 1024, analytic timing (Ramulator off) in both simulators so they are comparable:

| Configuration | B=1 | B=16 | B=64 |
|---|---|---|---|
| Duplex (4 dev, Logic-PIM) | 1.46 ms | 2.29 | 4.81 |
| GPU (4 dev, expert parallel) | 4.31 | 7.47 | 8.14 |
| GPU (1 dev, experts in HBM), original simulator | 7.58 | 27.9 | 30.7 |
| GPU (1 dev, experts in HBM), SSD simulator (cross-check) | 7.58 | 27.9 | 30.8 |
| GPU (1 dev, experts off-chip, C2C-900, on-demand) | 32.6 | 126.2 | 131.0 |
| GPU (1 dev, experts off-chip, PCIe4, on-demand) | 712.2 | 2791 | 2849 |

The two simulators agree on the single-GPU, experts-in-HBM step: 7.575 vs 7.576 ms at B=1, 27.93 vs 27.95 at B=16, and 30.67 vs 30.75 at B=64.

### 4.6 Energy (`plots/q2_fig6_energy_A100_b*.png`)
A100, batch 1, energy per generated token: Mixtral 1.82 J with all experts in HBM vs 8.1 J (C2C-900) and 94.4 J (PCIe4). The off-chip transfer itself costs 2.4 J per step for Mixtral (22.5 GB × 13.4 pJ/bit × 8 bit/B). The dominant term on slow links is **background power during the stall**: 127 W × 0.70 s = 89 J on PCIe4. Reducing exposed stalls is therefore also the main lever for energy.

## 5. Caveats
- HBM capacity is not enforced. The "all experts in HBM" baseline is an idealized reference: 90 GB of Mixtral experts alone exceeds 80 GB.
- The A100 HBM uses the HBM3 read energy (4.2 pJ/bit) of the simulator's energy model.
- Only the first decode step is simulated, and routing is i.i.d. uniform (see Q4). No expert is reused from HBM between steps, because the framework has no expert cache. That cache, plus prefetching into it, is the subject of the final deliverable.

## How to reproduce
```bash
cd results-ca        # from the repository root
python3 scripts/q2_offload.py                    # about 10 min on 28 processes (DeepSeek runs are slow)
python3 scripts/q2_offload.py --summarize-only   # recompute data/ from archived runs/
python3 scripts/plot_q2_q3.py
```
