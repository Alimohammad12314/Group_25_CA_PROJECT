# Simulator modifications

`setup.sh` at the repository root applies both patches (after the upstream patches) and builds the simulators. The `.patch` files document exactly what changed.

## `original_llmsim.patch` → `external/LLMSimulator` (used in Q1 and the Q2 ablation)
| File | Change | Why |
|---|---|---|
| `src/model/model_config.h` | new `llama3_70B` preset (80 layers, hidden 8192, FFN 28672, 64 heads, 8 KV heads) | Duplex Fig. 14 uses Llama3-70B; the original preset list only has 405B. Dimensions copied from the SSD-offloading copy. |
| `eval/test.cpp` | `glam` and `llama3_70B` selectable via `model_name` | `glam` was defined but not wired into the driver. |

Applied by `setup.sh` (manually: `cd external/LLMSimulator && git apply ../../results-ca/simulator_patches/original_llmsim.patch`).

## `ssd_offloading_llmsim.patch` → `LLMSimulator/` (Q2, Q3, Q4)
Relative to the submodule plus the authors' `patch/ssd_offloading.patch`:

| File | Change | Effect on results |
|---|---|---|
| `src/hardware/hardware_config.h` | optional `system.offload.offload_bandwidth_GBps` overrides the off-chip link bandwidth | Needed to model PCIe 4.0/5.0 host DRAM (32/64 GB/s). Without the key, behaviour is unchanged (450/900 GB/s). |
| `src/hardware/device.cpp` | `gpu_gen: A100` selects the HBM3 DRAM config file instead of an empty path, which crashed | Timing still uses the A100 preset (312 TFLOPS, 2.039 TB/s). The DRAM file only builds the unused Ramulator interface. |
| `src/scheduler/sequence.cpp` | (1) optional `simulation.routing` block: `model: uniform` (default, original RNG stream), `zipf`, or `correlated`, plus `log_file`, which writes every routing decision as `step,layer,seq,token,e_0..e_{k-1}`. (2) The routing layer counter now counts **MoE layers** (`num_layers/expert_freq − first_k_dense`) | (1) Used by Q4. (2) Fixes label drift for DeepSeek, whose 3 leading dense layers have no router. Uniform timing results are unaffected, because the draws do not depend on the layer index. |
| `src/scheduler/scheduler.cpp` | **bug fix:** `getRandomExpert` drew from `uniform_int(0, 256)` and took `% N`. The 257 values over-weight expert 0: 2× for N=256 (DeepSeek), 1.5× for N=128 (Maverick), +3% for N=8. Now `uniform_int(0, N−1)`, as in the original LLMSimulator. | Found in Q4: expert 0 had 7226 activations vs about 3648 expected in the DeepSeek uniform log. All Q2 and Q4 results were rerun after the fix. |
| `src/hardware/linear_impl.cpp`, `src/module/module_graph.cpp` | `config.yaml` is parsed once per process (`static const`) instead of on every linear layer and energy update | About 13× faster; outputs are bit-identical. Verified on three archived Q2 runs, maximum absolute difference 0.0. |

### The `correlated` routing model (Q4 only)
For token t of sequence s at MoE layer l, each of the k slots j is drawn as follows:
1. with probability `p_temporal`: reuse the expert in slot j of token t−1 at layer l (same sequence);
2. otherwise, with probability `p_spatial`: take π_l(e), where e is this token's slot-j expert at layer l−1 and π_l is a fixed random permutation per layer (a deterministic "affinity" between consecutive layers);
3. otherwise, or if the expert is already chosen: draw from the base distribution, which is uniform or Zipf(`zipf_s`) over a random per-layer popularity order.

The parameters are **assumptions** chosen to resemble ST-MoE-style locality. They are not measured from a real model.
