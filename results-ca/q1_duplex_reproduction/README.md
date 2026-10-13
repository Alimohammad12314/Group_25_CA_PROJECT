# Q1 — Reproducing the Duplex configuration

**Goal (mid-eval item 1):** reproduce the Duplex architecture (MICRO 2024) with the provided LLMSimulator and show a working simulation.

## Simulator and setup

| Item | Value |
|---|---|
| Simulator | Original LLMSimulator (scale-snu), submodule `external/LLMSimulator`, with the Ramulator 2.0 PIM patch, built in `build/` |
| Timing | `use_ramulator: on`: memory traffic replayed through the modified Ramulator (HBM3 timing, multi-bank reads for Logic-PIM) |
| Device | H100-class xPU, HBM3 80 GB, NVLink gen 4, FP16 (`precision_byte: 2`) |
| Mode | `decode_mode: on`: every sequence is at length `input_len` (= context) and generates one token per step |
| Steps simulated | `output_len: 4`, `iter: 3`; metrics are averaged over the simulated decode steps |
| Routing | synthetic, i.i.d. uniform top-k (`skewness: 0`), same as the paper |
| Models | Mixtral-8x7B, GLaM, Grok-1 (MoE), Llama3-70B (dense) |
| Grid | batch ∈ {32, 64, 128} × context ∈ {256, 1024, 4096} (paper Fig. 11 grid) |

### System → config mapping

| System (paper) | `num_device` | `processor_type` | `expert_tensor_degree` | `parallel_execution` |
|---|---|---|---|---|
| GPU (4× H100) | 4 | `GPU` | 1 (expert parallelism) | off |
| 2xGPU | 8 | `GPU` | 1 | off |
| Bank-PIM (Fig. 14) | 4 | `GPU+PIM` | 1 | off |
| Duplex (xPU *or* Logic-PIM) | 4 | `GPU+LOGIC` | 1 | off |
| Duplex+PE (co-processing) | 4 | `GPU+LOGIC` | 1 | **on** |
| Duplex+PE+ET (tensor-parallel experts) | 4 | `GPU+LOGIC` | **4** | on |

Non-expert layers use tensor parallelism 4 (`none_expert_tensor_degree: 4`) in every system. In `eval/test.cpp`, `GPU+LOGIC` sets the xPU as the high-Op/B unit and Logic-PIM as the low-Op/B unit. `parallel_execution` is read from the config: it is commented out in the driver, so we set it explicitly.

### Simulator changes (see `../simulator_patches/original_llmsim.patch`)
- Added a `llama3_70B` preset: 80 layers, hidden 8192, FFN 28672, 64 heads, GQA 8 KV heads. The dimensions are copied from the SSD-offloading copy of the simulator, which already has this preset.
- Wired the existing `glam` preset into the driver. It was defined in `model_config.h` but never selectable.

## Metrics: how each number is computed
For each run the simulator writes one CSV row per decode step. Times are in ns and energies in nJ; the scripts convert them to ms and J.
- **Step latency** = mean of the `latency` column over the decode steps.
- **Throughput** (tokens/s) = batch ÷ step latency. Each step produces one token per sequence.
- **Speedup vs GPU** = throughput ÷ GPU throughput at the same (model, batch, context). This is the normalization the paper uses in Fig. 11.
- **Latency breakdown** = per-step sums of the timeboard columns: `expert_ffn`, `atten_gen`+`atten_sum`, `qkvgen`+`o_proj`+`ffn`, `communication`, and `rope`+`layernorm`+`residual`.
- **Energy per token** = `total_energy` ÷ batch, split into the simulator's `fc/attn/moe × dram/comp` components. This is **dynamic energy only**; the simulator has no static or background power.

## Results

`data/q1_results.csv` holds all 207 runs. The raw config, log and CSV for every run are under `runs/`.

### Mixtral throughput normalized to GPU (`plots/q1_fig1_mixtral_throughput_fig11.png`)

| batch / ctx | 2xGPU | Bank-PIM | Duplex | Duplex+PE | Duplex+PE+ET |
|---|---|---|---|---|---|
| 32 / 256 | 1.90 | 1.95 | 2.89 | 2.90 | 2.99 |
| 32 / 1024 | 1.89 | 1.95 | 2.79 | 2.84 | 2.99 |
| 32 / 4096 | 1.91 | 2.04 | 2.94 | 2.95 | 3.14 |
| 64 / 256 | 1.89 | 1.13 | 2.00 | 2.02 | 2.78 |
| 64 / 1024 | 1.91 | 1.19 | 2.07 | 2.08 | 2.83 |
| 64 / 4096 | 1.90 | 1.34 | 2.20 | 2.21 | 2.90 |
| 128 / 256 | 1.86 | 1.01 | 1.17 | 1.84 | 1.80 |
| 128 / 1024 | 1.88 | 1.12 | 1.26 | 1.95 | 1.83 |
| 128 / 4096 | 1.92 | 1.39 | 1.56 | 2.23 | 2.12 |

### Comparison with numbers stated in the paper

| Quantity | Paper | This reproduction |
|---|---|---|
| Duplex vs GPU, Mixtral (max) | up to 2.51× | 2.94× (B=32, ctx 4096) |
| Duplex+PE+ET vs GPU (max, all models) | up to 2.67× | 3.14× Mixtral, 3.42× Grok-1 |
| Duplex, Mixtral, B=64, ctx 256 (Fig. 14) | 2.05× | 2.00× |
| Bank-PIM, Mixtral, B=32, ctx 4096 (Fig. 14) | 2.17× | 2.04× |
| Bank-PIM, Mixtral, B=64, ctx 256 (Fig. 14) | 1.18× | 1.13× |
| Gain shrinks as batch grows | yes | yes (B=128 Duplex: 1.17–1.56×) |
| Energy reduction, Mixtral | up to 33.28% | 42.9% (Duplex), 35.5% (PE+ET); dynamic energy only |
| Ordering GPU < 2xGPU < Duplex ≤ PE ≤ PE+ET | yes | yes at B = 32 and 64; at B = 128, PE+ET is slightly below PE |

The qualitative trends reproduce. The absolute speedups at B = 32 come out higher than the paper's. The likely reason is that our runs are decode-only, while the paper uses continuous batching with mixed prefill/decode stages: prefill tokens raise the MoE Op/B and favour the xPU.

### Observations to explain in the report
1. **Where the time goes** (`plots/q1_fig3_latency_breakdown.png`). On 4× H100, Mixtral at B=32 spends 8.66 of 9.74 ms (89%) in the expert FFNs. Each expert sees only about B·k/N = 32·2/8 = 8 tokens, so the layer is memory-bound (Op/B ≈ 8). Logic-PIM has 4× the internal bandwidth, which cuts the expert time from 8.66 to 2.69 ms.
2. **Why the Duplex gain shrinks at B = 128.** Tokens per expert rise to 32, so Op/B rises past Logic-PIM's compute-to-bandwidth ratio (8 Op/B) and Logic-PIM becomes compute-bound. **PE** (co-processing) then helps a lot: B=128/1024 goes from 1.26× to 1.95×, because experts are split between the xPU and Logic-PIM.
3. **ET** (tensor-parallel experts) helps at B = 32–64 by spreading each expert across 4 devices. It adds all-reduce communication (0.36 → 0.55 ms per step at B=32), which makes it slightly worse than PE alone at B = 128.
4. **Bank-PIM** (Fig. 14 comparison, `plots/q1_fig5_bankpim_fig14.png`) is competitive only at very low Op/B (B=32) and collapses at B=64. That matches the paper's argument that bank-level PIM suits Op/B < 1.
5. **Llama3-70B (dense)** gains only 1.0–1.5×. Its FFN Op/B equals the batch size, so the FFN stays on the xPU, and only decode attention (GQA, Op/B ≈ 8) runs on Logic-PIM. The gain grows with context (ctx 4096: 1.18–1.54×) because attention becomes a larger share of the step.
6. **Energy** (`plots/q1_fig4_energy_breakdown_fig15.png`). Duplex saves energy mainly on MoE DRAM: Logic-PIM reads weights through TSVs without crossing the HBM PHY to the xPU. The saving falls as batch grows, matching the paper's statement.

### Limitations / deviations (state these in the report)
- **Decode-only, not continuous batching.** The simulator's mixed mode (`decode_mode: off`) warms sequences to random positions and took more than 5 minutes per configuration even without Ramulator. A 300-iteration test under Ramulator showed no mixed stage in the first 240 steps. We therefore cannot reproduce the PE benefit that comes from mixed prefill stages.
- **Energy is the simulator's dynamic energy.** There is no background or static power, so energy *ratios* are comparable with Fig. 15 but absolute values are not.
- **Memory capacity is not enforced** (`exit_out_of_memory: off`). Grok-1 at FP16 on 4×80 GB would not fit physically; the paper does not state its Grok-1 precision or device count.
- The GLaM and Bank-PIM results (6–7×) have no counterpart in the paper's text and are not validated.

## How to reproduce
```bash
cd results-ca        # from the repository root
python3 scripts/q1_duplex.py      # 207 runs, about 3 minutes on 28 processes
python3 scripts/plot_q1.py
```

## Files
- `data/q1_results.csv`: one row per (model, system, batch, ctx), with every metric above.
- `plots/q1_fig1..5*.png`: Fig. 11 / 14 / 15 style figures.
- `runs/<model>_<system>_b<batch>_ctx<ctx>/`: exact `config.yaml`, simulator log and raw output CSV.
