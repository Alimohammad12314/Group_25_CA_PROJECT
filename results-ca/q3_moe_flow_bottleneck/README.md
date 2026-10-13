# Q3 — MoE execution flow and why expert loading becomes the bottleneck

**Goal (mid-eval item 3):** explain the MoE execution flow and analyse why expert loading becomes a performance bottleneck. All numbers come from the Q2 simulations (`../q2_offload_load_vs_exec/data/q2_summary.csv`) plus the analytic model below, and the two are checked against each other.

## 1. MoE decoder layer as executed by the simulator
The order of operations is taken from `src/module/decoder.cpp` (`MoEDecoder`) and `src/module/expert.cpp` (`ExpertFFN::forward`):

```
x ─► RMSNorm ─► Attention (QKV proj, RoPE, GQA attention over KV cache, O proj) ─► +residual
  ─► RMSNorm ─► gate_fn (router: h × N linear) ─► gate_update (top-k selection, counts tokens per expert)
  ─► moe_scatter (group tokens by expert) ─► moe_route
  ─► for each routed expert e with ≥1 token:  FFN_e(x) = W_down · (SiLU(W_gate·x) ⊙ W_up·x)
  ─► (all-reduce if experts are tensor-parallel) ─► moe_gather (weighted sum of top-k outputs)
  ─► shared expert FFN (Llama-4, DeepSeek only) ─► +residual ─► next layer
```

- Only experts that received at least one token are executed. `LinearExecutionGPU` returns zero cost for an empty input, so execution and loading both scale with the number of **unique** experts per layer, not with N.
- In the offloaded system (Q2), the weights of FFN_e live in off-chip DRAM. **Which experts are needed is known only after `gate_update` of the same layer**, i.e. a few microseconds before they are used.
- With on-demand fetching the timeline of one layer is therefore: *attention → router → [stall: copy the selected experts over the link] → expert FFN*, as in `plots/q3_fig1_layer_timeline_mixtral_A100_b1.png`. There is nothing independent to overlap the copy with, because the next layer's attention needs this layer's output.

## 2. Why loading costs more than executing: five reasons, each quantified

### (a) Bandwidth gap between HBM and the off-chip link
During decode an expert is executed from HBM at HBM speed, but loaded at link speed:

| GPU | HBM | PCIe4 ×16 | PCIe5 ×16 | C2C-450 | C2C-900 |
|---|---|---|---|---|---|
| A100 | 2039 GB/s | 32 (64× slower) | 64 (32×) | 450 (4.5×) | 900 (2.3×) |
| H100 | 3352 GB/s | 32 (105×) | 64 (52×) | 450 (7.4×) | 900 (3.7×) |

The simulated load/exec ratios (Q2 §4.2) equal these factors: 61–64× on A100 with PCIe4, 2.2–2.3× with C2C-900, and 3.6–3.7× on H100 with C2C-900.

### (b) Decode-time experts have very low arithmetic intensity
For one expert with P parameters receiving t tokens, in FP16:

```
FLOPs   = 2 · t · P              bytes from HBM ≈ 2 · P (weights dominate)
Op/B    = t                      (tokens per active expert)
exec    = max(2P / BW_HBM , 2tP / FLOPS)
load    = 2P / BW_link
load/exec = BW_HBM / BW_link                     if t < FLOPS/BW_HBM  (memory-bound)
          = FLOPS / (t · BW_link)                otherwise            (compute-bound)
```

During decode each sequence contributes one token, so t = B·k / (unique experts). In our sweep t ranges from 1 to 64. That is far below the HBM ridge (A100: 153 FLOP/B, H100: 295 FLOP/B), so execution is itself memory-bound and **both** load and execute are byte-moving operations. The ratio is simply the bandwidth ratio. Loading is hidden behind compute only when every active expert receives at least t* = FLOPS / BW_link tokens (`data/q3_break_even_table.csv`):

| GPU | PCIe4 | PCIe5 | C2C-450 | C2C-900 |
|---|---|---|---|---|
| A100 | 9750 tokens/expert | 4875 | 693 | 347 |
| H100 | 30919 | 15459 | 2199 | 1099 |

The decode batches simulated give at most 64 tokens per expert (Mixtral, B=256), so no decode configuration gets near break-even. `plots/q3_fig2_roofline_load_vs_exec.png` overlays the simulator points (markers) on the analytic curves (lines); every point lies on the flat, memory-bound plateau.

### (c) Experts are large, and each one is reused only within one layer and one step
One expert is 88 MB (DeepSeek-R1) to 1.2 GB (Grok-1). One decode step of Mixtral at batch 1 touches 2 experts × 32 layers = 22.5 GB of expert weights to produce a single token, so 22.5 GB/token crosses the link (`plots/q3_fig3_capacity_and_bytes_per_token.png`). Without an HBM expert cache, the same bytes are moved again at the next step.

### (d) The set of experts to load grows quickly with batch size
Under the simulator's routing, unique experts per layer = N·(1 − (1 − k/N)^B) (Q2 §4.3). For 8-expert models (Mixtral, Grok-1), all 8 experts of every layer are needed from B ≈ 16 on, so loading time saturates at the full expert footprint per step: 90 GB, i.e. 2.8 s on A100 PCIe4 or 100 ms with C2C-900, for Mixtral. Batching amortises the load over more tokens (Mixtral, C2C-900: loading time per generated token is 25 ms at B=1 vs 0.39 ms at B=256), but does not shorten the step. Fine-grained models (Maverick 128 experts, DeepSeek-R1 256) load only a subset at small batch, which is where a predictor or prefetcher has the most room.

### (e) Capacity forces the weights off-chip in the first place
All routed experts, FP16: Mixtral 90 GB, Grok-1 618 GB, Llama-4 Maverick 773 GB, DeepSeek-R1 1285 GB. All exceed an 80 GB A100/H100, even before attention weights and KV cache.

## 3. Consequence for the decode step (on-demand, no overlap)

| A100, batch 1 | step, all in HBM | step, on-demand C2C-900 | step, on-demand PCIe4 | stall share (PCIe4) |
|---|---|---|---|---|
| Mixtral | 12.5 ms | 37.5 ms (3.0×) | 717 ms (57.6×) | 98.3% |
| Grok-1 | 81.5 ms | 253 ms (3.1×) | 4914 ms (60.3×) | 98.3% |
| Llama-4 Maverick | 14.9 ms | 21.7 ms (1.45×) | 204 ms (13.6×) | 92.7% |
| DeepSeek-R1 | 34.6 ms | 79.3 ms (2.3×) | 1290 ms (37.2×) | 97.3% |

Even if loading were perfectly overlapped with execution, the step could not drop below the loading time (Q2 column `overlap_bound_step_ms`). With C2C-900 that bound is 25 ms for Mixtral at B=1, versus 37.5 ms for on-demand fetching. **Overlap alone recovers at most the execution time (about 33% here).

Caveat from the simulator's contention model: a copy running concurrently with compute takes its bandwidth out of HBM (HBM available = base − link). On A100 with C2C-900 only 1.14 of 2.04 TB/s remain, so everything that runs during the copy slows down. For Llama-4 Maverick at B=1 the overlapped bound (26.8 ms) is therefore *worse* than on-demand fetching (21.7 ms). A prefetcher must account for this interference: it pays off only when the bytes it moves are bytes that would otherwise be fetched on the critical path. The larger gains must come from not loading at all: keeping reused experts in HBM and prefetching only the experts that will be needed.** That requires predictable expert accesses, which Q4 examines.

## 4. Energy consequence
Energy per token rises from 1.8 J (Mixtral, all in HBM, A100, B=1) to 8.1 J (C2C-900) or 94 J (PCIe4). Of the PCIe4 figure, 2.4 J is the DRAM→HBM transfer itself and about 89 J is background power (GPU, HBM and host DRAM idling at 127 W) during the 0.70 s stall. Exposed stalls are therefore also the dominant energy cost (`../q2_offload_load_vs_exec/plots/q2_fig6_energy_A100_b*.png`).

## Files
- `plots/q3_fig1_layer_timeline_mixtral_A100_b1.png`: one MoE layer, resident vs on-demand for three links.
- `plots/q3_fig2_roofline_load_vs_exec.png`: analytic load/exec ratio vs tokens per expert, with simulator points.
- `plots/q3_fig3_capacity_and_bytes_per_token.png`: expert footprint vs 80 GB HBM, and off-chip bytes per token.
- `data/q3_break_even_table.csv`: ridge points and break-even tokens per expert for each GPU and link.
- Produced by `scripts/plot_q2_q3.py`.
