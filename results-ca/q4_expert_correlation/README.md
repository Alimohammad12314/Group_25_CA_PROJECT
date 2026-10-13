# Q4 — Temporal and spatial correlation of expert accesses

**Goal (mid-eval item 4):** analyse whether future expert requests can be predicted from previous ones (temporal: consecutive tokens; spatial: consecutive MoE layers), with suitable statistics and plots.

## 1. Data source: routing decisions logged from the simulator
The course requires everything to come from simulation, so no real model or trace was used. The routing data is what the **SSD-offloading LLMSimulator itself decides** while simulating. We patched `src/scheduler/sequence.cpp` to write every decision to a CSV (`simulation.routing.log_file`):

```
step,layer,seq,token,e_0,...,e_{k-1}      one row per (decode step, MoE layer, sequence)
```

| Setting | Value |
|---|---|
| Simulator | SSD-offloading LLMSimulator, single H100, GPU only |
| Models | Mixtral (8 experts, top-2, 32 MoE layers), Grok-1 (8, top-2, 64), Llama-4 Maverick (128, top-1, 24), DeepSeek-R1 (256, top-8, 57) |
| Workload | decode mode, batch 16, 128 decode steps, context 1024 → 2048 tokens per layer per model |
| Logged rows | 65 536 (Mixtral) to 116 736 (DeepSeek-R1) per scenario |

### How does the simulator choose experts?
Reading `Scheduler::getRandomExpert` and `BatchedSequence::update_expert`: in synthetic mode (`data: synthesis`), each token at each layer draws k distinct experts **independently and uniformly at random**. The original LLMSimulator also has a Zipf option (`skewness > 0`), but it too draws independently for every token. The simulator can replay real traces (`expert_data/experts_<model>_<dataset>.csv`), but none ship with it.

**Bug found while doing this analysis.** The SSD-offloading copy drew `uniform_int(0, 256) % N`. With 257 values, expert 0 is over-represented: 2× for DeepSeek (N=256) and 1.5× for Maverick (N=128). This appeared as one hot expert in the first DeepSeek log, with 7226 activations vs about 3648 expected, and 44% of layers failing the uniformity test. It was fixed to `uniform_int(0, N−1)`, as in the original simulator, and **all Q2 and Q4 results were regenerated after the fix**.

### Three routing scenarios
| Scenario | Config | What it represents |
|---|---|---|
| **uniform** | `model: uniform` | the simulator's default routing (Duplex paper: "uniform distribution") |
| **zipf** | `model: zipf, zipf_s: 1.0` | per-layer popularity skew (random hot experts per layer), still independent per token, like the original simulator's `skewness` option |
| **correlated** | `model: correlated, zipf_s: 0.8, p_temporal: 0.5, p_spatial: 0.6` | our extension: each slot reuses the previous token's expert at the same layer (prob. 0.5), else follows a fixed layer-to-layer expert map (prob. 0.6), else draws from the Zipf base. **The parameters are assumptions** chosen to resemble the locality reported by ST-MoE-style work; they are not measured. |

The first two scenarios are what the simulator *does*. The third is a controlled workload with *known* injected locality. It shows that the statistics below detect correlation when it exists, and it gives the predictor/prefetcher in the final phase something to exploit.

## 2. Statistics: definitions
Let E(s,l,t) be the set of k experts chosen for sequence s, layer l, token t.

| Statistic | Definition | Random reference |
|---|---|---|
| Activation share | p_l(e) = activations of e at layer l / all activations at l | 1/N |
| Normalised entropy | H(p_l)/log₂N, averaged over layers (1 = perfectly flat) | 1 |
| Top-20% share | share of activations taken by the 20% most popular experts of each layer | 0.2 |
| χ² uniformity | fraction of layers whose expert counts reject uniformity at p < 0.01 | 0 |
| **Temporal reuse(d)** | mean \|E(s,l,t) ∩ E(s,l,t+d)\| / k, same sequence and layer, token distance d | k/N |
| **Cross-layer MI** | mutual information (bits) between the slot-pair experts at layer l and l+1 of the same token | ≈ 0 (finite-sample bias, see shuffle) |
| **Cross-layer hit rate** | build the conditional table P(e_{l+1} \| e_l) on steps 0–63, predict the top-k for layer l+1 from layer l on steps 64–127, and measure the overlap | k/N |
| Pearson matrix | corr(1[i ∈ E(l)], 1[j ∈ E(l+1)]) over all tokens, an N×N matrix | 0 |
| **Batch-union overlap** | U(l,t) = union of E over the batch; mean \|U(l,t) ∩ U(l,t+1)\| / \|U(l,t+1)\|. This is the fraction of experts needed at step t+1 that are already needed at step t, which is what an HBM expert cache would hit. | depends on N, k, B |

**Shuffled baseline.** Every temporal statistic is recomputed after permuting the token order inside each (sequence, layer). Every spatial statistic is recomputed after permuting tokens at layer l+1. Shuffling keeps the frequency skew but destroys any ordering, so **observed − shuffled = exploitable correlation**. Without this baseline, Zipf skew alone would look like "temporal reuse".

## 3. Results (`data/q4_statistics.csv`, `data/q4_temporal_reuse_vs_lag.csv`)

### 3.1 Frequency distribution (`plots/q4_fig1_frequency_heatmap_*.png`, `plots/q4_fig2_expert_popularity_rank.png`)

| Model | uniform: entropy / top-20% / χ² layers | zipf | correlated |
|---|---|---|---|
| Mixtral | 1.000 / 0.13 / 0% | 0.907 / 0.32 / 100% | 0.982 / 0.19 / 100% |
| Grok-1 | 1.000 / 0.13 / 0% | 0.907 / 0.32 / 100% | 0.982 / 0.19 / 100% |
| Maverick | 0.994 / 0.27 / 0% | 0.788 / 0.71 / 100% | 0.927 / 0.48 / 100% |
| DeepSeek-R1 | 0.999 / 0.23 / 0% | 0.828 / 0.70 / 100% | 0.950 / 0.46 / 100% |

With 8 experts the top 20% is 1 expert (12.5%), so the uniform value is 0.13. The default routing is statistically flat in every layer of every model.

### 3.2 Temporal correlation (`plots/q4_fig3_temporal_reuse_vs_lag.png`, `plots/q4_fig4_transition_matrices_mixtral.png`)
Lag-1 reuse, observed (shuffled):

| Model | k/N | uniform | zipf | correlated |
|---|---|---|---|---|
| Mixtral | 0.250 | 0.250 (0.249) | 0.361 (0.364) | **0.654** (0.281) |
| Grok-1 | 0.250 | 0.251 (0.252) | 0.363 (0.362) | **0.654** (0.281) |
| Maverick | 0.008 | 0.008 (0.008) | 0.056 (0.054) | **0.574** (0.034) |
| DeepSeek-R1 | 0.031 | 0.031 (0.031) | 0.202 (0.203) | **0.583** (0.076) |

- **uniform:** reuse equals k/N at every lag, and the t→t+1 transition matrix is flat (about 0.125 everywhere for Mixtral). There is no temporal correlation.
- **zipf:** reuse is above k/N, but identical to its shuffled value at every lag. The extra reuse comes purely from popularity: the matrix shows bright *columns* (popular experts), not a diagonal. A frequency-based (LFU) predictor captures all of it.
- **correlated:** reuse is far above the shuffled value at lag 1 and decays with distance. Mixtral goes 0.65, 0.47, 0.33, 0.27 at d = 1, 2, 4, 8; DeepSeek goes 0.58, 0.35, 0.15, 0.07. The matrix shows a bright *diagonal*. The most recent tokens are the best predictor.

### 3.3 Spatial (cross-layer) correlation (`plots/q4_fig4_*` bottom row, `plots/q4_fig5_cross_layer_correlation_mixtral.png`)

| Model | cross-layer hit rate: uniform | zipf | correlated | MI excess over shuffle (bits): uniform / zipf / correlated |
|---|---|---|---|---|
| Mixtral | 0.250 (0.252) | 0.509 (0.506) | **0.480** (0.339) | 0.000 / 0.000 / **0.108** |
| Grok-1 | 0.250 (0.251) | 0.507 (0.506) | **0.474** (0.341) | 0.000 / 0.000 / **0.106** |
| Maverick | 0.008 (0.008) | 0.134 (0.130) | **0.301** (0.019) | −0.002 / 0.006 / **1.592** |
| DeepSeek-R1 | 0.031 (0.031) | 0.366 (0.365) | **0.209** (0.150) | 0.000 / 0.000 / **0.350** |

The Pearson matrices between layer 5 and 6 of Mixtral are all near 0 for uniform and Zipf (|r| ≤ 0.07). In the correlated scenario one strong entry per row appears (r ≈ 0.27–0.45), which is exactly the injected layer-to-layer map. Raw MI is biased upward for large N with 2048 samples (Maverick uniform shows 3.03 bits); the shuffled baseline removes that bias, so only the excess is meaningful.

The *absolute* hit rate under Zipf (0.51 for Mixtral) is higher than under correlated routing (0.48) because the Zipf scenario is more skewed. All of that Zipf hit rate is frequency, though: its shuffled value is the same.

### 3.4 Batch-level reuse (`plots/q4_fig6_predictability_summary.png`)
At batch 16:
- Mixtral and Grok-1: 99% of the experts needed at step t+1 were already needed at step t, in all scenarios. Unique experts per layer are 7.9 of 8 for uniform. **With 8 experts and batch 16, essentially all experts are needed at every step**, so the question is not *which* experts but how to avoid re-fetching them (residency or caching), as Q2/Q3 also showed.
- Maverick: overlap 0.12 (uniform), 0.30 (zipf), **0.63** (correlated, shuffled 0.19).
- DeepSeek-R1: 0.40 (uniform), 0.46 (zipf), **0.73** (correlated, shuffled 0.44).

Fine-grained models are where prediction can decide which subset to bring into HBM.

## 4. Conclusions to carry into the predictor design
1. **With the simulator's own routing (uniform, or Zipf-skewed), future experts cannot be predicted from past ones better than by frequency.** Every temporal and spatial statistic equals its shuffled baseline. This is a property of the synthetic workload (i.i.d. draws), not of real MoE models; ST-MoE and APEX report real locality.
2. Frequency skew (Zipf) is exploitable by a popularity-based policy, but it is not temporal or spatial correlation.
3. When locality is present (the correlated scenario), the statistics detect it clearly: lag-1 reuse is 2.3–17× the shuffled value, cross-layer hit rate is up to 16× the shuffled value, and batch-union overlap rises to 0.63–0.73 for fine-grained models. A training-free predictor can use (a) the previous token's experts at the same layer, which decays within about 4 tokens, and (b) the current layer's experts mapped to the next layer through an online-counted transition table.
4. Before the final phase we need the TA to agree on which routing workload the predictor and prefetcher are evaluated on. The default routing gives a predictor nothing to learn.

## How to reproduce
```bash
cd results-ca        # from the repository root
python3 scripts/q4_routing_runs.py     # 12 simulations, about 5 min (DeepSeek is the slowest)
python3 scripts/q4_analysis.py         # statistics + plots, about 20 s
```
`runs/<model>_<scenario>/` contains `config.yaml`, `routing_log.csv`, the simulator log and its output CSV.
