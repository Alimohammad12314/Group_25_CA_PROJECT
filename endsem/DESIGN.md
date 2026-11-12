# Design log: training-free expert prediction, prefetching and HBM management

Project 2 (MoE memory wall), Group 25. This file is the in-repository copy of the design log; it
records what was built, every change from the plan, and the measurement that forced it.
Results tables are in the last section and in `results/`.

## 1. Starting point

The input was the plan *MoE Memory Wall: Papers and Proposed Design for the Final Submission*:
a training-free prefetcher that puts ST-MoE's correlation tables, STEP's vote window, APEX's
adaptive top-(k+delta) budget and ProMoE's multi-layer lookahead in front of a bounded HBM expert
cache. All experts live off-chip; HBM holds a bounded cache; routing stays exact (no pruning).
The zero-cache timeline must reproduce the mid-eval on-demand numbers (Mixtral, A100, batch 1:
37.5 ms on C2C-900, 717 ms on PCIe 4.0).

## 2. Goals

- Maximise overlap accuracy (share of routed experts named in advance).
- Minimise token latency (exposed stall waiting on the DRAM-to-HBM link).
- Measure the prediction/prefetch trade-off: wasted traffic, HBM contention, energy, predictor
  time and memory.
- Report the metrics recent papers use: recall, full coverage, precision / wasted prefetches,
  timeliness, batch-union accuracy, stall, p95 latency, traffic, link utilisation, energy, EDP,
  predictor state size.

## 3. Design as built

Predictions depend only on routing history, never on timing, so they are computed once per
(model, batch, predictor configuration) as a *prediction stream* and replayed by every timeline run.

| # | Component | As built | Source idea |
|---|---|---|---|
| 1 | Traces | `[request, token, layer, top-k]` + prompt routing for warm-up; B requests decode together | plan step 1 |
| 2 | Temporal predictor | votes over the last W tokens per sequence and layer; W doubles after 2 windows > 75 % hit rate, halves < 40 % | STEP; ST-MoE history table |
| 3 | Cross-layer predictor | per (distance d, layer) counters for (expert at l, expert at l+d); row halving on overflow; 4-bit | ST-MoE; ProMoE / FineMoE distance |
| 4 | Popularity prior | sliding 64-step window of batch counts per layer | PopFetcher; LFU |
| 5 | Rank calibration | each predictor's rank -> probability via an online hit-rate-per-rank table | new (section 4) |
| 6 | Chooser | weight = EMA recall^2 per (layer, distance) over calibrated probabilities | tournament predictor; DAOP |
| 7 | Batch union | P(e) = 1 - prod_b (1 - q_b(e)) | ExpertFlow batch view |
| 8 | Adaptive budget | M = K_hat + tau-quantile of the last 64 oracle delta*; delta = 0 when recall < 40 % | APEX (empirical CDF); STEP off-switch |
| 9 | Lookahead | smallest d whose compute covers the predicted load, cap 4; crosses the token boundary | ProMoE stride; ExpertFlow adaptive step |
| 10 | Link | FIFO, or demand misses preempt prefetches at 16 MB chunks; late prefetches promoted; stale ones dropped | ProMoE; PreScope; SPICE |
| 11 | Link-time cap | prefetch only what the link can deliver before the target layer | SpecPrefetch |
| 12 | Expected-value gate | prefetch e only if p(e) >= c / (1 + c), c = (1 - 1/f) + link utilisation | new (section 4) |
| 13 | HBM cache | ratio x all experts; LRU / LFU / need / Belady; pins; early-layer bonus; need = P(used within 4 tokens) | MoE-Infinity; FATE; HybriMoE |
| 14 | Miss handling | run arrived experts while missing ones load | APEX correctness-preserving mode |
| 15 | Costs | LLMSimulator roofline + Q2 non-expert time; SSD-offloading contention (HBM BW - link BW) and energy | mid-eval Q2 |

## 4. Thought process (in the order it happened)

| Step | What I saw | What I changed | Why |
|---|---|---|---|
| Simulator | LLMServingSim works per batch and needs ASTRA-Sim; the Q2 numbers come from LLMSimulator / SSD-offloading | own trace-driven timeline here, costs from LLMSimulator and Q2 | END_SEM_PLAN D1 |
| Traces | Mixtral open; DeepSeek-R1 and Maverick gated (core12345/MoE_expert_selection_trace); no Grok-1 trace exists | real traces for 3 models; Grok-1 synthetic, fitted per layer to Mixtral | plan risk table |
| Validation gate | exact at batch 1; up to 8 % off at batch 4 | batch > 1 gated per unique expert | Q2 ran one decode step: its unique-expert count is one random draw (5.06 vs exact 5.47) |
| First combiner | mix scored 0.389 overlap, below cross-layer alone (0.427) | rank calibration before mixing | peaky temporal votes drowned the flatter but better cross-layer scores |
| Counter width | Mixtral: 2-bit 0.416, 4-bit 0.429, 8-bit 0.432 | 4-bit default; 2/8 reported | 2-bit saturating counters lose the ranking among frequent pairs |
| Combining rule | mean vs log-odds, gamma 2 vs 8 within 0.008 | kept hit-rate weighting (gamma 2) | after calibration the rule matters little; no tuning on the test trace |
| LRU at batch 4/16 | cache gave 1.00x | predicted-need eviction | each step sweeps layers cyclically; when a step's experts exceed the cache, LRU evicts exactly what comes next |
| Naive prefetch | rungs 3-5 at 0.62-0.95x of on-demand (batch 1) | priority link, link cap, stale cancel | wrong guesses blocked demand misses and polluted the cache |
| C2C-900 | full design 1.40x vs 1.51x for LRU + miss overlap with no prediction | expected-value gate | a 900 GB/s copy takes 44 % of A100 HBM bandwidth; at ~33 % precision, wasted copies cost more than correct ones save |
| Grok-1 generator | plain mixture: reuse 0.27, cross 0.31 vs Mixtral 0.36, 0.42 | sharpen the real transition tables (beta fitted per layer) | a token's hidden state drives both correlations together; a mixture dilutes them |
| DeepSeek-R1 files | 3 of 30 AIME files malformed; 10 queries with 1 decode token | cut at the first malformed token; keep >= 8 tokens | 94 queries with 128 tokens remain |
| Eviction horizon | Maverick batch 1: need eviction hit less than LRU (0.83 vs 0.85); batch 4 PCIe4: 1.22x slower than LRU + overlap | score = P(used within H tokens) = 1 - (1 - p_next)(1 - u)^H, H = 4 | skewed popularity makes experts reusable beyond the next token; H tuned on these traces (mean 0.847x, worst 1.016x of LRU + overlap) |
| Counter width, large N | DeepSeek-R1 / Maverick: 2-bit equal or better (0.270 vs 0.264) | 4-bit stays default, 2-bit reported | rows age faster with 2 bits; half the state |

Not used: hidden-state predictors (traces hold IDs only), trained predictors (plan is training-free),
expert pruning or substitution (changes outputs the simulator cannot score), integration into
LLMServingSim (needs GPU-profiled tables and an expert-level hook).

## 5. Changes made

All new files under `endsem/`; nothing else in the repository was modified. See the README's folder
map. Traces and streams are in `endsem/data/` (git-ignored).

## 6. Validation

- Gate (`results/e0_validation`): batch 1, 32 points, worst error 0.052 % (Mixtral A100: 37.5088 ms
  C2C-900, 717.0979 ms PCIe 4.0, identical to Q2); batch 4/16, 64 points, worst 0.14 % per unique expert.
- Traces: Mixtral `allenai/analysis_mixtral` (405 C4 documents, 32 layers, top-2/8); Llama-4 Maverick
  and DeepSeek-R1 (AWQ) from `core12345/MoE_expert_selection_trace` (AIME 2024, LiveCodeBench, MMLU;
  122 and 94 queries x 128 decode tokens; 24 and 57 MoE layers; top-1/128 and top-8/256); Grok-1 synthetic.
- No leakage: a prediction d layers ahead sees routing up to layer l - d of the current token only.
- 6 unit tests pass.

## 7. Results

See `results/` and the shared design log for tables and charts.
