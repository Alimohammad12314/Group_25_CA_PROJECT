# End-semester requirements: answers and evidence (Group 25)

Repository: <https://github.com/Alimohammad12314/Group_25_CA_PROJECT> · Members: Palak Mishra, Mohammad Ali, Shivansh Bhatnagar

For each requirement this file gives the answer, the key numbers and the file the numbers come from.

**Setup for every number.** Simulated A100-80GB, FP16, all routed experts in host DRAM, HBM holds a bounded expert cache.

**Traces.** Mixtral-8x7B, Llama-4 Maverick and DeepSeek-R1 use real Hugging Face routing traces. Grok-1 has no public trace, so it uses a synthetic one fitted to Mixtral.

Design log and full results: [`endsem/DESIGN.md`](endsem/DESIGN.md) · code: [`endsem/`](endsem/README.md) · figures: [README](README.md#results-at-a-glance).

| # | Requirement | Status |
|---|---|---|
| 1 | Prediction technique | ✅ |
| 2 | Prefetching mechanism | ✅ |
| 3 | Prefetching effectiveness | ✅ |
| 4 | HBM management beyond LRU | ✅ |
| 5 | Comparative evaluation | ✅ |
| 6 | Sensitivity analysis | ✅ |
| 7 | Energy analysis | ✅ |
| 8 | Documentation and reproducibility | ✅ repository; report PDF and ZIP submitted separately |
| ★ | Bonus: real routing in the LLM simulator | ✅ |

---

## 1. Prediction technique

*Implement a training-free expert prediction technique; evaluate precision, recall and other metrics against actual
routed experts, on real routing traces.*

**What we built** (`endsem/predictors/`). It is training-free and uses only the routing history.

| Predictor | Idea | Source |
|---|---|---|
| Temporal | experts used by the last W tokens of the same sequence; W adapts to the hit rate | STEP (ISCA'26), ST-MoE |
| Cross-layer | counter table "expert at layer l → expert at layer l+d", d = 1 … 4 | ST-MoE |
| Popularity | sliding 64-step window of expert counts per layer | PopFetcher (ATC'25) |

The three predictions are combined in four stages:
1. **Rank calibration (ours).** Each predictor's ranking becomes a probability.
2. **Chooser** (tournament predictor). Weights each predictor by its recent recall.
3. **Batch union.** P(e) = 1 − ∏ (1 − q_b(e)) over the batch's sequences.
4. **Adaptive budget** top-(k + δ) (APEX), with an off-switch when recall is low (STEP).

**Traces.**
- Mixtral: `allenai/analysis_mixtral` (405 C4 documents, 32 layers, top-2 of 8).
- Llama-4 Maverick and DeepSeek-R1: `core12345/MoE_expert_selection_trace` (AIME, LiveCodeBench, MMLU; top-1 of 128 and top-8 of 256).

**Accuracy** (`endsem/results/e2_predictor_accuracy/summary.csv`). One layer ahead, batch 1. *Overlap* is the share of the actually routed experts that were named in advance with a budget of exactly k.

| Model | Overlap | Random guess | Recall (adaptive budget) | Precision | Full coverage |
|---|---|---|---|---|---|
| Mixtral-8x7B | **43.5 %** | 25 % | 70.4 % | 33.2 % | 54.4 % |
| Grok-1 (synthetic) | 43.4 % | 25 % | 69.6 % | 32.5 % | 53.3 % |
| Llama-4 Maverick | **32.8 %** | 0.8 % (41× better) | 39.4 % | 25.5 % | 39.4 % |
| DeepSeek-R1 | **26.6 %** | 3.1 % (8.6× better) | 27.8 % | 25.9 % | 1.7 % |

- **Combining helps.** The combined predictor beats each single predictor on every model. Mixtral: cross-layer 42.7 %, temporal 35.7 %, popularity 31.6 %.
- **Accuracy holds with distance.** It barely drops from 1 to 4 layers ahead (Mixtral 43.5 → 42.3 %, DeepSeek 26.6 → 26.3 %).
- **Larger batches raise overlap.** DeepSeek goes 26.6 → 54.6 % at batch 16. Mixtral and Grok-1 reach 99 %, but that is trivial: nearly all 8 experts are needed every step.
- **We are close to the ceiling.** The traces contain only expert IDs. An ID-only lookup table fitted to the test data itself reaches only 46 % on Mixtral (52 % with the previous token), and our predictor reaches 43.5 %. Higher accuracy needs the model's hidden state, so we measured that on a real H100 NVL with the real Mixtral weights ([`endsem/gpu/README.md`](endsem/gpu/README.md)). Applying layer l+d's own router to the hidden state at layer l (training-free) gives next-layer overlap of **86 % at batch 1** (vs 42 % ID-only on the same tokens), 91 % at batch 4 and 99.8 % at batch 16; four layers ahead it is 73 % at batch 1. With our adaptive budget on top of those scores, batch-4 recall goes from 71 % to 96 % (full coverage 7 % → 83 %).
- **No published comparison exists.** No paper reports predictor accuracy on these traces. The trace authors report only locality statistics.

## 2. Prefetching mechanism

*Predictor-driven prefetching into HBM; explain when it is triggered and how transfers overlap with computation.*

Code: `endsem/engine/timeline.py`, `engine/link.py`. Diagram: [`endsem/architecture_v3.png`](endsem/architecture_v3.png).

**When prefetching is triggered**
- **Trigger.** It fires right after layer l's router decides which experts the batch needs.
- **Prediction.** The predictor then estimates P(expert) for layers l+1 … l+4.
- **Lookahead.** The prefetcher uses the smallest distance d ≤ 4 whose compute time covers the predicted load (ProMoE, ExpertFlow).
- **Link-time cap (SPICE).** It prefetches only what the link can deliver before the target layer.
- **Expected-value gate (ours).** An expert is prefetched only if p(e) ≥ c / (1 + c). Here c is the cost of a wrong copy: the HBM contention it causes plus the chance that a busy link delays a real miss.

**How transfers overlap with computation**
- Copies run on the PCIe / NVLink-C2C link while the GPU computes the current layers.
- **Demand misses go first.** A demand miss preempts a prefetch at the next 16 MB chunk boundary (ProMoE, PreScope).
- A prefetch still in flight when its layer arrives is promoted to demand priority.
- Prefetches whose target layer has already passed are dropped.
- **Miss overlap (APEX's correctness-preserving mode).** Experts run in arrival order, so arrived experts compute while missing ones still load. No expert is skipped or substituted.
- **Contention is modelled.** Copying slows compute through shared HBM bandwidth (SSD-offloading model: compute sees HBM bandwidth minus link bandwidth).

## 3. Prefetching effectiveness

*Coverage, timely and late prefetches, unused prefetches, HBM hit rate, additional traffic, exposed stall, expert-load
latency, end-to-end / token latency.*

Every simulation reports all of these (columns of `endsem/results/e3_ablation/ablation.csv`, also in e4, e6 and the bonus).

Example: complete design, batch 16, NVLink-C2C 900 GB/s, cache holds 25 % of all experts. Arrows show LRU + miss overlap → full design.

| Metric | Mixtral-8x7B | DeepSeek-R1 |
|---|---|---|
| Coverage (recall, distance 1) | 100 % | 99.1 % |
| Prefetches used | 99.4 % | 75.1 % |
| Unused (wasted) prefetches | 0.6 % | 24.9 % |
| Timely prefetches | 61 % (10.5 % of needs were late) | 100 % |
| HBM hit rate | 0 % → 35.1 % | 0.3 % → 34.0 % |
| Traffic per decode step | 89.1 → 72.8 GB | 477 → 317 GB |
| Exposed stall share of time | 28.6 % → 11.0 % | 20.5 % → 0.07 % |
| Expert-load latency: one expert, idle link | 0.39 ms | 0.10 ms |
| Expert-load latency: demand copy | 1.74 → 1.59 ms | 4.74 → 3.22 ms |
| Wait per needed expert | 1.74 → 0.92 ms | 4.73 → 2.13 ms |
| Token latency (mean) | 107.0 → 87.3 ms | 550.1 → 408.7 ms |
| Token latency (p95) | 107.8 → 88.3 ms | 576.7 → 433.0 ms |

- **Traffic falls instead of rising.** Better retention avoids more reloads than prefetching adds.
- **A demand copy takes several times the bare copy time.** All misses of a layer queue behind each other on the link.

## 4. HBM management beyond LRU

*Implement an eviction policy beyond LRU, compare with LRU, and analyse its impact on performance and prefetch effectiveness.*

**Policy: predicted-need eviction** (`endsem/engine/hbm.py`, `timeline.py`). The cache evicts the expert least likely to be used within the next 4 tokens:

    P(used within 4 tokens) = 1 − (1 − p_next) · (1 − usage rate)^4

- `p_next` comes from the predictor.
- `usage rate` = uses since loading ÷ (age + 4).
- Compared against LRU, LFU and Belady (the optimal, oracle policy).

**Effect on performance** (e3, 96 settings):
- Mean hit rate rises from 0.34 (LRU) to 0.44.
- Speedup over LRU + miss overlap is 1.18× on average (geometric mean), up to 2.25×.
- **Why LRU fails here.** A decode step walks the layers in a fixed cycle. When one step's experts exceed the cache, LRU evicts exactly the expert needed next. Mixtral and DeepSeek at batch 16 get a 0 % LRU hit rate.

**Effect on prefetching** (medians over e3):

| | Prefetch + LRU | Prefetch + predicted-need eviction (full design) |
|---|---|---|
| Prefetches used | 45 % | 56 % |
| Wasted traffic per token | 0.375 GB | 0.002 GB |

With LRU, prefetched experts evict each other before they are used.

**Where it loses.** It loses at batch 1 with 8-expert models and a small cache. Mixtral on PCIe 4.0 gets a 25 % hit rate against LRU's 35 % and runs at 0.87× the speed of LRU + overlap.

## 5. Comparative evaluation

*Baseline without prefetching, prediction without prefetching, prediction-driven prefetching, complete implementation;
contribution of each component.*

Data: `endsem/results/e3_ablation/ablation.csv`, 96 settings (4 models × batch 1/4/16 × 4 links × cache 25/50 %). Columns are geometric means; "best" is the best single setting.

| Configuration | vs on-demand | vs LRU + overlap | Best |
|---|---|---|---|
| On-demand, no cache | 1.00× | 0.53× | |
| **Baseline without prefetching:** HBM cache, LRU | 1.68× | 0.89× | 9.3× |
| **Baseline without prefetching:** LRU + miss overlap | 1.89× | 1.00× | 9.6× |
| **Prediction without prefetching** (prediction decides evictions only) | 2.23× | 1.18× | 10.3× |
| **Prediction-driven prefetching** (naive FIFO link → engineered, LRU cache) | 1.59× → 1.67× | 0.84× → 0.88× | 9.3× |
| **Complete implementation** | **2.24×** | **1.19×** | **10.3×** |
| Oracle predictor (upper bound) | 2.45× | 1.30× | 11.5× |

**Contribution of each component**

| Component | Gain |
|---|---|
| HBM cache | 1.68× over on-demand |
| Miss overlap | +12 % |
| Prediction-guided eviction | +18 %, the main gain |
| Prefetching on its own (LRU cache) | 0.84–0.88× of the baseline |
| Prefetching on top of predicted-need eviction | +0.7 % (at most 7 %, batch 16 on C2C) |
| Value gate | wasted traffic 0.375 → 0.002 GB/token; energy 12.3 → 9.5 J/token; no loss in speed |

- **Prefetching alone hurts.** About 55 % of copies are useless, and they block the link a demand miss needs.
- **The full design is at least as fast as LRU + overlap in 75 of 96 settings.**
- **By batch size:** 0.99× at batch 1, 1.28× at batch 4, 1.33× at batch 16.

## 6. Sensitivity analysis

*Multiple batch sizes and context lengths; lookahead and aggressiveness; when prefetching helps, gives limited gains,
or degrades performance.*

| Parameter | Result | Source |
|---|---|---|
| Batch 1 / 4 / 16 | full design vs best baseline: 0.99× / 1.28× / 1.33× | e3 |
| Context 256 – 4096 (fixed 25 % cache) | latency changes by < 2 %; expert loading is 10–1000× attention time | e6 |
| Context 256 – 4096 (cache = what fits in 80 GB) | see below | e6 |
| Lookahead cap 1 / 2 / 4 | latency changes by at most 6.7 % | e4 |
| Aggressiveness τ 0.6 / 0.75 / 0.9 | latency changes by < 0.05 %; adaptive budget wastes 0.045 GB/token vs 1.85–3.1 GB for fixed top-(k+x) | e4, e5 |
| Link: PCIe 4.0 / 5.0, NVLink-C2C 450 / 900 | all settings | e3, e4 |
| Cache: 0 / 25 / 50 / 75 % and physical | all settings | e4 |

**Context with the cache that physically fits.** The KV cache pushes experts out of HBM. Mixtral, batch 16, PCIe 4.0, from 256 to 4096 tokens:
- Share of experts that fit in HBM: 85 % → 76 %.
- Hit rate: 0.83 → 0.73.
- Latency: 501 → 782 ms.
- Energy: 4.4 → 6.7 J/token.
- Lead over LRU + overlap: 5.6× → 3.6×.

**Lookahead and τ barely matter** because the link-time cap and the value gate already limit how much gets prefetched.

**When prefetching improves performance, gives limited gains, or degrades it:**
- **Improves:**
  - batch 4–16 on large-expert models (DeepSeek-R1, Maverick);
  - 8-expert models at batch 16 on fast links;
  - any case where the cache cannot hold one step's experts.

  Gains over LRU + overlap: 1.2–1.5× at 25 % cache, and up to 5.6× with Mixtral's physical cache.
- **Limited:**
  - batch 1 on PCIe. One Mixtral expert takes about 11 ms to copy, but a layer computes in about 0.4 ms. The link-time cap therefore issues almost no prefetches.
  - shrinking caches (long context × large batch), which leave fewer slots to protect.
- **Degrades:**
  - naive prefetching on a FIFO link with LRU: 0.65× at Mixtral batch 1;
  - batch 1 with 8 experts and a small cache: the full design runs at 0.87–0.95× the speed of LRU + overlap.

  In both cases about two thirds of guesses are wrong. Wrong copies occupy the link and, on C2C, slow compute through HBM contention.

## 7. Energy analysis

*Energy consumption and savings using the available energy model; trade-off between fewer exposed stalls and extra traffic.*

**Energy model** (SSD-offloading, as in mid-eval Q2; `endsem/engine/energy.py`):
- link transfer: 13.4 pJ/bit (9.2 DDR5 read + 4.2 HBM write);
- dynamic energy: compute and HBM reads, from the Q2 base pass;
- background power: 70 W GPU + HBM, plus 57 W host DRAM, multiplied by elapsed time.

**Savings**
- Against the best baseline: 17 % less energy and 21 % fewer bytes per token (geometric mean).
- Against on-demand: 55–66 % less energy.
- Example, Mixtral batch 16 on PCIe 4.0 with the physical cache: 23.3 → 4.8 J per token.

**Trade-off between fewer stalls and more traffic**
- **Stalls dominate.** Background power burned while the GPU waits on the link is most of the energy. Link transfer energy is small: 89 GB per step is 9.5 J per step, about 0.6 J per token at batch 16.
- **So cutting stalls pays**, but only while prefetches are mostly used.
- **Without the value gate it backfires.** Prefetch + LRU uses more energy than LRU + overlap (12.3 vs 11.7 J/token median) because it moves 0.375 GB/token of wasted data and does not cut stalls.
- **With the gate**, wasted traffic is almost zero (0.002 GB/token) and energy falls to 9.5 J/token.

## 8. Documentation and reproducibility

| Item | Where |
|---|---|
| Setup instructions | [README: Setup on a new machine](README.md#setup-on-a-new-machine), `setup.sh` (patches and builds both simulators) |
| Commands to reproduce every experiment | [README step 6](README.md#6-end-semester-experiments), [`endsem/README.md`](endsem/README.md) |
| Configurations | `endsem/experiments/common.py` (ablation ladder, full design), `results-ca/configs/`, simulator patches with explanations in `results-ca/simulator_patches/` |
| Results | CSV tables in `endsem/results/<experiment>/`; figures in `docs/figures/`; design log `endsem/DESIGN.md` |
| References for every borrowed idea | [README: References](README.md#references-where-each-strategy-comes-from) (23 references, a table by paper) |
| Validation | `e0_validation_gate`: the timeline reproduces Q2 within 0.05 % (batch 1); 6 unit tests |
| Real-GPU validation | H100 NVL, real Mixtral-8x7B (`endsem/gpu/README.md`): trace reproduced for 98.7 % of (token, layer) pairs; hit rate and GB/token exact in all 33 timing-independent runs; latency within 3.1 % median over 77 runs with measured costs |
| Team contribution | commits by all three members in the GitHub history |
| Report PDF (GitHub link at the top) and ZIP of the code | submitted separately |

## ★ Bonus: realistic expert-routing traffic in the LLM simulator

*Model routing from real MoE traces in the LLM simulator and evaluate the mechanism on the resulting traffic.*

**What we did**
- Added `routing.model: trace` to the SSD-offloading LLMSimulator (`results-ca/simulator_patches/ssd_offloading_llmsim.patch`). It replays a recorded trace for any top-k; the simulator's built-in replay supports only top-2.
- The simulator runs 128 decode steps per configuration: 3 models (real traces only) × batch 1/4/16 × PCIe 4.0 / C2C-900, with real routing and, for comparison, its default uniform routing.
- Script: `endsem/experiments/bonus_llmsim_replay.py`. Results: `endsem/results/bonus_llmsim/`.

**Results**
- **Exact replay.** The simulator's own routing log equals the input trace for 100 % of decisions.
- **Our timeline matches the simulator** step by step: mean error ≤ 0.13 %, per-step correlation 0.999–1.000.
- **Real routing creates less traffic than uniform routing**, because sequences in a batch share experts. Maverick batch 16: 74.8 vs 91.2 GB per step (−18 %). DeepSeek-R1: −5 %. Mixtral: within 0.5 %.
- **On this traffic, the predictor finds structure only in real routing.** Overlap one layer ahead, real vs uniform: DeepSeek-R1 0.26 vs 0.03, Maverick 0.31 vs 0.01, Mixtral 0.39 vs 0.25.
- **The full design's speedup over on-demand, real vs uniform** (batch 16, PCIe 4.0): Maverick 3.53× vs 1.31×, DeepSeek-R1 1.53× vs 1.33×. The simulator's default synthetic routing underestimates the design by up to 2.7× on large-expert models.

## Evaluation emphasis: why, when it fails, and how the factors interact

- **Why the improvement occurs.** It comes from retention. LRU's interaction with the fixed layer cycle evicts the expert needed next; prediction-guided eviction keeps it. Prefetching adds on top only when copies can arrive in time and guesses are mostly right.
- **When prefetching fails.**
  - Copy time ≫ layer time (batch 1 on PCIe).
  - Low precision: about 30 % with ID-only prediction.
  - A FIFO link, where wrong copies block demand misses.
  - An LRU cache that evicts prefetched experts before use.
- **How the factors interact.**
  - **Prediction accuracy** sets how many copies are useful.
  - **Prefetch timing** (copy time vs compute window) sets how many can arrive in time.
  - **Memory traffic** sets the cost of each wrong guess: a busy link, HBM contention, 13.4 pJ/bit.
  - **HBM management** decides whether a correct guess survives until it is used.

  In every setting we measured, retention mattered most.
- **What would change the picture.** We tested it: the hidden-state predictor doubles batch-1 accuracy (42 → 86 %) but moves the full design only from 1.36x to 1.40x over on-demand with H100 costs, still below LRU + miss overlap (1.58x) and the oracle (1.72x). At batch 4-16 every predictor is already within 1 % of the oracle. On a PCIe-attached GPU the copy time, not prediction accuracy, is the limit once eviction is right.
