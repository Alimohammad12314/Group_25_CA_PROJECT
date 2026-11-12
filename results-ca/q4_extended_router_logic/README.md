# Q4-extended — Does the router's own logic create relations between expert choices?

**Question.** Q4 found no relation between expert choices, because the simulator picks experts with a dice roll. A real MoE router is not a dice roll: it **looks at the token** and picks the experts that fit it. Do relations over time and across layers appear once the simulator uses real router logic?

Everything is still simulation only: no model is downloaded and no real traces are used.

---

## In plain words

**How a real router decides.** Each token carries a list of numbers that describes it (its "meaning so far"). At every MoE layer the router gives each expert a score from that list and sends the token to the top-k experts. Two facts follow:
1. **Neighbouring words are similar** (same sentence, same topic), so their lists are similar, their scores are similar, and they tend to go to **the same experts**.
2. **Each layer changes the list only a little** (a layer adds a small correction to what came before). A token's list at layer 6 is still close to its list at layer 5, so its choice at layer 6 is **related** to its choice at layer 5.

**What we built.** We put exactly this logic into the simulator (`routing.model: router`). Each token gets a list of 64 numbers. The list drifts from word to word and is nudged by every layer, and every layer's router scores the experts and keeps the top-k. Two knobs control it:
- **token similarity ρ**: how alike consecutive words are (0 = unrelated, 0.99 = almost identical);
- **layer update α**: how much each layer changes the token (small = gentle nudge, large = big change).

We have no trained model, so the router weights are random and the knob values are **our assumptions**. The experiment shows *how* relations depend on them, not the exact values in a real model.

**What we found.**
- **Over time: a clear relation, created by the router logic itself.** With similar neighbouring words (ρ = 0.9), the next word reuses **72%** of the previous word's experts in Mixtral; pure chance is 25%. In DeepSeek it is **53%**, against a chance level of 3%. Nothing tells the router to repeat itself; it happens because similar words get similar scores. If neighbouring words are unrelated (ρ = 0), the relation disappears completely and behaves like the dice roll.
- **The relation fades with distance:** strongest for the previous word, weaker two words back, close to chance after about 10–16 words (unless the text stays on one topic).
- **Across layers: a real but weaker relation.** Knowing a token's experts at layer l helps guess its experts at layer l+1, beyond what popularity explains. DeepSeek: 26% correct guesses vs 10% with the order scrambled. The relation barely weakens over 2, 4 or 8 layers, because every layer sees the *same token*. But **"expert 3 on layer 5" does not mean "expert 3 on layer 6"**: each layer has its own router, so the link is a mapping, not the same number.
- **Popular experts appear on their own.** Even with no built-in favourites, some experts get several times their fair share of tokens in large-expert models (up to 6× in DeepSeek). Real MoE models also show such load imbalance.
- **For prefetching:** in DeepSeek (256 experts), with 16 users, **69%** of the experts needed at the next step are already needed now. The dice-roll simulator gives 40%. A cache or prefetcher can use this. Mixtral needs nearly all 8 experts every step anyway (99%), so guessing matters for models with many experts.

**Bottom line.** Expert choices are related **because of how the router works** (similar tokens give similar choices, and layers build on each other), not because of anything special added. The default simulator hides this by rolling dice. The strength of the relation depends on how similar neighbouring tokens are and how much each layer changes them.

---

## 1. Router model added to the simulator
Patched `src/scheduler/sequence.cpp` (see `../simulator_patches/`), selected with `simulation.routing.model: router`:

```
token content (per sequence s, token t):  x_t = ρ·x_{t-1} + sqrt(1-ρ²)·ε,   ε ~ N(0, I_64)
residual stream through MoE layers:       h_0 = x_t,   h_l = rmsnorm(h_{l-1} + α·tanh(A_l·h_{l-1}))
router at MoE layer l:                    score_l = W_l·h_l + b_l  →  top-k experts
```

| Symbol | Meaning | Config key | Values used |
|---|---|---|---|
| d | hidden size of the router model | `router_dim` | 64 |
| ρ | similarity of consecutive tokens | `token_similarity` | 0, 0.5, 0.8, 0.9, 0.95, 0.99 |
| α | strength of each layer's residual update | `layer_update` | 0.05, 0.25, 0.5, 1, 2 |
| W_l, A_l | router weights and layer transform, fixed random per layer (N(0, 1/d)) | `seed` | 2026 |
| b_l | per-expert router bias, N(0, σ_b), which creates favourite experts | `router_bias` | 0, 0.5 |

This is the gating rule of real MoE layers (linear router + top-k on the token's hidden state). The only synthetic parts are the random weights and the two knobs that stand in for real token statistics. Routing still drives the simulator's timing as before; the expert choices are logged as in Q4.

## 2. Runs
- **Sweep:** Mixtral (8 experts, top-2, 32 MoE layers) and DeepSeek-R1 (256, top-8, 57) × 6 ρ × 5 α.
- **Reference point** (ρ = 0.9, α = 0.5): all four models, router bias 0 and 0.5.
- Every run: single H100, decode mode, batch 16, 128 decode steps, context 1024. 66 runs in total; `scripts/q4x_router_runs.py`.

## 3. Statistics (`scripts/q4x_analysis.py`)
Each statistic is compared with a **shuffled baseline**: the same choices with the token order scrambled. Shuffling keeps "who is popular" but destroys "what comes next", so **real − shuffled** is the relation a predictor can use.

| Statistic | Definition |
|---|---|
| temporal reuse(d) | share of a token's experts that the same sequence uses again d tokens later, same layer; shuffled = token order permuted within each (sequence, layer) |
| cross-layer hit(d) | build a count table "experts at layer l → experts at layer l+d" on steps 0–63; for steps 64–127 predict layer l+d's top-k from layer l; share predicted correctly. Shuffled = tokens permuted independently per layer |
| same-id next layer | share of experts with the same index at layer l and l+1 |
| batch overlap | share of experts needed at step t+1 (union over the 16 sequences) already needed at step t |
| max expert share | load of the most popular expert per layer ÷ fair share (1 = perfectly balanced) |

## 4. Results (`data/q4x_statistics.csv`, `data/q4x_reuse_vs_lag.csv`)

### 4.1 Temporal relation grows with token similarity (`plots/q4x_fig1_temporal_vs_knobs.png`, `plots/q4x_fig3_temporal_reuse_vs_lag.png`)
Next-token reuse at α = 0.5, real (shuffled):

| ρ | 0 | 0.5 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| Mixtral (chance 0.25) | 0.254 (0.255) | 0.449 (0.262) | 0.625 (0.280) | **0.718** (0.307) | 0.793 (0.359) | 0.899 (0.557) |
| DeepSeek-R1 (chance 0.031) | 0.050 (0.049) | 0.184 (0.053) | 0.390 (0.065) | **0.527** (0.087) | 0.642 (0.128) | 0.821 (0.323) |

- ρ = 0 gives no relation at all (real = shuffled = about chance), the same as the dice-roll simulator.
- Reuse decays with token distance: Mixtral at ρ = 0.9 goes 0.72 → 0.52 → 0.41 → 0.32 at d = 1, 4, 8, 16.
- **Larger α weakens the temporal relation** (DeepSeek, ρ = 0.9: 0.59 at α = 0.05 down to 0.29 at α = 2). Big layer updates amplify small differences between neighbouring tokens.
- **The shuffled level rises above chance at high ρ** (DeepSeek, ρ = 0.99: 0.32 vs chance 0.03). When text stays on one topic, each *sequence* keeps its own favourite experts over the whole window, a per-request preference that survives shuffling. It is also usable for prefetching.

### 4.2 Cross-layer relation (`plots/q4x_fig2_crosslayer_vs_knobs.png`, `plots/q4x_fig4_crosslayer_vs_distance.png`, `plots/q4x_fig5_transition_tables_mixtral.png`)
Guess layer l+1 from layer l at ρ = 0.9, real (shuffled):

| α | 0.05 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| Mixtral | 0.311 (0.289) | 0.330 (0.290) | 0.373 (0.302) | 0.425 (0.306) | 0.419 (0.306) |
| DeepSeek-R1 | 0.108 (0.063) | 0.170 (0.078) | **0.262** (0.103) | 0.328 (0.140) | 0.354 (0.137) |

- **Same expert index on the next layer = chance** (Mixtral 0.246 vs 0.25; DeepSeek 0.032 vs 0.031). Each layer's router is different, so expert numbers do not line up between layers; the relation is a *mapping* that the count table learns.
- **The relation hardly decays across layers:** DeepSeek at α = 0.5 gives 0.262, 0.260, 0.255, 0.247 for d = 1, 2, 4, 8. All layers route the same token, whose content persists through the residual stream.
- **It gets stronger as α grows,** together with load imbalance (DeepSeek's most popular expert: 2.8× its fair share at α = 0.05, 6.2× at α = 1). Larger updates push token vectors toward a few typical states, so the same expert groups recur across layers. With tiny α each layer's router sees a raw random vector, and two independent routers give only weakly related choices.
- For Mixtral the cross-layer relation is weak (0.37 vs 0.30). With only 2 of 8 experts chosen, the choice carries little information about the token.

### 4.3 All four models at the reference point (ρ = 0.9, α = 0.5) (`plots/q4x_fig6_models_reference_point.png`)

| Model | Next-token reuse: dice / router (shuffled) | Cross-layer hit: router (shuffled) | Batch overlap: dice / router (shuffled) | Max expert share |
|---|---|---|---|---|
| Mixtral | 0.25 / **0.72** (0.31) | 0.37 (0.30) | 0.99 / 0.99 (0.99) | 1.3× |
| Grok-1 | 0.25 / **0.70** (0.31) | 0.42 (0.32) | 0.99 / 0.99 (0.98) | 1.4× |
| Llama-4 Maverick | 0.008 / **0.44** (0.04) | 0.05 (0.02) | 0.12 / **0.51** (0.20) | 5.5× |
| DeepSeek-R1 | 0.031 / **0.53** (0.09) | 0.26 (0.10) | 0.40 / **0.69** (0.49) | 4.5× |

"Dice" values are the simulator's default uniform routing, taken from Q4. A router bias of 0.5 (built-in favourite experts) raises every number a little but adds mostly popularity: the shuffled values rise just as much.

## 5. Conclusions
1. **The relations ST-MoE and APEX exploit follow directly from how routing works.** Similar consecutive tokens give temporal reuse, and the shared residual stream gives cross-layer relations. The simulator's default dice roll removes both, which is why Q4 found nothing.
2. **Temporal relation is the strongest signal:** up to 0.72 reuse for Mixtral and 0.53 for DeepSeek at ρ = 0.9, falling off over about 10 tokens. A predictor should rely mainly on the most recent tokens' experts.
3. **The cross-layer relation is real but weaker,** and it stretches over many layers. That supports *early* prediction (from layer l, guess layers l+1 … l+8), which gives a prefetcher time to fetch.
4. **Large-expert models benefit most.** For DeepSeek, 69% of next-step experts are already needed now (vs 40% with dice). For 8-expert models, nearly all experts are needed every step at batch 16.
5. **Limitation.** ρ, α and the router weights are assumptions; there are no trained weights. The *direction* of every effect follows from the mechanism, but the exact sizes depend on the knobs. The final phase should report predictor results across a range of ρ.

## How to reproduce
```bash
cd results-ca        # from the repository root
python3 scripts/q4x_router_runs.py              # 66 simulations, ~15 min on 28 cores (skips finished runs)
python3 scripts/q4x_analysis.py                 # statistics + figures, ~3 min
python3 scripts/q4x_analysis.py --plots-only    # figures only, from data/
```
`runs/<model>_rho<ρ>_alpha<α>_bias<b>/` holds each run's `config.yaml`, `routing_log.csv`, simulator log and output CSV. On GitHub the 30 DeepSeek `routing_log.csv` files (147 MB) are left out to keep the repository small; `q4x_router_runs.py` regenerates exactly those (it skips runs whose log exists), after which `q4x_analysis.py` recomputes everything. `--plots-only` needs only the committed Mixtral logs.
