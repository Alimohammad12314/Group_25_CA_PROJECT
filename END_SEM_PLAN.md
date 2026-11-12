# End-semester plan — Expert prediction, prefetching and HBM management (Group 25)

Goal: a training-free expert predictor, a prefetcher that moves predicted experts into GPU HBM ahead of time, and an HBM eviction policy better than LRU. Everything is evaluated on **real MoE routing traces** in simulation, explaining *why* it helps, *when* it fails, and what it costs in traffic and energy.

---

## 1. Decisions to take in week 1

| # | Decision | Recommendation |
|---|---|---|
| D1 | Simulator | Hybrid (section 2): our own trace-driven **prefetch engine**, with costs (time, energy) from **LLMSimulator / SSD-offloading** (course primary simulator, already patched and validated). **LLMServingSim 2.0** gets a 1-week feasibility check as an optional second backend. |
| D2 | Real traces | **Mixtral-8x7B: `allenai/analysis_mixtral`** (174 MB, C4 text, top-2 expert ids per token × 32 layers, 2048 tokens per document). Plus one **fine-grained** model (many experts) once its format is checked. Candidate: `mistrjirka/Predicting-Experts-for-MOE-Suite` (GLM, 1.4 GB). |
| D3 | Questions for the TA | Is LLMServingSim allowed as a second simulator? Which HBM budget should we use for experts? Is the trace-driven engine with LLMSimulator costs acceptable as "the LLM simulator"? |
| D4 | Deadline | Fill in the timeline in section 6. |

## 2. System to build

```
 real routing traces ──► workload builder (batch B, context L, decode order)
                               │  experts needed per (step, layer)
                               ▼
 ┌──────────────── prefetch engine (event timeline, per layer) ────────────────┐
 │ compute stream: attention → router → experts (wait until weights in HBM)    │
 │ copy stream:    DRAM → HBM transfers over the link (one queue, bandwidth)   │
 │ predictor ──► prefetcher (when, what, how many) ──► copy queue              │
 │ HBM manager: capacity C experts, eviction policy (LRU / ours / Belady)      │
 └──────────────────────────────────────────────────────────────────────────────┘
        ▲ per-op costs                                        │ metrics
 LLMSimulator / SSD-offloading:                               ▼
 attention & non-expert time, expert time vs tokens,   accuracy, coverage, timely/late,
 link time, HBM bandwidth loss during copies,          hit rate, traffic, stall, latency,
 energy per bit + background power                     energy
```

- **Cost tables** come from the simulators (like Q2) for every (model, GPU, batch, context, link): per-layer attention and non-expert time, expert execution time as a function of tokens per expert, transfer time = bytes ÷ link bandwidth, and the bandwidth loss while copying (the SSD-offloading contention model).
- **Validation gate:** with no cache and no prefetch, the engine must reproduce the Q2 on-demand step latency. Nothing else is trusted until this passes.
- **Bonus:** convert the real traces into LLMSimulator's own trace-replay format (`expert_data/experts_<model>_<dataset>.csv`, read by `Scheduler::initExpertList`). The simulator then executes realistic expert traffic directly, and its numbers cross-check the engine.

## 3. Components

### 3.1 Predictor (training-free; each one is a paper strategy)

| Predictor | Idea | Source |
|---|---|---|
| P0 Popularity | most frequent experts of this layer so far | baseline |
| P1 Temporal | experts this sequence used at the same layer for the last tokens | MoE-Infinity-style recency; our Q4-extended |
| P2 Cross-layer | online count table "experts at layer l → experts at layer l+d" | ST-MoE spatial correlation |
| P3 Request frequency | per-request expert activation counts | MoE-Infinity (activation matrix) |
| P4 Hybrid + adaptive count | weighted score of P1–P3; prefetch experts until cumulative confidence ≥ τ instead of a fixed k | APEX adaptive budget |
| Oracle | the true future experts | upper bound |

Pre-gated MoE and ProMoE need hidden states or trained predictors, which the id-only traces don't provide. List them as related work and as a limitation.

**Metrics:** precision, recall, F1, top-k hit rate. Each is computed per layer and per lookahead distance d = 1, 2, 4, 8, and for batch-union sets (which matter for serving).

### 3.2 Prefetcher
- **Trigger:** when layer l's router decides (and optionally at token start), predict layers l+1 … l+L (lookahead L).
- **What:** predicted experts not already in HBM, highest confidence first, capped by an aggressiveness knob (τ or extra-k) and a per-step byte budget.
- **Overlap:** transfers run on the copy stream while the compute stream runs layers l … l+L. A transfer that finishes before its expert is needed is **timely**. One still in flight is **late**, and only the remaining time is exposed. One whose expert is never used before eviction is **unused**.
- **Contention:** while a copy runs, compute sees reduced HBM bandwidth (SSD-offloading model). This is why aggressive prefetching can make things slower (we saw this in Q3).

### 3.3 HBM manager
- Capacity C experts (sweep 12.5–75% of all experts).
- **Baselines:** LRU, LFU, Belady (offline optimum, upper bound).
- **Our policy (to finalise):** *prediction-aware, cost-aware eviction*. Evict the expert with the lowest (predicted probability of use within the horizon × reload cost). Protect in-flight and not-yet-used prefetches. Keep a per-layer quota so one layer cannot flush the others.

### 3.4 Configurations (required comparison)

| Config | Prediction | Prefetch | HBM policy |
|---|---|---|---|
| C1 Baseline | – | – | LRU (on-demand) |
| C2 Prediction, no prefetch | ✓ (guides retention only) | – | prediction-guided |
| C3 Prediction + prefetch | ✓ | ✓ | LRU |
| C4 Complete | ✓ | ✓ | our policy |
| Bounds | oracle predictor / Belady | ✓ | – |

C1 → C2 → C3 → C4 shows the contribution of each component.

## 4. Experiments

- **Metrics (all configurations):**
  - prediction: precision, recall, F1;
  - prefetch: coverage, timely/late rate, unused prefetches;
  - memory: HBM hit rate, extra traffic (GB per token), exposed stall time;
  - latency: expert-load latency, token latency (TPOT) and end-to-end latency;
  - energy per token.
- **Sensitivity:**
  - batch {1, 4, 16, 32, 64};
  - context {256, 512, 1024, 2048} (the trace has 2048 tokens per document);
  - lookahead {1, 2, 4, 8};
  - aggressiveness (τ / extra-k);
  - HBM capacity;
  - link {PCIe4, PCIe5, C2C-450}.
- **Failure analysis (required):** where prefetching gives no gain or degrades performance. Expected cases:
  - Mixtral at batch ≥ 16, where almost every expert is needed anyway;
  - low prediction accuracy;
  - lookahead too short (late prefetches);
  - too aggressive, causing traffic and HBM contention;
  - tiny HBM budget, where prefetches evict each other.
- **Energy:** SSD-offloading energy model (DDR read + HBM write per bit, HBM and compute energy, background power × time). Compare the stall time saved against the extra traffic.

## 5. Work split (three equal parts; each person also writes their own report sections)

| Role | Owns | Deliverables |
|---|---|---|
| **A: Traces & prediction** | trace download and parsing, workload builder, correlation analysis on *real* traces (redo Q4 on real data), predictors P0–P4 + oracle, accuracy metrics | `traces/`, `predictors/`, accuracy plots, report sections "Traces", "Prediction" |
| **B: Engine & HBM** | cost tables from LLMSimulator, event engine (compute + copy streams, contention), prefetcher, HBM policies (LRU, LFU, Belady, ours), validation gate, LLMServingSim feasibility check | `engine/`, validation note, report sections "Prefetching", "HBM management" |
| **C: Experiments, energy, bonus** | experiment harness, configurations C1–C4, sensitivity sweeps, energy analysis, bonus (real traces in LLMSimulator), plots, reproducibility (`setup.sh`, commands, ZIP) | `experiments/`, figures, report sections "Evaluation", "Energy", "Reproducibility" |

**Shared:** the interface between parts (trace format → engine, predictor API → prefetcher), weekly integration, and everyone reviews everyone's sections. Track commits per person so contribution stays equal.

## 6. Timeline (adjust to the deadline)

| Week | A | B | C |
|---|---|---|---|
| 1 | download traces, parser, workload builder | cost tables, engine skeleton; LLMServingSim check | repo layout, harness, plotting style; TA questions |
| 2 | real-trace correlation, P0–P2 + metrics | on-demand engine **passes validation gate**, LRU cache | baseline C1 runs, energy hook |
| 3 | P3, P4 (adaptive), oracle | prefetcher (timely/late/unused), contention | C2/C3 runs, first sensitivity sweeps |
| 4 | prediction analysis per layer and lookahead | our HBM policy + Belady | C4, full sensitivity, energy, bonus |
| 5 | report: prediction | report: prefetch + HBM | report: evaluation, energy, reproducibility; final ZIP |

## 7. Risks

| Risk | Mitigation |
|---|---|
| Mixtral has only 8 experts: at large batch almost all are needed, so prefetching is moot | add a fine-grained trace (D2) and report this as an explained failure case |
| traces hold expert ids only (no hidden states) | stick to id-based predictors; note gate-based ones as a limitation |
| LLMServingSim needs GPU-profiled tables or lacks offloading hooks | keep it optional; LLMSimulator costs are the primary backend |
| engine diverges from the simulator | validation gate (section 2) and the bonus cross-check |

## 8. Reading list (prediction, prefetching, caching for MoE)

Venue check: ✅ confirmed by search · ◑ from memory, verify on DBLP/ACM/IEEE before citing · arXiv = preprint.

**Must-read: closest to our design**

| Paper | Venue | Why it matters for us |
|---|---|---|
| Pre-gated MoE (Hwang et al.) | ISCA 2024 ✅ | predicts next-layer experts one layer early; the canonical prefetch design |
| MoE-Infinity (Xue et al.) | arXiv (◑ later venue) | request-level activation tracing → frequency-based prefetch and caching |
| ProMoE: proactive caching for MoE serving | arXiv | sliding-window prefetch with a small predictor (input → experts) |
| Fast Inference of MoE LMs with Offloading (Eliseev & Mazur) | arXiv 2023 | LRU expert cache + speculative loading for Mixtral; natural LRU baseline |
| FineMoE (latency–memory trade-off) | EuroSys 2026 ✅ | prefetches experts CPU → GPU; recent strong related work |
| ST-MoE spatio-temporal prefetching (Zhao et al.) | arXiv 2026 (course ref [2]) | adjacent-layer + consecutive-token correlation, the basis of P1/P2 |
| APEX adaptive expert prefetching (Kanani et al.) | CODES 2026 (course ref [3]) | adaptive number of experts to prefetch, the basis of P4 |
| COPE: emulating MoE offloading policies | arXiv 2026 | batch-1 policy emulator; shows the gaps we fill (batch > 1, late prefetch, contention, eviction, energy) |

**Offloading / CPU–GPU systems (context and baselines)**

| Paper | Venue |
|---|---|
| KTransformers (CPU/GPU hybrid MoE inference) | SOSP 2025 ✅ |
| MoE-Lightning | ASPLOS 2025 ◑ |
| Fiddler (CPU executes cold experts) | ICLR 2025 ◑ |
| SiDA-MoE (hash-based expert prediction) | MLSys 2024 ◑ |
| AdapMoE (adaptive gating + prefetch) | ICCAD 2024 ◑ |
| HybriMoE | DAC 2025 ◑ |
| SwapMoE | ACL 2024 ◑ |
| Read-ME (router-decoupled MoE + scheduling) | NeurIPS 2024 ◑ |
| Lina (expert popularity in training/inference) | USENIX ATC 2023 ◑ |
| HOBBIT (mixed-precision expert offloading) | arXiv 2024 |
| ExpertFlow, DAOP, eMoE, SpecPrefetch | arXiv 2024–2026 |
| EdgeMoE | arXiv 2023 (◑ later journal) |

**Hardware / simulators**

| Paper | Venue |
|---|---|
| Duplex (course ref [1]) | MICRO 2024 ✅ |
| LLMServingSim | IISWC 2024 ◑; 2.0: arXiv 2511.07229 / 2602.23036 |
| DeepSpeed-MoE (MoE inference background) | ICML 2022 ◑ |

**Classic cache replacement (for our HBM policy)**

| Paper | Venue |
|---|---|
| Belady, optimal replacement (MIN) | IBM Systems Journal 1966 |
| LRFU (recency + frequency spectrum) | IEEE Trans. Computers 2001 |
| ARC: Adaptive Replacement Cache | USENIX FAST 2003 |
