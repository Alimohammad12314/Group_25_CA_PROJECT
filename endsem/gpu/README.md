# Real-GPU validation on an H100 (GPU.md)

Group 25. This folder is GPU.md carried out: the real Mixtral-8x7B runs on a real H100 with its experts in host
memory and a bounded expert cache in HBM. The same documents, policies and prediction streams as the engine are used,
and every real run is compared with the simulator. In the report this is the validation section, not the main results.

## Summary

- **The public trace is real Mixtral routing.** Our runtime reproduces it for 98.7 % of (token, layer) pairs and
  97.6 % of predicted tokens, on 32 documents. Caching and prefetching never change the output: router logits are
  bit-identical across cache sizes and policies.
- **The simulator's cache logic is exact.** On the GPU's own routing, hit rate and GB per token agree exactly for all
  33 timing-independent runs (C0, C1, C1b, C2).
- **Latency holds once costs are measured.** With costs measured on this machine the simulator is within 3.1 % median
  (12.8 % worst, C5 at batch 1) over all 77 runs. With the default costs (LLMSimulator/Q2, nominal PCIe 5.0) it is
  16.6 % too fast. GPU energy is 17.7 % above the calibrated model (43 % above the default one).
- **The hidden-state predictor (Phase B″) is far more accurate than ID-only prediction:** 86–88 % next-layer overlap at
  batch 1 against 42–44 %, and 91 % against 74–75 % at batch 4.
- **That accuracy buys little speed on PCIe.** At batch 1 the full design goes from 1.36× to 1.40× over on-demand in
  simulation. Live (C5) it is 1.28× against C4's 1.32×, because the predictor costs 11–15 ms per token in this Python
  runtime. Plain LRU + miss overlap stays fastest at batch 1 (1.56×). On a slow link the value gate blocks most
  prefetches, so better predictions have little to act on. The oracle reaches 1.67× (real) and 1.72× (simulated).
- **Real hardware adds costs the simulator does not model.** While waiting on the link the GPU and the CPU both clock
  down, so the kernels and launches that follow are 1.5–3× slower. Copies slow compute by 8 % (the simulator's formula
  predicts 2 %). The GPU idles at 91 W, not the 70 W the energy model assumes.

## Machine

| | |
|---|---|
| GPU | **NVIDIA H100 NVL** (94 GB HBM3, about 3.9 TB/s, 400 W): neither the PCIe nor the SXM card GPU.md expects. GPU 0 was used throughout. |
| Host link | **PCIe 5.0 x16** (measured 57.7 GB/s raw, 56.2 GB/s through our chunked scheduler) |
| CPU / NUMA | 2 × AMD EPYC 9354 (32 cores each), 2 NUMA nodes; GPU 0 is on node 0; every run used `numactl --cpunodebind=0 --membind=0` |
| Host RAM | 503 GB |
| Software | driver 575.57.08, CUDA 12.1 (torch 2.5.1+cu121), transformers 4.57.6, Python 3.10.2 |
| Access | GPU jobs go through the server's `gpurun` broker (48 GPU-h per week); about 7.5 GPU-h were used |
| Pinned memory | `ulimit -l` is 63 GB (8 GB inside broker jobs), but pinning 256 × 352 MB = 90 GB worked anyway |
| Clocks | not locked (no root) |

Full record: `results/env.txt` and `results/env.json`.

**One premise of GPU.md does not hold on this card.** Mixtral BF16 is 87 GiB and the H100 NVL has 93.6 GiB, so the
model fits with about 5 GB to spare. The offloaded setup is therefore set by the cache ratio (25 % / 50 % of experts in
HBM), not forced by memory size. On an 80 GB H100 it would be forced.

## How to run

```bash
cd endsem && . .venv/bin/activate
G="gpurun -g 1 gpu/numa_run.sh"                       # broker + NUMA binding of the assigned GPU
$G python -m gpu.env                                  # section 0
$G python -m gpu.check_routing --docs 32              # Phase B (+ correctness checks 1-3)
$G python -m gpu.microbench                           # Phase A -> results/calibration_h100.json
$G python -m gpu.collect_traces                       # Phase B' (live traces, not committed)
$G python -m gpu.hs_predictor                         # Phase B'' (experts resident; batch 16 runs out of memory)
$G python -m gpu.hs_predictor --offloaded --skip-check --runs 16x1024 16x1536
python -m gpu.run_e7 --gpu H100 --calibration gpu/results/calibration_h100.json      # e7, see "Differences"
$G python -m gpu.run_matrix --stages minimum context batch4 aggr                      # Phase C (resumable)
python -m gpu.compare --procs 24                      # Phase E -> results/compare*.csv, results/plots/
```

| File | What it does |
|---|---|
| `weights.py` | safetensors → non-expert weights on the GPU (3.2 GB) + one pinned 352 MB buffer per expert ([w1 \| w3 \| w2]) |
| `runtime.py` | Hugging Face Mixtral with every MoE block replaced by `OffloadedMoE` (same routing math). It holds the HBM slab, the copy scheduler `RealLink` (one thread, one copy stream, 16 MB chunks, demand first, ≤ 2 chunks in flight) and `RealTimeline` |
| `common.py` | configurations C0–C5/O (the engine's own), workloads with the decode starting at position ctx, cached streams |
| `microbench.py` | Phase A |
| `check_routing.py` | Phase B |
| `collect_traces.py` | Phase B′ |
| `hs_predictor.py` | Phase B″ (offline collection, and the live predictor for C5) |
| `run_matrix.py` | Phase C |
| `energy.py` | Phase D: NVML energy counter (RAPL is root-only here) |
| `compare.py` | Phase E |
| `run_e7.py` | runs `experiments/e7_hidden_state.py` with one fix |
| `diag.py`, `env.py`, `numa_run.sh` | diagnostics, machine record, launcher (refuses a GPU with a pending row remap) |

**How the runtime relates to the engine.** `RealTimeline` subclasses the engine's `Timeline`. Eviction scores,
`_make_room`, `_issue` (lookahead, link-time cap, value gate) and `_end_layer` (stale cancel, evict-after-use) are the
engine's code, running on the engine's `ExpertCache`. Only the per-layer loop is new: it issues real copies, waits on
CUDA events and measures time on the GPU. That is why the timing-independent policies can be compared exactly.

**Workload.** The allenai C4 documents the engine uses (seed 0), decoded teacher-forced: the prompt is the first ctx
tokens, then 128 steps each feed the document's true next token. The documents run in groups: 4 × 1 at batch 1,
2 × 4 at batch 4, 1 × 16 at batch 16. The cache, link and predictor state carry over between groups, as in the engine.
Prefill streams experts through two staging slots and never touches the decode cache, so decode starts cold, as the
engine assumes. Per-token latency comes from CUDA events. Engine comparisons use all steps; the CSV also has
`*_after_warmup` columns that drop the first 8 steps of each group.

## Phase B: is the public trace real Mixtral routing?

Teacher-forced over 32 whole documents (2048 tokens each, `results/check_routing_base.json`):

| Check | Result | GPU.md threshold |
|---|--:|--:|
| Predicted next token = trace's `predicted_token_ids` (32 docs) | **97.6 %** (first 8 docs: 98.2 %) | ≥ 97 % |
| Top-2 routing = trace's `exp_ids`, as sets (32 docs, all layers) | **98.7 %** (per layer 97.9–99.7 %, lowest in layers 22–27) | ≥ 95 % |
| The same through the decode path (KV cache, expert cache) | routing 98.9 %, tokens 98.4 % | |
| Invariance: C1 at 25 % vs C1 at 50 %, C4, C1b | router logits identical (max \|Δ\| = 0.0), tokens identical | identical |

Only the base checkpoint (`Mixtral-8x7B-v0.1`) was tested. Its agreement is well above threshold, so Instruct was not
downloaded. Across the 77 matrix runs, routing equals the trace for 97.95–98.91 % of (step, sequence, layer) entries.
It is bit-identical (100.0 %) between all runs of the same workload, so every configuration sees exactly the same
experts.

## Phase A: calibration (`results/calibration_h100.json`)

| Measurement | Value | Engine default |
|---|---|---|
| Link, raw pinned → HBM copy (16 / 64 / 352 MB; 2 streams) | 57.2 / 57.6 / 57.7 GB/s; 57.7 GB/s (one PCIe link, no gain from 2 streams) | 64 GB/s nominal |
| Link through our scheduler (16 MB chunks, 2 in flight) | **56.2 GB/s**; one 352 MB expert in **6.32 ms** (32/64 MB chunks: 56.8/57.3 GB/s); 24 µs of Python per chunk | 5.5 ms |
| One expert's compute t_exec(n), n = 1 … 64 tokens | **136–140 µs**, flat in n (memory-bound) | |
| t_exec right after an 8 ms idle | **194–203 µs** (+45 %) | |
| Non-expert time per layer (attention + norms + router + per-layer router sync) | batch 1: 452–455 µs at every context; batch 4: 477 µs; batch 16: 511 / 511 / 556 / 633 µs at ctx 256 / 512 / 1024 / 1536 | Q2 non-expert |
| The same right after an 8 ms idle | +100 µs (batch 1: 555–558 µs; batch 16 ctx 1536: 715 µs) | |
| Contention: expert GEMV with a concurrent 352 MB copy running | **1.08×** (n = 16: 1.075, n = 64: 1.095) | HBM/(HBM − link) = 1.02× |
| GPU power (NVML energy counter) | idle 90.6 W; copy only 99.3 W; compute-bound 394 W; copy + compute 288 W | 70 W background |

What this says:
- **Non-expert time is dominated by host launch overhead** (Python + PyTorch dispatch, about 50 kernels per layer).
  Context only matters at batch 16 (+24 % from 256 to 1536), which is why the context sweep below is flat.
- **Copies slow compute 4× more than the SSD-offloading formula predicts.**
- **Clock-down after idle.** When the GPU waits on the link for ≥ 6 ms, its SM clock drops (345 MHz at idle against
  1785 MHz). The kernels that follow run 1.5× slower (GEMV) to 3× slower (small kernels), and a 1-SM spinning kernel
  does not keep the clock up. The same happens on the CPU: a thread that blocks lets the `schedutil` governor drop its
  core to 1.5 GHz, and PyTorch launches then cost 17 µs instead of 7 µs. The runtime therefore spin-waits (`sleep(0)`,
  which releases the GIL) instead of blocking. That cut C0 at batch 1 from 555 to under 460 ms per token. Locking clocks needs root.
  The "calibrated after idle" profile in Phase E uses the post-idle numbers.

## Phase B′: live traces

Greedy generation (up to 128 tokens, stop at EOS) with every expert resident in HBM:

| Dataset | Prompts | Mean prompt / generated tokens | Reuse, lag 1 | Cross-layer hit, d = 1 | Time |
|---|--:|--:|--:|--:|--:|
| GSM8K (math) | 50 | 70 / 124 | 0.373 | 0.472 | 161 s |
| HumanEval (code) | 50 | 121 / 114 | 0.345 | 0.460 | 149 s |

These are close to the C4 trace's statistics. The traces (engine `Trace` format with source `live:H100:<dataset>`, plus
the full 8-way router probabilities per token and layer) and LLMSimulator replay files are in `endsem/data/traces/` and
`endsem/data/llmsim/`. They are **not committed**: traces are never committed. Regenerate them with
`python -m gpu.collect_traces`. Per-layer statistics are in `results/live_traces_summary.json`.

## Phase B″: hidden-state predictor

`hs_predictor.py` applies layer (l+d)'s own router to the hidden state at layer l. It is training-free, uses about
2 MB of router and norm weights on the GPU, and costs 4 small GEMMs per layer:
- **variant A:** `softmax(W_router[l+d] · RMSNorm_{l+d}(h_l))`, with h_l the residual stream entering layer l's MoE;
- **variant B:** `softmax(W_router[l+d] · x_l)`, with x_l layer l's own router input;
- **d = 0:** the previous token's real router probabilities.

On a tiny random Mixtral both variants match Hugging Face's own norm and router modules exactly.

**Variant check** (8 documents, batch 1, ctx 1024, `results/hs_variant_check.json`): overlap by distance.

| distance | A (renorm) | B (no renorm) |
|---|--:|--:|
| 0.0 | 0.376 | 0.376 |
| 1.0 | 0.858 | 0.854 |
| 2.0 | 0.803 | 0.796 |
| 3.0 | 0.757 | 0.749 |
| 4.0 | 0.721 | 0.713 |

**Variant A is kept.** The variants tie at d = 0 (the previous token, the same for both); A is better at every
d ≥ 1, by 0.4–0.8 points. Both variants were recorded in every run
(`results/hs_variants_all_runs.json`):

| run | A d=1 | B d=1 | routing = matrix runs | collected |
|---|--:|--:|--:|--:|
| 1x1024 | 0.862 | 0.858 | 1.0 | resident experts |
| 4x1024 | 0.909 | 0.906 | 1.0 | resident experts |
| 16x1024 | 0.998 | 0.997 | 1.0 | offloaded runtime, C1 at 50 % cache (resident run out of GPU memory) |
| 1x256 | 0.874 | 0.871 | 1.0 | resident experts |
| 4x256 | 0.91 | 0.907 | nan | resident experts |
| 16x256 | 0.998 | 0.998 | 1.0 | resident experts |
| 1x1536 | 0.875 | 0.873 | 1.0 | resident experts |
| 4x1536 | 0.91 | 0.907 | nan | resident experts |
| 16x1536 | 0.998 | 0.998 | 1.0 | offloaded runtime, C1 at 50 % cache (resident run out of GPU memory) |

Files: `results/hs_pred_b{1,4,16}_ctx{256,1024,1536}.npz`, laid out as GPU.md specifies (`q` float16
[S, B, 32, 5, 8], `A` int16 [S, B, 32, 2], `doc_ids`, `tokens_per_request` = 128, `offset` = ctx). The two batch-16
files at ctx 1024 and 1536 ran out of memory with all experts resident: only about 5 GB is left for their KV cache and
prefill. They were collected through the offloaded runtime instead (C1 at 50 %); the routing is identical either way.

### e7: hidden-state vs ID-only on the same tokens

`python -m gpu.run_e7 --gpu H100 --calibration gpu/results/calibration_h100.json`. The outputs are copied to
`results/e7_calibrated/` (`accuracy.csv`, `policies.csv`, `output.txt`), and an uncalibrated run is in
`results/e7_default/`.

**Overlap** (share of the batch's experts that are among the top-|U| predictions), by distance d:

| batch | ctx | predictor | d=0 | d=1 | d=2 | d=3 | d=4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | 256 | ID-only (ours) | 0.343 | 0.419 | 0.409 | 0.409 | 0.406 |
| 1 | 256 | hidden-state (H100) | 0.353 | 0.874 | 0.825 | 0.786 | 0.751 |
| 1 | 1024 | ID-only (ours) | 0.372 | 0.42 | 0.419 | 0.418 | 0.408 |
| 1 | 1024 | hidden-state (H100) | 0.381 | 0.862 | 0.809 | 0.763 | 0.729 |
| 1 | 1536 | ID-only (ours) | 0.352 | 0.438 | 0.432 | 0.432 | 0.422 |
| 1 | 1536 | hidden-state (H100) | 0.361 | 0.875 | 0.828 | 0.785 | 0.756 |
| 4 | 256 | ID-only (ours) | 0.717 | 0.739 | 0.736 | 0.736 | 0.736 |
| 4 | 256 | hidden-state (H100) | 0.718 | 0.91 | 0.892 | 0.876 | 0.86 |
| 4 | 1024 | ID-only (ours) | 0.724 | 0.746 | 0.746 | 0.745 | 0.74 |
| 4 | 1024 | hidden-state (H100) | 0.725 | 0.909 | 0.889 | 0.873 | 0.855 |
| 4 | 1536 | ID-only (ours) | 0.717 | 0.743 | 0.74 | 0.739 | 0.738 |
| 4 | 1536 | hidden-state (H100) | 0.721 | 0.91 | 0.891 | 0.872 | 0.857 |
| 16 | 256 | ID-only (ours) | 0.991 | 0.992 | 0.992 | 0.992 | 0.992 |
| 16 | 256 | hidden-state (H100) | 0.991 | 0.998 | 0.997 | 0.996 | 0.995 |
| 16 | 1024 | ID-only (ours) | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 |
| 16 | 1024 | hidden-state (H100) | 0.991 | 0.998 | 0.997 | 0.995 | 0.994 |
| 16 | 1536 | ID-only (ours) | 0.991 | 0.992 | 0.992 | 0.992 | 0.992 |
| 16 | 1536 | hidden-state (H100) | 0.99 | 0.998 | 0.998 | 0.996 | 0.994 |

**Recall / precision with the prefetch budget, d = 1:**

| batch | ctx | predictor | recall | precision | full coverage | budget | union |
|---|--:|--:|--:|--:|--:|--:|--:|
| 16 | 1024 | hidden-state (H100) | 0.971 | 0.993 | 0.767 | 7.725 | 7.908 |
| 16 | 1024 | ID-only (ours) | 1.0 | 0.988 | 1.0 | 8.0 | 7.908 |
| 16 | 1536 | hidden-state (H100) | 0.969 | 0.994 | 0.755 | 7.704 | 7.9 |
| 16 | 1536 | ID-only (ours) | 1.0 | 0.987 | 1.0 | 8.0 | 7.9 |
| 16 | 256 | hidden-state (H100) | 0.967 | 0.994 | 0.735 | 7.689 | 7.906 |
| 16 | 256 | ID-only (ours) | 1.0 | 0.988 | 1.0 | 8.0 | 7.906 |
| 1 | 1024 | hidden-state (H100) | 0.862 | 0.862 | 0.735 | 2.0 | 2.0 |
| 1 | 1024 | ID-only (ours) | 0.672 | 0.325 | 0.506 | 4.366 | 2.0 |
| 1 | 1536 | hidden-state (H100) | 0.875 | 0.875 | 0.757 | 2.0 | 2.0 |
| 1 | 1536 | ID-only (ours) | 0.712 | 0.326 | 0.556 | 4.639 | 2.0 |
| 1 | 256 | hidden-state (H100) | 0.874 | 0.874 | 0.758 | 2.0 | 2.0 |
| 1 | 256 | ID-only (ours) | 0.654 | 0.326 | 0.484 | 4.248 | 2.0 |
| 4 | 1024 | hidden-state (H100) | 0.709 | 0.958 | 0.074 | 3.976 | 5.487 |
| 4 | 1024 | ID-only (ours) | 0.986 | 0.689 | 0.952 | 7.857 | 5.487 |
| 4 | 1536 | hidden-state (H100) | 0.711 | 0.958 | 0.075 | 3.977 | 5.478 |
| 4 | 1536 | ID-only (ours) | 0.987 | 0.687 | 0.955 | 7.872 | 5.478 |
| 4 | 256 | hidden-state (H100) | 0.715 | 0.96 | 0.079 | 3.974 | 5.453 |
| 4 | 256 | ID-only (ours) | 0.987 | 0.683 | 0.955 | 7.875 | 5.453 |

Reading it:
- **Batch 1:** the hidden state roughly doubles accuracy at every distance (d = 1: 0.86–0.88 against 0.42–0.44). With
  a budget of exactly 2 it reaches 86–88 % recall at 86–88 % precision, against 65–71 % recall at 33 % precision for
  ID-only.
- **Batch 4:** overlap improves from 0.75 to 0.91. But e7's budget rule M = ⌈Σ P⌉ undershoots the union: about 4 slots
  for 5.5 needed experts, because sharp per-token softmaxes make the batch union look smaller than it is. So the
  hidden-state stream has lower recall (0.71) than ID-only (0.99) at higher precision (0.96 against 0.69). A budget of
  ⌈Σ P⌉ + 1 or the engine's adaptive budget would likely fix this; it was not tried.
- **Batch 16:** saturated. About 7.9 of 8 experts are needed per layer, so both predictors name almost all of them.
- **Context length (256–1536)** changes nothing.

**Speedup over on-demand, 25 % cache, calibrated costs (this H100, PCIe 5.0):**

| batch | ctx | link | LRU + overlap | predict-evict, ID | predict-evict, HS | full design, ID | full design, HS | oracle |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 256 | PCIe5 | 1.52 | 1.32 | 1.36 | 1.33 | 1.37 | 1.7 |
| 1 | 1024 | PCIe5 | 1.58 | 1.34 | 1.39 | 1.36 | 1.4 | 1.72 |
| 1 | 1536 | PCIe5 | 1.54 | 1.33 | 1.36 | 1.34 | 1.38 | 1.7 |
| 4 | 256 | PCIe5 | 1.02 | 1.31 | 1.31 | 1.31 | 1.31 | 1.32 |
| 4 | 1024 | PCIe5 | 1.02 | 1.32 | 1.32 | 1.32 | 1.32 | 1.33 |
| 4 | 1536 | PCIe5 | 1.02 | 1.31 | 1.31 | 1.32 | 1.31 | 1.33 |
| 16 | 256 | PCIe5 | 1.02 | 1.29 | 1.3 | 1.3 | 1.3 | 1.3 |
| 16 | 1024 | PCIe5 | 1.02 | 1.29 | 1.29 | 1.3 | 1.3 | 1.3 |
| 16 | 1536 | PCIe5 | 1.02 | 1.29 | 1.29 | 1.3 | 1.3 | 1.3 |

With `--calibration`, e7's per-link rows all use the measured PCIe 5.0 bandwidth, so only the PCIe5 rows are shown.
The uncalibrated run (engine defaults, ctx 1024) compares links:

| batch | ctx | link | LRU + overlap | predict-evict, ID | predict-evict, HS | full design, ID | full design, HS | oracle |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1024 | C2C-900 | 1.62 | 1.41 | 1.46 | 1.43 | 1.48 | 2.12 |
| 1 | 1024 | PCIe4 | 1.61 | 1.35 | 1.4 | 1.37 | 1.42 | 1.75 |
| 1 | 1024 | PCIe5 | 1.61 | 1.36 | 1.41 | 1.37 | 1.42 | 1.75 |
| 4 | 1024 | C2C-900 | 1.21 | 1.54 | 1.54 | 1.55 | 1.57 | 1.65 |
| 4 | 1024 | PCIe4 | 1.01 | 1.31 | 1.31 | 1.32 | 1.32 | 1.32 |
| 4 | 1024 | PCIe5 | 1.02 | 1.32 | 1.32 | 1.33 | 1.32 | 1.33 |
| 16 | 1024 | C2C-900 | 1.22 | 1.46 | 1.5 | 1.55 | 1.58 | 1.54 |
| 16 | 1024 | PCIe4 | 1.01 | 1.29 | 1.29 | 1.29 | 1.29 | 1.29 |
| 16 | 1024 | PCIe5 | 1.02 | 1.29 | 1.3 | 1.29 | 1.3 | 1.29 |

In simulation the hidden-state predictor adds 3–4 % at batch 1 on every link and 1–2 % on NVLink-C2C at batches
4–16. Prefetching is limited by the link, not by prediction quality: on PCIe the value gate blocks nearly every
speculative copy. LRU + overlap stays fastest at batch 1. At batch 4 and above on PCIe, predict-evict already reaches the oracle.

## Phase C: run matrix (`results/real_runs.csv`, 77 runs)

All sweep stages ran: the minimum matrix with all 8 configurations, the context sweep, batch 4 and the aggressiveness
sweep. C0 runs once per workload because it keeps no cache. Configurations are the engine's own
(`experiments/common.py`):
- C3 uses the `full` stream with `BASE_PREFETCH`;
- C4 and O use `FULL_SIM`;
- C5 is `FULL_SIM` driven by the hidden-state predictor computed live inside the decode.

Speedup is over C0. Energy is GPU-only (NVML).

| configuration | cache | ms/token | p95 | speedup | stall ms | hit | late | prefetch/token | prefetch acc. | GB/token | wasted GB | link util. | load lat. ms | GPU J/token | GPU W |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| C0 on-demand | none | 435.3 | 435.7 | 1.0 | 401.7 | 0.0 | 0.0 | 0.0 | 0.0 | 22.55 | 0.0 | 0.91 | 9.5 | 47.68 | 110 |
| C1 LRU | 25 % | 281.3 | 359.3 | 1.55 | 249.4 | 0.38 | 0.0 | 0.0 | 0.0 | 13.98 | 0.0 | 0.88 | 5.12 | 31.44 | 112 |
| C1 LRU | 50 % | 209.5 | 303.9 | 2.08 | 178.2 | 0.557 | 0.0 | 0.0 | 0.0 | 9.99 | 0.0 | 0.84 | 3.44 | 24.62 | 118 |
| C1b LRU + overlap | 25 % | 278.6 | 356.3 | 1.56 | 243.7 | 0.38 | 0.0 | 0.0 | 0.0 | 13.98 | 0.0 | 0.89 | 5.12 | 31.93 | 115 |
| C1b LRU + overlap | 50 % | 207.6 | 301.4 | 2.1 | 173.4 | 0.557 | 0.0 | 0.0 | 0.0 | 9.99 | 0.0 | 0.85 | 3.44 | 24.5 | 118 |
| C2 predict-evict | 25 % | 330.8 | 366.3 | 1.32 | 293.2 | 0.258 | 0.0 | 0.0 | 0.0 | 16.73 | 0.0 | 0.89 | 6.6 | 36.79 | 111 |
| C2 predict-evict | 50 % | 221.7 | 271.0 | 1.96 | 185.5 | 0.527 | 0.0 | 0.0 | 0.0 | 10.66 | 0.0 | 0.85 | 3.81 | 25.87 | 117 |
| C3 prefetch + LRU (naive) | 25 % | 1004.3 | 1156.7 | 0.43 | 943.2 | 0.0 | 0.669 | 139.43 | 0.31 | 56.58 | 34.03 | 0.99 | 23.39 | 105.42 | 105 |
| C3 prefetch + LRU (naive) | 50 % | 617.4 | 1153.2 | 0.71 | 564.2 | 0.383 | 0.354 | 81.37 | 0.36 | 34.58 | 18.39 | 0.99 | 12.38 | 65.98 | 107 |
| C4 full design | 25 % | 330.3 | 367.9 | 1.32 | 290.1 | 0.266 | 0.0 | 0.0 | 0.0 | 16.56 | 0.0 | 0.89 | 6.57 | 37.0 | 112 |
| C4 full design | 50 % | 219.1 | 267.0 | 1.99 | 180.5 | 0.539 | 0.0 | 0.0 | 0.0 | 10.38 | 0.0 | 0.84 | 3.72 | 25.73 | 117 |
| C5 full design, hidden-state | 25 % | 340.5 | 391.7 | 1.28 | 281.5 | 0.284 | 0.003 | 0.26 | 1.0 | 16.18 | 0.0 | 0.84 | 6.3 | 37.75 | 111 |
| C5 full design, hidden-state | 50 % | 223.0 | 282.9 | 1.95 | 172.9 | 0.555 | 0.007 | 0.52 | 1.0 | 10.07 | 0.0 | 0.8 | 3.45 | 26.07 | 117 |
| O oracle | 25 % | 260.9 | 318.9 | 1.67 | 218.9 | 0.427 | 0.062 | 3.98 | 1.0 | 12.93 | 0.0 | 0.88 | 4.6 | 30.11 | 115 |
| O oracle | 50 % | 156.4 | 218.7 | 2.78 | 115.6 | 0.675 | 0.112 | 7.16 | 1.0 | 7.34 | 0.0 | 0.83 | 2.22 | 19.53 | 125 |

| configuration | cache | ms/token | p95 | speedup | stall ms | hit | late | prefetch/token | prefetch acc. | GB/token | wasted GB | link util. | load lat. ms | GPU J/token | GPU W |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| C0 on-demand | none | 1156.2 | 1229.8 | 1.0 | 1103.0 | 0.0 | 0.0 | 0.0 | 0.0 | 61.89 | 0.0 | 0.94 | 20.95 | 31.19 | 108 |
| C1 LRU | 25 % | 1156.1 | 1228.6 | 1.0 | 1101.7 | 0.0 | 0.0 | 0.0 | 0.0 | 61.89 | 0.0 | 0.94 | 20.98 | 31.59 | 109 |
| C1 LRU | 50 % | 1156.8 | 1229.1 | 1.0 | 1101.7 | 0.0 | 0.0 | 0.0 | 0.0 | 61.89 | 0.0 | 0.94 | 21.0 | 31.49 | 109 |
| C1b LRU + overlap | 25 % | 1175.4 | 1248.9 | 0.98 | 1072.3 | 0.0 | 0.0 | 0.0 | 0.0 | 61.89 | 0.0 | 0.93 | 21.25 | 31.95 | 109 |
| C1b LRU + overlap | 50 % | 1172.7 | 1248.6 | 0.99 | 1074.3 | 0.0 | 0.0 | 0.0 | 0.0 | 61.89 | 0.0 | 0.94 | 21.25 | 31.8 | 108 |
| C2 predict-evict | 25 % | 885.5 | 962.7 | 1.31 | 818.9 | 0.236 | 0.0 | 0.0 | 0.0 | 47.28 | 0.0 | 0.94 | 13.69 | 24.51 | 111 |
| C2 predict-evict | 50 % | 593.2 | 665.7 | 1.95 | 532.5 | 0.496 | 0.0 | 0.0 | 0.0 | 31.19 | 0.0 | 0.93 | 6.97 | 17.28 | 117 |
| C3 prefetch + LRU (naive) | 25 % | 1588.8 | 1614.7 | 0.73 | 1505.1 | 0.0 | 0.986 | 251.31 | 0.69 | 89.42 | 27.53 | 0.99 | 28.05 | 42.5 | 107 |
| C3 prefetch + LRU (naive) | 50 % | 1589.2 | 1615.0 | 0.73 | 1504.9 | 0.0 | 0.986 | 251.31 | 0.69 | 89.42 | 27.53 | 0.99 | 28.07 | 42.57 | 107 |
| C4 full design | 25 % | 884.8 | 962.1 | 1.31 | 818.3 | 0.237 | 0.0 | 0.04 | 0.91 | 47.22 | 0.0 | 0.94 | 13.71 | 24.78 | 112 |
| C4 full design | 50 % | 593.0 | 662.8 | 1.95 | 529.0 | 0.499 | 0.0 | 0.11 | 0.74 | 31.0 | 0.01 | 0.92 | 6.89 | 17.28 | 117 |
| C5 full design, hidden-state | 25 % | 899.6 | 978.4 | 1.29 | 820.9 | 0.234 | 0.0 | 0.01 | 0.5 | 47.39 | 0.0 | 0.93 | 13.61 | 25.21 | 112 |
| C5 full design, hidden-state | 50 % | 607.2 | 675.8 | 1.9 | 531.6 | 0.496 | 0.001 | 0.29 | 0.97 | 31.22 | 0.0 | 0.91 | 6.94 | 17.77 | 117 |
| O oracle | 25 % | 881.8 | 955.7 | 1.31 | 814.3 | 0.239 | 0.006 | 1.17 | 1.0 | 47.14 | 0.0 | 0.94 | 13.98 | 24.67 | 112 |
| O oracle | 50 % | 596.6 | 672.9 | 1.94 | 516.8 | 0.505 | 0.008 | 1.71 | 1.0 | 30.78 | 0.0 | 0.91 | 6.81 | 17.37 | 116 |

| configuration | cache | ms/token | p95 | speedup | stall ms | hit | late | prefetch/token | prefetch acc. | GB/token | wasted GB | link util. | load lat. ms | GPU J/token | GPU W |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| C0 on-demand | none | 1652.5 | 1670.6 | 1.0 | 1586.5 | 0.0 | 0.0 | 0.0 | 0.0 | 89.16 | 0.0 | 0.95 | 28.15 | 11.4 | 110 |
| C1 LRU | 25 % | 1653.0 | 1671.6 | 1.0 | 1586.7 | 0.0 | 0.0 | 0.0 | 0.0 | 89.16 | 0.0 | 0.95 | 28.19 | 11.39 | 110 |
| C1 LRU | 50 % | 1653.9 | 1671.9 | 1.0 | 1586.5 | 0.0 | 0.0 | 0.0 | 0.0 | 89.16 | 0.0 | 0.95 | 28.22 | 11.38 | 110 |
| C1b LRU + overlap | 25 % | 1636.1 | 1653.8 | 1.01 | 1556.0 | 0.0 | 0.0 | 0.0 | 0.0 | 89.16 | 0.0 | 0.96 | 28.26 | 11.31 | 111 |
| C1b LRU + overlap | 50 % | 1636.2 | 1654.1 | 1.01 | 1556.1 | 0.0 | 0.0 | 0.0 | 0.0 | 89.16 | 0.0 | 0.96 | 28.26 | 11.22 | 110 |
| C2 predict-evict | 25 % | 1292.7 | 1305.7 | 1.28 | 1215.4 | 0.218 | 0.0 | 0.0 | 0.0 | 69.76 | 0.0 | 0.95 | 21.7 | 9.03 | 112 |
| C2 predict-evict | 50 % | 899.9 | 904.5 | 1.84 | 827.4 | 0.466 | 0.0 | 0.0 | 0.0 | 47.62 | 0.0 | 0.93 | 14.6 | 6.6 | 117 |
| C3 prefetch + LRU (naive) | 25 % | 1603.3 | 1609.0 | 1.03 | 1514.1 | 0.0 | 1.0 | 255.94 | 0.99 | 90.19 | 1.03 | 0.99 | 25.78 | 11.11 | 111 |
| C3 prefetch + LRU (naive) | 50 % | 1604.1 | 1609.4 | 1.03 | 1514.0 | 0.0 | 1.0 | 255.94 | 0.99 | 90.19 | 1.03 | 0.99 | 25.81 | 11.08 | 111 |
| C4 full design | 25 % | 1287.2 | 1301.8 | 1.28 | 1204.7 | 0.218 | 0.01 | 2.62 | 0.99 | 69.78 | 0.01 | 0.96 | 21.52 | 9.02 | 112 |
| C4 full design | 50 % | 888.7 | 894.5 | 1.86 | 809.6 | 0.469 | 0.012 | 4.09 | 0.96 | 47.67 | 0.06 | 0.95 | 14.04 | 6.52 | 117 |
| C5 full design, hidden-state | 25 % | 1296.5 | 1309.3 | 1.27 | 1197.9 | 0.224 | 0.004 | 2.53 | 1.0 | 69.75 | 0.0 | 0.95 | 20.5 | 9.15 | 113 |
| C5 full design, hidden-state | 50 % | 900.1 | 907.5 | 1.84 | 808.3 | 0.47 | 0.005 | 2.16 | 1.0 | 47.57 | 0.0 | 0.93 | 12.28 | 6.63 | 118 |
| O oracle | 25 % | 1286.7 | 1298.3 | 1.28 | 1203.2 | 0.218 | 0.012 | 2.95 | 1.0 | 69.77 | 0.0 | 0.96 | 21.6 | 9.05 | 113 |
| O oracle | 50 % | 883.4 | 885.7 | 1.87 | 803.4 | 0.468 | 0.019 | 5.5 | 1.0 | 47.68 | 0.0 | 0.95 | 14.01 | 6.5 | 118 |

**Context sweep** (ms per token, cache 25 %):

| batch | configuration | ctx 256 | ctx 512 | ctx 1024 | ctx 1536 |
|---|--:|--:|--:|--:|--:|
| 1 | C1 LRU | 293.0 | 295.1 | 281.3 | 289.9 |
| 1 | C2 predict-evict | 333.4 | 335.1 | 330.8 | 332.1 |
| 1 | C4 full design | 333.5 | 334.1 | 330.3 | 331.5 |
| 1 | O oracle | 264.0 | 264.3 | 260.9 | 263.5 |
| 16 | C1 LRU | 1654.6 | 1656.5 | 1653.0 | 1654.8 |
| 16 | C2 predict-evict | 1292.4 | 1295.7 | 1292.7 | 1300.6 |
| 16 | C4 full design | 1291.1 | 1293.1 | 1287.2 | 1310.9 |
| 16 | O oracle | 1290.5 | 1291.9 | 1286.7 | 1294.4 |

**Lookahead and aggressiveness** (C4, batch 16, ctx 1024, cache 25 %; τ = the adaptive budget's quantile):

| lookahead cap | tau | ms/token | hit | prefetch/token | GB/token |
|---|--:|--:|--:|--:|--:|
| 1.0 | 0.6 | 1303.7 | 0.218 | 1.55 | 69.77 |
| 1.0 | 0.75 | 1292.2 | 0.218 | 1.62 | 69.77 |
| 1.0 | 0.9 | 1318.7 | 0.218 | 0.99 | 69.77 |
| 2.0 | 0.6 | 1327.3 | 0.218 | 1.02 | 69.77 |
| 2.0 | 0.75 | 1327.1 | 0.218 | 1.02 | 69.77 |
| 2.0 | 0.9 | 1327.3 | 0.218 | 1.02 | 69.77 |
| 4.0 | 0.6 | 1324.5 | 0.221 | 1.98 | 69.78 |
| 4.0 | 0.75 | 1287.2 | 0.218 | 2.62 | 69.78 |
| 4.0 | 0.9 | 1326.0 | 0.221 | 1.98 | 69.78 |

Reading the matrix:
- **PCIe 5.0 sets the speed.** The link is busy 80–99 % of the time in every configuration, and latency follows
  GB/token almost linearly (18–19 ms per GB).
- **Batch 1:** retention helps most (LRU 1.55× at 25 % cache, 2.08× at 50 %). Miss overlap adds 1 %. The design's
  need eviction keeps fewer useful experts than LRU (hit 0.26–0.27 against 0.38). The full design issues no prefetches
  at all here, because the value gate rejects every candidate when the link is about 85 % busy. That is why C2 ≈ C4.
- **Batch 4 and 16:** LRU gets **no** hits. Each step sweeps the layers in order, so LRU always evicts exactly what
  comes next (DESIGN.md's finding, now seen on hardware). Need eviction fixes that: 1.28–1.31× at 25 % and
  1.84–1.95× at 50 %, equal to the oracle.
- **C3 (naive prefetch, FIFO link)** is slower than on-demand at batches 1 and 4 (0.43–0.73×). Wrong guesses queue in
  front of demand misses: 34 GB/token wasted at batch 1.
- **C5:** the live hidden-state predictor gets a slightly higher hit rate than C4 at batch 1 (0.284 against 0.266) and
  its prefetches are 100 % accurate. But it costs 11–15 ms per token (Python and three small GPU→host copies per
  layer, `hs_predictor_ms_per_token`), so it ends 1–3 % slower than C4.
- **Context barely matters**: within ±3 %, because non-expert time is launch-bound (Phase A).
- **Aggressiveness:** the default (lookahead ≤ 4, τ = 0.75) is best. Every other setting is 0.4–3 % slower and the
  hit rate does not change. At batch 16 the 25 % cache, not prediction, is the limit.

## Phase D: energy

GPU energy comes from NVML's total-energy counter around each decode, divided by tokens. **Host energy is not
reported**: RAPL `energy_uj` is root-only on this machine, and AMD EPYC exposes no DRAM domain.

- **Power.** The GPU draws 105–125 W in every configuration: idle (91 W) plus a little copy and compute. The card never
  gets near its 400 W limit, because the link starves it.
- **Energy per token** is therefore almost exactly latency × about 110 W / batch. It ranges from 47.7 J (C0, batch 1)
  to 6.5 J (O, batch 16 at 50 % cache). Every speedup is an equal energy saving.
- **Against the engine's model.** The engine's GPU-side energy (background + dynamic + the HBM-write half of the
  transfer energy) underestimates the measurement by 43 % with its 70 W background. With the measured 90.6 W idle and
  calibrated times it is still 17.7 % low. The rest is the power above idle while copying and computing
  (99–125 W measured, against 91 W modelled). That is worth one paragraph in the report: on an offloaded GPU, idle
  power is most of the energy.

## Phase E: real GPU vs simulation (`results/compare.csv`, `results/compare_summary.csv`)

Every real run was simulated on the same workload four ways:
- trace routing with default costs (what the report's simulations do);
- the GPU's own (live) routing with default costs;
- live routing with calibrated costs;
- live routing with costs calibrated right after idle.

C5 is simulated by replaying the recorded hidden-state predictions (`hs_pred_*.npz`).

**Median relative error |sim − real| / real (worst case in brackets), all 77 runs:**

| simulation | latency | stall | hit rate | GB/token | GPU energy |
|---|--:|--:|--:|--:|--:|
| trace routing, default costs | 16.6 % (28.6 %) | 11.5 % (12.7 %) | 0.1 % (100.0 %) | 0.0 % (2.1 %) | 43.4 % (52.3 %) |
| live routing, default costs | 16.6 % (28.8 %) | 11.5 % (12.8 %) | 0.0 % (100.0 %) | 0.0 % (2.4 %) | 43.4 % (52.5 %) |
| live routing, calibrated | 3.1 % (12.8 %) | 0.3 % (5.9 %) | 0.0 % (100.0 %) | 0.0 % (2.4 %) | 17.7 % (27.5 %) |
| live routing, calibrated after idle | 2.1 % (10.2 %) | 0.6 % (4.9 %) | 0.0 % (100.0 %) | 0.0 % (2.4 %) | 16.6 % (25.4 %) |

- **Exact check: 33 of 33 match.** Hit rate and GB/token agree exactly for every C0, C1, C1b and C2 run on live
  routing. The engine's cache logic and the runtime agree. The 100 % worst hit-rate error is C3 with a hit rate of
  0.0001 on the GPU against 0.0 simulated.
- **Default costs are 17 % too fast.** The link is 56.2 GB/s, not 64. Non-expert time is 0.45–0.63 ms per layer, not
  Q2's 0.24 ms. The SXM preset is not this card. C3, which is all link time, is off by 12 %.
- **Calibrated costs: 3.1 % median latency error**, and stall within 0.3 %. The timeline model holds. The after-idle
  profile is closer still (2.1 %), because a real offloaded GPU spends most of its time just after an idle wait.
- **The worst calibrated rows are C5** (11–13 % too fast at batch 1). The engine does not charge the live predictor's
  11–15 ms per token. Its `pred_latency_us` could be set from `hs_predictor_ms_per_token`.
- **Trace vs live routing** makes no difference: default costs give the same 16.6 % median error with either. Simulating
  on the public trace is as good as simulating on the routing the GPU actually produced.

**Latency, real vs simulated (cache 25 %, ctx 1024):**

| batch | configuration | real ms | sim default | sim calibrated | sim cal. after idle | real GPU J | sim cal. GPU J |
|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | C0 on-demand | 435.3 | 359.9 | 424.2 | 431.6 | 47.68 | 40.16 |
| 1 | C1 LRU | 281.3 | 226.0 | 271.9 | 279.2 | 31.44 | 26.06 |
| 1 | C1b LRU + overlap | 278.6 | 223.1 | 268.1 | 273.7 | 31.93 | 25.72 |
| 1 | C2 predict-evict | 330.8 | 266.3 | 316.8 | 322.3 | 36.79 | 30.22 |
| 1 | C3 prefetch + LRU (naive) | 1004.3 | 884.0 | 1006.4 | 1006.4 | 105.42 | 94.06 |
| 1 | C4 full design | 330.3 | 263.5 | 313.8 | 319.3 | 37.0 | 29.95 |
| 1 | C5 full design, hidden-state | 340.5 | 253.4 | 302.1 | 307.6 | 37.75 | 28.87 |
| 1 | O oracle | 260.9 | 205.8 | 246.7 | 251.5 | 30.11 | 23.74 |
| 4 | C0 on-demand | 1156.2 | 985.7 | 1139.9 | 1154.0 | 31.19 | 26.99 |
| 4 | C1 LRU | 1156.1 | 985.7 | 1139.9 | 1154.0 | 31.59 | 26.99 |
| 4 | C1b LRU + overlap | 1175.4 | 970.6 | 1120.5 | 1125.7 | 31.95 | 26.55 |
| 4 | C2 predict-evict | 885.5 | 742.1 | 860.6 | 865.9 | 24.51 | 20.54 |
| 4 | C3 prefetch + LRU (naive) | 1588.8 | 1397.1 | 1590.5 | 1590.5 | 42.5 | 37.43 |
| 4 | C4 full design | 884.8 | 741.6 | 859.7 | 865.0 | 24.78 | 20.51 |
| 4 | C5 full design, hidden-state | 899.6 | 743.2 | 862.6 | 867.8 | 25.21 | 20.58 |
| 4 | O oracle | 881.8 | 740.3 | 856.4 | 861.8 | 24.67 | 20.44 |
| 16 | C0 on-demand | 1652.5 | 1421.0 | 1638.2 | 1656.4 | 11.4 | 9.75 |
| 16 | C1 LRU | 1653.0 | 1421.0 | 1638.2 | 1656.4 | 11.39 | 9.75 |
| 16 | C1b LRU + overlap | 1636.1 | 1397.8 | 1608.2 | 1612.8 | 11.31 | 9.58 |
| 16 | C2 predict-evict | 1292.7 | 1098.3 | 1267.7 | 1274.4 | 9.03 | 7.62 |
| 16 | C3 prefetch + LRU (naive) | 1603.3 | 1409.2 | 1604.3 | 1604.3 | 11.11 | 9.57 |
| 16 | C4 full design | 1287.2 | 1096.5 | 1260.3 | 1267.2 | 9.02 | 7.57 |
| 16 | C5 full design, hidden-state | 1296.5 | 1096.0 | 1259.7 | 1267.8 | 9.15 | 7.57 |
| 16 | O oracle | 1286.7 | 1097.5 | 1260.3 | 1270.9 | 9.05 | 7.57 |

Plots (regenerated by `compare.py`, not committed): `results/plots/latency_real_vs_sim_b{1,4,16}.png` and
`latency_error_vs_batch_ctx.png`.

## Differences from GPU.md, failures, and why

1. **The GPU is an H100 NVL**, not PCIe or SXM. The default profile uses the engine's `H100` (SXM) preset; the
   calibrated profile replaces it. No `H100-NVL` preset was added to the engine.
2. **The model fits in HBM** (see Machine). The offload comes from the cache ratio, not from memory size.
3. **Clocks were not locked** (no root), and the GPU and CPU clock down during link waits (Phase A). The runtime
   spin-waits to keep the CPU clock up; the GPU effect is measured and reported, not removed.
4. **No host energy**: RAPL is root-only and AMD has no DRAM domain.
5. **GPU 1 of this server has uncorrectable HBM errors with a row remap pending.** It needs a reset by an admin and
   was never used; `numa_run.sh` refuses it.
6. **`CUBLAS_WORKSPACE_CONFIG` is not set.** The deterministic setting made expert GEMVs 50 % slower (193 µs against
   129 µs). Runs are still bit-identical (Phase B invariance), because all GEMMs run on one stream.
7. **Stall in the overlap configurations was mis-measured in the first matrix pass.** The "wait started" event was
   recorded after the host had already waited, so it missed most of each wait. Fixed, and the 50 affected runs (C1b,
   C2, C4, O) were rerun; the old file is not committed. Latency, hit rate, traffic and energy were not affected.
8. **B″ at batch 16** (ctx 1024 and 1536) was collected through the offloaded runtime; resident experts ran out of
   memory.
9. **`experiments/e7_hidden_state.py` fails on multi-group files.** Its `workload_from()` builds warm-up rows for one
   group, but the files hold several groups of 128 tokens, as GPU.md specifies. `gpu/run_e7.py` runs it with a fixed
   `workload_from` (no edit outside `gpu/`). Upstream one-line fix:
   `warm=np.full((S // tokens_per_request, B, 1, L, k), -1, np.int16)`.
10. **e7 with `--calibration` overrides every link's bandwidth** with the measured one, so its link rows are identical.
    It was run both ways (`results/e7_calibrated/`, `results/e7_default/`).
11. **C5's live stream.** Predictions are only available up to 4 layers ahead. For layers further ahead the live stream
    holds the previous token's real probabilities (no future information). When the engine replays the recorded
    files, it sees the same values for d ≤ 4.
12. **The `offset=ctx` workload** lives in `gpu/common.make_workload_at` instead of `traces/workload.py`, so that only
    `gpu/` changes. It picks exactly the same documents as `make_workload` (checked at offsets 64–1536).
13. **Not done:** section 9 (OLMoE secondary model), the warm-start variant (cache left filled by prefill), and the
    Instruct checkpoint (unnecessary, see Phase B).
14. **Phase B′ LLMSimulator files** use the old `Sum`/`Gen` format GPU.md first described. They stay uncommitted with
    the traces. The current route is `experiments/bonus_llmsim_replay.write_trace` on `data/traces/mixtral_live_*.npz`.
