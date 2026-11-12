# GPU.md — validating the expert prefetcher on a real H100 with real Mixtral and real traces

Group 25. Use this file on the H100 machine, either by hand or with a Claude Code session there
("follow endsem/GPU.md phase by phase"). It describes what to build, how to run it, and what to bring back.

**Why.** The project's results come from a trace-driven timeline (`endsem/engine/`) with costs from
LLMSimulator. The course requires simulation for the main results. This phase runs the **same
workload and the same policies on a real GPU with the real model** to show whether the simulator's
numbers hold, and where they don't. In the report it is *validation*, not a replacement.

**The setup is natural, not staged.** Mixtral-8x7B in BF16 is about 93 GB and does not fit in 80 GB.
Keeping experts in host memory and caching some of them in HBM is exactly the system we simulate.

**What you get.**
1. Measured costs (link bandwidth, expert time, attention time per batch and context, copy/compute
   contention) that calibrate the engine.
2. A check that the public trace is real Mixtral routing (our runtime reproduces it).
3. Real token latency, stall time, hit rate, traffic and GPU energy for every configuration the
   spec asks for (C1–C4 plus oracle), across batch sizes and **context lengths**.
4. A side-by-side table: real GPU vs simulation (default costs) vs simulation (calibrated costs).
5. Optional: new real traces with router probabilities, and LLMSimulator replay files (bonus).

---

## 0. Before anything: record the machine

Run these and save the output as `endsem/gpu/results/env.txt`. Several decisions below depend on it.

```bash
nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm,pci.bus_id --format=csv
nvidia-smi -q | grep -iA3 "PCIe Generation\|Link Width"   # host link generation and width
lscpu | grep -i "model name\|socket\|numa"
free -g                                                  # host RAM
ulimit -l                                                # pinned-memory limit (want "unlimited")
df -h .                                                  # need ~250 GB free
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

Decisions:

| Fact | If… | Then |
|---|---|---|
| GPU name | contains `PCIe` | H100 PCIe: HBM2e ~2.0 TB/s, 350 W. The engine's `H100` preset is the **SXM** (3.35 TB/s), so add an `H100-PCIe` preset (2.0e12 B/s, 756 TFLOPS FP16) or rely on the calibrated profile |
| GPU name | contains `SXM` / `HBM3` | the `H100` preset matches |
| PCIe generation | 5 | host link ≈ PCIe 5.0 x16 (the engine's `PCIe5`, 64 GB/s nominal, ~50–55 GB/s measured) |
| PCIe generation | 4 | use `PCIe4` |
| host RAM | ≥ 192 GB | full Mixtral offload (90 GB of experts pinned) |
| host RAM | 128–192 GB | full Mixtral works but load layer by layer (section 3); close other jobs |
| host RAM | < 128 GB | run only the first 16 of 32 layers (state this in the report) or use the secondary model (section 9) |
| `ulimit -l` | not unlimited | `sudo` set `memlock unlimited` in `/etc/security/limits.conf`, or pinning 90 GB fails |
| NUMA nodes | > 1 | run everything under `numactl --cpunodebind=N --membind=N` with N = the node the GPU hangs off (`nvidia-smi topo -m`); otherwise copy bandwidth drops |

## 1. Setup

```bash
git clone --recursive https://github.com/Alimohammad12314/ca-project.git && cd ca-project
# endsem/ must be in the repository; if it is not pushed yet, copy the folder over (without data/ and .venv/)
cd endsem
python -m venv .venv && . .venv/bin/activate
pip install "torch>=2.3" "transformers>=4.46,<5" safetensors accelerate huggingface_hub hf_transfer \
            numpy pandas scipy matplotlib pynvml pytest
export HF_HUB_ENABLE_HF_TRANSFER=1
# model: Apache-2.0, ~93 GB of safetensors
huggingface-cli download mistralai/Mixtral-8x7B-v0.1 --include "*.safetensors" "*.json" \
    --local-dir $HOME/models/Mixtral-8x7B-v0.1
# public trace (the one the engine uses) and the prediction streams
python -m traces.download mixtral
python -m pytest -q tests && python -m experiments.e0_validation_gate   # engine still sane here
```

The allenai trace was recorded on Mixtral-8x7B; section 5 checks which checkpoint matches. If agreement with
`-v0.1` is low, try `mistralai/Mixtral-8x7B-Instruct-v0.1`.

## 2. Files to create

```
endsem/gpu/
├── env.py            records the machine (section 0) as JSON
├── weights.py        safetensors -> GPU non-expert weights + pinned host experts (one contiguous buffer each)
├── runtime.py        Mixtral decode with an offloaded MoE block: cache, copy scheduler, policies
├── policies.py       eviction (lru | lfu | need | belady) and prefetch (none | naive | full | oracle)
├── microbench.py     Phase A: calibration measurements -> results/calibration_h100.json
├── check_routing.py  Phase B: our runtime's routing/predictions vs the allenai trace
├── collect_traces.py Phase B': live generation traces (+ router probabilities, LLMSimulator replay files)
├── run_matrix.py     Phase C: every configuration x batch x context x cache -> results/real_runs.csv
├── energy.py         NVML (and RAPL if readable) energy counters around a run
├── compare.py        Phase E: real vs engine (default and calibrated costs) -> results/compare.csv + plots
└── results/          everything you bring back (small CSV/JSON only)
```

## 3. The offloaded runtime (`weights.py`, `runtime.py`)

**Weights.** Read the safetensors shards directly with `safetensors.safe_open`, one tensor at a time,
so the full model is never in RAM twice.
- To the GPU: embeddings, `lm_head`, every `self_attn.*`, every `*layernorm*`, every `block_sparse_moe.gate.weight`
  (about 3.3 GB).
- To pinned host memory: for each (layer l, expert e), the tensors `w1`, `w3`, `w2` packed into **one contiguous
  pinned buffer** (`torch.empty(..., pin_memory=True)`), 352 MB each, so one expert is one DMA.
  256 buffers, about 90 GB. Free each source tensor right after copying it.

**Model.** Build `MixtralForCausalLM` on the `meta` device from the config, then load the GPU tensors with
`load_state_dict(..., assign=True)`. Replace every layer's `block_sparse_moe` with `OffloadedMoE(layer_idx, gate, runtime)`.
Attention, RoPE, RMSNorm and the KV cache (`DynamicCache`) stay Hugging Face's own code, so the attention math is the
real model's.

`OffloadedMoE.forward(hidden)` must reproduce Hugging Face's Mixtral MoE exactly:
`logits = gate(h)`; `w = softmax(logits, fp32)`; `topw, topi = w.topk(2)`; `topw /= topw.sum(-1)`;
`out = Σ topw · W2(silu(W1 h) * W3 h)` over the routed experts. It also:
1. records `topi` (and `logits`) for this layer and step: the live routing trace;
2. asks the cache for the union of experts needed by the batch;
3. executes experts as their weights arrive (strict mode: after all arrive).

**Expert cache in HBM.** Preallocate a slab of C slots × 352 MB (`C = round(ratio × 256)`) and keep a table
(layer, expert) → slot. The slab is allocated once, so caching never calls the allocator.

**Copy scheduler (one host thread, one copy stream).** It mirrors `engine/link.py`.
- Every transfer is split into **16 MB chunks** issued with `copy_(non_blocking=True)` from pinned memory on a dedicated
  `torch.cuda.Stream`.
- Two queues: **demand** chunks always go first; **prefetch** chunks run only when the demand queue is empty.
- Keep at most 2 chunks in flight, so a demand miss waits at most about two chunks behind a prefetch:
  this is the engine's "preempt at a chunk boundary".
- Record a CUDA event per finished transfer. The compute stream waits on that event before running the expert
  (`compute_stream.wait_event`).
- Time every transfer: issue time, first chunk start, finish. This gives **expert-load latency**.

**Prefetch decisions.** Two sources:
- **Replay mode (main).** Load the engine's prediction stream for the same workload
  (`experiments.common.get_stream(model, batch, PRED[...])`). The decode is teacher-forced (section 4), so routing equals
  the trace and the stream's predictions are exactly the ones the engine used. `P[s, l, d, e]` and the budget
  `M[s, l, d]` drive both prefetching and `need` eviction. Port the decision logic from `engine/timeline.py`:
  `_issue` (lookahead, link-time cap, value gate) and `_scores` (`need` eviction with horizon 4, early-layer bonus).
  Use **measured** link state (pending bytes, busy fraction) where the engine uses simulated state.
- **Oracle.** The true experts of the next layers, from the trace. This is the upper bound.

**Correctness checks (must pass before any timing).**
1. **Predicted tokens:** teacher-forced on the first 8 C4 documents (section 4), the argmax next token equals the
   trace's `predicted_token_ids` for ≥ 97% of positions. BF16 kernels differ slightly from the recording, so 100% is not
   expected.
2. **Routing:** the recorded top-2 equals the trace's `exp_ids` for ≥ 95% of (token, layer) pairs as sets.
3. **Invariance:** output logits are identical, up to BF16 noise, between cache ratio 25% and 50% and with prefetch
   on or off. Caching must never change results.

## 4. Workload: same documents as the engine, varied context

- Source: `allenai/analysis_mixtral` C4 documents (2048 tokens each, file line i = trace request `c4/i`).
- **Teacher-forced decode.** Prefill the first `ctx` tokens of each document. Then decode 128 steps one token at a time
  with the KV cache, feeding the **true next token** of the document instead of the sampled one. Compute per step is
  identical to real decoding, and routing equals the trace, so the engine and the GPU see the same experts.
- Batch B: B documents decoded together (all the same length, no padding).
- Contexts: **256, 512, 1024, 1536** (ctx + 128 ≤ 2048).
- Use the documents the engine picks: `traces.workload.make_workload(trace, batch, tokens=128, groups, seed=0)`.
  Add an `offset=ctx` argument there so decode starts at token `ctx` instead of the fixed 64 warm tokens. The prompt's
  last 64 tokens stay the predictor warm-up. Rebuild the engine streams for each `ctx` with the same function.
- After prefill, **flush the expert cache** before measuring decode, because the engine starts decode cold. Also report a
  warm-start variant (cache left as prefill filled it) as an extra.
- Timing: CUDA events around each decode step. Discard the first 8 steps (warm-up). Report mean and p95 per-token latency.

## 5. Phase A — calibration (`microbench.py` → `results/calibration_h100.json`)

| Measurement | How | Feeds |
|---|---|---|
| link bandwidth | pinned → HBM copy of 16 MB, 64 MB and 352 MB buffers, 50 repeats, single stream; also 2 concurrent streams | `link_bw` (replaces 64 GB/s nominal) |
| per-expert copy time | one 352 MB packed expert, demand path through the scheduler | `t_load_expert` |
| expert execution | real Mixtral expert, n tokens = 1, 2, 4, 8, 16, 32, 64 (CUDA events, 100 repeats) | `t_exec(n)` (replaces the calibrated roofline) |
| non-expert time per layer | real attention + norms + router at batch {1,4,16} × ctx {256,512,1024,1536}, experts excluded | `t_nonexpert_layer(batch, ctx)`: checks the engine's context sweep (e6), which takes these from LLMSimulator |
| contention | expert GEMM/GEMV loop alone vs with a concurrent 352 MB copy stream running; slowdown ratio | replaces SSD-offloading's factor HBM/(HBM − link), which predicts only ~1.6% for PCIe5 |
| idle and busy power | NVML power at idle, during copy only, during compute | energy model check |

JSON layout (one file; the engine reads it in Phase E):

```json
{"gpu": "H100-PCIe|H100-SXM", "link": "PCIe5", "link_GBps": 0.0,
 "t_exec_s": {"1": 0.0, "2": 0.0, "4": 0.0, "8": 0.0, "16": 0.0, "32": 0.0, "64": 0.0},
 "t_nonexpert_layer_s": {"b1_ctx256": 0.0, "...": 0.0},
 "contention_factor": 1.0, "power_W": {"idle": 0, "copy": 0, "compute": 0}, "env": {}}
```

## 6. Phase B — is the public trace real Mixtral routing? (`check_routing.py`)

Run the correctness checks 1–2 from section 3 on 32 documents. Report:
- token agreement with `predicted_token_ids`;
- routing agreement (per layer, as sets);
- whether agreement differs between the base and Instruct checkpoints.

This one table justifies using the trace for every simulated result.

**B′ (optional, needed for the bonus): live traces (`collect_traces.py`).** Free generation (greedy, 128 tokens) on real
prompts, for example `openai/gsm8k` (math) and `openai_humaneval` (code): 50 prompts each. Save in the engine's format
(`traces/format.py`, `Trace(model="mixtral", source="live:H100:gsm8k", ...)`) plus a parallel `.npz` with the
**router probabilities** (top-8 softmax per token and layer). These allow probability-based predictors (FineMoE-style)
that the public trace cannot support.

LLMSimulator replay of these live traces uses the patched simulator's `routing.model: trace` (any top-k); the engine side
is `experiments/bonus_llmsim_replay.py`, which writes the replay file from a workload (`write_trace`). No separate CSV
format is needed.

### B″ — hidden-state predictor (`hs_predictor.py` → `results/hs_pred_b{B}_ctx{ctx}.npz`) — **do this, high priority**

**Why.** The public traces contain expert IDs only. An ID-only lookup table fitted to the test data itself reaches only
46 % next-layer overlap on Mixtral (52 % with the previous token), and our predictor gets 43.5 %. Papers that report
80–90 % use the model's hidden state, which only this run has.

**What (training-free, uses the model's own weights).** Mixtral's residual stream changes little from layer to layer,
so layer (l+d)'s router applied to the hidden state at layer l is a good guess of layer (l+d)'s experts:

```
h_l      = residual stream entering layer l's MoE block (after attention + residual add), one vector per token
q[l+d]   = softmax( W_router[l+d] · RMSNorm_{l+d}(h_l) )      for d = 1 … 4, l + d < 32
           W_router[l+d] = model.layers[l+d].block_sparse_moe.gate.weight        (8 × 4096)
           RMSNorm_{l+d} = model.layers[l+d].post_attention_layernorm           (its own weight)
q[l, d=0]= the previous token's real router probabilities at layer l
```

- All 32 routers + norms are about 2 MB: keep them on the GPU. Extra cost: 4 small GEMVs per layer per token.
- Also try variant **B** (no renorm: `W_router[l+d] · x_l`, where `x_l` is layer l's own router input) on 8 documents and
  keep whichever has higher d = 1 overlap. Report both numbers.
- Record, per decode step s, sequence b, target layer l and distance d, **all 8 probabilities** (float16), and the
  routing the model actually chose. Save one file per (batch, ctx):

```python
np.savez("results/hs_pred_b{B}_ctx{ctx}.npz",
         q=q,             # float16 [S, B, L=32, D+1=5, N=8]   q[s, b, l, d] = prediction FOR layer l made d layers earlier
         A=A,             # int16   [S, B, L, 2]               experts the router actually picked
         doc_ids=doc_ids, tokens_per_request=128, offset=ctx)
```

- Same documents and teacher-forced decode as section 4. Minimum: batch {1, 4, 16} × ctx 1024. Then ctx {256, 1536}.
- Size: batch 16 is 128 × 16 × 32 × 5 × 8 × 2 B ≈ 2.6 MB per file.
- **Evaluate here:** `python -m experiments.e7_hidden_state --gpu H100 --calibration gpu/results/calibration_h100.json`
  prints overlap / recall / precision by distance for the hidden-state and ID-only predictors on the same tokens, and
  the speedups of every policy with each predictor. Paste these tables into `gpu/README.md`.
- **Use it live (C5 in Phase C):** in the runtime's replay mode, drive the prefetcher with these predictions computed
  on the fly instead of the engine stream (same batch union `1 − ∏(1 − q_b)` and budget `ceil(Σ P)`).

## 7. Phase C — the run matrix (`run_matrix.py` → `results/real_runs.csv`)

Configurations, the same as the engine (`experiments/common.py`):

| Name | Spec item | Eviction | Prefetch | Miss overlap |
|---|---|---|---|---|
| C0 on-demand | baseline | evict after use (no retention) | – | no |
| C1 LRU | baseline without prefetching | LRU | – | no |
| C1b LRU + overlap | stronger baseline | LRU | – | yes |
| C2 predict-evict | prediction without prefetching | need (stream `full`) | – | yes |
| C3 prefetch + LRU | prediction-driven prefetching | LRU | naive: top-M, lookahead 1, FIFO | no |
| C4 full design | complete implementation | need | full: adaptive lookahead ≤ 4, demand priority, link cap, stale cancel, value gate | yes |
| C5 full design, hidden-state predictor | complete implementation, better predictor | need (scores from B″) | full, driven by B″ predictions computed live | yes |
| O oracle | upper bound | need | oracle (true experts) | yes |

Sweep, in priority order (stop wherever time runs out):
1. **Minimum:** batch {1, 16} × ctx 1024 × cache {25%, 50%} × all 8 configurations (C0–C5, O).
2. **Context:** batch {1, 16} × ctx {256, 512, 1536} × cache 25% × {C1, C2, C4, O}.
3. **Batch 4** at ctx 1024.
4. **Lookahead / aggressiveness:** C4 with lookahead cap {1, 2, 4} and τ {0.6, 0.75, 0.9} at batch 16, ctx 1024, cache 25%.

Rough time: on-demand at batch 1 is about 22.5 GB per token over about 50 GB/s, roughly 0.45 s per token, so about
1 minute per 128 tokens. Cached configurations are faster. The minimum matrix takes under an hour, everything under 4 hours.

Columns per run (use the **same names as the engine's results**, so `compare.py` can join them):
`model, batch, ctx, link, cache_ratio, config, token_latency_ms, p95_token_latency_ms, stall_ms_per_token,
stall_fraction, hit_rate, late_rate, miss_rate, prefetch_issued_per_token, prefetch_accuracy, prefetch_timely_frac,
prefetch_wasted_frac, GB_per_token, wasted_GB_per_token, link_utilization, expert_load_latency_ms_mean,
expert_load_latency_ms_p95, gpu_energy_J_per_token, host_energy_J_per_token`.

Definitions (identical to `engine/timeline.py`):
- **hit**: resident and arrived when the router asks;
- **late**: a prefetch still in flight;
- **miss**: a demand fetch;
- **stall**: time the compute stream waits on a copy event;
- **wasted**: a prefetched expert evicted or left unused;
- **expert-load latency**: time from the moment an expert is needed until its transfer finishes (0 for hits).

## 8. Phase D — energy (`energy.py`)

- **GPU:** `pynvml.nvmlDeviceGetTotalEnergyConsumption` (mJ) before and after each run, divided by tokens. Use this
  counter, not sampled power.
- **Host:** RAPL `energy_uj` under `/sys/class/powercap/intel-rapl:*` (package and DRAM domains) if readable. On AMD or
  without permission, report GPU only and say so.
- **Compare** with the engine's energy model, which assumes 70 W GPU + HBM background, 57 W host DRAM and 13.4 pJ/bit
  transfer energy. A finding in either direction is worth a report paragraph: measured busy, idle and stall power vs
  those assumptions.

## 9. Optional secondary model: the fine-grained case

The simulation shows that prefetching only pays off when many experts are small and only some are needed per step
(DeepSeek-like). On real hardware, `allenai/OLMoE-1B-7B-0924` (64 experts, top-8, 16 layers, about 12.6 MB per expert,
14 GB total) covers that case:
- force the cache to 25/50% of experts even though it would fit;
- collect live routing with `collect_traces.py`;
- add an `olmoe` entry to `engine/costs.py` that uses only the calibrated profile (there are no Q2 numbers for it).

Expect Python overhead per small transfer to matter here. Measure it and report it.

## 10. Phase E — compare with the simulation (`compare.py`)

1. The engine already accepts a measured profile: `SimConfig(gpu="H100", link="PCIe5", context=ctx,
   calibration="gpu/results/calibration_h100.json")` (passed through to `CostModel`). It overrides `link_bw`,
   `t_exec(n)` (interpolated between measured n), `t_nonexpert_layer` (nearest measured batch, linear in ctx) and
   `contention_factor`. Keys missing from the JSON keep the LLMSimulator defaults. Without calibration, `context=`
   uses LLMSimulator's base pass at that context (`results/context_base/`).
2. For every row of `real_runs.csv`, run the engine on the same workload (same documents, batch, ctx, cache, config)
   twice: default costs (`gpu=H100` or `H100-PCIe`, `link=PCIe5`) and calibrated costs.
3. Report relative error |sim − real| / real for token latency, stall, hit rate, GB per token and energy.

What to expect, and how to read it:
- **C0–C2 hit rate and GB per token must match exactly.** Those policies don't depend on timing, so any difference is a
  bug in the engine's cache logic or in the runtime. Fix it before going on.
- **Latency with calibrated costs within ~10–15%** means the timeline model holds. Default costs will be further off,
  because nominal 64 GB/s is not the ~50 GB/s measured and H100 PCIe is not SXM. Report both.
- **Prefetch configurations (C3, C4, O)** depend on timing, so expect larger gaps. Explain them: chunk issue overhead,
  real copy-engine concurrency, contention.

Plots: real vs simulated token latency per configuration (bars, both cost profiles), and error vs batch and vs context.

## 11. Bring back

Commit `endsem/gpu/*.py` and `endsem/gpu/results/` (CSV/JSON, `hs_pred_*.npz`, plus `env.txt`), and write the
measured tables into `endsem/gpu/README.md`. **Never commit** model weights,
traces or streams. Keep plots out of git as well (they regenerate). Send the `results/` folder back so the report
tables can be built from it.

## 12. Pitfalls

- **Pinned memory:** pin per expert (256 × 352 MB), not one 90 GB block, or allocation can fail. Check `ulimit -l`.
- **Pageable vs pinned:** a copy from pageable memory runs at about half speed and is synchronous. Assert
  `buf.is_pinned()`.
- **NUMA:** wrong-node memory costs 20–40% of link bandwidth. Use `numactl` (section 0).
- **Clocks:** for repeatable timing, consider `sudo nvidia-smi -lgc <max>` (lock clocks) and record that you did.
- **Don't use `generate()`:** it adds sampling, logits processing and Python overhead per step. Write the decode loop
  yourself.
- **Synchronisation:** CUDA events, not `time.time()`. Warm up kernels first. No `torch.compile` (changes over runs).
- **BF16 numerics:** routing ties can flip between kernels. That's why the checks use ≥ 95–97%, not equality.
- **Course rule:** the report's main results stay simulation results. This phase is a validation section
  ("Does the simulator hold up on real hardware?").
