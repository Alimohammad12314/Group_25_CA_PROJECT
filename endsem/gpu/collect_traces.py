"""Phase B': live routing traces from free generation, with router probabilities, + LLMSimulator files.

Greedy generation (up to 128 tokens, stop at EOS) on real prompts, every expert resident in HBM
(Mixtral BF16 fits in an H100 NVL's 94 GB, so collection runs at full speed):

  gsm8k      openai/gsm8k (main, test): "Question: ...\\nAnswer:"
  humaneval  openai/openai_humaneval (test): the function prompt

Outputs (data/, not committed: they are traces)
  data/traces/mixtral_live_<ds>.npz        engine Trace format, source "live:H100:<ds>"
  data/traces/mixtral_live_<ds>_probs.npz  router softmax over all 8 experts per token and layer
                                           (decode [R, T, L, 8], warm [R, 64, L, 8]; float16)
  data/llmsim/experts_mixtral_<ds>.csv     LLMSimulator Scheduler::initExpertList replay file:
                                           seq_i / Lin P / Sum / P*L lines layer-major / Gen /
                                           T*L lines token-major / Lout T
  results/live_traces_summary.json         sizes, lengths, reuse and cross-layer statistics

    gpurun -g 1 gpu/numa_run.sh python -m gpu.collect_traces [--n 50] [--tokens 128]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from traces.format import Trace

from .common import MODEL_DIR, RESULTS, ROOT
from .runtime import Runtime

WARM = 64
TRACES = os.path.join(ROOT, "data", "traces")
LLMSIM = os.path.join(ROOT, "data", "llmsim")


def prompts(ds, n):
    from datasets import load_dataset
    if ds == "gsm8k":
        d = load_dataset("openai/gsm8k", "main", split="test")
        return [f"Question: {r['question']}\nAnswer:" for r in d.select(range(n))]
    if ds == "humaneval":
        d = load_dataset("openai/openai_humaneval", split="test")
        return [r["prompt"] for r in d.select(range(n))]
    raise ValueError(ds)


@torch.inference_mode()
def generate(rt, tok, text, max_new):
    ids = tok(text, return_tensors="pt").input_ids.to(rt.device)
    P = ids.shape[1]
    rt.prefill_topi, rt.prefill_probs = [], []
    kv, hid = rt.prefill(ids)
    nxt = rt.model.lm_head(hid[:, -1]).argmax(-1)
    out = []
    for t in range(max_new):
        out.append(int(nxt))
        if int(nxt) == tok.eos_token_id:
            break
        o = rt.model.model(input_ids=nxt[:, None], past_key_values=kv, use_cache=True,
                           cache_position=torch.tensor([P + t], device=rt.device))
        nxt = rt.model.lm_head(o.last_hidden_state[:, -1]).argmax(-1)
    L = rt.L
    topi, probs = rt.prefill_topi, rt.prefill_probs
    rt.prefill_topi = rt.prefill_probs = None
    warm = np.stack(topi[:L], 1)                         # [P, L, 2]
    warm_p = np.stack(probs[:L], 1)                      # [P, L, 8]
    rest = topi[L:]
    T = len(rest) // L
    dec = np.stack([np.concatenate(rest[s * L:(s + 1) * L], 0) for s in range(T)]) if T else \
        np.zeros((0, L, 2), np.int8)                     # [T, L, 2]
    dec_p = np.stack([np.concatenate(probs[L + s * L:L + (s + 1) * L], 0) for s in range(T)]) if T else \
        np.zeros((0, L, 8), np.float16)
    return dict(warm=warm, warm_p=warm_p, dec=dec, dec_p=dec_p, text=tok.decode(out), P=P)


def write_llmsim(path, seqs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            P, T, L = s["warm"].shape[0], s["dec"].shape[0], s["warm"].shape[1]
            f.write(f"seq_{i}\nLin {P}\nSum\n")
            for l in range(L):                           # layer-major
                for p in range(P):
                    a, b = s["warm"][p, l]
                    f.write(f"{a} {b}\n")
            f.write("Gen\n")
            for t in range(T):                           # token-major
                for l in range(L):
                    a, b = s["dec"][t, l]
                    f.write(f"{a} {b}\n")
            f.write(f"Lout {T}\n")


def to_trace(ds, seqs, L):
    R = len(seqs)
    T = max(len(s["dec"]) for s in seqs)
    decode = np.full((R, T, L, 2), -1, np.int16)
    dprob = np.zeros((R, T, L, 8), np.float16)
    warm = np.full((R, WARM, L, 2), -1, np.int16)
    wprob = np.zeros((R, WARM, L, 8), np.float16)
    for i, s in enumerate(seqs):
        d = s["dec"]
        decode[i, :len(d)] = d
        dprob[i, :len(d)] = s["dec_p"]
        w = s["warm"][-WARM:]
        warm[i, WARM - len(w):] = w
        wprob[i, WARM - len(w):] = s["warm_p"][-WARM:]
    tr = Trace(model="mixtral", source=f"live:H100:{ds}", num_experts=8, top_k=2, decode=decode,
               dec_len=np.array([len(s["dec"]) for s in seqs]), warm=warm,
               warm_len=np.array([min(len(s["warm"]), WARM) for s in seqs]), tags=[f"{ds}/{i}" for i in range(R)])
    return tr, dprob, wprob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    texts = {ds: prompts(ds, args.n) for ds in args.datasets}
    rt = Runtime(args.model_dir, slab_slots=0, resident=True)
    summary = {}
    for ds in args.datasets:
        t0 = time.time()
        seqs = []
        for i, text in enumerate(texts[ds]):
            seqs.append(generate(rt, tok, text, args.tokens))
            if i % 10 == 0:
                print(f"  {ds} {i}: prompt {seqs[-1]['P']} tokens, generated {len(seqs[-1]['dec'])}: "
                      f"{seqs[-1]['text'][:60]!r}", flush=True)
        tr, dprob, wprob = to_trace(ds, seqs, rt.L)
        os.makedirs(TRACES, exist_ok=True)
        tr.save(os.path.join(TRACES, f"mixtral_live_{ds}.npz"))
        np.savez_compressed(os.path.join(TRACES, f"mixtral_live_{ds}_probs.npz"), decode=dprob, warm=wprob)
        write_llmsim(os.path.join(LLMSIM, f"experts_mixtral_{ds}.csv"), seqs)
        from traces.stats import all_stats
        try:
            st = all_stats(tr)
        except Exception as exc:          # statistics are a summary only; keep the trace
            st = {"error": f"{type(exc).__name__}: {exc}"}
        summary[ds] = dict(requests=len(seqs), mean_prompt=float(np.mean([s["P"] for s in seqs])),
                           mean_decode=float(tr.dec_len.mean()), seconds=time.time() - t0,
                           stats=st, sample=seqs[0]["text"][:200])
        print(ds, json.dumps(summary[ds], default=float)[:400], flush=True)
    rt.close()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "live_traces_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
