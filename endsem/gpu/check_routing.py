"""Phase B: is the public trace real Mixtral routing? (and correctness checks 1-3 of GPU.md section 3)

  1. predicted tokens   teacher-forced over whole documents, argmax next token vs the trace's
                        predicted_token_ids (>= 97 % expected; BF16 kernels differ from the recording)
  2. routing            our top-2 vs the trace's exp_ids as sets, per (token, layer) (>= 95 %)
  3. invariance         router logits at every layer and the predicted tokens are identical between
                        cache 25 % and 50 % and with prefetch on or off (caching never changes results)

Checks 1-2 run on the prefill path (whole 2048-token documents, experts streamed through staging
slots) and again on the decode path (KV cache, one token at a time, the runtime's expert cache).

    gpurun -g 1 gpu/numa_run.sh python -m gpu.check_routing [--docs 32] [--model-dir ...] [--tag base]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from engine.timeline import SimConfig
from experiments.common import get_trace

from .common import CALIBRATION, MODEL_DIR, RESULTS, doc_tokens, gpu_workload, sim_overrides, stream_for
from .runtime import Runtime


def set_agree(a, b):
    """a, b: [..., 2] expert pairs -> bool array, equal as sets."""
    return (np.sort(a, -1) == np.sort(b, -1)).all(-1)


def prefill_check(rt, ids, pred, full, docs, batch):
    tok_hits, tok_n = 0, 0
    route = np.zeros((len(docs), full.shape[1], rt.L), bool)
    first8 = [0, 0]
    for i in range(0, len(docs), batch):
        sel = docs[i:i + batch]
        x = torch.as_tensor(ids[sel], dtype=torch.long, device=rt.device)
        rt.prefill_topi = []
        with torch.inference_mode():
            _, hid = rt.prefill(x)
            am = []
            for c in range(0, hid.shape[1], 512):
                am.append(rt.model.lm_head(hid[:, c:c + 512]).argmax(-1))
            am = torch.cat(am, 1).cpu().numpy()
        B, T = x.shape
        topi = np.stack(rt.prefill_topi).reshape(rt.L, B, T, 2).transpose(1, 2, 0, 3)   # [B, T, L, 2]
        rt.prefill_topi = None
        ok = am == pred[sel]
        tok_hits += ok.sum()
        tok_n += ok.size
        for b, d in enumerate(sel):
            if d < 8:
                first8[0] += ok[b].sum()
                first8[1] += ok[b].size
        route[i:i + len(sel)] = set_agree(topi, full[sel])
        print(f"  docs {i}-{i + len(sel) - 1}: tokens {ok.mean():.4f}, routing {route[i:i + len(sel)].mean():.4f}",
              flush=True)
    return dict(token_agreement=tok_hits / tok_n, token_agreement_first8=first8[0] / max(first8[1], 1),
                routing_agreement=float(route.mean()),
                routing_agreement_per_layer=route.mean(axis=(0, 1)).round(4).tolist(),
                routing_agreement_first_64=float(route[:, :64].mean()))


def decode_runs(rt, ids, pred, full, ctx, tokens):
    """Short decodes through the runtime: decode-path agreement + invariance across cache/prefetch."""
    wl = gpu_workload(1, ctx, groups=2, tokens=tokens)
    needed = wl.needed()
    out = {}
    runs = {}
    for name, ratio in (("C1", 0.25), ("C1", 0.5), ("C4", 0.25), ("C1b", 0.25)):
        pred_name, kw = sim_overrides(name, ratio)
        stream = stream_for(wl, pred_name, f"ctx{ctx}") if pred_name else None
        cal = CALIBRATION if os.path.exists(CALIBRATION) else None
        cfg = SimConfig(gpu="H100", link="PCIe5", calibration=cal, context=ctx, **kw)
        res, live, p = rt.run(wl, needed, stream, cfg, ids, ctx, record_logits=True, progress=False)
        runs[(name, ratio)] = (live, p)
        print(f"  decode {name} cache {ratio}: {res['token_latency_ms']:.1f} ms/token, hit {res['hit_rate']:.3f}",
              flush=True)
    live, p = runs[("C1", 0.25)]
    S, B = live.shape[:2]
    reqs = wl.requests.reshape(-1)
    T = wl.tokens_per_request
    tr = np.stack([full[reqs[s // T * B + b], ctx + s % T] for s in range(S) for b in range(B)]).reshape(S, B, rt.L, 2)
    tp = np.stack([pred[reqs[s // T * B + b], ctx + s % T] for s in range(S) for b in range(B)]).reshape(S, B)
    out["decode_routing_agreement"] = float(set_agree(live, tr).mean())
    out["decode_token_agreement"] = float((p["decode"] == tp).mean())
    ref_live, ref_p = runs[("C1", 0.25)]
    inv = {}
    for key, (lv, pp) in runs.items():
        if key == ("C1", 0.25):
            continue
        inv[f"{key[0]}@{key[1]}"] = dict(
            routing_identical=bool(np.array_equal(lv, ref_live)),
            tokens_identical=bool(np.array_equal(pp["decode"], ref_p["decode"])),
            router_logits_max_abs_diff=float(np.abs(pp["logits"] - ref_p["logits"]).max()))
    out["invariance_vs_C1@0.25"] = inv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--docs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--skip-decode", action="store_true")
    args = ap.parse_args()
    ids, pred = doc_tokens()
    tr = get_trace("mixtral")
    full = np.concatenate([tr.warm, tr.decode], axis=1)          # [R, 2048, L, 2]
    rt = Runtime(args.model_dir, slab_slots=128)
    t0 = time.time()
    res = dict(model_dir=os.path.basename(args.model_dir.rstrip("/")), docs=args.docs)
    res.update(prefill_check(rt, ids, pred, full, list(range(args.docs)), args.batch))
    res["prefill_seconds"] = time.time() - t0
    print(json.dumps({k: v for k, v in res.items() if k != "routing_agreement_per_layer"}, indent=1), flush=True)
    if not args.skip_decode:
        res.update(decode_runs(rt, ids, pred, full, args.ctx, args.decode_tokens))
    rt.close()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"check_routing_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
