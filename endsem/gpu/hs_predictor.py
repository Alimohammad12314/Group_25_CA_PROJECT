"""Phase B'': training-free hidden-state expert predictor -> results/hs_pred_b{B}_ctx{ctx}.npz

Mixtral's residual stream changes little from layer to layer, so layer (l+d)'s own router applied to the hidden
state at layer l is a good guess of layer (l+d)'s experts (uses only the model's weights, nothing trained):

  h_l       residual stream entering layer l's MoE block (input of post_attention_layernorm), one vector per token
  variant A q[l+d] = softmax( W_router[l+d] . RMSNorm_{l+d}(h_l) )        RMSNorm with layer l+d's own weight
  variant B q[l+d] = softmax( W_router[l+d] . x_l )                         x_l = layer l's own router input
  d = 0     the previous token's real router probabilities at layer l (prompt's last token at a group start)

All 32 routers + norms (~2 MB) stay on the GPU; the cost is 4 small GEMMs per layer per step.

File layout (what experiments/e7_hidden_state.py reads):
  q        float16 [S, B, L, D+1, N]  q[s, b, l, d] = prediction FOR layer l made d layers earlier
                                      (entries with l - d < 0 are never read; they hold the d = 0 values)
  A        int16   [S, B, L, 2]       experts the router actually picked (live routing)
  doc_ids  [groups * B]               allenai C4 line numbers (trace request ids), group-major
  tokens_per_request = 128, offset = ctx

Two uses:
  offline   python -m gpu.hs_predictor   (all experts resident in HBM: Mixtral fits in an H100 NVL's 94 GB)
            1. variant check on 8 documents (batch 1, ctx 1024): d = 1..4 overlap of A and B -> keep the better
            2. hs_pred files for batch {1, 4, 16} x ctx 1024, then ctx {256, 1536}
  live      C5 in run_matrix: the same predictions computed during the offloaded decode feed a LiveStream that
            the engine's prefetcher and need-eviction read instead of the replayed ID-only stream
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import MODEL_DIR, RESULTS, doc_tokens, gpu_workload

D = 4
VARIANT_FILE = os.path.join(RESULTS, "hs_variant_check.json")


def chosen_variant():
    if os.path.exists(VARIANT_FILE):
        with open(VARIANT_FILE) as f:
            return json.load(f)["chosen"]
    return "A"


class LiveStream:
    """Prediction stream filled while the model runs (engine layout P[s, l, d, e], M[s, l, d]).

    Batch union P = 1 - prod_b (1 - q_b) and budget M = ceil(sum P) clipped to [k, N], as e7 does offline.
    Before step s starts, every P[s, l, :] holds the previous token's real probabilities at layer l
    (placeholder); the hidden-state predictions overwrite P[s, l + d, d] as layer l runs. Layers more
    than D ahead therefore keep the previous-token values: nothing from the future is used.
    """

    def __init__(self, S, L, N, k):
        self.P = np.zeros((S, L, D + 1, N), np.float16)
        self.M = np.full((S, L, D + 1), k, np.int16)
        self.k, self.N = k, N

    def put(self, s, l, ds, q):
        """q: [B, N] per-sequence probabilities (numpy float32) for target (s, l) at distances ds."""
        P = 1.0 - np.prod(1.0 - np.clip(q, 0.0, 1.0), axis=0)
        M = int(np.clip(np.ceil(P.sum()), self.k, self.N))
        self.P[s, l, ds] = P.astype(np.float16)
        self.M[s, l, ds] = M


class HSPredictor:
    """Forward hooks on the runtime's model: captures h_l, computes both variants at every MoE layer."""

    def __init__(self, rt, variant="A", live=None, record=False):
        self.rt, self.variant, self.live, self.record = rt, variant, live, record
        layers = rt.model.model.layers
        self.L, self.N = rt.L, rt.N
        self.G = torch.stack([layers[t].block_sparse_moe.gate.weight for t in range(self.L)])        # [L, N, H]
        self.w = torch.stack([layers[t].post_attention_layernorm.weight for t in range(self.L)])     # [L, H]
        self.eps = layers[0].post_attention_layernorm.variance_epsilon
        self.h = [None] * self.L
        self.hooks = [layers[l].post_attention_layernorm.register_forward_pre_hook(self._grab(l))
                      for l in range(self.L)]
        self.prev = [None] * self.L          # previous token's router probs per layer, GPU [B, N] fp32
        self.buf = None                      # per-step GPU buffers [2 variants, B, L, D+1, N]
        self.rec = None
        self.seconds = 0.0

    def _grab(self, l):
        def hook(_mod, args):
            self.h[l] = args[0]
        return hook

    def close(self):
        for h in self.hooks:
            h.remove()

    # --- run bookkeeping ------------------------------------------------------------------------
    def begin_run(self, S, B):
        if self.record:
            self.rec = np.zeros((2, S, B, self.L, D + 1, self.N), np.float16)

    def begin_group(self, s0):
        self.s0 = s0

    def end_step(self, s):
        if self.buf is None or self.rec is None:
            return
        b = self.buf
        for l in range(self.L):          # entries never predicted (l - d < 0) hold the d = 0 values
            for d in range(l + 1, D + 1):
                b[:, :, l, d] = b[:, :, l, 0]
        if self.rec is not None:
            self.rec[:, s] = b.to(torch.float16).cpu().numpy()

    # --- the predictor ----------------------------------------------------------------------------
    def predict(self, l, h, x):
        """Both variants for targets l+1 .. l+D: (qA, qB) GPU fp32 [d, T, N], and the valid d count."""
        n = min(D, self.L - 1 - l)
        if n <= 0:
            return None, None, 0
        G = self.G[l + 1:l + 1 + n]                                               # [n, N, H]
        hf = h.float()
        hn = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + self.eps)).to(h.dtype)   # HF RMSNorm core
        xa = self.w[l + 1:l + 1 + n, None, :] * hn[None]                          # [n, T, H] each layer's weight
        qa = torch.softmax(torch.bmm(xa, G.transpose(1, 2)).float(), -1)          # variant A
        qb = torch.softmax(torch.einsum("th,dnh->dtn", x, G).float(), -1)         # variant B
        return qa, qb, n

    def on_router(self, l, x, probs):
        """Called by OffloadedMoE with layer l's router input x [T, H] and its softmax probs [T, N]."""
        rt = self.rt
        t0 = time.perf_counter()
        if rt.mode == "prefill":
            T = x.shape[0]
            B = self.B_hint
            last = probs.view(B, T // B, -1)[:, -1].float()                       # prompt's last token
            self.prev[l] = last
            if self.live is not None:
                self.live.put(self.s0, l, slice(0, D + 1), last.cpu().numpy())
            return
        if rt.mode != "decode":
            return
        s = rt.step
        B = x.shape[0]
        if self.buf is None or self.buf.shape[1] != B:
            self.buf = torch.zeros(2, B, self.L, D + 1, self.N, device=x.device)
        h = self.h[l].reshape(-1, x.shape[-1])
        qa, qb, n = self.predict(l, h, x)
        self.buf[:, :, l, 0] = self.prev[l]
        if n:
            for d in range(1, n + 1):
                self.buf[0, :, l + d, d] = qa[d - 1]
                self.buf[1, :, l + d, d] = qb[d - 1]
        cur = probs.float()
        self.prev[l] = cur
        if self.live is not None:
            S = self.live.P.shape[0]
            q = (qa if self.variant == "A" else qb)
            if n:
                qh = q.cpu().numpy()
                for d in range(1, n + 1):
                    self.live.put(s, l + d, d, qh[d - 1])
            if s + 1 < S:                     # placeholder for the next token: this token's real probs
                self.live.put(s + 1, l, slice(0, D + 1), cur.cpu().numpy())
        self.seconds += time.perf_counter() - t0


# --------------------------------------------------------------------------------------------------
# offline collection (resident experts, teacher-forced decode of the section-4 workload)
# --------------------------------------------------------------------------------------------------
@torch.inference_mode()
def collect(rt, hs, wl, docs, ctx):
    S, B, T = wl.S, wl.B, wl.tokens_per_request
    A = np.zeros((S, B, rt.L, rt.k), np.int16)
    hs.begin_run(S, B)
    rt.record = A
    for g, g0 in enumerate(range(0, S, T)):
        ids = torch.as_tensor(docs[wl.requests[g]], dtype=torch.long, device=rt.device)
        hs.begin_group(g0)
        hs.B_hint = B
        rt.mode = "prefill"
        kv, _ = rt.prefill(ids[:, :ctx])
        rt.mode = "decode"
        for t in range(T):
            s = g0 + t
            rt.step = s
            rt.model.model(input_ids=ids[:, ctx + t:ctx + t + 1], past_key_values=kv, use_cache=True,
                           cache_position=torch.tensor([ctx + t], device=rt.device))
            hs.end_step(s)
        del kv
    rt.record = None
    rt.mode = "prefill"
    return hs.rec, A


@torch.inference_mode()
def collect_offloaded(rt, hs, wl, docs, ctx, cache_ratio=0.5):
    """Same recording through the offloaded runtime (LRU cache, C1): for batches whose KV cache does not fit
    next to all 256 resident experts. Routing is identical to the resident and matrix runs."""
    from engine.timeline import SimConfig
    from .common import sim_overrides
    _, kw = sim_overrides("C1", cache_ratio)
    hs.begin_run(wl.S, wl.B)
    _, live, _ = rt.run(wl, wl.needed(), None, SimConfig(gpu="H100", link="PCIe5", context=ctx, **kw), docs, ctx,
                        progress=False)
    return hs.rec, live


def overlap_by_distance(q, A):
    """e7's overlap (share of the batch's experts in the top-|U| union scores), per distance."""
    from experiments.e7_hidden_state import score, stream_from_q, workload_from
    wl = workload_from(A, 128)
    t = score(stream_from_q(q, A.shape[-1]), wl)
    return {int(r.distance): dict(overlap=float(r.overlap), recall=float(r.recall), precision=float(r.precision))
            for r in t.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--runs", nargs="*", default=["1x1024", "4x1024", "16x1024", "1x256", "4x256", "16x256",
                                                  "1x1536", "4x1536", "16x1536"])
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--offloaded", action="store_true",
                    help="experts in host memory, 50 %% HBM cache (batch 16 does not fit next to resident experts)")
    args = ap.parse_args()
    from .runtime import Runtime
    ids, _ = doc_tokens()
    if args.offloaded:
        rt = Runtime(args.model_dir, slab_slots=128)
    else:
        rt = Runtime(args.model_dir, slab_slots=0, resident=True)
    hs = HSPredictor(rt, record=True)
    rt.hs = hs
    os.makedirs(RESULTS, exist_ok=True)
    if not args.skip_check:
        # variant check: 8 documents, batch 1, ctx 1024 (the same seed-0 documents the runs use)
        wl = gpu_workload(1, 1024, groups=8)
        rec, A = collect(rt, hs, wl, ids, 1024)
        res = {v: overlap_by_distance(rec[i], A) for i, v in enumerate("AB")}
        chosen = "A" if res["A"][1]["overlap"] >= res["B"][1]["overlap"] else "B"
        out = dict(documents=wl.requests.reshape(-1).tolist(), batch=1, ctx=1024, chosen=chosen,
                   variant_A="softmax(W_router[l+d] . RMSNorm_{l+d}(h_l))",
                   variant_B="softmax(W_router[l+d] . x_l), x_l = layer l's own router input", **res)
        with open(VARIANT_FILE, "w") as f:
            json.dump(out, f, indent=1)
        print("variant check (d: overlap A / B):",
              {d: (round(res["A"][d]["overlap"], 3), round(res["B"][d]["overlap"], 3)) for d in range(D + 1)},
              "-> chosen", chosen, flush=True)
    var = chosen_variant()
    both = {}
    for spec in args.runs:
        b, ctx = (int(x) for x in spec.split("x"))
        t0 = time.time()
        wl = gpu_workload(b, ctx)
        try:
            rec, A = (collect_offloaded if args.offloaded else collect)(rt, hs, wl, ids, ctx)
        except torch.OutOfMemoryError as exc:            # all experts resident leaves ~5 GB for KV/activations
            print(f"{spec}: out of GPU memory, skipped ({str(exc)[:120]})", flush=True)
            both[spec] = dict(error="out of GPU memory with all experts resident")
            rt.mode = "prefill"
            torch.cuda.empty_cache()
            continue
        q = rec[0 if var == "A" else 1]
        path = os.path.join(RESULTS, f"hs_pred_b{b}_ctx{ctx}.npz")
        np.savez(path, q=q, A=A, doc_ids=wl.requests.reshape(-1), tokens_per_request=wl.tokens_per_request,
                 offset=ctx)
        both[spec] = {v: overlap_by_distance(rec[i], A) for i, v in enumerate("AB")}
        live = os.path.join(os.path.dirname(RESULTS), "..", "data", "gpu", f"live_routing_b{b}_ctx{ctx}_g{wl.S // 128}.npz")
        same = None
        if os.path.exists(live):
            same = float((np.load(live)["A"] == A).all(-1).mean())
        both[spec]["routing_vs_offloaded_runs"] = same
        both[spec]["routing_vs_trace"] = float((np.sort(A, -1) == np.sort(wl.A, -1)).all(-1).mean())
        print(f"{path}: {time.time() - t0:.0f}s, d1 overlap A {both[spec]['A'][1]['overlap']:.3f} "
              f"B {both[spec]['B'][1]['overlap']:.3f}, routing = offloaded runs {same}", flush=True)
    path = os.path.join(RESULTS, "hs_variants_all_runs.json")
    prev = {}
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f).get("runs", {})
    for spec, r in both.items():
        if "error" not in r or spec not in prev:
            if args.offloaded and "error" not in r:
                r["collected_with"] = "offloaded runtime, C1 at 50 % cache (resident run out of GPU memory)"
            prev[spec] = r
    with open(path, "w") as f:
        json.dump(dict(chosen=var, runs=prev), f, indent=1)
    hs.close()
    rt.close()


if __name__ == "__main__":
    main()
