"""Calibrated synthetic routing, for models with no public trace (Grok-1).

The generator copies the real structure of a reference trace (Mixtral, same 8 experts top-2) per
layer, instead of assuming it as Q4's 'correlated' mode did:

  pi_l     expert popularity at layer l
  T_l      real temporal transition P(expert at token t+1 | expert at token t), same layer
  X_l      real cross-layer transition P(expert at layer l | expert at layer l-1), same token

Each of the k slots of token t at layer l is drawn from the mixture
    a_l * T_l[slot's expert at token t-1] + b_l * X_l[slot's expert at layer l-1] + (1-a_l-b_l) * pi_l
without repeating an expert within the token, where T_l and X_l are first sharpened to the power
beta_l and renormalised. A plain mixture dilutes the real correlation (first version: reuse 0.27 and
cross-layer hit 0.31 against Mixtral's 0.36 and 0.42) because a real token's hidden state drives both
at once; sharpening restores the concentration. (a_l, b_l, beta_l) are grid-searched per layer so that
the synthetic lag-1 reuse and cross-layer hit rate match the reference layer's. Layers of a deeper
model map to reference layers by relative depth (Grok-1 layer i of 64 -> Mixtral layer of 32).

    python -m traces.synthetic grok1
"""
import argparse
import os

import numpy as np

from .format import Trace

GRID = [(a, b) for a in np.linspace(0, 1, 11) for b in np.linspace(0, 1, 11) if a + b <= 1.0 + 1e-9]
BETAS = [1.0, 2.0, 3.0, 5.0, 8.0]


def profile(trace, max_requests=200, max_tokens=512):
    dec = trace.decode[:max_requests, :max_tokens]
    R, T, L, k = dec.shape
    N = trace.num_experts
    pi = np.zeros((L, N))
    Tm = np.full((L, N, N), 1e-3)
    Xm = np.full((L, N, N), 1e-3)
    for l in range(L):
        e = dec[:, :, l, :]
        pi[l] = np.bincount(e.ravel(), minlength=N) + 1e-3
        a, b = e[:, :-1, :], e[:, 1:, :]
        for i in range(k):
            for j in range(k):
                np.add.at(Tm[l], (a[..., i].ravel(), b[..., j].ravel()), 1)
        if l > 0:
            p = dec[:, :, l - 1, :]
            for i in range(k):
                for j in range(k):
                    np.add.at(Xm[l], (p[..., i].ravel(), e[..., j].ravel()), 1)
    pi /= pi.sum(1, keepdims=True)
    Tm /= Tm.sum(2, keepdims=True)
    Xm /= Xm.sum(2, keepdims=True)
    reuse, cross = layer_stats(dec, Xm)
    return dict(pi=pi, T=Tm, X=Xm, reuse=reuse, cross=cross, N=N, k=k)


def layer_stats(dec, Xm):
    """Per layer: lag-1 reuse, and top-k hit of predicting layer l from layer l-1 with X_l."""
    R, T, L, k = dec.shape
    reuse = np.zeros(L)
    cross = np.zeros(L)
    for l in range(L):
        e = dec[:, :, l, :]
        inter = (e[:, 1:, :, None] == e[:, :-1, None, :]).any(-1).sum(-1)
        reuse[l] = inter.mean() / k
        if l > 0:
            cross[l] = _cross_hit(dec[:, :, l - 1, :], e, Xm[l], k)
    return reuse, cross


def _sharpen(M, beta):
    S = M ** beta
    return S / S.sum(axis=-1, keepdims=True)


def _cross_hit(prev, cur, X, k):
    score = X[prev].sum(axis=2)                                   # [R, T, N]
    top = np.argsort(-score, axis=-1)[..., :k]
    return (top[..., :, None] == cur[..., None, :]).any(-1).sum(-1).mean() / k


def _sample_layer(rng, pi, T, X, prev_layer, a, b, R, Tn, k, N):
    """Generate one layer for R sequences x Tn tokens."""
    out = np.zeros((R, Tn, k), np.int16)
    for t in range(Tn):
        taken = np.zeros((R, N), bool)
        for j in range(k):
            p = np.broadcast_to((1 - a - b) * pi, (R, N)).copy()
            if t > 0 and a > 0:
                p += a * T[out[:, t - 1, j]]
            else:
                p += a * pi
            if prev_layer is not None and b > 0:
                p += b * X[prev_layer[:, t, j]]
            else:
                p += b * pi
            p[taken] = 0
            p /= p.sum(1, keepdims=True)
            u = rng.random((R, 1))
            choice = (p.cumsum(1) < u).sum(1).clip(0, N - 1)
            out[:, t, j] = choice
            taken[np.arange(R), choice] = True
    return out


def calibrate_and_generate(prof, L_out, R, Tn, seed=0, fit_R=256, fit_T=24):
    rng = np.random.default_rng(seed)
    N, k = prof["N"], prof["k"]
    L_ref = len(prof["pi"])
    src = np.round(np.arange(L_out) * (L_ref - 1) / max(L_out - 1, 1)).astype(int)
    params, out = [], np.zeros((R, Tn, L_out, k), np.int16)
    fit_prev = None
    for l in range(L_out):
        s = src[l]
        pi, T, X = prof["pi"][s], prof["T"][s], prof["X"][s]
        target_r, target_c = prof["reuse"][s], prof["cross"][s]
        best, best_err = (0.0, 0.0, 1.0), np.inf
        for beta in BETAS:
            Ts, Xs = _sharpen(T, beta), _sharpen(X, beta)
            for a, b in GRID:
                if l == 0 and b > 0:
                    continue
                g = _sample_layer(rng, pi, Ts, Xs, fit_prev, a, b, fit_R, fit_T, k, N)
                r = (g[:, 1:, :, None] == g[:, :-1, None, :]).any(-1).sum(-1).mean() / k
                err = (r - target_r) ** 2
                if l > 0:
                    err += (_cross_hit(fit_prev, g, X, k) - target_c) ** 2
                if err < best_err:
                    best, best_err = (a, b, beta), err
        a, b, beta = best
        Ts, Xs = _sharpen(T, beta), _sharpen(X, beta)
        fit_prev = _sample_layer(rng, pi, Ts, Xs, fit_prev, a, b, fit_R, fit_T, k, N)
        prev = out[:, :, l - 1, :] if l > 0 else None
        out[:, :, l, :] = _sample_layer(rng, pi, Ts, Xs, prev, a, b, R, Tn, k, N)
        params.append(dict(layer=l, ref_layer=int(s), a=a, b=b, beta=beta, fit_error=float(best_err)))
    return out, params


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["grok1"])
    ap.add_argument("--reference", default="mixtral")
    ap.add_argument("--requests", type=int, default=80)
    ap.add_argument("--tokens", type=int, default=256 + 64)
    args = ap.parse_args(argv)
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "traces")
    ref = Trace.load(os.path.join(here, f"{args.reference}.npz"))
    prof = profile(ref)
    L_out = {"grok1": 64}[args.model]
    dec, params = calibrate_and_generate(prof, L_out, args.requests, args.tokens)
    warm = 64
    tr = Trace(model=args.model, source=f"synthetic: calibrated per layer to {ref.source}",
               num_experts=ref.num_experts, top_k=ref.top_k, decode=dec[:, warm:],
               dec_len=np.full(args.requests, args.tokens - warm), warm=dec[:, :warm],
               warm_len=np.full(args.requests, warm), tags=[f"synthetic/{i}" for i in range(args.requests)])
    tr.save(os.path.join(here, f"{args.model}.npz"))
    import pandas as pd
    pd.DataFrame(params).to_csv(os.path.join(here, f"{args.model}_calibration.csv"), index=False)
    print(tr.describe())
    print(pd.DataFrame(params)[["a", "b", "beta", "fit_error"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
