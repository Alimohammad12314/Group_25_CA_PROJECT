"""Q4-extended step 2: how expert-choice relations emerge from router logic.

Reads runs/<model>_rho<r>_alpha<a>_bias<b>/routing_log.csv (q4x_router_runs.py) and measures,
always against a shuffled baseline (same popularity, order destroyed):
  temporal reuse(d)   share of experts the same sequence reuses d tokens later, same layer
  cross-layer hit(d)  predict the experts of layer l+d from layer l with a count table built on
                      the first half of the steps, score on the second half
  same-id next layer  share of experts with the same index at layer l+1 (expected: chance)
  batch overlap       share of experts needed at step t+1 that are already needed at step t
                      (union over the batch)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import CAT, DIV, INK2, SEQ, plt, save  # noqa: E402
from q4x_router_runs import ALPHAS, MODELS, REF_ALPHA, REF_RHO, RHOS, SWEEP_MODELS, run_name  # noqa: E402
from simrun import RESULTS  # noqa: E402

Q = os.path.join(RESULTS, "q4_extended_router_logic")
P = os.path.join(Q, "plots")
NEXP = {"mixtral": 8, "grok1": 8, "llama4_maverick": 128, "deepseekR1": 256}
LAGS = list(range(1, 17))
DISTS = [1, 2, 4, 8]
RNG = np.random.default_rng(0)


def load(name):
    raw = np.loadtxt(os.path.join(Q, "runs", name, "routing_log.csv"), delimiter=",", skiprows=1, dtype=np.int64)
    S, L = raw[:, 0].max() + 1, raw[:, 1].max() + 1
    seqs = np.unique(raw[:, 2])
    A = np.full((S, L, len(seqs), raw.shape[1] - 4), -1, dtype=np.int32)
    A[raw[:, 0], raw[:, 1], np.searchsorted(seqs, raw[:, 2])] = raw[:, 4:]
    return A


def onehot(A, N):
    H = np.zeros(A.shape[:-1] + (N,), dtype=np.int8)
    np.put_along_axis(H, A, 1, axis=-1)
    return H


def reuse(H, k, d):
    return (H[d:] * H[:-d]).sum(-1).mean() / k


def shuffle_time(H):
    Hs = H.copy()
    S, L, B, _ = H.shape
    for l in range(L):
        for b in range(B):
            Hs[:, l, b] = H[RNG.permutation(S), l, b]
    return Hs


def shuffle_tokens(A):
    """Permute tokens independently per layer: keeps popularity, breaks the same-token link."""
    S, L, B, k = A.shape
    As = A.copy()
    for l in range(L):
        As[:, l] = A[:, l].reshape(S * B, k)[RNG.permutation(S * B)].reshape(S, B, k)
    return As


def pair_table(A, N, l, m):
    J = np.zeros((N, N))
    x, y = A[:, l].reshape(-1, A.shape[-1]), A[:, m].reshape(-1, A.shape[-1])
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            np.add.at(J, (x[:, i], y[:, j]), 1)
    return J


def cross_hit(A, N, d):
    """Predict layer l+d's top-k from layer l's experts (table from first half of steps)."""
    S, L, _, k = A.shape
    half = S // 2
    H = onehot(A, N)
    hits = []
    for l in range(L - d):
        J = pair_table(A[:half], N, l, l + d)
        pred = np.argsort(-(H[half:, l].astype(float) @ J), axis=-1)[..., :k]
        hits.append(np.take_along_axis(H[half:, l + d], pred, -1).sum(-1).mean() / k)
    return float(np.mean(hits))


def stats(model, rho, alpha, bias):
    N = NEXP[model]
    A = load(run_name(model, rho, alpha, bias))
    S, L, B, k = A.shape
    H = onehot(A, N)
    Hs = shuffle_time(H)
    As = shuffle_tokens(A)
    U, Us = H.max(axis=2), Hs.max(axis=2)
    share = H.sum(axis=(0, 2)) / H.sum(axis=(0, 2)).sum(1, keepdims=True)    # [layer, N]
    row = dict(model=model, rho=rho, alpha=alpha, bias=bias, N=N, top_k=k, chance=k / N,
               reuse_lag1=reuse(H, k, 1), reuse_lag1_shuffled=reuse(Hs, k, 1),
               same_id_next_layer=(H[:, 1:] * H[:, :-1]).sum(-1).mean() / k,
               batch_overlap=((U[1:] * U[:-1]).sum(-1) / U[1:].sum(-1)).mean(),
               batch_overlap_shuffled=((Us[1:] * Us[:-1]).sum(-1) / Us[1:].sum(-1)).mean(),
               unique_experts_per_layer_step=U.sum(-1).mean(),
               max_expert_share_x_fair=share.max(1).mean() * N)
    for d in DISTS:
        row[f"cross_hit_d{d}"] = cross_hit(A, N, d)
        row[f"cross_hit_d{d}_shuffled"] = cross_hit(As, N, d) if d == 1 else np.nan
    lags = [dict(model=model, rho=rho, alpha=alpha, bias=bias, lag=g, reuse=reuse(H, k, g)) for g in LAGS]
    return row, lags, (A, H)


def main():
    os.makedirs(P, exist_ok=True)
    os.makedirs(os.path.join(Q, "data"), exist_ok=True)
    pts = sorted({(m, r, a, 0.0) for m in SWEEP_MODELS for r in RHOS for a in ALPHAS} |
                 {(m, REF_RHO, REF_ALPHA, b) for m in MODELS for b in (0.0, 0.5)})
    rows, lags, cache = [], [], {}
    for p in pts:
        r, lg, c = stats(*p)
        rows.append(r)
        lags += lg
        if p[0] == "mixtral" and p[1] == REF_RHO:
            cache[p] = c
        print(p, f"reuse {r['reuse_lag1']:.3f}/{r['reuse_lag1_shuffled']:.3f}",
              f"cross {r['cross_hit_d1']:.3f}/{r['cross_hit_d1_shuffled']:.3f}", flush=True)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(Q, "data", "q4x_statistics.csv"), index=False)
    lg = pd.DataFrame(lags)
    lg.to_csv(os.path.join(Q, "data", "q4x_reuse_vs_lag.csv"), index=False)
    plots(st, lg, cache)


def shades(n):
    return [plt.get_cmap(SEQ)(v) for v in np.linspace(0.35, 1.0, n)]


def heat(ax, M, rows, cols, title, vmax):
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r:g}" for r in rows])
    ax.set_xlabel("layer update α  (how much each layer changes the token)")
    ax.set_ylabel("token similarity ρ")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 * vmax else "#0b0b0b")
    return im


def plots(st, lg, cache):
    sw = st[st.bias == 0]
    # F1 / F2: knob maps (excess over shuffled baseline)
    for col, sh, fname, label in (("reuse_lag1", "reuse_lag1_shuffled", "q4x_fig1_temporal_vs_knobs.png",
                                   "next token reuses experts: real − shuffled"),
                                  ("cross_hit_d1", "cross_hit_d1_shuffled", "q4x_fig2_crosslayer_vs_knobs.png",
                                   "layer l → l+1 guess hit rate: real − shuffled")):
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
        fig.subplots_adjust(wspace=0.35)
        for ax, m in zip(axs, SWEEP_MODELS):
            d = sw[sw.model == m]
            M = np.array([[float((d[(d.rho == r) & (d.alpha == a)][col] - d[(d.rho == r) & (d.alpha == a)][sh]).iloc[0])
                           for a in ALPHAS] for r in RHOS])
            im = heat(ax, M, RHOS, ALPHAS, f"{m}: {label}", vmax=max(0.05, M.max()))
            fig.colorbar(im, ax=ax, fraction=0.04)
        save(fig, os.path.join(P, fname))

    # F3: temporal reuse vs token distance for each rho (alpha = ref)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    for ax, m in zip(axs, SWEEP_MODELS):
        for c, r in zip(shades(len(RHOS)), RHOS):
            x = lg[(lg.model == m) & (lg.rho == r) & (lg.alpha == REF_ALPHA) & (lg.bias == 0)]
            ax.plot(x.lag, x.reuse, color=c, label=f"ρ = {r:g}")
        ch = st[st.model == m].chance.iloc[0]
        ax.axhline(ch, color=INK2, ls="--", lw=1)
        ax.text(16, ch, " chance", va="bottom", fontsize=8, color=INK2)
        ax.set_title(f"{m}: same sequence, same layer, α = {REF_ALPHA:g}")
        ax.set_xlabel("token distance d")
    axs[0].set_ylabel("share of experts reused d tokens later")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.1),
               title="token similarity", title_fontsize=8, fontsize=8)
    save(fig, os.path.join(P, "q4x_fig3_temporal_reuse_vs_lag.png"))

    # F4: cross-layer predictability vs layer distance for each alpha (rho = ref)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    for ax, m in zip(axs, SWEEP_MODELS):
        d = sw[(sw.model == m) & (sw.rho == REF_RHO)]
        for c, a in zip(shades(len(ALPHAS))[::-1], ALPHAS):
            x = d[d.alpha == a].iloc[0]
            ax.plot(DISTS, [x[f"cross_hit_d{k}"] for k in DISTS], marker="o", color=c, label=f"α = {a:g}")
        ax.axhline(d.cross_hit_d1_shuffled.mean(), color=INK2, ls="--", lw=1)
        ax.text(8, d.cross_hit_d1_shuffled.mean(), " shuffled", va="bottom", fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTS)
        ax.set_xticklabels([str(k) for k in DISTS])
        ax.set_title(f"{m}: guess layer l+d from layer l, ρ = {REF_RHO:g}")
        ax.set_xlabel("layer distance d")
    axs[0].set_ylabel("guess hit rate")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.1),
               title="layer update", title_fontsize=8, fontsize=8)
    save(fig, os.path.join(P, "q4x_fig4_crosslayer_vs_distance.png"))

    # F5: Mixtral tables at the reference point and with a large layer update
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.4)
    A, H = cache[("mixtral", REF_RHO, REF_ALPHA, 0.0)]
    T = pair_table(np.concatenate([A[:-1], A[1:]], axis=1)[:, [0, A.shape[1]]], 8, 0, 1)
    J1 = pair_table(A, 8, 0, 1)
    A2, _ = cache[("mixtral", REF_RHO, 2.0, 0.0)]
    J2 = pair_table(A2, 8, 0, 1)
    for ax, M, ttl in ((axs[0], T, f"next token, same layer (ρ={REF_RHO:g})"),
                       (axs[1], J1, f"layer 0 → layer 1, same token (α={REF_ALPHA:g})"),
                       (axs[2], J2, "layer 0 → layer 1, same token (α=2)")):
        Mn = M / M.sum(1, keepdims=True)
        im = ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=0.6)
        ax.set_title(f"Mixtral: {ttl}", fontsize=10)
        ax.set_xlabel("next expert")
        ax.set_ylabel("current expert")
        ax.grid(False)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.35 else "#0b0b0b")
        fig.colorbar(im, ax=ax, fraction=0.045, label="P(next | current)")
    save(fig, os.path.join(P, "q4x_fig5_transition_tables_mixtral.png"))

    # F6: all four models at the reference point (with and without popular experts)
    ref = st[(st.rho == REF_RHO) & (st.alpha == REF_ALPHA)]
    q4 = pd.read_csv(os.path.join(RESULTS, "q4_expert_correlation", "data", "q4_statistics.csv"))
    uni = q4[q4.scenario == "uniform"].set_index("model")
    metrics = [("reuse_lag1", "reuse_lag1_shuffled", "Next token reuses experts", "reuse_lag1"),
               ("cross_hit_d1", "cross_hit_d1_shuffled", "Guess next layer from this layer", "cross_layer_hit"),
               ("batch_overlap", "batch_overlap_shuffled", "Batch: experts still needed next step", "batch_step_overlap")]
    labels = ["simulator default (dice)", "router, no bias", "router, bias 0.5"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
    x = np.arange(len(MODELS))
    w = 0.26
    for ax, (obs, sh, ttl, q4col) in zip(axs, metrics):
        vals = [uni.loc[MODELS, q4col].values,
                ref[ref.bias == 0].set_index("model").loc[MODELS, obs].values,
                ref[ref.bias == 0.5].set_index("model").loc[MODELS, obs].values]
        shv = [None, ref[ref.bias == 0].set_index("model").loc[MODELS, sh].values,
               ref[ref.bias == 0.5].set_index("model").loc[MODELS, sh].values]
        for i, (v, s, lab) in enumerate(zip(vals, shv, labels)):
            ax.bar(x + (i - 1) * w, v, w * 0.9, color=CAT[i], label=lab)
            if s is not None:
                ax.scatter(x + (i - 1) * w, s, marker="_", s=180, color="#0b0b0b", zorder=3,
                           label="shuffled baseline" if i == 1 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(ttl)
    axs[0].legend(fontsize=8)
    save(fig, os.path.join(P, "q4x_fig6_models_reference_point.png"))


def replot():
    """Rebuild figures from data/ without recomputing statistics (needs the two Mixtral runs for F5)."""
    st = pd.read_csv(os.path.join(Q, "data", "q4x_statistics.csv"))
    lg = pd.read_csv(os.path.join(Q, "data", "q4x_reuse_vs_lag.csv"))
    cache = {}
    for a in ALPHAS:
        A = load(run_name("mixtral", REF_RHO, a, 0.0))
        cache[("mixtral", REF_RHO, a, 0.0)] = (A, onehot(A, 8))
    plots(st, lg, cache)


if __name__ == "__main__":
    replot() if "--plots-only" in sys.argv else main()
