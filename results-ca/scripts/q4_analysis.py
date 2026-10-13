"""Q4 step 2: temporal / spatial correlation statistics + plots from the simulator routing logs.

Definitions (k = top-k, N = routed experts, E(s,l,t) = expert set of sequence s, layer l, token t):
  activation share     p_l(e) = #tokens routed to e at layer l / #tokens
  temporal reuse(d)    mean |E(s,l,t) ∩ E(s,l,t+d)| / k            (same sequence, same layer)
  spatial transition   P(e' at layer l+1 | e at layer l)           (same token)
  cross-layer MI       mutual information (bits) of the (e_l, e_{l+1}) slot-pair distribution
  Pearson matrix       corr(1[e_i in E(l)], 1[e_j in E(l+1)]) over all tokens
  batch overlap        |U(l,t) ∩ U(l,t+1)| / |U(l,t+1)|, U = union over the batch
Every statistic is also computed on a SHUFFLED copy (token order permuted inside each
(sequence, layer) for temporal; layer-(l+1) tokens permuted for spatial). Shuffling keeps
the frequency skew but destroys ordering, so  (observed - shuffled)  is the correlation that
a predictor could exploit beyond "pick the popular experts".
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chisquare

sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import CAT, DIV, INK2, SEQ, color_map, plt, save  # noqa: E402
from simrun import RESULTS  # noqa: E402

Q = os.path.join(RESULTS, "q4_expert_correlation")
P = os.path.join(Q, "plots")
os.makedirs(P, exist_ok=True)
os.makedirs(os.path.join(Q, "data"), exist_ok=True)
MODELS = ["mixtral", "grok1", "llama4_maverick", "deepseekR1"]
NEXP = {"mixtral": 8, "grok1": 8, "llama4_maverick": 128, "deepseekR1": 256}
SCEN = ["uniform", "zipf", "correlated"]
SCOL = color_map(SCEN)
RNG = np.random.default_rng(0)
LAGS = list(range(1, 17))


def load(model, scen):
    f = os.path.join(Q, "runs", f"{model}_{scen}", "routing_log.csv")
    raw = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.int64)
    steps, layers, seqs = raw[:, 0], raw[:, 1], raw[:, 2]
    S, L = steps.max() + 1, layers.max() + 1
    seq_ids = np.unique(seqs)
    sidx = np.searchsorted(seq_ids, seqs)
    k = raw.shape[1] - 4
    A = np.full((S, L, len(seq_ids), k), -1, dtype=np.int32)   # [step, layer, seq, slot]
    A[steps, layers, sidx] = raw[:, 4:]
    return A


def onehot(A, N):
    """[..., k] expert ids -> [..., N] 0/1 membership."""
    H = np.zeros(A.shape[:-1] + (N,), dtype=np.int8)
    np.put_along_axis(H, A, 1, axis=-1)
    return H


def temporal_reuse(H, k, lag):
    return (H[lag:] * H[:-lag]).sum(-1).mean() / k          # H: [step, layer, seq, N]


def shuffle_time(H):
    Hs = H.copy()
    S, L, B, _ = H.shape
    for l in range(L):
        for b in range(B):
            Hs[:, l, b] = H[RNG.permutation(S), l, b]
    return Hs


def mutual_info(A, N, l):
    """MI (bits) of slot-pair joint distribution between layer l and l+1 for the same token."""
    x = A[:, l].reshape(-1, A.shape[-1])
    y = A[:, l + 1].reshape(-1, A.shape[-1])
    J = np.zeros((N, N))
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            np.add.at(J, (x[:, i], y[:, j]), 1)
    J /= J.sum()
    px, py = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    nz = J > 0
    return float((J[nz] * np.log2(J[nz] / (px @ py)[nz])).sum()), J


def cross_layer_hit(A, H, N, k):
    """Training-free-style predictability bound: successor table fit on first half of steps,
    predict layer l+1 top-k from layer l experts, overlap measured on second half."""
    S, L = A.shape[0], A.shape[1]
    half = S // 2
    hits = []
    for l in range(L - 1):
        _, J = mutual_info(A[:half], N, l)
        score = H[half:, l].astype(float) @ J                     # [step, seq, N]
        pred = np.argsort(-score, axis=-1)[..., :k]
        hits.append(np.take_along_axis(H[half:, l + 1], pred, -1).sum(-1).mean() / k)
    return float(np.mean(hits))


def main():
    rows, curves, cache = [], [], {}
    for m in MODELS:
        N = NEXP[m]
        for sc in SCEN:
            A = load(m, sc)
            S, L, B, k = A.shape
            H = onehot(A, N)
            cache[(m, sc)] = (A, H)
            cnt = H.sum(axis=(0, 2))                               # [layer, N]
            share = cnt / cnt.sum(1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -(np.where(share > 0, share * np.log2(share), 0)).sum(1) / np.log2(N)
            pvals = [chisquare(c).pvalue for c in cnt]
            top20 = np.mean([np.sort(s)[::-1][:max(1, N // 5)].sum() for s in share])
            Hs = shuffle_time(H)
            for lag in LAGS:
                curves.append(dict(model=m, scenario=sc, lag=lag,
                                   reuse=temporal_reuse(H, k, lag), reuse_shuffled=temporal_reuse(Hs, k, lag)))
            mi = np.mean([mutual_info(A, N, l)[0] for l in range(L - 1)])
            Ash = A.copy()
            for l in range(L):
                Ash[:, l] = A[:, l].reshape(S * B, k)[RNG.permutation(S * B)].reshape(S, B, k)
            mi_sh = np.mean([mutual_info(Ash, N, l)[0] for l in range(L - 1)])
            U = H.max(axis=2)                                      # batch union [step, layer, N]
            inter = (U[1:] * U[:-1]).sum(-1)
            bover = (inter / U[1:].sum(-1)).mean()
            Us = shuffle_time(H).max(axis=2)
            bover_sh = ((Us[1:] * Us[:-1]).sum(-1) / Us[1:].sum(-1)).mean()
            rows.append(dict(
                model=m, scenario=sc, N=N, top_k=k, moe_layers=L, steps=S, batch=B,
                tokens_per_layer=S * B,
                norm_entropy_mean=ent.mean(), top20pct_expert_share=top20,
                frac_layers_nonuniform_p01=np.mean(np.array(pvals) < 0.01),
                random_reuse_k_over_N=k / N,
                reuse_lag1=temporal_reuse(H, k, 1), reuse_lag1_shuffled=temporal_reuse(Hs, k, 1),
                cross_layer_MI_bits=mi, cross_layer_MI_shuffled_bits=mi_sh,
                cross_layer_hit=cross_layer_hit(A, H, N, k),
                cross_layer_hit_shuffled=cross_layer_hit(Ash, onehot(Ash, N), N, k),
                unique_experts_per_layer_step=U.sum(-1).mean(),
                batch_step_overlap=bover, batch_step_overlap_shuffled=bover_sh))
            print(rows[-1])
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(Q, "data", "q4_statistics.csv"), index=False)
    cv = pd.DataFrame(curves)
    cv.to_csv(os.path.join(Q, "data", "q4_temporal_reuse_vs_lag.csv"), index=False)
    plots(st, cv, cache)


def plots(st, cv, cache):
    # F1: activation-share heatmaps, layer x expert
    for m in [x for x in ("mixtral", "deepseekR1") if (x, "uniform") in cache]:
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.6 if m == "mixtral" else 4.2))
        fig.subplots_adjust(wspace=0.45)
        for ax, sc in zip(axs, SCEN):
            H = cache[(m, sc)][1]
            cnt = H.sum(axis=(0, 2)).astype(float)
            share = cnt / cnt.sum(1, keepdims=True)
            im = ax.imshow(share, aspect="auto", cmap=SEQ, vmin=0, vmax=np.percentile(share, 99.5),
                           interpolation="nearest")
            ax.set_title(f"{m} - {sc}")
            ax.set_xlabel("expert id")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.04, label="share of activations")
        axs[0].set_ylabel("MoE layer")
        save(fig, os.path.join(P, f"q4_fig1_frequency_heatmap_{m}.png"))

    # F2: rank-ordered expert popularity
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, m in zip(axs, MODELS):
        N = NEXP[m]
        if (m, "uniform") not in cache:
            continue
        for sc in SCEN:
            H = cache[(m, sc)][1]
            cnt = H.sum(axis=(0, 2)).astype(float)
            share = np.sort(cnt / cnt.sum(1, keepdims=True), axis=1)[:, ::-1].mean(0)
            ax.plot(np.arange(1, N + 1), share, color=SCOL[sc], label=sc, marker="o" if N <= 8 else None)
        ax.axhline(1 / N, color=INK2, ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xscale("log" if N > 8 else "linear")
        ax.set_title(m)
        ax.set_xlabel("expert popularity rank (per layer)")
    axs[0].set_ylabel("share of activations")
    axs[0].legend()
    save(fig, os.path.join(P, "q4_fig2_expert_popularity_rank.png"))

    # F3: temporal reuse vs lag
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
    for ax, m in zip(axs, MODELS):
        for sc in SCEN:
            c = cv[(cv.model == m) & (cv.scenario == sc)]
            ax.plot(c.lag, c.reuse, color=SCOL[sc], label=f"{sc}")
            ax.plot(c.lag, c.reuse_shuffled, color=SCOL[sc], ls=":", lw=1.5)
        r = st[st.model == m].iloc[0]
        ax.axhline(r.random_reuse_k_over_N, color=INK2, ls="--", lw=1)
        ax.text(16, r.random_reuse_k_over_N, " k/N", va="center", fontsize=8, color=INK2)
        ax.set_title(m)
        ax.set_xlabel("token distance d")
    axs[0].set_ylabel("|E(t) ∩ E(t+d)| / k   (same seq, same layer)")
    axs[0].legend(title="solid: observed, dotted: shuffled", fontsize=8, title_fontsize=8)
    save(fig, os.path.join(P, "q4_fig3_temporal_reuse_vs_lag.png"))

    # F4: transition matrices for Mixtral (temporal: layer 0, t->t+1; spatial: layer 0->1)
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5))
    fig.subplots_adjust(wspace=0.35, hspace=0.45)
    for col, sc in enumerate(SCEN):
        A, H = cache[("mixtral", sc)]
        N = 8
        T = np.zeros((N, N))
        x, y = A[:-1, 0].reshape(-1, A.shape[-1]), A[1:, 0].reshape(-1, A.shape[-1])
        for i in range(x.shape[1]):
            for j in range(y.shape[1]):
                np.add.at(T, (x[:, i], y[:, j]), 1)
        _, J = mutual_info(A, N, 0)
        for row, (M, ttl) in enumerate(((T, "temporal: layer 0, token t -> t+1"),
                                        (J, "spatial: layer 0 -> layer 1, same token"))):
            Mn = M / M.sum(1, keepdims=True)
            ax = axs[row, col]
            im = ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=max(0.5, Mn.max()))
            ax.set_title(f"{sc}\n{ttl}", fontsize=9)
            ax.set_xlabel("next expert")
            ax.set_ylabel("current expert")
            ax.grid(False)
            for i in range(N):
                for j in range(N):
                    ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if Mn[i, j] > 0.35 else "#0b0b0b")
            fig.colorbar(im, ax=ax, fraction=0.045, label="P(next | current)")
    save(fig, os.path.join(P, "q4_fig4_transition_matrices_mixtral.png"))

    # F5: Pearson correlation layer l experts vs layer l+1 experts (Mixtral, layer 5->6)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    fig.subplots_adjust(wspace=0.5)
    for ax, sc in zip(axs, SCEN):
        H = cache[("mixtral", sc)][1].astype(float)
        X = H[:, 5].reshape(-1, 8)
        Y = H[:, 6].reshape(-1, 8)
        C = np.corrcoef(np.hstack([X, Y]).T)[:8, 8:]
        im = ax.imshow(C, cmap=DIV, vmin=-1, vmax=1)
        ax.set_title(f"{sc}: corr(expert i @ L5, expert j @ L6)", fontsize=9)
        ax.set_xlabel("expert j at layer 6")
        ax.set_ylabel("expert i at layer 5")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045, label="Pearson r")
    save(fig, os.path.join(P, "q4_fig5_cross_layer_correlation_mixtral.png"))

    # F6: summary of predictability (observed vs shuffled) per model x scenario
    metrics = [("reuse_lag1", "reuse_lag1_shuffled", "Temporal reuse (lag 1)"),
               ("cross_layer_hit", "cross_layer_hit_shuffled", "Cross-layer top-k hit rate"),
               ("batch_step_overlap", "batch_step_overlap_shuffled", "Batch-union overlap step t->t+1")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(MODELS))
    w = 0.26
    for ax, (obs, sh, ttl) in zip(axs, metrics):
        for i, sc in enumerate(SCEN):
            s = st[st.scenario == sc].set_index("model").loc[MODELS]
            ax.bar(x + (i - 1) * w, s[obs], w * 0.9, color=SCOL[sc], label=sc)
            ax.scatter(x + (i - 1) * w, s[sh], marker="_", s=180, color="#0b0b0b", zorder=3,
                       label="shuffled baseline" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=8)
        ax.set_title(ttl)
        ax.set_ylim(0, 1.05)
    axs[0].legend(fontsize=8)
    save(fig, os.path.join(P, "q4_fig6_predictability_summary.png"))


if __name__ == "__main__":
    main()
