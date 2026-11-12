"""Q4 statistics, rerun on real routing traces.

Every statistic is computed per MoE layer and next to a shuffled baseline. Shuffling keeps
expert popularity but destroys order, so (observed - shuffled) is the part a predictor can
exploit beyond plain frequency.

  lag-d reuse      mean |E(t) & E(t+d)| / k, same request and layer
  cross-layer hit  top-k predicted for layer l+d from layer l's experts with a conditional count
                   table built on the first half of the requests, scored on the second half
  entropy          popularity entropy / log2(N); 1 = perfectly flat
  top-20% share    share of activations taken by the 20% most used experts of the layer
"""
import numpy as np


def _onehot(dec, valid, N):
    """dec [R, T, L, k] -> bool [R, T, L, N]; padded tokens are all-False."""
    R, T, L, k = dec.shape
    oh = np.zeros((R, T, L, N + 1), bool)
    idx = np.where(dec < 0, N, dec)
    r, t, l, j = np.indices(dec.shape)
    oh[r, t, l, idx] = True
    oh = oh[..., :N]
    oh &= valid[:, :, None, None]
    return oh


def _shuffle_time(oh, valid, rng):
    """Permute valid tokens independently inside every (request, layer)."""
    out = oh.copy()
    R, T, L, _ = oh.shape
    for r in range(R):
        n = int(valid[r].sum())
        for l in range(L):
            out[r, :n, l] = oh[r, rng.permutation(n), l]
    return out


def temporal_reuse(oh, valid, k, lags=(1, 2, 4, 8)):
    """{lag: [L] mean overlap fraction}"""
    res = {}
    for d in lags:
        both = valid[:, d:] & valid[:, :-d]
        inter = (oh[:, d:] & oh[:, :-d]).sum(-1) / k          # [R, T-d, L]
        res[d] = (inter * both[..., None]).sum((0, 1)) / max(both.sum(), 1)
    return res


def cross_layer_hit(oh, valid, k, d=1, shuffle_dst=None):
    """[L-d] hit rate of predicting layer l+d's top-k from layer l with a train/test split."""
    R = oh.shape[0]
    tr, te = slice(0, R // 2), slice(R // 2, R)
    dst_all = oh if shuffle_dst is None else shuffle_dst
    L = oh.shape[2]
    out = np.zeros(L - d)
    for l in range(L - d):
        src_tr = oh[tr, :, l].reshape(-1, oh.shape[-1]).astype(np.float32)
        dst_tr = dst_all[tr, :, l + d].reshape(-1, oh.shape[-1]).astype(np.float32)
        table = src_tr.T @ dst_tr                                 # [N, N] co-occurrence counts
        src_te = oh[te, :, l].reshape(-1, oh.shape[-1]).astype(np.float32)
        dst_te = dst_all[te, :, l + d].reshape(-1, oh.shape[-1])
        mask = valid[te].reshape(-1)
        score = src_te[mask] @ table
        score += 1e-6 * table.sum(0)                              # break ties by popularity
        top = np.argpartition(-score, k - 1, axis=1)[:, :k]
        hit = np.take_along_axis(dst_te[mask], top, axis=1).sum(1) / k
        out[l] = hit.mean()
    return out


def popularity(oh, valid):
    counts = (oh & valid[:, :, None, None]).sum((0, 1)).astype(float)  # [L, N]
    p = counts / counts.sum(1, keepdims=True)
    N = counts.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(p > 0, p * np.log2(p), 0)).sum(1) / np.log2(N)
    top = max(1, int(round(0.2 * N)))
    share = np.sort(p, axis=1)[:, ::-1][:, :top].sum(1)
    return ent, share


def all_stats(trace, max_requests=120, max_tokens=512, seed=0):
    """Per-layer table (list of dicts) for one trace."""
    rng = np.random.default_rng(seed)
    R = min(trace.num_requests, max_requests)
    pick = rng.permutation(trace.num_requests)[:R]
    dec = trace.decode[pick, :max_tokens]
    valid = np.arange(dec.shape[1])[None, :] < np.minimum(trace.dec_len[pick], max_tokens)[:, None]
    N, k, L = trace.num_experts, trace.top_k, trace.num_layers
    oh = _onehot(dec, valid, N)
    sh = _shuffle_time(oh, valid, rng)
    reuse, reuse_sh = temporal_reuse(oh, valid, k), temporal_reuse(sh, valid, k)
    # Cross-layer baseline: permute which token at layer l+d is paired with layer l.
    perm = _shuffle_time(oh, valid, np.random.default_rng(seed + 1))
    cross = {d: cross_layer_hit(oh, valid, k, d) for d in (1, 2, 4)}
    cross_sh = {d: cross_layer_hit(oh, valid, k, d, shuffle_dst=perm) for d in (1, 2, 4)}
    ent, share = popularity(oh, valid)
    rows = []
    for l in range(L):
        row = dict(model=trace.model, layer=l, random=k / N, entropy=ent[l], top20_share=share[l])
        for d in reuse:
            row[f"reuse_lag{d}"] = reuse[d][l]
            row[f"reuse_lag{d}_shuffled"] = reuse_sh[d][l]
        for d in cross:
            row[f"cross_d{d}"] = cross[d][l] if l < L - d else np.nan
            row[f"cross_d{d}_shuffled"] = cross_sh[d][l] if l < L - d else np.nan
        rows.append(row)
    return rows
