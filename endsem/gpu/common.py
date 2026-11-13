"""Shared definitions for the H100 validation: configurations, workloads, prediction streams, paths.

The configurations are the engine's own (experiments/common.py), so a GPU run and a simulation of the
same name use the same eviction policy, prefetcher and prediction stream.

Workload: the allenai C4 documents the engine uses, decoded teacher-forced for 128 tokens starting at
document position ctx (the context length). Groups of B documents run one after another; the expert
cache, the link and the predictor tables carry over between groups, exactly as in the engine.
"""
import hashlib
import json
import os

import numpy as np

from experiments.common import BASE_PREFETCH, FULL_SIM, PRED, get_trace
from predictors.stream import build_stream
from traces.workload import Workload

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(ROOT, "data", "gpu")
STREAMS = os.path.join(ROOT, "data", "streams")
MODEL_DIR = os.environ.get("MIXTRAL_DIR", os.path.expanduser("~/models/Mixtral-8x7B-v0.1"))
CALIBRATION = os.path.join(RESULTS, "calibration_h100.json")

TOKENS = 128
GROUPS = {1: 4, 4: 2, 16: 1}          # request groups per batch size (B x groups documents per run)
CONTEXTS = [256, 512, 1024, 1536]
EXPERT_CHUNK_MB = 16.0

NEED = dict(eviction="need", early_bonus=0.05, need_horizon=4)
# name -> (spec item, predictor stream or None, timeline overrides, uses retention)
CONFIGS = {
    "C0": ("on-demand", None, dict(), False),
    "C1": ("LRU", None, dict(eviction="lru"), True),
    "C1b": ("LRU + overlap", None, dict(eviction="lru", miss_overlap=True), True),
    "C2": ("predict-evict", "full", dict(NEED, miss_overlap=True), True),
    "C3": ("prefetch + LRU", "full", dict(BASE_PREFETCH), True),
    "C4": ("full design", "full", dict(FULL_SIM), True),
    "C5": ("full design, hidden-state predictor", "hidden", dict(FULL_SIM), True),
    "O": ("oracle", "oracle", dict(FULL_SIM), True),
}


def sim_overrides(config, cache_ratio):
    _, pred, kw, retention = CONFIGS[config]
    kw = dict(kw)
    kw["cache_ratio"] = cache_ratio if retention else 0.0
    return pred, kw


def groups_for(batch):
    return GROUPS.get(batch, max(1, 16 // batch))


def make_workload_at(trace, batch, tokens, groups, seed=0, offset=None):
    """traces.workload.make_workload with decode starting at document position `offset` (the context).

    Same document choice as make_workload (seed 0 permutation of the eligible requests); the warm-up rows
    are the 64 tokens before the decode start. Only for traces whose prompt and decode rows are one
    contiguous document (the allenai Mixtral C4 trace). offset = None or 64 gives make_workload exactly.
    """
    P = trace.warm.shape[1]
    shift = 0 if offset is None else offset - P
    if shift < 0:
        raise ValueError(f"offset {offset} is smaller than the {P} warm-up tokens")
    if shift and not np.all(trace.warm_len == P):
        raise ValueError("offset needs traces with full, contiguous prompt rows")
    eligible = np.nonzero(trace.dec_len >= tokens + shift)[0]
    need = batch * groups
    if len(eligible) < need:
        raise ValueError(f"need {need} requests with >= {tokens + shift} decode tokens, trace has {len(eligible)}")
    pick = np.random.default_rng(seed).permutation(eligible)[:need].reshape(groups, batch)
    if shift == 0:
        A, warm = trace.decode[pick, :tokens], trace.warm[pick]
    else:
        full = np.concatenate([trace.warm[pick], trace.decode[pick, :shift + tokens]], axis=2)
        A, warm = full[:, :, offset:offset + tokens], full[:, :, offset - P:offset]
    A = A.transpose(0, 2, 1, 3, 4).reshape(groups * tokens, batch, trace.num_layers, trace.top_k)
    gs = np.zeros(groups * tokens, bool)
    gs[::tokens] = True
    wl = Workload(model=trace.model, source=trace.source, num_experts=trace.num_experts, top_k=trace.top_k,
                  A=np.ascontiguousarray(A), group_start=gs, warm=np.ascontiguousarray(warm),
                  tokens_per_request=tokens)
    wl.requests = pick                       # trace request ids [G, B]
    return wl


def gpu_workload(batch, ctx, groups=None, tokens=TOKENS):
    """The trace workload: same documents as the engine (seed 0), decode starting at position ctx."""
    groups = groups or groups_for(batch)
    return make_workload_at(get_trace("mixtral"), batch=batch, tokens=tokens, groups=groups, seed=0, offset=ctx)


def live_workload(wl: Workload, A_live):
    """The same workload with the routing our runtime actually produced ([S, B, L, k])."""
    out = Workload(model=wl.model, source="live:H100:" + wl.source, num_experts=wl.num_experts,
                   top_k=wl.top_k, A=np.ascontiguousarray(A_live.astype(np.int16)),
                   group_start=wl.group_start, warm=wl.warm, tokens_per_request=wl.tokens_per_request)
    out.requests = wl.requests
    return out


def _stream_path(tag, wl, cfg):
    ident = json.dumps([tag, wl.B, wl.S, int(wl.tokens_per_request), wl.requests.tolist(),
                        hashlib.sha1(wl.A.tobytes()).hexdigest(), cfg.key()])
    h = hashlib.sha1(ident.encode()).hexdigest()[:12]
    return os.path.join(STREAMS, f"gpu_{tag}_b{wl.B}_{cfg.name}_{h}.npz")


def stream_for(wl, pred_name, tag, pred_cfg=None):
    """Prediction stream for a workload, cached on disk (keyed by the routing itself).
    pred_cfg overrides PRED[pred_name] (for the tau sweep)."""
    cfg = pred_cfg or PRED[pred_name]
    path = _stream_path(tag, wl, cfg)
    if os.path.exists(path):
        z = np.load(path)
        return _Stream(z["P"], z["M"])
    st = build_stream(wl, cfg)
    os.makedirs(STREAMS, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, P=st.P, M=st.M)
    os.replace(tmp, path)
    return _Stream(st.P, st.M)


class _Stream:
    """The two arrays the timeline reads (P[s, l, d, e], M[s, l, d])."""

    def __init__(self, P, M):
        self.P, self.M = P, M


def live_path(batch, ctx, groups):
    return os.path.join(DATA, f"live_routing_b{batch}_ctx{ctx}_g{groups}.npz")


def load_live(batch, ctx, groups):
    p = live_path(batch, ctx, groups)
    if not os.path.exists(p):
        return None
    return np.load(p)["A"]


def save_live(batch, ctx, groups, A, logits=None):
    os.makedirs(DATA, exist_ok=True)
    kw = dict(A=A)
    if logits is not None:
        kw["logits"] = logits
    np.savez_compressed(live_path(batch, ctx, groups), **kw)


def doc_tokens():
    """input_ids and predicted_token_ids of every allenai C4 document (file line i = request c4/i)."""
    path = os.path.join(DATA, "c4_tokens.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["input_ids"], z["predicted"]
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("allenai/analysis_mixtral", "c4_results.jsonl", repo_type="dataset")
    ids, pred = [], []
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["input_ids"])
            pred.append(d["predicted_token_ids"])
    ids, pred = np.asarray(ids, np.int32), np.asarray(pred, np.int32)
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(path, input_ids=ids, predicted=pred)
    return ids, pred
