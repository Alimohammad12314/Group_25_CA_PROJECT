"""Shared setup for every experiment: workloads, cached prediction streams, configurations.

The configurations below are the single source of truth for the ablation ladder and the
"full design" used everywhere else.
"""
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from engine.timeline import SimConfig, simulate
from predictors.stream import PredictorConfig, Stream, build_stream
from traces.format import Trace
from traces.workload import make_workload

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRACES = os.path.join(ROOT, "data", "traces")
STREAMS = os.path.join(ROOT, "data", "streams")
RESULTS = os.path.join(ROOT, "results")

MODELS = ["mixtral", "grok1", "llama4_maverick", "deepseekR1"]
LINKS = ["PCIe4", "PCIe5", "C2C-450", "C2C-900"]
BATCHES = [1, 4, 16]
CACHE_RATIOS = [0.0, 0.25, 0.5, 0.75]

# Decode tokens per request and request groups per batch size (about 2048 / 1024 / 512 steps).
WORKLOAD = {
    "mixtral": dict(tokens=256, groups={1: 8, 4: 4, 16: 2}),
    "grok1": dict(tokens=256, groups={1: 8, 4: 4, 16: 2}),
    "llama4_maverick": dict(tokens=128, groups={1: 16, 4: 8, 16: 4}),
    "deepseekR1": dict(tokens=128, groups={1: 16, 4: 8, 16: 4}),
}

# --- predictor configurations ---------------------------------------------------------------
PRED = {
    "prev_token": PredictorConfig(name="prev_token", predictors=("temporal",), w_init=1,
                                  temporal_adaptive=False, chooser="equal", budget="topk"),
    "temporal": PredictorConfig(name="temporal", predictors=("temporal",), chooser="equal", budget="topk"),
    "cross": PredictorConfig(name="cross", predictors=("cross",), chooser="equal", budget="topk"),
    "popularity": PredictorConfig(name="popularity", predictors=("pop",), chooser="equal", budget="topk"),
    "equal_topk": PredictorConfig(name="equal_topk", chooser="equal", budget="topk"),
    "chooser_topk": PredictorConfig(name="chooser_topk", budget="topk"),
    "chooser_topk+2": PredictorConfig(name="chooser_topk+2", budget="topk+2"),
    "full": PredictorConfig(name="full"),
    "oracle": PredictorConfig(name="oracle", budget="oracle"),
}

# --- timeline configurations ----------------------------------------------------------------
BASE_PREFETCH = dict(prefetch=True, lookahead="fixed", lookahead_cap=1, link_mode="fifo", eviction="lru")
NO_GATE_SIM = dict(prefetch=True, lookahead="adaptive", lookahead_cap=4, link_mode="prio", link_cap=True,
                   stale_cancel=True, miss_overlap=True, eviction="need", early_bonus=0.05,
                   need_horizon=4)
FULL_SIM = dict(NO_GATE_SIM, value_gate=True)

# rung name -> (predictor config or None, timeline overrides, uses retention)
LADDER = [
    ("1 on-demand, no cache", None, dict(), False),
    ("2 + HBM cache (LRU)", None, dict(eviction="lru"), True),
    ("3 + temporal predictor", "temporal", BASE_PREFETCH, True),
    ("4 + cross-layer + popularity", "equal_topk", BASE_PREFETCH, True),
    ("5 + chooser", "chooser_topk", BASE_PREFETCH, True),
    ("6 + adaptive budget", "full", dict(BASE_PREFETCH, link_cap=True), True),
    ("7 + lookahead + preemption", "full", dict(BASE_PREFETCH, link_cap=True, lookahead="adaptive",
                                                 lookahead_cap=4, link_mode="prio", stale_cancel=True), True),
    ("8 + need eviction + miss overlap", "full", NO_GATE_SIM, True),
    ("9 + value gate = full design", "full", FULL_SIM, True),
    ("10 oracle predictor", "oracle", FULL_SIM, True),
    ("11 oracle + Belady", "oracle", dict(FULL_SIM, eviction="belady"), True),
]
# Reference rows outside the ladder, so the gain from prediction is not conflated with the rest.
EXTRA = [
    ("LRU + miss overlap (no prediction)", None, dict(eviction="lru", miss_overlap=True), True),
    ("LFU (no prediction)", None, dict(eviction="lfu"), True),
    ("LFU + miss overlap (no prediction)", None, dict(eviction="lfu", miss_overlap=True), True),
    ("prediction for eviction only (no prefetch)", "full",
     dict(eviction="need", early_bonus=0.05, need_horizon=4, miss_overlap=True), True),
    ("full design, no retention", "full", FULL_SIM, False),
    ("oracle, no retention", "oracle", FULL_SIM, False),
]


def available_models():
    return [m for m in MODELS if os.path.exists(os.path.join(TRACES, f"{m}.npz"))]


_TRACE_CACHE = {}


def get_trace(model):
    if model not in _TRACE_CACHE:
        _TRACE_CACHE[model] = Trace.load(os.path.join(TRACES, f"{model}.npz"))
    return _TRACE_CACHE[model]


_WL_CACHE = {}


def get_workload(model, batch):
    key = (model, batch)
    if key not in _WL_CACHE:
        w = WORKLOAD[model]
        wl = make_workload(get_trace(model), batch=batch, tokens=w["tokens"], groups=w["groups"][batch])
        _WL_CACHE[key] = (wl, wl.needed())
    return _WL_CACHE[key]


def stream_path(model, batch, cfg: PredictorConfig):
    w = WORKLOAD[model]
    ident = json.dumps([model, batch, w["tokens"], w["groups"][batch], cfg.key()])
    h = hashlib.sha1(ident.encode()).hexdigest()[:12]
    return os.path.join(STREAMS, f"{model}_b{batch}_{cfg.name}_{h}.npz")


def get_stream(model, batch, cfg: PredictorConfig):
    path = stream_path(model, batch, cfg)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        metrics = pd.read_json(io.StringIO(str(z["metrics"])))
        prec = pd.read_json(io.StringIO(str(z["prec"])))
        return Stream(cfg, z["P"], z["M"], metrics, prec, float(z["sec"]), json.loads(str(z["bits"])),
                      z["W"])
    wl, _ = get_workload(model, batch)
    st = build_stream(wl, cfg)
    os.makedirs(STREAMS, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, P=st.P, M=st.M, metrics=np.array(st.metrics.to_json()),
             prec=np.array(st.predictor_recall.to_json()), sec=np.array(st.seconds_per_layer_event),
             bits=np.array(json.dumps(st.state_bits)), W=st.temporal_window)
    os.replace(tmp, path)
    return st


def run_sim(model, batch, pred_name, sim_kwargs, pred_cfg=None):
    """One timeline run; pred_cfg overrides PRED[pred_name] (for sweeps over tau, bits, ...)."""
    wl, needed = get_workload(model, batch)
    stream = None
    if pred_name is not None or pred_cfg is not None:
        stream = get_stream(model, batch, pred_cfg or PRED[pred_name])
    return simulate(wl, needed, stream, SimConfig(**sim_kwargs))


def _job(args):
    tag, model, batch, pred_name, sim_kwargs, pred_cfg = args
    try:
        r = run_sim(model, batch, pred_name, sim_kwargs, pred_cfg)
    except Exception as exc:          # keep the sweep going; report the failure in the table
        r = dict(model=model, batch=batch, error=f"{type(exc).__name__}: {exc}")
    r.update(tag)
    return r


def _stream_job(args):
    model, batch, cfg = args
    st = get_stream(model, batch, cfg)
    return model, batch, cfg.name, st.seconds_per_layer_event


def prebuild_streams(jobs, procs=28):
    """Build every needed (model, batch, predictor config) stream once, in parallel."""
    todo = sorted({(m, b, c.key()): (m, b, c) for m, b, c in jobs}.values(), key=lambda x: (x[0], x[1]))
    todo = [j for j in todo if not os.path.exists(stream_path(*j))]
    if not todo:
        return
    print(f"building {len(todo)} prediction streams", flush=True)
    with ProcessPoolExecutor(min(procs, len(todo))) as ex:
        for model, batch, name, sec in ex.map(_stream_job, todo):
            print(f"  stream {model} b={batch} {name}: {sec * 1e6:.0f} us per layer event", flush=True)


def run_many(jobs, procs=28):
    """jobs: list of (tag dict, model, batch, pred_name, sim_kwargs, pred_cfg). Returns a DataFrame."""
    stream_jobs = []
    for _tag, model, batch, pred_name, _kw, pred_cfg in jobs:
        if pred_cfg is not None:
            stream_jobs.append((model, batch, pred_cfg))
        elif pred_name is not None:
            stream_jobs.append((model, batch, PRED[pred_name]))
    prebuild_streams(stream_jobs, procs)
    rows = []
    with ProcessPoolExecutor(procs) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} runs done", flush=True)
    return pd.DataFrame(rows)


def out_dir(name):
    d = os.path.join(RESULTS, name)
    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    return d


def sim_kwargs(gpu, link, ratio, overrides):
    kw = dict(gpu=gpu, link=link, cache_ratio=ratio)
    kw.update(overrides)
    return kw


__all__ = ["replace"]
