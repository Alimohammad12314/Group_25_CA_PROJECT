"""Download real routing traces from Hugging Face and convert them to the common format.

    python -m traces.download mixtral            # open, no login
    python -m traces.download deepseekR1         # gated: accept the dataset terms, then `hf auth login`
    python -m traces.download llama4_maverick    # gated, same dataset

Sources
  mixtral          allenai/analysis_mixtral, c4_results.jsonl: 405 C4 documents x 2048 tokens,
                   top-2 of 8 experts at all 32 layers (teacher-forced text, the Mixtral paper's data).
  deepseekR1       core12345/MoE_expert_selection_trace (arXiv 2510.05497): per query, prefill routing
  llama4_maverick  plus up to 128 decode tokens, on aime_2024 (math), livecodebench (code), mmlu.

Mixtral documents have no prompt/answer split, so the first WARM_TOKENS tokens of each document
are treated as the prompt and the rest as decode tokens.
"""
import argparse
import json
import os
import sys

import numpy as np

from .format import Trace

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "traces")
WARM_TOKENS = 64

GATED_REPO = "core12345/MoE_expert_selection_trace"
GATED_MODELS = {
    "deepseekR1": dict(hf="cognitivecomputations/DeepSeek-R1-AWQ", N=256, k=8, sim_layers=57),
    "llama4_maverick": dict(hf="meta-llama/Llama-4-Maverick-17B-128E-Instruct", N=128, k=1, sim_layers=24),
}
# Benchmarks to pull per model and at most how many queries from each (keeps downloads small).
GATED_BENCHMARKS = [("HuggingFaceH4/aime_2024", 30), ("livecodebench", 40), ("mmlu", 60)]


def out_path(model):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, f"{model}.npz")


def convert_mixtral():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("allenai/analysis_mixtral", "c4_results.jsonl", repo_type="dataset")
    docs = []
    with open(path) as f:
        for line in f:
            exp = np.asarray(json.loads(line)["exp_ids"], dtype=np.int16)  # [T, k, L]
            docs.append(exp.transpose(0, 2, 1))                             # -> [T, L, k]
    arr = np.stack(docs)                                                    # [R, 2048, 32, 2]
    R, T = arr.shape[:2]
    trace = Trace(model="mixtral", source="allenai/analysis_mixtral (C4)", num_experts=8, top_k=2,
                  decode=arr[:, WARM_TOKENS:], dec_len=np.full(R, T - WARM_TOKENS),
                  warm=arr[:, :WARM_TOKENS], warm_len=np.full(R, WARM_TOKENS),
                  tags=[f"c4/{i}" for i in range(R)])
    return trace


def _list_query_files(api, root, limit):
    """All .json files under root (benchmark folders nest subjects one level deeper)."""
    files = []
    for entry in api.list_repo_tree(GATED_REPO, path_in_repo=root, repo_type="dataset", recursive=True):
        if entry.path.endswith(".json"):
            files.append(entry.path)
    files.sort()
    if len(files) > limit:                       # spread the picks across subjects
        idx = np.linspace(0, len(files) - 1, limit).round().astype(int)
        files = [files[i] for i in idx]
    return files


def _ids(value, k):
    """One layer's selection -> int16 [k], or None if it is missing or malformed."""
    try:
        a = np.asarray(value, dtype=np.int16).reshape(-1)
    except (TypeError, ValueError):
        return None
    return a[:k] if a.size >= k else None


def _parse_query(path, sim_layers, k):
    """One query JSON -> (warm [P, L, k], decode [T, L, k]) restricted to MoE layers.

    A few files are malformed (None inside a layer's selection, a missing layer key). A decode
    sequence is cut at its first malformed token; malformed prompt tokens are dropped.
    """
    with open(path) as f:
        tokens = json.load(f)
    layer_keys = [key for key in tokens[0] if tokens[0][key] is not None]
    layer_keys.sort(key=lambda s: int(s))
    layer_keys = layer_keys[:sim_layers]         # match LLMSimulator's MoE layer count
    L = len(layer_keys)
    prefill = []
    rows0 = [tokens[0].get(key) for key in layer_keys]
    n_prompt = min((len(r) for r in rows0 if isinstance(r, list)), default=0)
    for p in range(n_prompt):
        sel = [_ids(r[p], k) if isinstance(r, list) and p < len(r) else None for r in rows0]
        if all(s is not None for s in sel):
            prefill.append(np.stack(sel))
    decode = []
    for tok in tokens[1:]:
        sel = [_ids(tok.get(key), k) for key in layer_keys]
        if any(s is None for s in sel):
            break
        decode.append(np.stack(sel))
    empty = np.zeros((0, L, k), np.int16)
    return (np.stack(prefill) if prefill else empty), (np.stack(decode) if decode else empty)


def convert_gated(model):
    from huggingface_hub import HfApi, hf_hub_download
    spec = GATED_MODELS[model]
    api = HfApi()
    warms, decs, tags = [], [], []
    for bench, limit in GATED_BENCHMARKS:
        files = _list_query_files(api, f"{spec['hf']}/{bench}", limit)
        print(f"  {bench}: {len(files)} queries", flush=True)
        for rel in files:
            local = hf_hub_download(GATED_REPO, rel, repo_type="dataset")
            w, d = _parse_query(local, spec["sim_layers"], spec["k"])
            if len(d) < 8:
                continue
            warms.append(w[-WARM_TOKENS:])
            decs.append(d)
            tags.append(rel.split(spec["hf"] + "/")[1])
    L, k = decs[0].shape[1], spec["k"]
    R, T, P = len(decs), max(len(d) for d in decs), WARM_TOKENS
    decode = np.full((R, T, L, k), -1, np.int16)
    warm = np.full((R, P, L, k), -1, np.int16)
    for i, (w, d) in enumerate(zip(warms, decs)):
        decode[i, :len(d)] = d
        warm[i, P - len(w):] = w
    return Trace(model=model, source=f"{GATED_REPO} ({spec['hf']})", num_experts=spec["N"], top_k=k,
                 decode=decode, dec_len=np.array([len(d) for d in decs]), warm=warm,
                 warm_len=np.array([len(w) for w in warms]), tags=tags)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", choices=["mixtral", *GATED_MODELS])
    args = ap.parse_args(argv)
    for model in args.models:
        print(f"[{model}] downloading and converting", flush=True)
        try:
            trace = convert_mixtral() if model == "mixtral" else convert_gated(model)
        except Exception as exc:                       # gated repo without a login lands here
            print(f"[{model}] FAILED: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
            continue
        trace.save(out_path(model))
        print(f"[{model}] saved {out_path(model)}\n  {trace.describe()}")


if __name__ == "__main__":
    main()
