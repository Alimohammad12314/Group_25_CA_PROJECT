"""Mixtral safetensors -> non-expert weights on the GPU + experts in pinned host memory.

Shards are read one tensor at a time (safetensors.safe_open), so the model is never in RAM twice.

  GPU          embeddings, lm_head, every self_attn.*, every *layernorm*, final norm, every router
               (block_sparse_moe.gate.weight): about 3.3 GB
  pinned host  one contiguous buffer per (layer, expert) holding [w1 | w3 | w2] (352 MB), so one expert
               is one DMA source; 256 buffers, about 90 GB

An expert buffer, viewed on the GPU: w1 = [ffn, hidden], w3 = [ffn, hidden], w2 = [hidden, ffn]
(the Hugging Face nn.Linear layouts, so F.linear gives exactly MixtralBlockSparseTop2MLP's math).
"""
import json
import os
import re
import time

import torch
from safetensors import safe_open

EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight")


def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def expert_numel(cfg):
    return 3 * cfg["hidden_size"] * cfg["intermediate_size"]


def expert_views(buf, cfg):
    """Split a packed expert buffer (1-D, host or GPU) into (w1, w3, w2) views."""
    H, F = cfg["hidden_size"], cfg["intermediate_size"]
    n = H * F
    return buf[:n].view(F, H), buf[n:2 * n].view(F, H), buf[2 * n:3 * n].view(H, F)


def _shards(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx) as f:
        wmap = json.load(f)["weight_map"]
    by_file = {}
    for name, fn in wmap.items():
        by_file.setdefault(fn, []).append(name)
    return by_file


def _drop_page_cache(path):
    """Release the shard's page cache: under numactl --membind it would fill the GPU's NUMA node,
    and every later allocation (the pinned experts included) would then wait on reclaim."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def load_weights(model_dir, device="cuda", layers=None, verbose=True, gpu_slab=None):
    """Returns (gpu_state_dict, host_experts) with host_experts[(l, e)] a pinned 1-D bf16 tensor.

    layers: number of decoder layers to load (None = all); fewer layers if host RAM is short.
    gpu_slab: optional [L * N, expert numel] GPU tensor; experts are then copied straight into row
              l * N + e (everything resident in HBM, nothing pinned) and host_experts is empty.
    """
    cfg = load_config(model_dir)
    L = cfg["num_hidden_layers"] if layers is None else layers
    n_exp = expert_numel(cfg)
    H, F = cfg["hidden_size"], cfg["intermediate_size"]
    offs = {"w1": 0, "w3": H * F, "w2": 2 * H * F}
    host = {}
    gpu = {}
    t0 = time.time()
    done = 0
    for fn, names in sorted(_shards(model_dir).items()):
        path = os.path.join(model_dir, fn)
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in names:
                mo = EXPERT_RE.match(name)
                if mo:
                    l, e, w = int(mo.group(1)), int(mo.group(2)), mo.group(3)
                    if l >= L:
                        continue
                    if gpu_slab is not None:
                        t = f.get_tensor(name)
                        row = gpu_slab[l * cfg["num_local_experts"] + e]
                        row[offs[w]:offs[w] + t.numel()].copy_(t.reshape(-1))
                        del t
                        continue
                    buf = host.get((l, e))
                    if buf is None:
                        buf = torch.empty(n_exp, dtype=torch.bfloat16, pin_memory=True)
                        host[(l, e)] = buf
                    t = f.get_tensor(name)
                    buf[offs[w]:offs[w] + t.numel()].copy_(t.reshape(-1))
                    del t
                    continue
                lm = re.match(r"model\.layers\.(\d+)\.", name)
                if lm and int(lm.group(1)) >= L:
                    continue
                gpu[name] = f.get_tensor(name).to(device, non_blocking=False)
        _drop_page_cache(path)
        done += 1
        if verbose:
            print(f"  shard {done}: {fn}  experts {len(host)}  {time.time() - t0:.0f}s", flush=True)
    for buf in host.values():
        assert buf.is_pinned(), "expert buffer is not pinned (pageable copies run at half speed)"
    return cfg, gpu, host
