"""Phase A: calibration measurements on the real GPU -> results/calibration_h100.json

  link bandwidth        pinned -> HBM copies of 16, 64 and 352 MB (50 repeats), one stream and two
  scheduler             one 352 MB expert through RealLink (demand path): latency and throughput, by
                        chunk size and in-flight depth; Python cost per chunk
  t_exec(n)             one real Mixtral expert (gather + w1/w3/w2 + routing weight) at n tokens
  t_nonexpert_layer     real attention + norms + router (+ the per-layer router sync), experts
                        replaced by zeros, at batch x ctx; per layer
  contention            expert loop alone vs with a concurrent 352 MB copy stream: slowdown ratio
  power                 NVML energy counter: idle, copy only, compute only, both

    gpurun -g 1 gpu/numa_run.sh python -m gpu.microbench
"""
import argparse
import json
import os
import threading
import time

import numpy as np
import torch

from engine.link import DEMAND, Transfer

from .common import CALIBRATION, MODEL_DIR, RESULTS
from .energy import GPUEnergy
from .env import collect
from .runtime import Runtime, _ev

MB = 1 << 20
IDLE_S = 0.008          # a typical demand wait per layer (two 352 MB experts at ~56 GB/s = 12.5 ms)


def spin(seconds):
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        pass


def gpu_time_after_idle(fn, reps, idle_s):
    """GPU time of fn right after the GPU sat idle for idle_s (host spinning, like a link wait)."""
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        spin(idle_s)
        a, b = _ev(), _ev()
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e-3)
    return float(np.median(ts))


def gpu_time(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = _ev(), _ev()
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e-3 / reps


def bench_link(rt):
    src = rt.host[0]
    src2 = rt.host[1]
    dst, dst2 = rt.slab[0], rt.slab[1]
    out = {}
    for mb in (16, 64, 352):
        n = min(mb * MB // 2, src.numel())
        t = gpu_time(lambda: dst[:n].copy_(src[:n], non_blocking=True), 50)
        out[f"single_{mb}MB_GBps"] = 2 * n / t / 1e9
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.cuda.stream(s1):
            dst.copy_(src, non_blocking=True)
        with torch.cuda.stream(s2):
            dst2.copy_(src2, non_blocking=True)
    torch.cuda.synchronize()
    out["two_streams_352MB_GBps"] = 2 * 20 * src.numel() * 2 / (time.perf_counter() - t0) / 1e9
    print("link", json.dumps({k: round(v, 2) for k, v in out.items()}), flush=True)
    return out


def bench_scheduler(rt, n_experts=24):
    """Demand transfers through the real scheduler, nothing else running."""
    link = rt.link
    nb = rt.numel * 2
    out = {}
    for chunk_MB in (16, 32, 64):
        for depth in (2, 4):
            link.chunk_el = int(chunk_MB * 1e6) // 2
            link.depth = depth
            link.reset("prio", 55e9)
            # single-expert latency (enqueue -> last chunk done), 8 repeats
            lats = []
            for r in range(8):
                tr = Transfer(r, nb, DEMAND, 0.0, 0)
                need = _ev()
                need.record()
                link.enqueue(tr)
                link.bind(tr, r % 4)
                ev = link.wait_issued(tr)
                ev.synchronize()
                lats.append(need.elapsed_time(ev) * 1e-3)
            link.drain()
            # throughput: a queue of n_experts back to back
            link.reset("prio", 55e9)
            busy0 = link.busy_time
            trs = []
            torch.cuda.synchronize()
            start = _ev()
            start.record()
            t0 = time.perf_counter()
            for r in range(n_experts):
                tr = Transfer(r % 200, nb, DEMAND, 0.0, 0)
                link.enqueue(tr)
                link.bind(tr, r % 8)
                trs.append(tr)
            last = link.wait_issued(trs[-1])
            last.synchronize()
            wall = time.perf_counter() - t0
            gpu = start.elapsed_time(last) * 1e-3
            link.drain()
            key = f"chunk{chunk_MB}_depth{depth}"
            out[key] = dict(latency_ms=float(np.median(lats) * 1e3),
                            throughput_GBps=n_experts * nb / gpu / 1e9,
                            busy_GBps=n_experts * nb / max(link.busy_time - busy0, 1e-9) / 1e9,
                            host_wall_GBps=n_experts * nb / wall / 1e9,
                            chunks=link.chunks)
            print("scheduler", key, json.dumps({k: round(v, 3) for k, v in out[key].items()}), flush=True)
    link.chunk_el = int(16e6) // 2
    link.depth = 2
    link.reset("prio", 55e9)
    # Python cost of issuing one chunk (2 events + copy launch), measured in isolation
    src, dst = rt.host[0], rt.slab[0]
    n = int(16e6) // 2
    t0 = time.perf_counter()
    for _ in range(200):
        e0, e1 = _ev(), _ev()
        e0.record(link.stream)
        with torch.cuda.stream(link.stream):
            dst[:n].copy_(src[:n], non_blocking=True)
        e1.record(link.stream)
    out["python_issue_us_per_chunk"] = (time.perf_counter() - t0) / 200 * 1e6
    torch.cuda.synchronize()
    return out


def load_expert(rt, slot, l=0, e=0):
    rt.slab[slot].copy_(rt.host_le[(l, e)])
    torch.cuda.synchronize()


@torch.inference_mode()
def bench_exec(rt, ns=(1, 2, 4, 8, 16, 32, 64)):
    """Per expert as the runtime runs it: rows/weights already gathered once per layer."""
    load_expert(rt, 0)
    H = rt.H
    out, out_wall, out_idle = {}, {}, {}
    for n in ns:
        x = torch.randn(n, H, dtype=torch.bfloat16, device=rt.device)
        w = torch.rand(n, 1, dtype=torch.bfloat16, device=rt.device)
        fn = lambda: rt._expert(rt.slab[0], x, w)
        out[str(n)] = gpu_time(fn, 100)
        # host-paced (each call synchronised): what a layer sees when the host is not ahead
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
            torch.cuda.synchronize()
        out_wall[str(n)] = (time.perf_counter() - t0) / 50
        out_idle[str(n)] = gpu_time_after_idle(fn, 30, IDLE_S)
    print("t_exec_us", {k: round(v * 1e6, 1) for k, v in out.items()}, flush=True)
    print("t_exec_us_after_idle", {k: round(v * 1e6, 1) for k, v in out_idle.items()}, flush=True)
    return out, out_wall, out_idle


@torch.inference_mode()
def bench_nonexpert(rt, batches=(1, 4, 16), ctxs=(256, 512, 1024, 1536), steps=24, docs=None):
    from transformers import DynamicCache
    out, out_idle = {}, {}
    rt.mode = "null"
    for b in batches:
        for c in ctxs:
            ids = torch.as_tensor(docs[:b, :c + steps + 9], dtype=torch.long, device=rt.device)
            kv = DynamicCache()
            rt.model.model(input_ids=ids[:, :c], past_key_values=kv, use_cache=True)
            times = []
            for t in range(steps):
                a, e = _ev(), _ev()
                a.record()
                pos = torch.tensor([c + t], device=rt.device)
                o = rt.model.model(input_ids=ids[:, c + t:c + t + 1], past_key_values=kv, use_cache=True,
                                   cache_position=pos)
                rt.model.lm_head(o.last_hidden_state[:, -1]).argmax(-1)
                e.record()
                e.synchronize()
                times.append(a.elapsed_time(e) * 1e-3)
            # the same steps with the GPU idle for IDLE_S before every layer (host spinning)
            rt.null_idle_s = IDLE_S
            idle = []
            for t in range(steps, steps + 8):
                a, e = _ev(), _ev()
                a.record()
                pos = torch.tensor([c + t], device=rt.device)
                o = rt.model.model(input_ids=ids[:, c + t:c + t + 1], past_key_values=kv, use_cache=True,
                                   cache_position=pos)
                rt.model.lm_head(o.last_hidden_state[:, -1]).argmax(-1)
                e.record()
                e.synchronize()
                idle.append(a.elapsed_time(e) * 1e-3 - rt.L * IDLE_S)
            rt.null_idle_s = 0.0
            del kv
            key = f"b{b}_ctx{c}"
            out[key] = float(np.median(times[4:])) / rt.L
            out_idle[key] = float(np.median(idle)) / rt.L
            print(f"nonexpert {key}: {out[key] * 1e6:.0f} us/layer, after idle {out_idle[key] * 1e6:.0f}", flush=True)
    rt.mode = "prefill"
    return out, out_idle


@torch.inference_mode()
def bench_contention(rt):
    load_expert(rt, 2)
    H = rt.H
    res = {}
    stop = threading.Event()
    cs = torch.cuda.Stream()

    def copier():
        torch.cuda.set_device(rt.device)
        while not stop.is_set():
            with torch.cuda.stream(cs):
                for _ in range(4):
                    rt.slab[5].copy_(rt.host[7], non_blocking=True)
            cs.synchronize()

    for n in (1, 16, 64):
        x = torch.randn(n, H, dtype=torch.bfloat16, device=rt.device)
        w = torch.ones(n, 1, dtype=torch.bfloat16, device=rt.device)
        fn = lambda: rt._expert(rt.slab[2], x, w)
        alone = gpu_time(fn, 300)
        stop.clear()
        th = threading.Thread(target=copier)
        th.start()
        time.sleep(0.05)
        busy = gpu_time(fn, 300)
        stop.set()
        th.join()
        torch.cuda.synchronize()
        res[str(n)] = dict(alone_us=alone * 1e6, with_copy_us=busy * 1e6, factor=busy / alone)
        print("contention", n, json.dumps({k: round(v, 3) for k, v in res[str(n)].items()}), flush=True)
    return res


@torch.inference_mode()
def bench_power(rt, energy):
    load_expert(rt, 2)
    x = torch.randn(1, rt.H, dtype=torch.bfloat16, device=rt.device)
    w = torch.ones(1, 1, dtype=torch.bfloat16, device=rt.device)
    cs = torch.cuda.Stream()
    torch.cuda.synchronize()
    time.sleep(2)
    out = {"idle": energy.sample_power(4.0)}

    def copy_burst():
        with torch.cuda.stream(cs):
            for _ in range(8):
                rt.slab[5].copy_(rt.host[7], non_blocking=True)
        cs.synchronize()

    def compute_burst():
        for _ in range(200):
            rt._expert(rt.slab[2], x, w)
        torch.cuda.synchronize()

    def both():
        with torch.cuda.stream(cs):
            for _ in range(8):
                rt.slab[5].copy_(rt.host[7], non_blocking=True)
        compute_burst()
        cs.synchronize()

    out["copy"] = energy.sample_power(4.0, copy_burst)
    out["compute"] = energy.sample_power(4.0, compute_burst)
    out["copy_and_compute"] = energy.sample_power(4.0, both)
    torch.cuda.synchronize()
    time.sleep(1)
    out["idle_after"] = energy.sample_power(3.0)
    print("power_W", {k: round(v, 1) for k, v in out.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: link scheduler exec nonexpert contention power")
    ap.add_argument("--out", default=CALIBRATION)
    args = ap.parse_args()
    from .common import doc_tokens
    ids, _ = doc_tokens()
    rt = Runtime(args.model_dir, slab_slots=16)
    energy = GPUEnergy()
    want = set(args.only or ["link", "scheduler", "exec", "nonexpert", "contention", "power"])
    cal = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            cal = json.load(f)
    name = torch.cuda.get_device_name()
    cal.update(gpu="H100-NVL" if "NVL" in name else ("H100-PCIe" if "PCIe" in name else "H100-SXM"),
               gpu_name=name, model="mixtral", link="PCIe5", env=collect())
    if "link" in want:
        cal["link_bench"] = bench_link(rt)
    if "scheduler" in want:
        cal["scheduler_bench"] = bench_scheduler(rt)
        sb = cal["scheduler_bench"]["chunk16_depth2"]
        cal["link_GBps"] = sb["throughput_GBps"]          # what the runtime's scheduler achieves
        cal["t_load_expert_s"] = sb["latency_ms"] * 1e-3
    if "exec" in want:
        cal["t_exec_s"], cal["t_exec_hostpaced_s"], cal["t_exec_after_idle_s"] = bench_exec(rt)
    if "nonexpert" in want:
        cal["t_nonexpert_layer_s"], cal["t_nonexpert_layer_after_idle_s"] = bench_nonexpert(rt, docs=ids)
    if "contention" in want:
        cal["contention_bench"] = bench_contention(rt)
        cal["contention_factor"] = cal["contention_bench"]["1"]["factor"]
    if "power" in want:
        cal["power_W"] = bench_power(rt, energy)
    rt.close()
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cal, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
