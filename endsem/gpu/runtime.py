"""Mixtral decode on a real GPU with an offloaded MoE block: HBM expert cache, copy scheduler, policies.

Attention, RoPE, RMSNorm and the KV cache are Hugging Face's own modules. Every layer's
block_sparse_moe is replaced by OffloadedMoE, which reproduces MixtralSparseMoeBlock exactly and asks
the runtime for the experts it needs.

The cache and prefetch decisions are the engine's own code: RealTimeline subclasses engine.Timeline and
reuses its eviction scores, _make_room, _issue (lookahead, link-time cap, value gate), _end_layer
(stale cancel, evict-after-use) and the same ExpertCache. Only the per-layer loop is rewritten, with the
simulated link replaced by RealLink (real copies) and simulated time by CUDA events.

RealLink mirrors engine/link.py: one host thread, one copy stream, transfers split into 16 MB chunks
from pinned memory; demand chunks always go first (prio mode), prefetch chunks only when no demand is
queued; at most `depth` chunks in flight, so a demand miss waits at most ~depth chunks behind a prefetch
(the engine's "preempt at a chunk boundary").

Expert outputs are accumulated in a fixed (ascending expert) order whatever order the experts arrive
in, so caching and prefetching can never change the model's output.
"""
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from engine.hbm import ExpertCache
from engine.link import DEMAND, SPEC, Transfer
from engine.timeline import SimConfig, Timeline

from .weights import expert_numel, expert_views, load_weights

sys.setswitchinterval(1e-4)    # the copy thread must get the GIL quickly when a chunk finishes


def _ev():
    return torch.cuda.Event(enable_timing=True)


# --------------------------------------------------------------------------------------------------
# the real host -> HBM link
# --------------------------------------------------------------------------------------------------
class RealLink:
    def __init__(self, host, slab, chunk_bytes, device, depth=2):
        self.host, self.slab = host, slab
        self.chunk_el = int(chunk_bytes) // 2
        self.chunk = self.chunk_el * 2
        self.depth = depth
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.cv = threading.Condition()
        self.mode = "prio"
        self.bw = 55e9
        self.spin = True
        self._stop = False
        self.reset()
        self.thread = threading.Thread(target=self._run, daemon=True, name="copy-scheduler")
        self.thread.start()

    def reset(self, mode="prio", bw=None):
        with getattr(self, "cv", threading.Condition()):
            self.mode = mode
            if bw:
                self.bw = bw
            self.qd, self.qs = deque(), deque()
            self.inflight = deque()          # (tr, nbytes, ev_start, ev_end, last)
            self.dst, self.issued, self.last_ev = {}, {}, {}
            self.pending_bytes = 0.0
            self.bytes_moved = 0.0
            self.bytes_issued = 0.0
            self.busy_time = 0.0
            self.chunks = 0

    # --- API used by the engine's timeline code ---------------------------------------------
    def enqueue(self, tr):
        with self.cv:
            q = self.qd if (self.mode == "prio" and tr.prio == DEMAND) else self.qs
            q.append(tr)
            self.pending_bytes += tr.nbytes - tr.done
            self.issued[tr] = threading.Event()
            self.cv.notify()

    def bind(self, tr, slot):
        with self.cv:
            self.dst[tr] = self.slab[slot]
            self.cv.notify()

    def promote(self, tr):
        with self.cv:
            if tr.prio == DEMAND or tr.finish is not None or tr.cancelled:
                return
            tr.prio = DEMAND
            if self.mode != "prio" or tr.done >= tr.nbytes:
                return
            self.qs.remove(tr)
            if tr.done > 0:
                self.qd.appendleft(tr)       # finish the partially moved expert first
            else:
                self.qd.append(tr)
            self.cv.notify()

    def cancel(self, tr):
        with self.cv:
            if tr.finish is not None or tr.cancelled:
                return
            tr.cancelled = True
            if tr.done < tr.nbytes:          # chunks already issued complete anyway (counted as moved)
                for q in (self.qd, self.qs):
                    try:
                        q.remove(tr)
                        break
                    except ValueError:
                        pass
                self.pending_bytes -= tr.nbytes - tr.done
            self.dst.pop(tr, None)
            ev = self.issued.get(tr)
            if ev is not None:
                ev.set()

    # --- the scheduler thread -----------------------------------------------------------------
    def _head(self):
        tr = None
        if self.qd:
            tr = self.qd[0]
        elif self.qs:
            tr = self.qs[0]
        if tr is None or tr not in self.dst:
            return None
        return tr

    def _issue_chunk(self, tr):
        off = int(tr.done) // 2
        n = min(self.chunk_el, int(tr.nbytes) // 2 - off)
        if tr.start is None:
            tr.start = time.perf_counter()
        src = self.host[tr.key]
        dst = self.dst[tr]
        e0, e1 = _ev(), _ev()
        e0.record(self.stream)
        with torch.cuda.stream(self.stream):
            dst[off:off + n].copy_(src[off:off + n], non_blocking=True)
        e1.record(self.stream)
        tr.done += 2 * n
        self.bytes_issued += 2 * n
        self.chunks += 1
        last = tr.done >= tr.nbytes
        if last:
            q = self.qd if (self.qd and self.qd[0] is tr) else self.qs
            q.popleft()
            self.dst.pop(tr, None)
            self.last_ev[tr] = e1
            self.issued[tr].set()
        self.inflight.append((tr, 2 * n, e0, e1, last))

    def _prune(self):
        while self.inflight and self.inflight[0][3].query():
            tr, n, e0, e1, last = self.inflight.popleft()
            self.busy_time += e0.elapsed_time(e1) * 1e-3
            self.bytes_moved += n
            self.pending_bytes -= n
            if last and tr.finish is None:
                tr.finish = time.perf_counter()

    def _run(self):
        torch.cuda.set_device(self.device)
        while True:
            wait_ev = None
            with self.cv:
                if self._stop:
                    return
                self._prune()
                if len(self.inflight) >= self.depth:
                    wait_ev = self.inflight[0][3]
                else:
                    tr = self._head()
                    if tr is not None:
                        self._issue_chunk(tr)
                        continue
                    if self.inflight:
                        wait_ev = self.inflight[0][3]
                    else:
                        self.cv.notify_all()          # drain() waiters
                        self.cv.wait()
                        continue
            wait_ev.synchronize()                     # releases the GIL

    # --- helpers for the runtime ------------------------------------------------------------------
    def poll(self):
        with self.cv:
            self._prune()

    def order(self, trs):
        """trs sorted in the order the link will finish them: in flight, demand queue, prefetch queue."""
        with self.cv:
            pos = {t: 1 + i for i, t in enumerate(self.qd)}
            n = len(self.qd)
            pos.update({t: 1 + n + i for i, t in enumerate(self.qs)})
        return sorted(trs, key=lambda t: pos.get(t, 0))

    def wait_issued(self, tr):
        """Spin (sleep(0) releases the GIL each turn) rather than block: a blocked thread lets the
        CPU governor drop its core's clock, and every later PyTorch launch runs ~2.5x slower."""
        ev = self.issued[tr]
        if self.spin:
            while not ev.is_set():
                time.sleep(0)
        else:
            ev.wait()
        return self.last_ev.get(tr)

    def drain(self):
        with self.cv:
            while self.qd or self.qs or self.inflight:
                if not (self.qd or self.qs) and self.inflight:
                    ev = self.inflight[-1][3]
                    self.cv.release()
                    try:
                        ev.synchronize()
                    finally:
                        self.cv.acquire()
                    self._prune()
                    continue
                self.cv.wait(0.01)
                self._prune()

    def stop(self):
        with self.cv:
            self._stop = True
            self.cv.notify_all()


class RealCache(ExpertCache):
    """The engine's cache; inserting a slot binds the transfer to that slot's HBM memory."""

    def __init__(self, capacity, L, N, link, slab_slots):
        super().__init__(capacity, L, N)
        self.link, self.slab_slots = link, slab_slots

    def insert(self, key, transfer, prefetched, pin_g, now_g):
        slot = super().insert(key, transfer, prefetched, pin_g, now_g)
        if slot >= self.slab_slots:
            raise RuntimeError(f"slot {slot} beyond the {self.slab_slots}-slot HBM slab")
        self.link.bind(transfer, slot)
        return slot


# --------------------------------------------------------------------------------------------------
# the engine timeline, driven by real copies and real compute
# --------------------------------------------------------------------------------------------------
class RealTimeline(Timeline):
    def __init__(self, wl, needed, stream, cfg: SimConfig, rt):
        super().__init__(wl, needed, stream, cfg)
        self.rt = rt
        rt.link.drain()
        cap = self.cache.cap
        self.cache = RealCache(cap, self.L, self.N, rt.link, rt.slab_slots)
        rt.link.reset(cfg.link_mode, self.cost.link_bw)
        self.link = rt.link
        self.ebytes = self.cost.expert_bytes
        self._util_mark = (0.0, time.perf_counter())
        self.step_ms = np.zeros(self.S)
        self.step_stall_ms = np.zeros(self.S)
        self.load_lat = []                 # expert-load latency (s) per needed expert, 0 for hits
        self.host_layer_s = []             # host time from router result to last expert launched
        self._pend = []                    # events to resolve at the end of the step
        self._lat = []
        self._marks = []                   # (router-done event, MoE-done event) per layer of the step
        self.nonexpert_ms, self.moe_ms = [], []

    # one MoE layer of decode step s ------------------------------------------------------------------
    def layer(self, s, j, experts, counts, run_expert):
        """experts: sorted unique experts the batch routed to; run_expert(slot_row, e) launches one."""
        cfg, cost, c, link, m = self.cfg, self.cost, self.cache, self.link, self.m
        L, N = self.L, self.N
        g = s * L + j
        h0 = time.perf_counter()
        need_ev = _ev()
        need_ev.record()
        self._need_ev = need_ev
        link.poll()
        slots, missing, waits = [], [], {}
        for i, e in enumerate(experts):
            key = j * N + int(e)
            slot = c.index.get(key)
            if slot is not None:
                tr = c.transfer[slot]
                if tr is None or tr.finish is not None:
                    m["hits"] += 1
                    self.layer_hits[j] += 1
                    if c.prefetched[slot] and not c.used[slot]:
                        m["prefetch_timely"] += 1
                    self.load_lat.append(0.0)
                else:
                    m["late"] += 1
                    if c.prefetched[slot] and not c.used[slot]:
                        m["prefetch_late"] += 1
                    link.promote(tr)
                    waits[i] = tr
                c.pin[slot] = max(c.pin[slot], g)
            else:
                missing.append((i, key))
            slots.append(slot)
        if missing:
            m["misses"] += len(missing)
            if self._make_room(len(missing), s, j, g, time.perf_counter(), "demand") < len(missing):
                raise RuntimeError("no slot for a demand miss")
            for i, key in missing:
                tr = Transfer(key, self.ebytes, DEMAND, h0, g)
                link.enqueue(tr)
                slots[i] = c.insert(key, tr, False, g, g)
                waits[i] = tr
        m["needs"] += len(experts)
        self.layer_needs[j] += len(experts)

        if cfg.prefetch:
            self._issue(s, j, g, time.perf_counter())

        # execute: arrived experts first, then the others in link order (overlap), or all after all
        # arrive (strict). Stall = time the compute stream waits on a copy event.
        ready = [i for i in range(len(experts)) if i not in waits]
        by_tr = {waits[i]: i for i in waits}
        pending = [by_tr[t] for t in link.order(list(by_tr))]
        if cfg.miss_overlap:
            for i in ready:
                run_expert(slots[i], int(experts[i]))
            for i in pending:
                tr = waits[i]
                # record "wait starts" before the host waits for the transfer to be issued: the GPU
                # reaches it as soon as the experts launched so far are done, so a -> b is the whole
                # time the compute stream sits idle for this expert
                a, b = _ev(), _ev()
                a.record()
                ev = link.wait_issued(tr)
                torch.cuda.current_stream().wait_event(ev)
                b.record()
                self._pend.append((a, b))
                self._lat.append((need_ev, ev))
                run_expert(slots[i], int(experts[i]))
        else:
            if pending:
                a, b = _ev(), _ev()
                a.record()
                for i in pending:
                    ev = link.wait_issued(waits[i])
                    torch.cuda.current_stream().wait_event(ev)
                    self._lat.append((need_ev, ev))
                b.record()
                self._pend.append((a, b))
            for i in range(len(experts)):
                run_expert(slots[i], int(experts[i]))
        self.host_layer_s.append(time.perf_counter() - h0)

        m["exec_experts"] += len(experts)
        m["exec_tokens"] += int(np.sum(counts))
        layer_compute = cost.t_nonexpert_layer + sum(cost.t_exec(int(n)) for n in counts)
        self.t_layer_est += 0.1 * (layer_compute - self.t_layer_est)
        now = time.perf_counter()
        moved0, t0 = self._util_mark
        if now > t0:
            util = min(1.0, (link.bytes_moved - moved0) / link.bw / (now - t0))
            self.link_util += 0.1 * (util - self.link_util)
        self._util_mark = (link.bytes_moved, now)

        for slot in slots:
            if c.prefetched[slot] and not c.used[slot]:
                m["prefetch_used"] += 1
                m["useful_prefetch_bytes"] += self.ebytes
            c.used[slot] = True
            c.last_use[slot] = g
            c.freq[slot] += 1
            tr = c.transfer[slot]
            if tr is not None and tr.finish is not None:
                c.ready_at[slot] = tr.finish
                c.inflight.discard(slot)
        self._end_layer(s, j, g, experts)

    def mark_moe_done(self):
        e = _ev()
        e.record()
        self._marks.append((self._need_ev, e))

    def end_step(self, s, ev0, ev1):
        ev1.synchronize()
        self.step_ms[s] = ev0.elapsed_time(ev1)
        mk = self._marks
        if len(mk) >= 2:
            self.nonexpert_ms.append(np.mean([mk[i][1].elapsed_time(mk[i + 1][0]) for i in range(len(mk) - 1)]))
            self.moe_ms.append(np.mean([a.elapsed_time(b) for a, b in mk]))
        self._marks = []
        st = sum(a.elapsed_time(b) for a, b in self._pend)
        self.step_stall_ms[s] = st
        for need_ev, done_ev in self._lat:
            self.load_lat.append(max(need_ev.elapsed_time(done_ev), 0.0) * 1e-3)
        self._pend, self._lat = [], []
        # drop bookkeeping of finished transfers
        link = self.link
        with link.cv:
            for tr in [t for t in link.last_ev if t.finish is not None or t.cancelled]:
                link.last_ev.pop(tr, None)
                link.issued.pop(tr, None)

    def results(self, energy_J=None):
        m, cfg = self.m, self.cfg
        S, B = self.S, self.B
        self.link.drain()
        for slot in list(self.cache.index.values()):
            if self.cache.prefetched[slot] and not self.cache.used[slot]:
                m["prefetch_wasted"] += 1
        t_total = self.step_ms.sum() * 1e-3
        stall = self.step_stall_ms.sum() * 1e-3
        bytes_moved = self.link.bytes_moved
        useful = m["misses"] * self.ebytes + m["useful_prefetch_bytes"]
        lat = np.asarray(self.load_lat) * 1e3
        lat_loads = lat[lat > 0]
        T = self.wl.tokens_per_request
        keep = np.ones(S, bool)
        for g0 in range(0, S, T):
            keep[g0:g0 + 8] = False            # first 8 steps of each group = warm-up
        if not keep.any():
            keep[:] = True
        res = dict(model=self.wl.model, batch=B, steps=S, config_eviction=cfg.eviction,
                   prefetch=cfg.prefetch, miss_overlap=cfg.miss_overlap, link_mode=cfg.link_mode,
                   lookahead=cfg.lookahead, lookahead_cap=cfg.lookahead_cap, value_gate=cfg.value_gate,
                   cache_slots=self.cache.cap if self.retention else 0,
                   token_latency_ms=1e3 * t_total / S,
                   p95_token_latency_ms=float(np.percentile(self.step_ms, 95)),
                   token_latency_ms_after_warmup=float(self.step_ms[keep].mean()),
                   p95_token_latency_ms_after_warmup=float(np.percentile(self.step_ms[keep], 95)),
                   tokens_per_s=S * B / t_total,
                   stall_ms_per_token=1e3 * stall / S,
                   stall_fraction=stall / t_total,
                   hit_rate=m["hits"] / max(m["needs"], 1),
                   late_rate=m["late"] / max(m["needs"], 1),
                   miss_rate=m["misses"] / max(m["needs"], 1),
                   prefetch_issued_per_token=m["prefetch_issued"] / S,
                   prefetch_accuracy=m["prefetch_used"] / max(m["prefetch_issued"], 1),
                   prefetch_wasted_frac=m["prefetch_wasted"] / max(m["prefetch_issued"], 1),
                   prefetch_timely_frac=m["prefetch_timely"] / max(m["prefetch_timely"] + m["prefetch_late"], 1),
                   stale_cancelled_per_token=m["stale_cancelled"] / S,
                   GB_per_token=bytes_moved / S / 1e9,
                   useful_GB_per_token=min(useful, bytes_moved) / S / 1e9,
                   wasted_GB_per_token=max(bytes_moved - useful, 0.0) / S / 1e9,
                   link_utilization=self.link.busy_time / t_total,
                   link_GBps_achieved=bytes_moved / max(self.link.busy_time, 1e-12) / 1e9,
                   mean_lookahead=m["lookahead_sum"] / max(S * self.L, 1) if cfg.prefetch else 0.0,
                   expert_load_latency_ms_mean=float(lat.mean()) if len(lat) else 0.0,
                   expert_load_latency_ms_p95=float(np.percentile(lat, 95)) if len(lat) else 0.0,
                   expert_load_latency_ms_mean_loads=float(lat_loads.mean()) if len(lat_loads) else 0.0,
                   host_decision_ms_per_token=1e3 * float(np.sum(self.host_layer_s)) / S,
                   measured_nonexpert_ms_per_layer=float(np.mean(self.nonexpert_ms)) if self.nonexpert_ms else 0.0,
                   measured_moe_ms_per_layer=float(np.mean(self.moe_ms)) if self.moe_ms else 0.0,
                   measured_exec_ms_per_layer=(float(np.mean(self.moe_ms)) - 1e3 * stall / S / self.L)
                   if self.moe_ms else 0.0,
                   needs=int(m["needs"]), hits=int(m["hits"]), late=int(m["late"]), misses=int(m["misses"]))
        if energy_J is not None:
            res["gpu_energy_J_per_token"] = energy_J / (S * B)
            res["gpu_power_W"] = energy_J / t_total if t_total > 0 else float("nan")
        res["layer_hit_rate"] = (self.layer_hits / np.maximum(self.layer_needs, 1)).tolist()
        return res


# --------------------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------------------
class OffloadedMoE(nn.Module):
    """Drop-in for MixtralSparseMoeBlock: same routing math, experts served by the runtime."""

    def __init__(self, layer_idx, hidden, n_experts, top_k, rt):
        super().__init__()
        self.layer_idx, self.top_k = layer_idx, top_k
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self._rt = [rt]                        # list: keep the runtime out of the module tree

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        x = hidden_states.view(-1, H)
        logits = self.gate(x)
        w = F.softmax(logits, dim=1, dtype=torch.float)
        topw, topi = torch.topk(w, self.top_k, dim=-1)
        topw /= topw.sum(dim=-1, keepdim=True)
        topw = topw.to(x.dtype)
        rt = self._rt[0]
        if rt.hs is not None:                  # Phase B'': hidden-state predictions for layers l+1 .. l+4
            rt.hs.on_router(self.layer_idx, x, w)
        out = rt.moe(self.layer_idx, x, topw, topi, logits)
        return out.view(B, T, H), logits


class Runtime:
    """Loads Mixtral once; runs prefill (experts streamed through staging slots) and decode (cache)."""

    def __init__(self, model_dir, slab_slots, device="cuda:0", layers=None, depth=2, chunk_MB=16.0,
                 verbose=True, resident=False):
        """resident=True: every expert lives in HBM (Mixtral fits in an H100 NVL's 94 GB); no pinned
        copies, no link. Used to collect live traces quickly."""
        from transformers import MixtralConfig, MixtralForCausalLM
        from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding
        torch.cuda.set_device(device)
        self.device = torch.device(device)
        t0 = time.time()
        from .weights import load_config
        cfg0 = load_config(model_dir)
        self.resident = resident
        res_slab = None
        if resident:
            Lr = cfg0["num_hidden_layers"] if layers is None else layers
            res_slab = torch.empty(Lr * cfg0["num_local_experts"], expert_numel(cfg0), dtype=torch.bfloat16,
                                   device=device)
        cfg, gpu_sd, host = load_weights(model_dir, device, layers, verbose, gpu_slab=res_slab)
        self.cfg = cfg
        self.L = cfg["num_hidden_layers"] if layers is None else layers
        self.N, self.k = cfg["num_local_experts"], cfg["num_experts_per_tok"]
        self.H = cfg["hidden_size"]
        hf = MixtralConfig.from_pretrained(model_dir)
        hf.num_hidden_layers = self.L
        hf._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = MixtralForCausalLM(hf)
        for l, layer in enumerate(model.model.layers):
            layer.block_sparse_moe = OffloadedMoE(l, self.H, self.N, self.k, self)
        missing, unexpected = model.load_state_dict(gpu_sd, strict=False, assign=True)
        assert not unexpected, unexpected[:5]
        assert not missing, missing[:5]
        model.model.rotary_emb = MixtralRotaryEmbedding(hf, device=self.device)
        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            assert p.device.type == "cuda", f"{n} still on {p.device}"
        model.eval()
        self.model = model
        self.hf_config = hf
        # host experts keyed like the engine: layer * N + expert
        self.host = {l * self.N + e: buf for (l, e), buf in host.items()}
        self.host_le = host
        self.numel = expert_numel(cfg)
        if resident:
            self.slab_slots = self.L * self.N
            self.slab = res_slab
            self.staging = None
            self.link = None
        else:
            self.slab_slots = slab_slots
            self.slab = torch.empty(slab_slots + 2, self.numel, dtype=torch.bfloat16, device=self.device)
            self.staging = [self.slab[slab_slots], self.slab[slab_slots + 1]]
            self.link = RealLink(self.host, self.slab, chunk_MB * 1e6, self.device, depth)
        self.prefill_stream = torch.cuda.Stream(self.device)
        self.mode = "prefill"
        self.null_idle_s = 0.0
        self.hs = None                   # HSPredictor (gpu/hs_predictor.py) when B'' / C5 is on
        self.timeline = None
        self.step = 0
        self.record = None              # decode routing [S, B, L, k] of the current run
        self.record_logits = None
        self.prefill_topi = None         # list per layer of prefill routing (when recording)
        self._idx_pin = torch.empty(1 << 16, dtype=torch.int64, pin_memory=True)
        self.load_seconds = time.time() - t0
        if verbose:
            gb = sum(t.numel() * t.element_size() for t in gpu_sd.values()) / 1e9
            print(f"loaded in {self.load_seconds:.0f}s: {gb:.2f} GB non-expert on GPU, "
                  f"{len(self.host)} experts pinned, slab {slab_slots} slots", flush=True)

    # --- expert math (MixtralBlockSparseTop2MLP, then the routing weight) ---------------------------
    def _expert(self, buf, xs, wts):
        w1, w3, w2 = expert_views(buf, self.cfg)
        return F.linear(F.silu(F.linear(xs, w1)) * F.linear(xs, w3), w2) * wts

    def _assignments(self, topi_h, experts):
        """Per expert: (token rows, top-k position) in HF's order, as GPU index tensors (one H2D)."""
        rows, kpos, spans = [], [], {}
        o = 0
        for e in experts:
            kk, tt = np.nonzero((topi_h == e).T)        # HF: torch.where(mask[e]) -> (k, token)
            rows.append(tt)
            kpos.append(kk)
            spans[int(e)] = (o, o + len(tt))
            o += len(tt)
        flat = np.concatenate(rows + kpos).astype(np.int64)
        n = len(flat)
        if n > self._idx_pin.numel():
            self._idx_pin = torch.empty(2 * n, dtype=torch.int64, pin_memory=True)
        self._idx_pin[:n].copy_(torch.from_numpy(flat))
        dev = self._idx_pin[:n].to(self.device, non_blocking=True)
        return dev[:o], dev[o:], spans

    def moe(self, l, x, topw, topi, logits):
        if self.resident:
            return self._moe_resident(l, x, topw, topi, logits)
        if self.mode == "decode":
            return self._moe_decode(l, x, topw, topi, logits)
        if self.mode == "null":
            topi.cpu()
            if self.null_idle_s:
                t = time.perf_counter()
                while time.perf_counter() - t < self.null_idle_s:
                    pass
            return torch.zeros_like(x)
        return self._moe_prefill(l, x, topw, topi)

    def _combine(self, x, outs, rows, spans):
        final = torch.zeros_like(x)
        for e in sorted(outs):                           # fixed order = HF's ascending expert loop
            a, b = spans[e]
            final.index_add_(0, rows[a:b], outs[e].to(x.dtype))
        return final

    def _moe_resident(self, l, x, topw, topi, logits):
        topi_h = topi.cpu().numpy()
        if self.record is not None and self.mode == "decode":
            self.record[self.step, :, l, :] = topi_h
        if self.prefill_topi is not None:
            self.prefill_topi.append(topi_h.astype(np.int8))
            self.prefill_probs.append(torch.softmax(logits.float(), -1).cpu().numpy().astype(np.float16))
        experts = np.unique(topi_h)
        rows, kpos, spans = self._assignments(topi_h, experts)
        xg, wg = x[rows], topw[rows, kpos].unsqueeze(-1)
        outs = {}
        for e in experts:
            a, b = spans[int(e)]
            outs[int(e)] = self._expert(self.slab[l * self.N + int(e)], xg[a:b], wg[a:b])
        return self._combine(x, outs, rows, spans)

    def _moe_prefill(self, l, x, topw, topi):
        topi_h = topi.cpu().numpy()
        if self.prefill_topi is not None:
            self.prefill_topi.append(topi_h.astype(np.int8))
        experts = np.unique(topi_h)
        rows, kpos, spans = self._assignments(topi_h, experts)
        xg, wg = x[rows], topw[rows, kpos].unsqueeze(-1)    # one gather per layer, sliced per expert
        cur = torch.cuda.current_stream()
        cs = self.prefill_stream
        outs, freed = {}, [None, None]
        for i, e in enumerate(experts):
            buf = self.staging[i % 2]
            if freed[i % 2] is not None:
                cs.wait_event(freed[i % 2])
            with torch.cuda.stream(cs):
                buf.copy_(self.host_le[(l, int(e))], non_blocking=True)
            arrived = torch.cuda.Event()
            arrived.record(cs)
            cur.wait_event(arrived)
            a, b = spans[int(e)]
            outs[int(e)] = self._expert(buf, xg[a:b], wg[a:b])
            done = torch.cuda.Event()
            done.record(cur)
            freed[i % 2] = done
        return self._combine(x, outs, rows, spans)

    def _moe_decode(self, l, x, topw, topi, logits):
        topi_h = topi.cpu().numpy()                      # the router result: one sync per layer
        if self.record is not None:
            self.record[self.step, :, l, :] = topi_h
        if self.record_logits is not None:
            self.record_logits[self.step, :, l, :] = logits.float().cpu().numpy()
        experts, counts = np.unique(topi_h, return_counts=True)
        rows, kpos, spans = self._assignments(topi_h, experts)
        xg, wg = x[rows], topw[rows, kpos].unsqueeze(-1)    # one gather per layer, sliced per expert
        outs = {}

        def run_expert(slot, e):
            a, b = spans[e]
            outs[e] = self._expert(self.slab[slot], xg[a:b], wg[a:b])

        self.timeline.layer(self.step, l, experts, counts, run_expert)
        out = self._combine(x, outs, rows, spans)
        self.timeline.mark_moe_done()
        return out

    # --- decode driver ------------------------------------------------------------------------------
    @torch.inference_mode()
    def prefill(self, ids):
        """ids: int tensor [B, T] on the GPU. Returns (KV cache, logits of every position)."""
        from transformers import DynamicCache
        if self.link is not None:
            self.link.drain()
        self.mode = "prefill"
        kv = DynamicCache()
        out = self.model.model(input_ids=ids, past_key_values=kv, use_cache=True)
        return kv, out.last_hidden_state

    @torch.inference_mode()
    def run(self, wl, needed, stream, cfg: SimConfig, docs, ctx, energy=None, record_logits=False,
            progress=True):
        """Teacher-forced decode of every group of wl; returns (results dict, live routing, predictions).

        docs: int32 [R, 2048] input ids of every trace request (file line = request id).
        """
        tl = RealTimeline(wl, needed, stream, cfg, self)
        self.timeline = tl
        S, B, T = wl.S, wl.B, wl.tokens_per_request
        self.record = np.zeros((S, B, self.L, self.k), np.int16)
        self.record_logits = np.zeros((S, B, self.L, self.N), np.float32) if record_logits else None
        preds = np.zeros((S, B), np.int64)
        prefill_last = []
        e_total = 0.0
        lm = self.model.lm_head
        t0 = time.time()
        for g, g0 in enumerate(range(0, S, T)):
            req = wl.requests[g]
            ids = torch.as_tensor(docs[req], dtype=torch.long, device=self.device)   # [B, 2048]
            if self.hs is not None:
                self.hs.begin_group(g0)
                self.hs.B_hint = B
            kv, hid = self.prefill(ids[:, :ctx])
            prefill_last.append(lm(hid[:, -1]).argmax(-1).cpu().numpy())
            del hid
            torch.cuda.synchronize()
            self.mode = "decode"
            e0 = energy.read() if energy else None
            step_preds = []
            for t in range(T):
                s = g0 + t
                self.step = s
                ev0, ev1 = _ev(), _ev()
                ev0.record()
                pos = torch.tensor([ctx + t], device=self.device)
                out = self.model.model(input_ids=ids[:, ctx + t:ctx + t + 1], past_key_values=kv,
                                       use_cache=True, cache_position=pos)
                step_preds.append(lm(out.last_hidden_state[:, -1]).argmax(-1))
                ev1.record()
                tl.end_step(s, ev0, ev1)
                if self.hs is not None:
                    self.hs.end_step(s)
            if energy:
                e_total += energy.read() - e0
            preds[g0:g0 + T] = torch.stack(step_preds).cpu().numpy()
            self.mode = "prefill"
            del kv
            if progress:
                done = g0 + T
                print(f"    group {g + 1}/{S // T}: {tl.step_ms[g0:done].mean():.1f} ms/token, "
                      f"{time.time() - t0:.0f}s", flush=True)
        res = tl.results(e_total if energy else None)
        self.timeline = None
        live = self.record
        self.record = None
        logits = self.record_logits
        self.record_logits = None
        return res, live, dict(decode=preds, prefill_last=np.stack(prefill_last), logits=logits)

    def close(self):
        if self.link is not None:
            self.link.stop()
