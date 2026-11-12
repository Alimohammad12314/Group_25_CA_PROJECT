"""The host-DRAM -> HBM link (PCIe or NVLink-C2C): one transfer at a time, in chunks.

Two modes
  fifo   one queue in arrival order: a demand miss waits behind every speculative transfer queued
         before it (the naive prefetcher)
  prio   two queues (ProMoE, PreScope): demand misses go first, and preempt a speculative transfer
         at its next chunk boundary; the preempted transfer resumes afterwards

The link is simulated lazily. New transfers only arrive at compute event boundaries, so the link
is advanced either up to "now" (stopping at a chunk boundary, so a demand arriving now can still
preempt) or until a specific transfer completes when compute must wait for it.
"""
import math
from collections import deque

DEMAND, SPEC = 0, 1


class Transfer:
    __slots__ = ("key", "nbytes", "prio", "arrival", "done", "finish", "target_g", "cancelled",
                 "was_spec", "start")

    def __init__(self, key, nbytes, prio, arrival, target_g):
        self.key, self.nbytes, self.prio, self.arrival = key, nbytes, prio, arrival
        self.done, self.finish, self.target_g = 0.0, None, target_g
        self.cancelled, self.was_spec, self.start = False, prio == SPEC, None


class Link:
    def __init__(self, bandwidth, chunk_bytes, mode="prio"):
        self.bw, self.chunk, self.mode = bandwidth, chunk_bytes, mode
        self.t = 0.0                    # time up to which link work has been scheduled
        self.qd, self.qs = deque(), deque()
        self.pending_bytes = 0.0
        self.busy_time = 0.0
        self.bytes_moved = 0.0

    # --- queue management ------------------------------------------------------------------
    def enqueue(self, tr):
        q = self.qd if (self.mode == "prio" and tr.prio == DEMAND) else self.qs
        q.append(tr)
        self.pending_bytes += tr.nbytes - tr.done

    def promote(self, tr):
        """A queued speculative transfer is now needed by the running layer: make it a demand."""
        if tr.prio == DEMAND or tr.finish is not None:
            return
        tr.prio = DEMAND
        if self.mode != "prio":
            return
        self.qs.remove(tr)
        if tr.done > 0:
            self.qd.appendleft(tr)       # finish the partially moved expert first
        else:
            self.qd.append(tr)

    def cancel(self, tr):
        """Drop a speculative transfer; bytes already moved stay counted as moved (wasted)."""
        if tr.finish is not None or tr.cancelled:
            return
        tr.cancelled = True
        for q in (self.qd, self.qs):
            try:
                q.remove(tr)
                break
            except ValueError:
                pass
        self.pending_bytes -= tr.nbytes - tr.done

    # --- simulation ------------------------------------------------------------------------
    def _select(self):
        if self.qd and self.qd[0].arrival <= self.t + 1e-15:
            return self.qd[0], self.qd
        if self.qs and self.qs[0].arrival <= self.t + 1e-15:
            return self.qs[0], self.qs
        return None, None

    def _next_arrival(self):
        cands = [q[0].arrival for q in (self.qd, self.qs) if q]
        return min(cands) if cands else None

    def _advance(self, limit=None, until=None):
        while True:
            if until is not None and (until.finish is not None or until.cancelled):
                return
            tr, q = self._select()
            if tr is None:
                nxt = self._next_arrival()
                if nxt is None or (limit is not None and nxt >= limit):
                    return
                self.t = max(self.t, nxt)
                continue
            if limit is not None and self.t >= limit:
                return
            if tr.start is None:
                tr.start = self.t
            remaining = tr.nbytes - tr.done
            seg = remaining
            if tr.prio == SPEC and self.mode == "prio" and self.qd:
                seg = min(seg, self._to_boundary(tr, self.qd[0].arrival))
            if limit is not None:
                seg = min(seg, self._to_boundary(tr, limit))
            dt = seg / self.bw
            self.t += dt
            tr.done += seg
            self.busy_time += dt
            self.bytes_moved += seg
            self.pending_bytes -= seg
            if tr.done >= tr.nbytes - 1e-6:
                tr.finish = self.t
                q.popleft()

    def _to_boundary(self, tr, when):
        """Bytes of tr that run before stopping at the first chunk boundary at or after `when`."""
        if when <= self.t:
            ahead = tr.done
        else:
            ahead = tr.done + (when - self.t) * self.bw
        boundary = math.ceil(ahead / self.chunk - 1e-9) * self.chunk
        return max(min(boundary, tr.nbytes) - tr.done, min(self.chunk, tr.nbytes - tr.done))

    def run_until(self, t):
        self._advance(limit=t)

    def finish_time(self, tr):
        if tr.finish is None:
            self._advance(until=tr)
        return tr.finish

    def busy_until(self, t):
        """When the link drains if nothing else arrives (for the HBM contention model)."""
        if self.pending_bytes <= 1e-6:
            return min(self.t, t) if self.t <= t else self.t
        return max(self.t, t) + self.pending_bytes / self.bw
