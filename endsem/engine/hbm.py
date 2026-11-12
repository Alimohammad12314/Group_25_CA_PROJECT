"""HBM expert cache: a bounded set of (layer, expert) slots.

Every slot is free, in flight (its transfer has not finished) or ready. A slot is pinned while it
is needed by the running layer or by a future layer a prefetch was issued for; pinned and
in-flight slots are never chosen as victims (except as a last resort for a demand miss).

Eviction policies (lowest score is evicted first)
  lru     least recently used or loaded
  lfu     least frequently used, then least recent (Lin et al. found LFU better than LRU)
  need    lowest predicted need at the expert's next possible use, discounted by how far away that
          use is, plus a bonus for the first layers (MoE-Infinity, FATE: early layers predict worst);
          ties broken by frequency. Scores come from the prediction stream (see timeline.py)
  belady  farthest actual next use (offline optimum; needs the future, upper bound only)

Scores are computed vectorised over all slots, so a policy costs the same at 64 or 4000 slots.
"""
import numpy as np

FREE, BUSY = 0, 1


class ExpertCache:
    def __init__(self, capacity, L, N):
        self.cap, self.L, self.N = capacity, L, N
        self.key = np.full(capacity, -1, np.int64)       # layer * N + expert
        self.state = np.zeros(capacity, np.int8)
        self.ready_at = np.full(capacity, np.inf)
        self.last_use = np.full(capacity, -1, np.int64)
        self.load_g = np.full(capacity, -1, np.int64)
        self.freq = np.zeros(capacity, np.int64)
        self.pin = np.full(capacity, -1, np.int64)        # pinned while current global layer <= pin
        self.prefetched = np.zeros(capacity, bool)
        self.used = np.zeros(capacity, bool)
        self.transfer = [None] * capacity
        self.index = {}
        self.free = list(range(capacity - 1, -1, -1))
        self.inflight = set()

    def __len__(self):
        return len(self.index)

    def insert(self, key, transfer, prefetched, pin_g, now_g):
        slot = self.free.pop()
        self.key[slot], self.state[slot] = key, BUSY
        self.ready_at[slot] = np.inf
        self.last_use[slot], self.load_g[slot], self.freq[slot] = -1, now_g, 0
        self.pin[slot], self.prefetched[slot], self.used[slot] = pin_g, prefetched, False
        self.transfer[slot] = transfer
        self.index[key] = slot
        self.inflight.add(slot)
        return slot

    def remove(self, slot):
        del self.index[int(self.key[slot])]
        self.key[slot], self.state[slot] = -1, FREE
        self.ready_at[slot], self.pin[slot] = np.inf, -1
        self.transfer[slot] = None
        self.inflight.discard(slot)
        self.free.append(slot)

    def refresh(self, now_t):
        """Mark finished transfers ready (lazily, only when a victim has to be chosen)."""
        done = []
        for slot in self.inflight:
            tr = self.transfer[slot]
            if tr is not None and tr.finish is not None and tr.finish <= now_t:
                self.ready_at[slot] = tr.finish
                done.append(slot)
        for slot in done:
            self.inflight.discard(slot)

    def evictable(self, now_g, now_t):
        return np.nonzero((self.state == BUSY) & (self.pin < now_g) & (self.ready_at <= now_t))[0]

    def fallback(self, now_g):
        """Last resort for a demand miss: speculative slots for future layers, not yet used."""
        return np.nonzero((self.state == BUSY) & self.prefetched & ~self.used & (self.pin >= now_g)
                          & (self.pin != now_g))[0]
