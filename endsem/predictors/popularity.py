"""Popularity prior (PopFetcher sliding window [README ref 5]; classic LFU).

Per layer, the activation count of each expert over the last `window` decode steps, summed over
the batch. The same score is given to every sequence.
"""
import numpy as np


class PopularityPredictor:
    def __init__(self, L, N, window=64):
        self.L, self.N, self.window = L, N, window
        self.ring = np.zeros((L, window, N), np.int32)
        self.pos = np.zeros(L, int)
        self.counts = np.zeros((L, N), np.int64)

    def push(self, l, experts):
        e = experts[experts >= 0]
        c = np.bincount(e, minlength=self.N)
        p = self.pos[l]
        self.counts[l] += c - self.ring[l, p]
        self.ring[l, p] = c
        self.pos[l] = (p + 1) % self.window

    def warm(self, warm):
        """Count a new group's prompt tokens. warm [B, P, L, k]."""
        for p in range(warm.shape[1]):
            for l in range(self.L):
                self.push(l, warm[:, p, l])

    def scores(self, l):
        return self.counts[l].astype(np.float64)

    def state_bits(self):
        return self.L * self.window * self.N * 8 + self.L * self.N * 16
