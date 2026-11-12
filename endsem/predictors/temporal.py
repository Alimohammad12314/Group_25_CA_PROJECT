"""Temporal predictor (STEP's adaptive vote window [README ref 23]; ST-MoE's History Table is the W = 1 case).

Per sequence and layer, every expert gets one vote for each appearance in the last W tokens,
with a small bonus for recency so ties go to the newest token. W is adapted per layer:
it doubles after `patience` evaluation windows with hit rate above `hi`, and halves after
one window below `lo` (a stable layer can look further back; a drifting one should not).
"""
import numpy as np


class TemporalPredictor:
    def __init__(self, L, N, k, B, w_init=4, w_max=16, adaptive=True, hi=0.75, lo=0.40,
                 eval_tokens=16, patience=2):
        self.L, self.N, self.k, self.B = L, N, k, B
        self.w_max, self.adaptive, self.hi, self.lo = w_max, adaptive, hi, lo
        self.eval_tokens, self.patience = eval_tokens, patience
        self.W = np.full(L, min(w_init, w_max), int)
        self.hist = np.full((B, L, w_max, k), -1, np.int16)   # index -1 = newest token
        self._acc = np.zeros(L)
        self._n = np.zeros(L, int)
        self._streak = np.zeros(L, int)
        self._bidx = np.repeat(np.arange(B), 1)

    def reset(self, warm):
        """New requests: history = their last prompt tokens. warm [B, P, L, k] (-1 padded)."""
        self.hist.fill(-1)
        P = warm.shape[1]
        take = min(P, self.w_max)
        if take:
            self.hist[:, :, -take:, :] = warm[:, -take:].transpose(0, 2, 1, 3)

    def scores(self, l):
        """[B, N] vote counts for layer l."""
        W = self.W[l]
        h = self.hist[:, l, -W:, :]                                # [B, W, k]
        recency = 1.0 + 0.01 * np.arange(1, W + 1)                 # newest gets the largest weight
        weights = np.broadcast_to(recency[None, :, None], h.shape)
        out = np.zeros((self.B, self.N + 1))
        idx = np.where(h < 0, self.N, h)
        b = np.broadcast_to(np.arange(self.B)[:, None, None], h.shape)
        np.add.at(out, (b.ravel(), idx.ravel()), weights.ravel())
        return out[:, :self.N]

    def push(self, l, experts):
        """Record token's experts at layer l. experts [B, k]."""
        self.hist[:, l, :-1] = self.hist[:, l, 1:]
        self.hist[:, l, -1] = experts

    def feedback(self, l, recall):
        """Hit rate of this predictor's top-k at layer l for one token (mean over the batch)."""
        if not self.adaptive:
            return
        self._acc[l] += recall
        self._n[l] += 1
        if self._n[l] < self.eval_tokens:
            return
        rate = self._acc[l] / self._n[l]
        self._acc[l], self._n[l] = 0.0, 0
        if rate > self.hi:
            self._streak[l] += 1
            if self._streak[l] >= self.patience:
                self.W[l] = min(self.W[l] * 2, self.w_max)
                self._streak[l] = 0
        else:
            self._streak[l] = 0
            if rate < self.lo:
                self.W[l] = max(self.W[l] // 2, 1)

    def state_bits(self):
        """History of W_max tokens x k ids per layer and sequence, plus a W per layer."""
        id_bits = int(np.ceil(np.log2(self.N)))
        return self.B * self.L * self.w_max * self.k * id_bits + self.L * 5
