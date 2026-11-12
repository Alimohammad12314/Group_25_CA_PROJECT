"""Cross-layer predictor (ST-MoE correlation table; distance d from ProMoE and FineMoE).

One table per (distance d, source layer l): counter[e_src, e_dst] counts how often expert
e_dst was routed at layer l+d for a token that used e_src at layer l. The score of e_dst for a
new token is the sum of counters over the token's experts at layer l.

Counters are `bits` wide. When an increment would overflow, the whole row is halved: this keeps
relative frequencies (an LFU with aging) instead of a plain saturating counter, which with
2 bits loses the ranking among frequent pairs. `bits` is an ablation knob (2, 4, 8).
Tables are updated with the router's real choice after every layer, so nothing is trained.
"""
import numpy as np


class CrossLayerPredictor:
    def __init__(self, L, N, D, bits=4):
        self.L, self.N, self.D, self.bits = L, N, D, bits
        self.max = (1 << bits) - 1
        self.C = np.zeros((D + 1, L, N, N), np.uint16)

    def scores(self, l_src, d, src):
        """src [B, k] experts at layer l_src -> [B, N] scores for layer l_src + d."""
        table = self.C[d, l_src].astype(np.float32)
        valid = src >= 0
        rows = table[np.where(valid, src, 0)]                      # [B, k, N]
        rows *= valid[..., None]
        return rows.sum(axis=1)

    def update(self, l_src, d, src, dst):
        table = self.C[d, l_src]
        B, k = src.shape
        s = np.repeat(src, k, axis=1).ravel()                      # every (src slot, dst slot) pair
        t = np.tile(dst, (1, k)).ravel()
        ok = (s >= 0) & (t >= 0)
        np.add.at(table, (s[ok], t[ok]), 1)
        rows = np.unique(s[ok])
        over = rows[table[rows].max(axis=1) > self.max]
        while len(over):
            table[over] >>= 1
            over = over[table[over].max(axis=1) > self.max]

    def state_bits(self, distances=None):
        d = self.D if distances is None else distances
        return d * self.L * self.N * self.N * self.bits
