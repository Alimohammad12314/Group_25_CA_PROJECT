"""Chooser: weights the predictors per layer by their own recent hit rate.

Like a tournament branch predictor, but soft: weight_p = recall_p ** gamma, normalised, where
recall_p is an exponential moving average of predictor p's top-k recall at that (target layer,
distance). gamma = 1 is proportional weighting; a large gamma approaches picking the single best
predictor. 'equal' mode keeps fixed equal weights (the ablation rung before the chooser).
Weights start equal (all recalls initialised to the same value).
"""
import numpy as np

PREDICTORS = ("temporal", "cross", "pop")


class Chooser:
    def __init__(self, L, D, enabled, mode="adaptive", alpha=0.05, gamma=2.0):
        self.mode, self.alpha, self.gamma = mode, alpha, gamma
        self.enabled = [p in enabled for p in PREDICTORS]
        self.rec = [[[0.5, 0.5, 0.5] for _ in range(D + 1)] for _ in range(L)]

    def weights(self, l, d):
        w = []
        for i, on in enumerate(self.enabled):
            if not on or (i == 1 and d == 0):     # no cross-layer signal at a token boundary
                w.append(0.0)
            elif self.mode == "equal":
                w.append(1.0)
            else:
                w.append(max(self.rec[l][d][i], 1e-3) ** self.gamma)
        s = sum(w)
        return np.array([x / s for x in w]) if s > 0 else np.zeros(3)

    def update(self, l, d, recalls):
        """recalls: 3 recalls, None where a predictor was not evaluated."""
        r = self.rec[l][d]
        for i, x in enumerate(recalls):
            if x is not None:
                r[i] += self.alpha * (x - r[i])
