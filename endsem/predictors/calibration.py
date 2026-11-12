"""Rank calibration: turn any predictor's ranking into probabilities, online and training-free.

The three predictors score experts on unrelated scales: temporal votes are peaky (2-4 experts get
all the votes), cross-layer counters are flat, popularity is nearly uniform on real Mixtral. Mixing
their normalised scores lets the peakiest signal dominate whatever its accuracy (first version:
the mix scored below cross-layer alone).

Instead, for every predictor and layer we count how often the expert it ranked r-th was actually
routed: p(r) = hits[r] / count[r], with a weak prior of k/N. Tied scores share the best rank, so
"no votes" is one bin with its own learned probability. These are the same kind of online counters
as the predictors themselves; nothing is trained offline.
"""
import numpy as np


def competition_rank(scores):
    """[B, N] scores -> [B, N] int ranks (0 = best); equal scores share the best position."""
    B, N = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")
    sv = np.take_along_axis(scores, order, axis=1)
    new = np.ones((B, N), bool)
    new[:, 1:] = sv[:, 1:] != sv[:, :-1]
    first = np.maximum.accumulate(np.where(new, np.arange(N), 0), axis=1)
    ranks = np.empty((B, N), np.int64)
    np.put_along_axis(ranks, order, first, axis=1)
    return ranks


class RankCalibrator:
    def __init__(self, n_tables, N, k, prior_weight=4.0, halve_at=2048):
        self.N = N
        self.prior = k / N
        self.prior_weight = prior_weight
        self.halve_at = halve_at
        self.hits = np.zeros((n_tables, N))
        self.counts = np.zeros((n_tables, N))

    def table(self, i):
        return (self.hits[i] + self.prior_weight * self.prior) / (self.counts[i] + self.prior_weight)

    def prob(self, i, ranks):
        return self.table(i)[ranks]

    def update(self, i, ranks, member):
        self.counts[i] += np.bincount(ranks.ravel(), minlength=self.N)
        self.hits[i] += np.bincount(ranks[member], minlength=self.N)
        if self.counts[i, 0] > self.halve_at:      # age old evidence so the table can drift
            self.counts[i] *= 0.5
            self.hits[i] *= 0.5
