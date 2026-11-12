"""Adaptive prefetch budget: how many experts to fetch for one layer (APEX top-(k+delta)).

APEX learns a CDF of how far down the predictor's ranking the routed experts fall. Here the CDF
is empirical and online: after each router result we record the oracle delta*, the smallest
extra budget that would have covered every needed expert:

    r*      = 1 + lowest rank (0-based) of any needed expert in the predicted ranking
    K_hat   = round(sum of predicted need probabilities)   (= k at batch 1; the expected union size above)
    delta*  = r* - K_hat

The budget for the next prediction of that layer is K_hat + the tau-quantile of the last `hist`
delta* values, so tau is the target full-coverage rate. When the layer's recent recall falls under
`off_threshold` the extra budget is set to zero (off-switch, our design): extra guesses from a predictor
that is mostly wrong are mostly wasted traffic. The link-time cap (SpecPrefetch) is applied later,
in the timeline, because it depends on how busy the link is.
"""
import math
from collections import deque


class AdaptiveBudget:
    def __init__(self, L, D, N, tau=0.75, hist=64, min_samples=8, off_threshold=0.40, alpha=0.05):
        self.L, self.D, self.N = L, D, N
        self.tau, self.min_samples, self.off_threshold, self.alpha = tau, min_samples, off_threshold, alpha
        self.deltas = [[deque(maxlen=hist) for _ in range(D + 1)] for _ in range(L)]
        self.recall = [[0.5] * (D + 1) for _ in range(L)]

    def size(self, l, d, p_sum):
        k_hat = min(max(int(round(p_sum)), 1), self.N)
        h = self.deltas[l][d]
        extra = 0
        if len(h) >= self.min_samples:
            ordered = sorted(h)
            extra = ordered[min(int(math.ceil(self.tau * len(ordered))) - 1, len(ordered) - 1)]
        if self.recall[l][d] < self.off_threshold:
            extra = min(extra, 0)
        return min(max(k_hat + extra, 1), self.N), k_hat

    def update(self, l, d, pos_of_needed, k_hat):
        """pos_of_needed: rank positions (0-based) of the routed experts in the predicted order."""
        r_star = int(pos_of_needed.max()) + 1
        self.deltas[l][d].append(r_star - k_hat)
        rec = float((pos_of_needed < k_hat).mean())
        self.recall[l][d] += self.alpha * (rec - self.recall[l][d])
        return r_star - k_hat
