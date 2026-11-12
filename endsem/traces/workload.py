"""Turn a trace into a decode workload: batches of requests stepping token by token.

A workload has S decode steps. At every step all B sequences of the batch produce one token,
so MoE layer l of step s needs the union of the B sequences' experts. Requests are run in
groups of B for T steps each; the next group starts when a group finishes (a new batch of
requests arrives at the server). Predictor tables persist across groups, per-sequence
history does not.
"""
from dataclasses import dataclass

import numpy as np

from .format import Trace


@dataclass
class Workload:
    model: str
    source: str
    num_experts: int
    top_k: int
    A: np.ndarray            # [S, B, L, k] int16 experts of every sequence at every step and layer
    group_start: np.ndarray  # [S] bool, True where a new group of requests starts
    warm: np.ndarray         # [G, B, P, L, k] prompt routing of each group (-1 padded)
    tokens_per_request: int

    @property
    def S(self):
        return self.A.shape[0]

    @property
    def B(self):
        return self.A.shape[1]

    @property
    def L(self):
        return self.A.shape[2]

    def group_of(self, s):
        return s // self.tokens_per_request

    def needed(self):
        """Per (s, l): sorted unique experts of the batch and how many tokens each serves."""
        out = []
        N = self.num_experts
        for s in range(self.S):
            row = []
            for l in range(self.L):
                counts = np.bincount(self.A[s, :, l, :].ravel(), minlength=N)
                experts = np.nonzero(counts)[0]
                row.append((experts, counts[experts]))
            out.append(row)
        return out


def make_workload(trace: Trace, batch: int, tokens: int, groups: int, seed: int = 0) -> Workload:
    eligible = np.nonzero(trace.dec_len >= tokens)[0]
    need = batch * groups
    if len(eligible) < need:
        raise ValueError(f"{trace.model}: need {need} requests with >= {tokens} decode tokens, "
                         f"trace has {len(eligible)}")
    rng = np.random.default_rng(seed)
    pick = rng.permutation(eligible)[:need].reshape(groups, batch)
    A = trace.decode[pick, :tokens]                      # [G, B, T, L, k]
    A = A.transpose(0, 2, 1, 3, 4).reshape(groups * tokens, batch, trace.num_layers, trace.top_k)
    group_start = np.zeros(groups * tokens, bool)
    group_start[::tokens] = True
    return Workload(model=trace.model, source=trace.source, num_experts=trace.num_experts,
                    top_k=trace.top_k, A=np.ascontiguousarray(A), group_start=group_start,
                    warm=trace.warm[pick], tokens_per_request=tokens)


def make_uniform_workload(model, num_experts, top_k, num_layers, batch, steps, seed=0) -> Workload:
    """LLMSimulator's default routing: k distinct experts, uniform and independent per token."""
    rng = np.random.default_rng(seed)
    keys = rng.random((steps, batch, num_layers, num_experts))
    A = np.argsort(keys, axis=-1)[..., :top_k].astype(np.int16)
    warm = np.full((1, batch, 1, num_layers, top_k), -1, np.int16)
    gs = np.zeros(steps, bool)
    gs[0] = True
    return Workload(model=model, source="synthetic:uniform", num_experts=num_experts, top_k=top_k,
                    A=A, group_start=gs, warm=warm, tokens_per_request=steps)
