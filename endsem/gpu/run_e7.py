"""Run experiments/e7_hidden_state.py with one fix, without editing it (gpu/ is the only folder we commit).

e7's workload_from() builds warm-up rows for a single request group (warm shape [1, B, 1, L, k]), but the
hs_pred files hold several groups of tokens_per_request = 128 steps (GPU.md: batch 1 = 4 groups, batch 4 = 2),
so the ID-only stream builder indexes warm[1] and fails. The fix gives every group an empty (-1) warm-up row,
which is what e7 already does for the first group. Upstream one-line fix in workload_from:
    warm=np.full((S // tokens_per_request, B, 1, L, k), -1, np.int16)

    python -m gpu.run_e7 --gpu H100 --calibration gpu/results/calibration_h100.json
"""
import sys

import numpy as np

from experiments import e7_hidden_state as e7
from traces.workload import Workload


def workload_from(A, tokens_per_request, model="mixtral", num_experts=8):
    S, B, L, k = A.shape
    gs = np.zeros(S, bool)
    gs[::tokens_per_request] = True
    G = -(-S // tokens_per_request)
    return Workload(model=model, source="live:H100", num_experts=num_experts, top_k=k, A=A.astype(np.int16),
                    group_start=gs, warm=np.full((G, B, 1, L, k), -1, np.int16), tokens_per_request=tokens_per_request)


if __name__ == "__main__":
    e7.workload_from = workload_from
    e7.main(sys.argv[1:])
