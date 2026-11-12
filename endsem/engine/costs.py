"""Time and energy costs, taken from LLMSimulator (SSD-offloading copy) and the Q2 runs.

Nothing here is fitted to the prefetcher's results. Sources:

  model shapes        LLMSimulator src/model/model_config.h (expert = 3 matrices of hidden x d_expert, FP16)
  GPU peak / HBM BW   LLMSimulator src/hardware/hardware_config.h (A100: 312 TFLOPS, 2.039 TB/s; H100: 989.4, 3.352)
  link bandwidth      Q2 sweep: PCIe 4.0 32 GB/s, PCIe 5.0 64, NVLink-C2C 450 and 900 (decimal GB/s)
  non-expert time     Q2 base pass: non_expert_ms per decode step (attention, router, dense and shared FFN)
  expert exec time    LLMSimulator's roofline per linear layer: max(2mkn / peak, (mk + kn + mn) * 2 B / HBM BW),
                      scaled by one constant per (GPU, model) so batch 1 matches Q2's expert_ffn exactly
  contention          SSD-offloading: while the link copies into HBM, compute sees HBM BW - link BW
  energy              Q2: 9.2 pJ/bit DDR5 read + 4.2 pJ/bit HBM write per transferred bit, HBM read
                      4.2 pJ/bit per executed expert, compute energy per token-expert from Q2's moe_comp,
                      non-expert dynamic energy from Q2's base pass, background 70 W GPU+HBM + 57 W host DRAM
"""
import functools
import os

import numpy as np
import pandas as pd

Q2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results-ca",
                      "q2_offload_load_vs_exec", "data")

GPUS = {
    "A100": dict(peak_flops=312.0e12, hbm_bw=2.039e12, hbm_bytes=80e9),
    "H100": dict(peak_flops=989.4e12, hbm_bw=3.352e12, hbm_bytes=80e9),
}
LINKS = {"PCIe4": 32e9, "PCIe5": 64e9, "C2C-450": 450e9, "C2C-900": 900e9}
Q2_LINK_NAMES = {"PCIe4": "PCIe4-x16", "PCIe5": "PCIe5-x16", "C2C-450": "C2C-450", "C2C-900": "C2C-900"}

# hidden, expert intermediate, routed experts, top-k, MoE layers (LLMSimulator presets).
# total_params_B: published parameter counts, used only to estimate non-expert weights for the
# "what physically fits in 80 GB" annotation.  kv_bytes_per_token: K+V cache bytes per token, all layers.
MODELS = {
    "mixtral": dict(hidden=4096, d_expert=14336, N=8, k=2, moe_layers=32, total_params_B=46.7,
                    kv_bytes_per_token=2 * 8 * 128 * 32 * 2),
    "grok1": dict(hidden=6144, d_expert=32768, N=8, k=2, moe_layers=64, total_params_B=314.0,
                  kv_bytes_per_token=2 * 8 * 128 * 64 * 2),
    "llama4_maverick": dict(hidden=5120, d_expert=8192, N=128, k=1, moe_layers=24, total_params_B=400.0,
                            kv_bytes_per_token=2 * 8 * 128 * 48 * 2),
    "deepseekR1": dict(hidden=7168, d_expert=2048, N=256, k=8, moe_layers=57, total_params_B=671.0,
                       kv_bytes_per_token=(512 + 64) * 61 * 2),
}
PRECISION_BYTES = 2
E_LINK_J_PER_BIT = (9.2 + 4.2) * 1e-12
E_HBM_READ_J_PER_BIT = 4.2e-12
P_BG_GPU_W = 70.0
P_BG_HOST_W = 57.0
CONTEXT_TOKENS = 1024          # Q2 runs used context 1024


@functools.lru_cache(maxsize=None)
def _q2_tables():
    summary = pd.read_csv(os.path.join(Q2_DIR, "q2_summary.csv"))
    runs = pd.read_csv(os.path.join(Q2_DIR, "q2_runs_long.csv"))
    return summary, runs


def expert_bytes(model):
    m = MODELS[model]
    return 3 * m["hidden"] * m["d_expert"] * PRECISION_BYTES


def _roofline_expert_s(model, gpu, n_tokens):
    """LLMSimulator linear_impl.cpp roofline for gate, up and down projections of one expert."""
    m, g = MODELS[model], GPUS[gpu]
    h, d, b = m["hidden"], m["d_expert"], PRECISION_BYTES
    total = 0.0
    for (rows, inner, cols) in ((n_tokens, h, d), (n_tokens, h, d), (n_tokens, d, h)):
        compute = 2.0 * rows * inner * cols / g["peak_flops"]
        memory = (rows * inner + inner * cols + rows * cols) * b / g["hbm_bw"]
        total += max(compute, memory)
    return total


class CostModel:
    """All costs for one (GPU, model, link, batch) point. Times in seconds, energy in joules."""

    def __init__(self, gpu, model, link, batch):
        self.gpu, self.model, self.link, self.batch = gpu, model, link, batch
        m = MODELS[model]
        self.N, self.k, self.L = m["N"], m["k"], m["moe_layers"]
        self.expert_bytes = expert_bytes(model)
        self.link_bw = LINKS[link]
        self.hbm_bw = GPUS[gpu]["hbm_bw"]
        self.t_load_expert = self.expert_bytes / self.link_bw
        self.contention_factor = self.hbm_bw / (self.hbm_bw - self.link_bw)

        summary, runs = _q2_tables()
        rows = summary[(summary.gpu == gpu) & (summary.model == model) & (summary.link == Q2_LINK_NAMES[link])]
        self._q2 = rows.set_index("batch")
        base = runs[(runs.gpu == gpu) & (runs.model == model) & (runs["mode"] == "base")].set_index("batch")
        self.t_nonexpert_layer = self._interp(rows.batch.values, rows.non_expert_ms.values, batch) * 1e-3 / self.L

        # Calibrate the roofline to Q2's batch-1 expert time (one constant per GPU and model).
        b1 = rows[rows.batch == 1].iloc[0]
        q2_one_expert = b1.exec_ms * 1e-3 / (self.L * self.k)
        self.exec_scale = q2_one_expert / _roofline_expert_s(model, gpu, 1)
        max_tok = max(batch * self.k, 1)
        self._t_exec = np.array([0.0] + [self.exec_scale * _roofline_expert_s(model, gpu, n)
                                         for n in range(1, max_tok + 1)])

        # Energy: non-expert dynamic per step and compute energy per token-expert, from Q2's base pass.
        ne_cols = ["fc_dram", "fc_comp", "attn_dram", "attn_comp", "ffn_dram", "ffn_comp",
                   "shared_moe_hbm", "shared_moe_comp"]
        ne_nj = base[ne_cols].sum(axis=1)
        self.e_nonexpert_step = self._interp(ne_nj.index.values, ne_nj.values, batch) * 1e-9
        b1r = base.loc[1]
        self.e_comp_token_expert = b1r.moe_comp * 1e-9 / (self.L * self.k)
        self.e_hbm_expert = self.expert_bytes * 8 * E_HBM_READ_J_PER_BIT
        self.e_link_byte = 8 * E_LINK_J_PER_BIT

    @staticmethod
    def _interp(xs, ys, x):
        order = np.argsort(xs)
        return float(np.interp(np.log(x), np.log(np.asarray(xs, float)[order]), np.asarray(ys, float)[order]))

    def t_exec(self, n_tokens):
        return self._t_exec[n_tokens]

    def background_power(self, offloaded=True):
        return P_BG_GPU_W + (P_BG_HOST_W if offloaded else 0.0)

    def q2(self, column):
        """Q2 value at this point's batch (only defined for batches Q2 ran: 1, 4, 16, 64, 256)."""
        return float(self._q2.loc[self.batch, column])

    def physical_cache_ratio(self):
        """Share of all routed experts that fits in HBM after non-expert weights and the KV cache."""
        m = MODELS[self.model]
        all_experts = self.N * self.L * self.expert_bytes
        nonexpert = max(m["total_params_B"] * 1e9 * PRECISION_BYTES - all_experts, 0.0)
        kv = m["kv_bytes_per_token"] * CONTEXT_TOKENS * self.batch
        free = GPUS[self.gpu]["hbm_bytes"] - nonexpert - kv
        return float(np.clip(free / all_experts, 0.0, 1.0))
