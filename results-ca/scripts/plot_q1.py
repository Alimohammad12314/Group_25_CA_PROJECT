"""Figures for Q1 (Duplex reproduction)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import CAT, INK2, color_map, plt, save  # noqa: E402
from simrun import RESULTS  # noqa: E402

Q = os.path.join(RESULTS, "q1_duplex_reproduction")
P = os.path.join(Q, "plots")
os.makedirs(P, exist_ok=True)
d = pd.read_csv(os.path.join(Q, "data", "q1_results.csv"))
SYS = ["GPU", "2xGPU", "Bank-PIM", "Duplex", "Duplex+PE", "Duplex+PE+ET"]
COL = color_map(SYS)


def grouped(ax, sub, systems, xkeys, xlabels, value):
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        vals = [sub[(sub.system == s) & (sub.batch == b) & (sub.ctx == c)][value].mean() for b, c in xkeys]
        ax.bar(np.arange(len(xkeys)) + (i - (len(systems) - 1) / 2) * w, vals, w * 0.92,
               color=COL[s], label=s)
    ax.set_xticks(np.arange(len(xkeys)))
    ax.set_xticklabels(xlabels)


# Fig 1: Mixtral normalized throughput (paper Fig. 11 grid)
sys11 = ["GPU", "2xGPU", "Duplex", "Duplex+PE", "Duplex+PE+ET"]
m = d[d.model == "mixtral"]
xk = [(b, c) for b in (32, 64, 128) for c in (256, 1024, 4096)]
fig, ax = plt.subplots(figsize=(11, 4))
grouped(ax, m, sys11, xk, [f"B={b}\nctx {c}" for b, c in xk], "speedup_vs_GPU")
ax.axhline(1, color=INK2, lw=1)
ax.set_ylabel("Throughput normalized to GPU (x)")
ax.set_title("Mixtral-8x7B decode throughput, 4x H100 vs Duplex variants (LLMSimulator, Ramulator on)")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
save(fig, os.path.join(P, "q1_fig1_mixtral_throughput_fig11.png"))

# Fig 2: all models, batch x system at ctx 1024
models = ["mixtral", "glam", "grok1", "llama3_70B"]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for ax, mod in zip(axs, models):
    sub = d[(d.model == mod) & (d.ctx == 1024)]
    systems = [s for s in sys11 if s in set(sub.system)]
    grouped(ax, sub, systems, [(b, 1024) for b in (32, 64, 128)], ["B=32", "B=64", "B=128"], "speedup_vs_GPU")
    ax.axhline(1, color=INK2, lw=1)
    ax.set_title(mod)
axs[0].set_ylabel("Throughput vs GPU (x), ctx 1024")
fig.legend(*axs[0].get_legend_handles_labels(), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.1))
save(fig, os.path.join(P, "q1_fig2_all_models_speedup.png"))

# Fig 3: latency breakdown of one decode step, Mixtral B=32/128 ctx 1024
parts = [("expert_ffn_ms", "MoE experts"), ("attention_ms", "Attention"), ("fc_ms", "QKV/O-proj"),
         ("communication_ms", "Communication"), ("other_ms", "Norm/residual")]
fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=False)
fig.subplots_adjust(wspace=0.4)
for ax, b in zip(axs, (32, 128)):
    sub = m[(m.batch == b) & (m.ctx == 1024)].set_index("system").loc[SYS]
    left = np.zeros(len(SYS))
    for i, (c, lab) in enumerate(parts):
        ax.barh(SYS, sub[c].values, left=left, color=CAT[i], label=lab, height=0.6,
                edgecolor="white", linewidth=1)
        left += sub[c].values
    for y, v in enumerate(sub.step_latency_ms.values):
        ax.text(v, y, f" {v:.2f} ms", va="center", fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("Decode-step latency (ms)")
    ax.set_title(f"Mixtral, batch {b}, ctx 1024")
fig.legend(*axs[0].get_legend_handles_labels(), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.08))
save(fig, os.path.join(P, "q1_fig3_latency_breakdown.png"))

# Fig 4: energy per token breakdown (paper Fig. 15)
eparts = [("e_moe_dram_J", "MoE DRAM"), ("e_moe_comp_J", "MoE compute"), ("e_attn_dram_J", "Attn DRAM"),
          ("e_attn_comp_J", "Attn compute"), ("e_fc_dram_J", "FC DRAM"), ("e_fc_comp_J", "FC compute")]
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for ax, b in zip(axs, (32, 64, 128)):
    sub = m[(m.batch == b) & (m.ctx == 1024)].set_index("system").loc[SYS]
    bottom = np.zeros(len(SYS))
    for i, (c, lab) in enumerate(eparts):
        v = sub[c].values / b * 1e3
        ax.bar(SYS, v, bottom=bottom, color=CAT[i], label=lab, width=0.65, edgecolor="white", linewidth=1)
        bottom += v
    ax.set_title(f"Mixtral, batch {b}")
    ax.tick_params(axis="x", rotation=35)
axs[0].set_ylabel("Dynamic energy per token (mJ)")
axs[-1].legend(fontsize=8)
save(fig, os.path.join(P, "q1_fig4_energy_breakdown_fig15.png"))

# Fig 5: Bank-PIM comparison (paper Fig. 14)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, mod in zip(axs, ("mixtral", "llama3_70B")):
    sub = d[d.model == mod]
    xk = [(b, c) for b in (32, 64) for c in (256, 4096)]
    grouped(ax, sub, ["GPU", "Bank-PIM", "Duplex"], xk, [f"B={b}\nctx {c}" for b, c in xk], "speedup_vs_GPU")
    ax.axhline(1, color=INK2, lw=1)
    ax.set_title(mod)
axs[0].set_ylabel("Throughput vs GPU (x)")
axs[0].legend()
save(fig, os.path.join(P, "q1_fig5_bankpim_fig14.png"))
