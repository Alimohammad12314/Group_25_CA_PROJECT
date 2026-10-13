"""Figures for Q2 (expert loading vs execution) and Q3 (why loading is the bottleneck)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import CAT, INK, INK2, color_map, plt, save  # noqa: E402
from simrun import RESULTS  # noqa: E402

Q2 = os.path.join(RESULTS, "q2_offload_load_vs_exec")
Q3 = os.path.join(RESULTS, "q3_moe_flow_bottleneck")
P2, P3 = os.path.join(Q2, "plots"), os.path.join(Q3, "plots")
os.makedirs(P2, exist_ok=True)
os.makedirs(P3, exist_ok=True)
os.makedirs(os.path.join(Q3, "data"), exist_ok=True)
s = pd.read_csv(os.path.join(Q2, "data", "q2_summary.csv"))
abl = pd.read_csv(os.path.join(Q2, "data", "q2_ablation_mixtral_H100.csv"))
MODELS = ["mixtral", "grok1", "llama4_maverick", "deepseekR1"]
LINKS = ["PCIe4-x16", "PCIe5-x16", "C2C-450", "C2C-900"]
LCOL = color_map(LINKS)
GPU_SPEC = {"A100": dict(flops=312e12, hbm=2.039e12), "H100": dict(flops=989.4e12, hbm=3.352e12)}

for gpu in ("A100", "H100"):
    g = s[s.gpu == gpu]
    # F1: load vs exec time per decode step
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, m in zip(axs, MODELS):
        d = g[g.model == m]
        for ln in LINKS:
            x = d[d.link == ln]
            ax.plot(x.batch, x.load_ms, marker="o", color=LCOL[ln], label=f"load, {ln}")
        e = d[d.link == "PCIe4-x16"]
        ax.plot(e.batch, e.exec_ms, marker="s", color=INK, ls="--", label="execute (experts in HBM)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{m}  ({e.N_experts.iloc[0]} experts, top-{e.top_k.iloc[0]}, "
                     f"{e.expert_MB.iloc[0]:.0f} MB each)", fontsize=9)
        ax.set_xlabel("batch size")
    axs[0].set_ylabel("time per decode step (ms)")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f"{gpu}: routed-expert loading from off-chip DRAM vs expert execution", y=1.02)
    save(fig, os.path.join(P2, f"q2_fig1_load_vs_exec_{gpu}.png"))

    # F2: ratio load / exec
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
    for ax, m in zip(axs, MODELS):
        d = g[g.model == m]
        for ln in LINKS:
            x = d[d.link == ln]
            ax.plot(x.batch, x.load_over_exec, marker="o", color=LCOL[ln], label=ln)
        ax.axhline(1, color=INK2, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(m)
        ax.set_xlabel("batch size")
    axs[0].set_ylabel("loading time / execution time")
    axs[0].legend(fontsize=8)
    save(fig, os.path.join(P2, f"q2_fig2_load_exec_ratio_{gpu}.png"))

# F3: unique experts per layer: simulator vs uniform-routing formula
g = s[(s.gpu == "A100") & (s.link == "PCIe4-x16")]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
mc = color_map(MODELS)
for m in MODELS:
    d = g[g.model == m]
    axs[0].plot(d.batch, d.unique_experts_per_layer_sim / d.N_experts, marker="o", color=mc[m], label=m)
    bb = np.logspace(0, np.log2(256), 60, base=2)
    N, k = d.N_experts.iloc[0], d.top_k.iloc[0]
    axs[0].plot(bb, 1 - (1 - k / N) ** bb, color=mc[m], ls=":", lw=1.5)
    axs[1].plot(d.batch, d.offchip_GB_per_step, marker="o", color=mc[m], label=m)
    axs[1].axhline(d.all_experts_GB.iloc[0], color=mc[m], ls=":", lw=1)
for ax in axs:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
axs[0].set_ylabel("fraction of experts loaded per layer")
axs[0].set_title("markers: simulator, dotted: 1-(1-k/N)^B (uniform routing)", fontsize=9)
axs[1].set_yscale("log")
axs[1].set_ylabel("off-chip bytes per decode step (GB)")
axs[1].set_title("dotted: size of all routed experts", fontsize=9)
axs[0].legend(fontsize=8)
save(fig, os.path.join(P2, "q2_fig3_unique_experts_and_traffic.png"))

# F4: decode-step latency breakdown on-demand (A100), batch 1 and 64
for b in (1, 64):
    g = s[(s.gpu == "A100") & (s.batch == b)]
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, m in zip(axs, MODELS):
        d = g[g.model == m].set_index("link").loc[LINKS]
        labels = ["HBM\n(ideal)"] + [ln.replace("-", "\n") for ln in LINKS]
        ne = np.r_[d.non_expert_ms.iloc[0], d.non_expert_ms.values]
        ex = np.r_[d.exec_ms.iloc[0], d.exec_ms.values]
        st = np.r_[0, d.exposed_stall_ms.values]
        ax.bar(labels, ne, color=CAT[0], label="non-expert (attention, projections)", width=0.65)
        ax.bar(labels, ex, bottom=ne, color=CAT[2], label="expert execution", width=0.65)
        ax.bar(labels, st, bottom=ne + ex, color=CAT[1], label="exposed expert-load stall", width=0.65)
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5 * ne.min())
        ax.set_title(m)
        for i, (a, bb, c) in enumerate(zip(ne, ex, st)):
            ax.text(i, a + bb + c, f"{a + bb + c:.0f}" if a + bb + c >= 10 else f"{a + bb + c:.1f}",
                    ha="center", va="bottom", fontsize=7, color=INK2)
    axs[0].set_ylabel("decode-step latency (ms, log)")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"A100, batch {b}: on-demand expert loading (no overlap)", y=1.02)
    save(fig, os.path.join(P2, f"q2_fig4_step_breakdown_A100_b{b}.png"))

# F5: ablation from Duplex to single-GPU off-chip (H100, Mixtral)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.2), sharey=True)
for ax, b in zip(axs, (1, 16, 64)):
    d = abl[abl.batch == b].reset_index(drop=True)
    y = np.arange(len(d))
    ax.scatter(d.step_ms, y, color=[CAT[1] if "off-chip" in sname else CAT[0] for sname in d.system], s=50, zorder=3)
    ax.hlines(y, 0.1, d.step_ms, color="#c3c2b7", lw=1)
    for yi, v in zip(y, d.step_ms):
        ax.text(v * 1.15, yi, f"{v:.2f} ms", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(d.system, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1, 20000)
    ax.set_title(f"Mixtral, H100, batch {b}")
    ax.set_xlabel("decode-step latency (ms, log)")
axs[0].invert_yaxis()
save(fig, os.path.join(P2, "q2_fig5_ablation_duplex_to_offload.png"))

# F6: energy per token, A100
for b in (1, 64):
    g = s[(s.gpu == "A100") & (s.batch == b)]
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, m in zip(axs, MODELS):
        d = g[g.model == m].set_index("link").loc[LINKS]
        labels = ["HBM\n(ideal)"] + [ln.replace("-", "\n") for ln in LINKS]
        dyn = np.r_[d.energy_dynamic_base_J.iloc[0], d.energy_dynamic_base_J.values] / b
        bg_base = d.energy_base_J_per_token.iloc[0] - d.energy_dynamic_base_J.iloc[0] / b
        xfer = np.r_[0, d.energy_offload_transfer_J.values] / b
        bg = np.r_[bg_base, d.energy_ondemand_J_per_token.values - (d.energy_dynamic_base_J.values + d.energy_offload_transfer_J.values) / b]
        ax.bar(labels, dyn, color=CAT[0], label="dynamic: HBM + compute", width=0.65)
        ax.bar(labels, xfer, bottom=dyn, color=CAT[1], label="off-chip DRAM read + HBM write", width=0.65)
        ax.bar(labels, bg, bottom=dyn + xfer, color=CAT[3], label="background (static) x latency", width=0.65)
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5 * dyn.min())
        ax.set_title(m)
    axs[0].set_ylabel("energy per token (J, log)")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"A100, batch {b}: energy per generated token", y=1.02)
    save(fig, os.path.join(P2, f"q2_fig6_energy_A100_b{b}.png"))

# ---------------- Q3 ----------------
# F1: per-layer timeline of one MoE decoder layer (Mixtral, A100, batch 1)
d = s[(s.gpu == "A100") & (s.model == "mixtral") & (s.batch == 1)].set_index("link")
L = 32
ne = d.non_expert_ms.iloc[0] / L
ex = d.exec_ms.iloc[0] / L
fig, ax = plt.subplots(figsize=(13, 3.4))
rows = [("experts resident in HBM", 0), ("on-demand, C2C-900", d.loc["C2C-900", "load_per_layer_ms"]),
        ("on-demand, PCIe5-x16", d.loc["PCIe5-x16", "load_per_layer_ms"]),
        ("on-demand, PCIe4-x16", d.loc["PCIe4-x16", "load_per_layer_ms"])]
for i, (name, ld) in enumerate(rows):
    ax.barh(i, ne, left=0, color=CAT[0], height=0.55, label="attention + router" if i == 0 else None)
    ax.barh(i, ld, left=ne, color=CAT[1], height=0.55, label="load top-2 experts over link (stall)" if i == 3 else None)
    ax.barh(i, ex, left=ne + ld, color=CAT[2], height=0.55, label="expert FFN compute" if i == 0 else None)
    ax.text(ne + ld + ex, i, f"  {ne + ld + ex:.2f} ms/layer", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.invert_yaxis()
ax.set_xscale("symlog", linthresh=0.5)
ax.set_xlabel("time within one MoE decoder layer (ms, symlog)")
ax.set_title("Mixtral on A100, batch 1: the router decides the experts, then the GPU must wait for them")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
save(fig, os.path.join(P3, "q3_fig1_layer_timeline_mixtral_A100_b1.png"))

# F2: roofline-style explanation: load/exec ratio vs tokens per active expert
fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
mk = {"mixtral": "o", "grok1": "s", "llama4_maverick": "^", "deepseekR1": "D"}
roof = []
for ax, gpu in zip(axs, ("A100", "H100")):
    spec = GPU_SPEC[gpu]
    t = np.logspace(0, 4.3, 200)
    for ln, bw in zip(LINKS, (32e9, 64e9, 450e9, 900e9)):
        hbm_eff = spec["hbm"] - bw
        exec_per_byte = np.maximum(1 / spec["hbm"], t / spec["flops"])   # s per weight byte, FP16: 2*t*P flops / 2P bytes
        ratio = (1 / bw) / exec_per_byte
        ax.plot(t, ratio, color=LCOL[ln], label=f"{ln}: break-even t* = {spec['flops'] / bw:,.0f}")
        roof.append(dict(gpu=gpu, link=ln, link_GBps=bw / 1e9, ridge_tokens=spec["flops"] / spec["hbm"],
                         break_even_tokens_per_expert=spec["flops"] / bw,
                         ratio_at_1_token=spec["hbm"] / bw))
        g = s[(s.gpu == gpu) & (s.link == ln)]
        for m in MODELS:
            x = g[g.model == m]
            ax.scatter(x.tokens_per_active_expert, x.load_over_exec, marker=mk[m], color=LCOL[ln],
                       edgecolor="white", s=40, zorder=3)
    ax.axhline(1, color=INK2, lw=1)
    ax.axvline(spec["flops"] / spec["hbm"], color=INK2, ls=":", lw=1)
    ax.text(spec["flops"] / spec["hbm"], 0.02, " HBM ridge", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens per active expert (= arithmetic intensity, FLOP/B, FP16)")
    ax.set_title(f"{gpu}: lines analytic, markers simulator (o mixtral, s grok1, ^ maverick, D deepseek)", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left")
axs[0].set_ylabel("expert loading time / execution time")
save(fig, os.path.join(P3, "q3_fig2_roofline_load_vs_exec.png"))
pd.DataFrame(roof).to_csv(os.path.join(Q3, "data", "q3_break_even_table.csv"), index=False)

# F3: capacity view: expert weights vs 80 GB HBM, and bytes needed per token at batch 1
g = s[(s.gpu == "A100") & (s.link == "PCIe4-x16")]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
a = g[g.batch == 1].set_index("model").loc[MODELS]
axs[0].bar(MODELS, a.all_experts_GB, color=CAT[0], width=0.6)
axs[0].axhline(80, color=CAT[7], lw=1.5)
axs[0].text(3.3, 80, " 80 GB HBM", color=CAT[7], va="bottom", fontsize=8)
axs[0].set_yscale("log")
axs[0].set_ylabel("all routed-expert weights, FP16 (GB)")
axs[0].set_title("Routed experts do not fit in one GPU's HBM")
for b, c in zip((1, 16, 256), (CAT[1], CAT[2], CAT[6])):
    x = g[g.batch == b].set_index("model").loc[MODELS]
    axs[1].plot(MODELS, x.offchip_GB_per_token, marker="o", color=c, label=f"batch {b}")
axs[1].set_yscale("log")
axs[1].set_ylabel("off-chip bytes per generated token (GB)")
axs[1].set_title("Bytes that must cross the link per token (on-demand)")
axs[1].legend(fontsize=8)
save(fig, os.path.join(P3, "q3_fig3_capacity_and_bytes_per_token.png"))
