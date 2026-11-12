"""Draw the end-semester architecture as endsem/architecture.png.

    python -m experiments.architecture_diagram

Boxes are grouped by role (predictor, prefetch + link, HBM cache, decode timeline, inputs/outputs);
the small grey line in each box names the paper the idea comes from (see the README references).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .common import ROOT

OUT = os.path.join(ROOT, "architecture.png")
SURFACE, INK, INK2, MUTED, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#77766f", "#c9c8c2"
FILL = {   # light tints of the reference categorical hues, one per role
    "input": "#f0efec", "pred": "#e3eefb", "prefetch": "#fde9e0", "cache": "#ddf3ea",
    "time": "#fdf1d6", "out": "#f0efec", "bonus": "#f6e6ee",
}
LINE = {"input": "#8a8984", "pred": "#2a78d6", "prefetch": "#eb6834", "cache": "#1baf7a",
        "time": "#c98500", "out": "#8a8984", "bonus": "#d55181"}


def box(ax, x, y, w, h, title, body=None, src=None, role="input", title_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
                                facecolor=FILL[role], edgecolor=LINE[role], linewidth=1.4))
    ty = y + h - 1.15
    ax.text(x + 0.9, ty, title, fontsize=title_size, fontweight="bold", color=INK, va="top", ha="left")
    if body:
        ax.text(x + 0.9, ty - 2.0, body, fontsize=8, color=INK2, va="top", ha="left", linespacing=1.3)
    if src:
        ax.text(x + w - 0.7, y + 0.6, src, fontsize=7.5, color=MUTED, va="bottom", ha="right", style="italic")
    return dict(x=x, y=y, w=w, h=h)


def anchor(b, side, frac=0.5):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    return {"top": (x + w * frac, y + h + 0.25), "bottom": (x + w * frac, y - 0.25),
            "left": (x - 0.25, y + h * frac), "right": (x + w + 0.25, y + h * frac)}[side]


def arrow(ax, p, q, label=None, color=INK2, dashed=False, rad=0.0, lx=0.0, ly=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=color,
                                 linestyle=(0, (4, 3)) if dashed else "solid",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=7.6, color=INK2, ha=ha,
                va="center", bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))


def group(ax, x, y, w, h, label, role):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="none", edgecolor=LINE[role], linewidth=1.0, linestyle=(0, (2, 2))))
    ax.text(x + 0.8, y + h + 0.5, label, fontsize=10, fontweight="bold", color=LINE[role], va="bottom")


def main():
    fig, ax = plt.subplots(figsize=(17.5, 12.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 109)
    ax.set_ylim(-2, 75)
    ax.axis("off")

    ax.text(1, 74.5, "End-semester architecture: training-free expert prediction, prefetching and HBM management",
            fontsize=14, fontweight="bold", color=INK, va="top")
    ax.text(1, 72.0, "All routed experts live in host DRAM; HBM holds a bounded expert cache. Grey italic = paper the idea "
            "comes from (numbers = README references). Routing is never changed: every routed expert is executed.",
            fontsize=9, color=INK2, va="top")

    # ---- inputs (top row) ---------------------------------------------------------------------
    traces = box(ax, 1, 57, 24, 10, "Real routing traces (Hugging Face)",
                 "Mixtral-8x7B: allenai/analysis_mixtral\nDeepSeek-R1, Llama-4: core12345/...\nGrok-1: synthetic, fitted to Mixtral",
                 "OLMoE [21]; Orders in Chaos [15]", "input")
    wl = box(ax, 29, 57, 22, 10, "Workload builder",
             "B requests decode together, step by step;\nprompt routing warms the predictors;\nneeded[s][l] = batch union of experts",
             None, "input")
    bonus = box(ax, 55, 57, 23, 10, "Bonus: LLMSimulator replay",
                "routing.model: trace replays the traces\ninside the simulator (100 % exact); its\nrouting log can replace the workload",
                "our simulator patch", "bonus")
    costs = box(ax, 82, 57, 26, 10, "Cost model",
                "LLMSimulator / SSD-offloading (Q2): expert\nload + execution, attention vs context,\ncontention, energy; optional H100\ncalibration (GPU.md)",
                "Duplex [18]; SSD-offloading [19]", "input")

    # ---- 1 predictor (left column) ------------------------------------------------------------
    group(ax, 0.6, 6.4, 31.0, 45.8, "1  Predictor  (training-free)", "pred")
    t = box(ax, 1.5, 42.2, 11.2, 8.8, "Temporal", "votes over the\nlast W tokens;\nW adapts", "STEP [23], ST-MoE [1]", "pred", 9)
    c = box(ax, 13.5, 42.2, 9.0, 8.8, "Cross-layer", "experts at l →\nl+d, 4-bit\ncounters", "ST-MoE [1]", "pred", 9)
    p = box(ax, 23.3, 42.2, 7.5, 8.8, "Popularity", "64-step\nwindow\nper layer", "PopFetcher [5]", "pred", 9)
    cal = box(ax, 1.5, 34.0, 29.3, 6.6, "Rank calibration", "each predictor's rank → probability (online)",
              "ours", "pred", 9)
    ch = box(ax, 1.5, 25.8, 29.3, 6.6, "Chooser", "weight = recent recall², per layer and distance",
             "tournament predictor [16]", "pred", 9)
    bu = box(ax, 1.5, 17.6, 29.3, 6.6, "Batch union", "P(e) = 1 − ∏_b (1 − q_b(e)) over the batch",
             "ExpertFlow [22], DAOP [9]", "pred", 9)
    bd = box(ax, 1.5, 7.4, 29.3, 8.6, "Adaptive budget", "M = top-(k + δ), δ from the empirical CDF;\nδ = 0 when recall < 40 % (off-switch)",
             "APEX [2], STEP [23]", "pred", 9)

    # ---- 3 prefetcher, link, 4 cache (middle column) ------------------------------------------
    group(ax, 35.2, 32.8, 34.6, 19.4, "3  Prefetcher  (after each router decision)", "prefetch")
    la = box(ax, 36.0, 42.2, 16.6, 8.8, "Lookahead d ≤ 4", "smallest d whose\ncompute covers the\npredicted load",
             "ProMoE [4], ExpertFlow [11]", "prefetch", 9)
    cap = box(ax, 53.4, 42.2, 15.6, 8.8, "Link-time cap", "only what the link\ncan deliver before\nlayer l + d",
              "SPICE [12]", "prefetch", 9)
    gate = box(ax, 36.0, 33.8, 33.0, 6.8, "Expected-value gate", "prefetch e only if p(e) ≥ c / (1 + c)",
               "ours (cost view: PreScope [10])", "prefetch", 9)
    link = box(ax, 36.0, 21.7, 33.0, 8.6, "DRAM → HBM link (PCIe 4/5, NVLink-C2C)",
               "demand first; a miss preempts a prefetch at the\nnext 16 MB chunk; stale prefetches dropped",
               "ProMoE [4], PreScope [10]", "prefetch", 9)
    group(ax, 35.2, 6.8, 34.6, 12.4, "4  HBM expert cache", "cache")
    cache = box(ax, 36.0, 7.8, 33.0, 10.4, "Predicted-need eviction",
                "evict the expert least likely to be used within\n4 tokens: 1 − (1 − p_next)(1 − usage rate)^4\nbaselines: LRU [7], LFU, Belady oracle [17]",
                "ours; MoE-Infinity [6], HybriMoE [8]", "cache", 9)

    # ---- 2 decode timeline (right column) -----------------------------------------------------
    group(ax, 72.6, 6.8, 31.2, 45.4, "2  Decode timeline  (per layer)", "time")
    t1 = box(ax, 73.4, 42.4, 29.6, 8.6, "① Non-expert compute",
             "attention, norms, router; slowed by HBM\ncontention while the link copies", "SSD-offloading [19]", "time", 9)
    t2 = box(ax, 73.4, 33.9, 29.6, 6.6, "② Router decides (from the trace)",
             "the batch needs experts U at this layer", None, "time", 9)
    t3 = box(ax, 73.4, 21.7, 29.6, 8.6, "③ HBM lookup for each expert in U",
             "hit: resident · late: in flight → promoted\nmiss: demand copy, evict a victim", None, "time", 9)
    t4 = box(ax, 73.4, 7.8, 29.6, 10.4, "④ Execute experts",
             "in arrival order while missing ones still\nload (miss overlap); nothing is skipped\nor substituted",
             "APEX [2]", "time", 9)

    # ---- outputs ------------------------------------------------------------------------------
    box(ax, 1, -1.6, 103, 6.4, "Outputs  (results/*.csv, docs/figures)",
        "coverage, precision/recall · timely / late / unused prefetches · HBM hit rate · extra traffic · exposed stall · "
        "expert-load latency · token latency (mean, p95) · energy (13.4 pJ/bit transfer + compute + background)",
        None, "out", 9.5)

    # ---- arrows -------------------------------------------------------------------------------
    arrow(ax, anchor(traces, "right"), anchor(wl, "left"))
    arrow(ax, anchor(bonus, "left"), anchor(wl, "right"), None, LINE["bonus"], dashed=True)
    arrow(ax, anchor(wl, "bottom", 0.2), (25.0, 52.5), "routing history", lx=-5.5, ly=0.4)
    arrow(ax, anchor(wl, "bottom", 0.8), (93.0, 52.5), "routed experts", rad=0.08, lx=8, ly=0.2)
    arrow(ax, anchor(costs, "bottom", 0.73), (101.0, 52.5), "time & energy per operation", lx=1.2, ha="left")

    for b in (t, c, p):
        arrow(ax, anchor(b, "bottom"), (b["x"] + b["w"] / 2, cal["y"] + cal["h"] + 0.25), color=LINE["pred"])
    for a, b in ((cal, ch), (ch, bu), (bu, bd)):
        arrow(ax, anchor(a, "bottom"), anchor(b, "top"), color=LINE["pred"])
    arrow(ax, anchor(t1, "bottom"), anchor(t2, "top"), color=LINE["time"])
    arrow(ax, anchor(t2, "bottom"), anchor(t3, "top"), color=LINE["time"])
    arrow(ax, anchor(t3, "bottom"), anchor(t4, "top"), color=LINE["time"])
    arrow(ax, anchor(t4, "right", 0.6), anchor(t1, "right", 0.5), "next layer,\nnext token", LINE["time"], rad=0.32,
          lx=3.2, ha="left")

    arrow(ax, anchor(bd, "right", 0.8), anchor(gate, "left", 0.4), "prediction\nstream", LINE["pred"], rad=0.25,
          lx=-0.2, ly=-2)
    arrow(ax, anchor(bd, "right", 0.35), anchor(cache, "left", 0.4), "eviction\nscores", LINE["pred"], lx=0.1, ly=1.6)
    arrow(ax, anchor(gate, "bottom", 0.5), anchor(link, "top", 0.5), "speculative copies", LINE["prefetch"], lx=7)
    arrow(ax, anchor(link, "bottom", 0.5), anchor(cache, "top", 0.5), "weights arrive", LINE["cache"], lx=6)
    arrow(ax, anchor(t2, "left", 0.5), anchor(gate, "right", 0.5), "trigger", LINE["prefetch"], ly=1.0)
    arrow(ax, anchor(t3, "left", 0.65), anchor(link, "right", 0.65), "demand miss", LINE["prefetch"], ly=1.1)
    arrow(ax, anchor(cache, "right", 0.7), anchor(t3, "left", 0.2), "resident?", LINE["cache"], ly=-1.3)
    arrow(ax, anchor(t4, "bottom", 0.5), (88.2, 5.0), "metrics", INK2, lx=3)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
