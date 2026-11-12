"""Draw the end-semester architecture as endsem/architecture_v3.png (paper-style block diagram).

    python -m experiments.architecture_diagram

Top row = control path (predictions), bottom row = data path (expert weights). Numbered steps are
explained in the figure caption in the README; the source paper of each idea is in the README table.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .common import ROOT

OUT = os.path.join(ROOT, "architecture_v3.png")
INK, INK2, EDGE, GROUP = "#1d1d1b", "#55544f", "#3d3d3a", "#f3f3f1"
CTRL = "#2a78d6"      # control path (predictions)
DATA = "#6b6a65"      # data path (expert weights)
plt.rcParams["font.family"] = "DejaVu Sans"


def rbox(ax, x, y, w, h, fill="white", edge=EDGE, lw=1.2, r=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))


def label(ax, x, y, s, size=10, weight="normal", color=INK, ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, linespacing=1.25)


def step(ax, x, y, n, color=INK):
    ax.add_patch(plt.Circle((x, y), 1.25, facecolor=color, edgecolor="none", zorder=6))
    ax.text(x, y - 0.05, str(n), fontsize=8.5, fontweight="bold", color="white", ha="center", va="center", zorder=7)


def arrow(ax, p, q, color, lw=1.4, head=10, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=z))


def experts_icon(ax, x, y, cols=4, rows=3, w=2.2, h=1.5, gap=0.55, fill="#d9d8d3"):
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x + j * (w + gap), y + i * (h + gap)), w, h, facecolor=fill,
                                   edgecolor="none", zorder=3))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(-17, 46)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- data path (bottom row) ---------------------------------------------------------------
    rbox(ax, 1, 2, 18, 20, fill=GROUP, edge="#bdbcb6", lw=1.0, r=1.2, z=1)
    label(ax, 10, 19.8, "Host DRAM", 10.5, "bold")
    experts_icon(ax, 4.0, 8.4)
    label(ax, 10, 5.4, "all routed experts", 8.5, color=INK2)

    rbox(ax, 33, 2, 66, 20, fill=GROUP, edge="#bdbcb6", lw=1.0, r=1.2, z=1)
    label(ax, 35, 19.8, "GPU", 10.5, "bold", ha="left")

    # link (step 5)
    arrow(ax, (19.4, 12), (32.6, 12), DATA, lw=5, head=18, z=3)
    step(ax, 21.8, 8.6, 5)
    label(ax, 27.4, 8.6, "PCIe / C2C", 8.8)
    label(ax, 26, 5.0, "demand misses first,\n16 MB chunks", 7.6, color=INK2)

    # HBM cache
    rbox(ax, 35, 4.5, 21, 13)
    label(ax, 45.5, 15.4, "HBM expert cache", 9.5, "bold")
    experts_icon(ax, 38.6, 9.2, cols=5, rows=2, w=2.2, h=1.4, gap=0.6, fill="#c7dcf5")
    label(ax, 45.5, 6.6, "keeps experts needed\nin the next 4 tokens", 7.8, color=INK2)

    # compute (step 1 = router, step 7 = experts run on HBM weights)
    rbox(ax, 61, 4.5, 36, 13)
    label(ax, 62.8, 15.6, "Compute", 9.5, "bold", ha="left")
    for x, name in ((62.8, "Attention"), (74.4, "Router"), (86.0, "Experts")):
        rbox(ax, x, 6.6, 9.6, 5.6, lw=1.0, r=0.6)
        label(ax, x + 4.8, 9.4, name, 8.8)
    arrow(ax, (72.5, 9.4), (74.3, 9.4), EDGE, lw=1.1, head=8)
    arrow(ax, (84.1, 9.4), (85.9, 9.4), EDGE, lw=1.1, head=8)
    step(ax, 74.4, 12.2, 1)
    # weights from the cache into the Experts box (runs under the compute row)
    ax.plot([56.2, 58.6, 58.6, 90.8], [8.0, 8.0, 3.4, 3.4], color=DATA, lw=2.6, zorder=3, solid_capstyle="butt")
    arrow(ax, (90.8, 3.4), (90.8, 6.4), DATA, lw=2.6, head=11, z=3)
    step(ax, 70.0, 3.4, 7)

    # ---- control path (top row) ---------------------------------------------------------------
    rbox(ax, 20, 27, 26, 16, edge=CTRL)
    label(ax, 33, 40.6, "Prefetcher", 10.5, "bold")
    label(ax, 33, 33.6, "lookahead  d ≤ 4 layers\nlink-time cap\nexpected-value gate", 8.6, color=INK2)

    rbox(ax, 52, 27, 47, 16, edge=CTRL)
    label(ax, 75.5, 40.6, "Predictor  (training-free)", 10.5, "bold")
    for x, name in ((55.0, "Temporal"), (69.6, "Cross-layer"), (84.2, "Popularity")):
        rbox(ax, x, 33.6, 13.0, 4.2, fill="#eef4fc", edge=CTRL, lw=1.0, r=0.6)
        label(ax, x + 6.5, 35.7, name, 8.8)
    rbox(ax, 55.0, 28.4, 42.2, 3.8, fill="#eef4fc", edge=CTRL, lw=1.0, r=0.6)
    label(ax, 76.1, 30.3, "calibrate  →  weight  →  batch union  →  budget top-(k+δ)", 8.4)

    # ---- control arrows, numbered in the order they happen ----------------------------------
    arrow(ax, (79.2, 12.4), (79.2, 26.8), CTRL)                      # 2 router -> predictor
    step(ax, 79.2, 22.0, 2)
    label(ax, 80.9, 22.0, "router's choice\n(history)", 7.8, color=CTRL, ha="left")
    arrow(ax, (51.8, 35.5), (46.2, 35.5), CTRL)                      # 3 predictor -> prefetcher
    step(ax, 49.0, 38.4, 3)
    arrow(ax, (26.0, 26.8), (26.0, 13.2), CTRL)                      # 4 prefetcher -> link
    step(ax, 26.0, 21.6, 4)
    label(ax, 24.3, 24.6, "prefetch", 7.8, color=CTRL, ha="right")
    arrow(ax, (57.0, 26.8), (51.0, 17.7), CTRL)                      # 6 predictor -> cache
    step(ax, 54.0, 22.25, 6)
    label(ax, 55.7, 22.0, "eviction\nscores", 7.8, color=CTRL, ha="left")

    # ---- legend -------------------------------------------------------------------------------
    arrow(ax, (1.5, 40.5), (5.5, 40.5), CTRL)
    label(ax, 6.4, 40.5, "control path", 8.4, color=INK2, ha="left")
    arrow(ax, (1.5, 36.5), (5.5, 36.5), DATA, lw=4, head=12)
    label(ax, 6.4, 36.5, "data path", 8.4, color=INK2, ha="left")

    # ---- key: what happens at each step, for one MoE layer --------------------------------------
    ax.plot([1, 99], [-0.6, -0.6], color="#d6d5cf", lw=0.8)
    label(ax, 1, -2.4, "One MoE layer, step by step", 9.5, "bold", ha="left")
    steps = [
        (1, "Router picks this layer's experts (replayed from the real trace)."),
        (2, "Its choice is sent to the predictor as history."),
        (3, "Predictor estimates P(expert) for layers l+1 … l+4."),
        (4, "Prefetcher requests the likely experts early."),
        (5, "Link copies weights DRAM → HBM; demand misses go first."),
        (6, "Predictor's scores decide what the HBM cache evicts."),
        (7, "Experts run on weights already in HBM; next layer starts at 1."),
    ]
    for i, (n, text) in enumerate(steps):
        col, row = (0, i) if i < 4 else (1, i - 4)
        x, y = 2.4 + col * 50, -5.4 - row * 2.9
        step(ax, x, y, n)
        label(ax, x + 2.0, y, text, 8.6, color=INK2, ha="left")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
