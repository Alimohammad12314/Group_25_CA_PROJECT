"""Draw the end-semester architecture as endsem/architecture_v2.png (paper-style block diagram).

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

OUT = os.path.join(ROOT, "architecture_v2.png")
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
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- data path (bottom row) ---------------------------------------------------------------
    rbox(ax, 1, 2, 18, 20, fill=GROUP, edge="#bdbcb6", lw=1.0, r=1.2, z=1)
    label(ax, 10, 19.8, "Host DRAM", 10.5, "bold")
    experts_icon(ax, 4.0, 8.4)
    label(ax, 10, 5.4, "all routed experts", 8.5, color=INK2)

    rbox(ax, 33, 2, 66, 20, fill=GROUP, edge="#bdbcb6", lw=1.0, r=1.2, z=1)
    label(ax, 35, 19.8, "GPU", 10.5, "bold", ha="left")

    # link
    arrow(ax, (19.4, 12), (32.6, 12), DATA, lw=5, head=18, z=3)
    step(ax, 21.8, 8.6, 4, DATA)
    label(ax, 27.4, 8.6, "PCIe / C2C", 8.8)
    label(ax, 26, 5.0, "demand misses first,\n16 MB chunks", 7.6, color=INK2)

    # HBM cache
    rbox(ax, 35, 4.5, 21, 13)
    step(ax, 37.2, 15.4, 5)
    label(ax, 46.8, 15.4, "HBM expert cache", 9.5, "bold")
    experts_icon(ax, 38.6, 9.2, cols=5, rows=2, w=2.2, h=1.4, gap=0.6, fill="#c7dcf5")
    label(ax, 45.5, 6.6, "keep experts needed\nin the next 4 tokens", 7.8, color=INK2)

    # compute
    arrow(ax, (56.3, 11), (60.7, 11), DATA, lw=3, head=12, z=3)
    rbox(ax, 61, 4.5, 36, 13)
    label(ax, 62.8, 15.6, "Compute", 9.5, "bold", ha="left")
    for x, name in ((62.8, "Attention"), (74.4, "Router"), (86.0, "Experts")):
        rbox(ax, x, 6.6, 9.6, 5.6, lw=1.0, r=0.6)
        label(ax, x + 4.8, 9.4, name, 8.8)
    arrow(ax, (72.5, 9.4), (74.3, 9.4), EDGE, lw=1.1, head=8)
    arrow(ax, (84.1, 9.4), (85.9, 9.4), EDGE, lw=1.1, head=8)
    step(ax, 74.4, 12.2, 1)
    step(ax, 86.0, 12.2, 6)

    # ---- control path (top row) ---------------------------------------------------------------
    rbox(ax, 20, 27, 26, 16, edge=CTRL)
    step(ax, 22.4, 40.6, 3, CTRL)
    label(ax, 34, 40.6, "Prefetcher", 10.5, "bold")
    label(ax, 33, 33.6, "lookahead  d ≤ 4 layers\nlink-time cap\nexpected-value gate", 8.6, color=INK2)

    rbox(ax, 52, 27, 47, 16, edge=CTRL)
    step(ax, 54.4, 40.6, 2, CTRL)
    label(ax, 75.5, 40.6, "Predictor  (training-free)", 10.5, "bold")
    for x, name in ((55.0, "Temporal"), (69.6, "Cross-layer"), (84.2, "Popularity")):
        rbox(ax, x, 33.6, 13.0, 4.2, fill="#eef4fc", edge=CTRL, lw=1.0, r=0.6)
        label(ax, x + 6.5, 35.7, name, 8.8)
    rbox(ax, 55.0, 28.4, 42.2, 3.8, fill="#eef4fc", edge=CTRL, lw=1.0, r=0.6)
    label(ax, 76.1, 30.3, "calibrate  →  weight  →  batch union  →  budget top-(k+δ)", 8.4)

    # ---- control arrows -----------------------------------------------------------------------
    arrow(ax, (81.8, 12.4), (81.8, 26.8), CTRL)
    label(ax, 82.6, 24.2, "routed experts", 7.8, color=CTRL, ha="left")
    arrow(ax, (51.8, 35.5), (46.2, 35.5), CTRL)
    label(ax, 49.0, 37.6, "P(expert)", 7.6, color=CTRL)
    arrow(ax, (26.0, 26.8), (26.0, 13.2), CTRL)
    label(ax, 26.8, 22.5, "prefetch", 7.8, color=CTRL, ha="left")
    arrow(ax, (57.0, 26.8), (51.0, 17.7), CTRL)
    label(ax, 56.6, 22.6, "eviction\nscores", 7.8, color=CTRL, ha="left")

    # ---- legend -------------------------------------------------------------------------------
    arrow(ax, (1.5, 40.5), (5.5, 40.5), CTRL)
    label(ax, 6.4, 40.5, "control path", 8.4, color=INK2, ha="left")
    arrow(ax, (1.5, 36.5), (5.5, 36.5), DATA, lw=4, head=12)
    label(ax, 6.4, 36.5, "data path", 8.4, color=INK2, ha="left")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
