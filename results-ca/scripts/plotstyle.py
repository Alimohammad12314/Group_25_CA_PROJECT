"""Shared matplotlib style for all report figures (light mode, print friendly)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# validated categorical order (dataviz reference palette, light mode) - never cycled
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = "Blues"        # sequential: one hue, light -> dark
DIV = "RdBu_r"       # diverging: two hues, neutral midpoint

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 180, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def color_map(keys):
    """Fixed entity -> colour mapping (colour follows the entity, not its rank)."""
    return {k: CAT[i] for i, k in enumerate(keys)}


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
