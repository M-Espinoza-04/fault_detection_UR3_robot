"""Shared matplotlib style: one palette, recessive grid, no dual axes."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

# categorical slots, always used in this order (colour follows the entity)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
GRAY = "#9c9c9c"
INK = "#191919"
INK2 = "#666666"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SET_COLORS = {"train": BLUE, "val": ORANGE, "test": AQUA, "embargo": GRAY}
DIVERGING = "RdBu_r"      # blue <-> red, neutral midpoint
SEQUENTIAL = "Blues"


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelcolor": INK, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def save(fig, name):
    path = C.FIG / name
    fig.savefig(path)
    plt.close(fig)
    return path
