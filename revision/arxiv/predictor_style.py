"""Shared visual encodings for the active Experiment 1 plots."""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Four colors from the Okabe–Ito colorblind-friendly palette.
COLORS = {
    "DiT": "#0072B2",
    "DPD5": "#E69F00",
    "DPD6": "#009E73",
    "Learned PLD": "#CC79A7",
}
LABELS = {"DiT": "MLP", "DPD5": "SCA", "DPD6": "SMLP", "Learned PLD": "LPLD"}
HEAD_MARKERS = {1: "o", 8: "^"}


def efficiency_legend(families):
    families = [family for family in TRAJECTORY_FAMILIES if family in families]
    handles = [
        Patch(facecolor=COLORS[family], label=LABELS[family]) for family in families
    ]
    handles += [
        Line2D(
            [],
            [],
            color="#333333",
            marker=HEAD_MARKERS[h],
            linestyle=":" if h == 1 else "-",
            markersize=8,
            label=rf"$H={h}$",
        )
        for h in (1, 8)
    ]
    return handles


TRAJECTORY_FAMILIES = ("Learned PLD", "DPD5", "DPD6", "DiT")


EFFICIENCY_METRICS = (
    ("vp_velocity_loss", "VP velocity loss"),
    ("denoising_auc", "Log-$\\sigma$ AUC MSE"),
    ("fid", "FID-50k"),
)


def format_loss_axes(axes):
    """Display the two loss axes in units of 10^-2 without changing metric data."""
    from matplotlib.ticker import FuncFormatter

    for ax, label in zip(axes, (r"VP velocity loss", r"Log-$\sigma$ AUC MSE")):
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value * 100:.2f}")
        )
        ax.set_ylabel(label + r" ($10^{-2}$)", fontsize=15)


def efficiency_layout():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.7))
    fig.subplots_adjust(left=0.065, right=0.885, bottom=0.19, top=0.96, wspace=0.25)
    for ax, (_, label) in zip(axes, EFFICIENCY_METRICS):
        ax.set_xlabel("Trainable parameters (M)", fontsize=14)
        ax.set_ylabel(label, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.22)
    format_loss_axes(axes)
    return fig, axes


def add_efficiency_legend(fig, families):
    handles = efficiency_legend(families)
    return fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.90, 0.575),
        ncol=1,
        frameon=False,
        fontsize=13,
        columnspacing=1.1,
        handletextpad=0.5,
    )
