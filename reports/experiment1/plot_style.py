"""Shared visual encodings for the active Experiment 1 plots."""
import re

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Four colors from the Okabe–Ito colorblind-friendly palette.
COLORS = {"DiT": "#0072B2", "DPD5": "#E69F00",
          "DPD6": "#009E73", "Learned PLD": "#CC79A7"}
LABELS = {"DiT": "MLP", "DPD5": "SCA", "DPD6": "SMLP", "Learned PLD": "LPLD"}
HEAD_MARKERS = {1: "o", 8: "^"}


def head_count(name):
    if name.startswith("DiT"):
        return 1
    match = re.search(r"H=?([18])\b", name)
    if match is None:
        raise ValueError(f"Missing supported predictor head count: {name}")
    return int(match[1])


def width_linestyle(name, family):
    match = re.search(r"(?:F|W|R=|d_h'=)(\d+)", name)
    if match is None:
        raise ValueError(f"Missing memory width or size: {name}")
    widths = (640, 1024, 1792) if family == "DiT" else (
        (128, 256, 512) if family == "Learned PLD" else (256, 512, 1024))
    return ("-", "--", ":")[widths.index(int(match[1]))]


def efficiency_legend(families):
    families = [family for family in TRAJECTORY_FAMILIES if family in families]
    handles = [Patch(facecolor=COLORS[family], label=LABELS[family]) for family in families]
    handles += [Line2D([], [], color="#333333", marker=HEAD_MARKERS[h],
                       linestyle=":" if h == 1 else "-", markersize=8,
                       label=rf"$H={h}$")
                for h in (1, 8)]
    return handles


TRAJECTORY_FAMILIES = ("Learned PLD", "DPD5", "DPD6", "DiT")


def trajectory_order(item):
    name, family = item
    width = re.search(r"(?:F|W|R=|d_h'=)(\d+)", name)
    return TRAJECTORY_FAMILIES.index(family), head_count(name), int(width[1])


def trajectory_label(name):
    """Presentation only; recovery identity stays in metric data and report prose."""
    return name.removesuffix(" recovery").replace("DiT", "MLP")


def trajectory_layout(*, wspace=.36):
    """Three metric panels with space for a grouped legend below."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.9))
    fig.subplots_adjust(left=.075, right=.99, bottom=.43, top=.965, wspace=wspace)
    for ax in axes:
        ax.tick_params(labelsize=12)
        ax.grid(alpha=.22)
        ax.set_xlim(20, 405)
        ax.set_xlabel("Training epoch", fontsize=14)
    return fig, axes


def trajectory_legend(fig, models):
    """One bottom legend group per family, with no repeated model acronyms."""
    models = sorted(models, key=trajectory_order)
    families = [family for family in TRAJECTORY_FAMILIES
                if any(f == family for _, f in models)]
    legends = []
    for index, family in enumerate(families):
        handles = [Line2D([], [], color=COLORS[family],
                          marker=HEAD_MARKERS[head_count(name)],
                          linestyle=width_linestyle(name, family),
                          linewidth=1.8, markersize=5,
                          label=trajectory_label(name).split(" ", 1)[1])
                   for name, f in models if f == family]
        center = .06 + .93 * (index + .5) / len(families)
        legend = fig.legend(handles=handles, title="MLP" if family == "DiT" else LABELS[family],
                            loc="upper center", bbox_to_anchor=(center, .305),
                            fontsize=13, title_fontsize=14, frameon=False,
                            handlelength=2, handletextpad=.6, labelspacing=.5,
                            borderaxespad=0)
        legend.get_title().set_color(COLORS[family])
        legend.get_title().set_fontweight("bold")
        legends.append(legend)
    return legends


EFFICIENCY_METRICS = (("vp_velocity_loss", "VP velocity loss"),
                      ("denoising_auc", "Log-$\\sigma$ AUC MSE"),
                      ("fid", "FID-50k"))


def format_loss_axes(axes):
    """Display the two loss axes in units of 10^-2 without changing metric data."""
    from matplotlib.ticker import FuncFormatter
    for ax, label in zip(axes, (r"VP velocity loss", r"Log-$\sigma$ AUC MSE")):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.2f}"))
        ax.set_ylabel(label + r" ($10^{-2}$)", fontsize=15)


def efficiency_layout():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.7))
    fig.subplots_adjust(left=.065, right=.885, bottom=.19, top=.96, wspace=.25)
    for ax, (_, label) in zip(axes, EFFICIENCY_METRICS):
        ax.set_xlabel("Trainable parameters (M)", fontsize=14)
        ax.set_ylabel(label, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=.22)
    format_loss_axes(axes)
    return fig, axes


def add_efficiency_legend(fig, families):
    handles = efficiency_legend(families)
    return fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.90, .575),
                      ncol=1, frameon=False, fontsize=13,
                      columnspacing=1.1, handletextpad=.5)
