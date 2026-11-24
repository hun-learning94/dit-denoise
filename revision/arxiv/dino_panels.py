"""Presentation-only endpoint maps from saved paired DINO measurements.

No inference or scheduler changes. AAHQ and AFHQ use epoch100; the completed
FFHQ epoch50 comparison is included at the user's November 13 request.
"""

from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

FAMILIES = ("uvit64", "dit128", "lfm256")
TARGETS = ("aahq", "afhq")
DISPLAY_TARGETS = (*TARGETS, "ffhq")
BITS = ("100", "001", "011")
LABELS = {"100": "QK", "001": "MLP", "011": "VO+MLP"}
COLORS = {"100": "#0072B2", "001": "#E69F00", "011": "#D55E00"}
ENDPOINT_COLORS = {"source": "#008B5A", "target": "#B23A8A"}
MARKERS = {"uvit64": "o", "dit128": "s", "lfm256": "^"}
NAMES = {"uvit64": "U-ViT64", "dit128": "DiT128", "lfm256": "LFM256"}


def draw(fig, records, norm_matched, width, height, header=0):

    ink, gray = "#203040", "#657484"

    def text(x, y, value, **kw):
        return fig.text(
            x / width, 1 - (y + header) / height, value, va="center", color=ink, **kw
        )

    text(438, 10, "(c) DINO", ha="center", fontsize=8.1, weight="bold")
    # Two compact columns: component colors and backbone markers. Entries
    # align by row only for space; color and marker remain independent keys.
    for bits, family, y in zip(BITS, FAMILIES, [25, 35, 45]):
        fig.lines.append(
            Line2D(
                [400 / width, 406 / width],
                [1 - (y + header) / height] * 2,
                transform=fig.transFigure,
                color=COLORS[bits],
                lw=1.7,
            )
        )
        text(410, y, LABELS[bits], ha="left", fontsize=6.3)
        fig.lines.append(
            Line2D(
                [442 / width],
                [1 - (y + header) / height],
                transform=fig.transFigure,
                marker=MARKERS[family],
                ms=3.8,
                ls="",
                color=gray,
                markerfacecolor="white",
                markeredgewidth=0.7,
            )
        )
        text(448, y, NAMES[family], ha="left", fontsize=6.3)

    square_pt = 44
    panel_step = 58
    square_axes = []
    for i, (target, label) in enumerate(
        zip(DISPLAY_TARGETS, ["AAHQ", "AFHQ-v2", "FFHQ"])
    ):
        top = 64 + i * panel_step
        title = "→ " + label + ", epoch " + str(50 if target == "ffhq" else 100)
        text(
            423 + square_pt / 2,
            top - 7,
            title,
            ha="center",
            fontsize=6.3,
            weight="bold",
        )
        ax = fig.add_axes(
            [
                423 / width,
                (height - header - top - square_pt) / height,
                square_pt / width,
                square_pt / height,
            ]
        )
        ax.set_box_aspect(1)
        square_axes.append(ax)
        ax.plot([0, 1.2], [0, 1.2], color="#B5BDC5", ls=(0, (2, 2)), lw=0.6, zorder=1)
        ax.scatter(
            [0, 1],
            [1, 0],
            marker="X",
            c=[ENDPOINT_COLORS["source"], ENDPOINT_COLORS["target"]],
            s=15,
            linewidths=0.25,
            zorder=4,
        )
        ax.text(
            0.025,
            1.025,
            "Source",
            fontsize=6.1,
            color=ENDPOINT_COLORS["source"],
            va="bottom",
        )
        ax.text(
            1,
            0.07,
            "Target",
            fontsize=6.1,
            color=ENDPOINT_COLORS["target"],
            va="bottom",
            ha="right",
        )
        for family in FAMILIES:
            for record in [
                r for r in records if r["target"] == target and r["family"] == family
            ]:
                x, y = record["source_mean"], record["joint_mean"]
                ax.plot(
                    x,
                    y,
                    marker=MARKERS[family],
                    linestyle="none",
                    color=COLORS[record["bits"]],
                    mfc="white",
                    mew=0.75,
                    ms=3.6,
                    zorder=5,
                )
        ax.set(xlim=(-0.045, 1.2), ylim=(-0.045, 1.2), axisbelow=True)
        ax.xaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
        ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0", "0.5", "1"]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0", "0.5", "1"]))
        ax.tick_params(
            labelsize=6.2,
            pad=1,
            length=1.5,
            width=0.5,
            labelbottom=(i == len(DISPLAY_TARGETS) - 1),
            colors=gray,
        )
        ax.spines[["top", "right"]].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set(color="#B5BDC5", linewidth=0.5)
        ax.grid(alpha=0.13, linewidth=0.4)
    text(
        400,
        64 + panel_step + square_pt / 2,
        "Distance to Target (normalized)",
        ha="center",
        rotation=90,
        fontsize=7,
    )
    # Anchor the axis label to its plot, independently of any footer legends.
    text(
        446,
        height - header - 7
        if norm_matched
        else 64 + (len(DISPLAY_TARGETS) - 1) * panel_step + square_pt + 18,
        "Distance to Source",
        ha="center",
        fontsize=6.6,
    )
