"""Preview saved spatial endpoint distances in the style of Figure 1(c)."""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from settings import DATA, FONTS, OUT
import dino_panels as style

BITS = style.BITS
TARGETS = ("aahq", "afhq", "ffhq")
EPOCHS = {"aahq": 100, "afhq": 100, "ffhq": 50}
TARGET_NAMES = {"aahq": "AAHQ", "afhq": "AFHQ-v2", "ffhq": "FFHQ"}
with (DATA / "spatial_dino_distances.csv").open() as stream:
    spatial = list(csv.DictReader(stream))
for r in spatial:
    for key in ("source_mean", "target_mean"):
        r[key] = float(r[key])
for font in ["Cantarell-Regular.otf", "Cantarell-Bold.otf"]:
    font_manager.fontManager.addfont(str(FONTS / font))
font_manager.fontManager.ttflist = [
    f
    for f in font_manager.fontManager.ttflist
    if Path(f.fname).name != "Cantarell-VF.otf"
]
font_manager.fontManager._findfont_cached.cache_clear()
plt.rcParams.update(
    {
        "font.family": "Cantarell",
        "font.size": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "pdf.compression": 9,
    }
)


def panel(ax, rows, target):
    ax.plot([0, 1.2], [0, 1.2], color="#AEB9C3", ls=(0, (3, 3)), lw=0.8, zorder=1)
    ax.scatter(
        [0, 1],
        [1, 0],
        marker="X",
        c=[style.ENDPOINT_COLORS["source"], style.ENDPOINT_COLORS["target"]],
        s=55,
        linewidths=0.6,
        zorder=4,
    )
    ax.text(0.025, 1.04, "Source", color=style.ENDPOINT_COLORS["source"], fontsize=9)
    ax.text(
        1.03,
        0.04,
        "Target",
        color=style.ENDPOINT_COLORS["target"],
        fontsize=9,
        ha="right",
    )
    for row in rows:
        if row["target"] != target:
            continue
        ax.plot(
            row["source_mean"],
            row["target_mean"],
            marker=style.MARKERS[row["family"]],
            ls="",
            color=style.COLORS[row["bits"]],
            mfc="white",
            ms=7.3,
            mew=1.4,
            zorder=5,
        )
    ax.set(
        xlim=(-0.045, 1.2),
        ylim=(-0.045, 1.2),
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
        title=f"{TARGET_NAMES[target]} · epoch {EPOCHS[target]}",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set(color="#AEB9C3", linewidth=0.8)
    ax.tick_params(colors="#657484", labelsize=9, length=3)
    ax.grid(alpha=0.12, linewidth=0.6)


def save(fig, name):
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUT / (name + "." + suffix), dpi=240, facecolor="white")
    plt.close(fig)


# Compact appendix version: all nine settings in one row of square panels.
fig, axes = plt.subplots(1, 3, figsize=(5.2, 2.0), sharex=True, sharey=True)
component_handles = [
    Line2D([0], [0], color=style.COLORS[b], lw=1.5, label=style.LABELS[b]) for b in BITS
]
family_handles = [
    Line2D(
        [0],
        [0],
        marker=style.MARKERS[f],
        ls="",
        color="#657484",
        mfc="white",
        mew=0.9,
        ms=4.3,
        label=style.NAMES[f],
    )
    for f in style.FAMILIES
]
fig.legend(
    handles=component_handles + family_handles,
    loc="upper center",
    bbox_to_anchor=(0.52, 1.015),
    ncol=6,
    frameon=False,
    fontsize=7.3,
    handlelength=1.1,
    handletextpad=0.45,
    columnspacing=1.1,
)
for ax, target in zip(axes, TARGETS):
    panel(ax, spatial, target)
    ax.title.set_fontsize(8)
    ax.tick_params(labelsize=7, length=2, pad=2)
    for artist in ax.lines[1:]:
        artist.set_markersize(4.6)
        artist.set_markeredgewidth(0.9)
    for text in ax.texts:
        text.set_fontsize(7)
    ax.collections[0].set_sizes([25])
    ax.set_xlabel("Distance to Source", fontsize=8, labelpad=3)
fig.supylabel("Distance to Target", x=0.009, fontsize=8)
fig.subplots_adjust(left=0.07, right=0.995, top=0.76, bottom=0.21, wspace=0.22)
save(fig, "spatial_dino_1x3")
