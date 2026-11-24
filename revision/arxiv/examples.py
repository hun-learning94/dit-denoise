"""Compact, three-backbone face-transfer examples from verified saved pixels."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np

from settings import INPUTS, FONTS, OUT

MODELS = ["uvit64", "dit128", "lfm256"]
CODES = ["100", "001", "011", "111"]
RESOLUTIONS = dict(uvit64=64, dit128=128, lfm256=256)
BLUE, RED, INK, GRAY = "#0072B2", "#D55E00", "#203040", "#657484"
SOURCE_ENDPOINT, TARGET_ENDPOINT = "#008B5A", "#B23A8A"
SOURCE_GRAY = "#8C969F"
ICON_FONT = FONTS / "FontAwesome.ttf"
W, H, TILE, PITCH, TOP = 480, 194, 33, 35, 33
PANEL_X = [35, 184, 333]


def row_y(row):
    return TOP + row * 35 + (4 if row == 3 else 0)


def icon(ax, glyph, cx, cy, color, w=6, h=7):
    path = TextPath(
        (0, 0), glyph, size=1, prop=font_manager.FontProperties(fname=str(ICON_FONT))
    )
    bounds = path.get_extents()
    transform = (
        Affine2D()
        .translate(-bounds.x0, -bounds.y0)
        .scale(w / bounds.width, -h / bounds.height)
        .translate(cx - w / 2, cy + h / 2)
        + ax.transData
    )
    ax.add_patch(
        PathPatch(
            path, transform=transform, facecolor=color, edgecolor="none", zorder=11
        )
    )


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
        "font.size": 6.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

for case in json.loads((INPUTS / "examples.json").read_text()):
    tiles = case["tiles"]
    with np.load(INPUTS / f"{case['key']}.npz") as images:
        for index, tile in enumerate(tiles):
            tile["image"] = images[str(index)]
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")

    def text(x, y, label, size=6.5, color=INK, ha="center", weight="normal"):
        return ax.text(
            x,
            y,
            label,
            fontsize=size,
            color=color,
            ha=ha,
            va="center",
            weight=weight,
            zorder=12,
        )

    offset = (4 - len(case["epochs"])) * PITCH / 2
    for i, (model, x) in enumerate(zip(MODELS, PANEL_X)):
        source = ["U-ViT CelebA64", "DiT CelebA128", "LFM CelebA-HQ256"][i]
        text(x + 69, 7.5, f"({chr(97 + i)}) {source}", 7.0, weight="bold")
        target = (
            case["label"]
            + ("-" if case["key"] == "afhq_v2" else "")
            + str(RESOLUTIONS[model])
        )
        text(x + 69, 17.5, f"→ {target}", 6.7)
        for col, epoch in enumerate(case["epochs"]):
            text(
                x + offset + col * PITCH + TILE / 2,
                28,
                "Source" if epoch == 0 else f"Epoch {epoch}",
                6.4,
            )
    text(13, 28, "Weights", 6.2, GRAY)
    for row, code in enumerate(CODES):
        for index, (label, bit) in enumerate(zip(["QK", "VO", "MLP"], code)):
            adapted = bit == "1"
            color = (BLUE if label == "QK" else RED) if adapted else SOURCE_GRAY
            fill = ("#EAF4FA" if label == "QK" else "#FFF2E8") if adapted else "#F2F4F5"
            y = row_y(row) + 1.8 + index * 10.1
            ax.add_patch(
                FancyBboxPatch(
                    (1.3, y),
                    19.5,
                    9.0,
                    boxstyle="round,pad=0,rounding_size=1.2",
                    facecolor=fill,
                    edgecolor=color,
                    lw=0.55,
                )
            )
            text(11.05, y + 4.5, label, 6.1, color, weight="bold")
            if code == "111":
                icon(ax, "\uf06d", 26.5, y + 4.5, "#E65C35")
            elif adapted:
                icon(ax, "\uf0ad", 26.5, y + 4.5, GRAY)
    separator = (row_y(2) + TILE + row_y(3)) / 2
    ax.plot([1, W - 5], [separator, separator], lw=0.6, color="#CDD3DA")
    for tile in tiles:
        x = PANEL_X[tile["model"]] + offset + tile["col"] * PITCH
        y = row_y(tile["row"])
        style = (0, (2.6, 1.7)) if tile["row"] < 3 else "solid"
        ax.imshow(
            tile["image"],
            extent=(x, x + TILE, y + TILE, y),
            interpolation="nearest",
            zorder=3,
        )
        ax.add_patch(
            Rectangle(
                (x, y),
                TILE,
                TILE,
                fill=False,
                edgecolor="#AAB5C1",
                lw=0.45,
                linestyle=style,
                zorder=5,
            )
        )
        if tile["frame"]:
            ax.add_patch(
                Rectangle(
                    (x - 0.55, y - 0.55),
                    TILE + 1.1,
                    TILE + 1.1,
                    fill=False,
                    edgecolor=tile["frame"],
                    lw=1.2,
                    linestyle=style,
                    zorder=7,
                )
            )
    # Use two empty Source columns for compact border keys.
    for panel, color, dashed, y, label in [
        (0, SOURCE_ENDPOINT, False, 47, "Source"),
        (0, TARGET_ENDPOINT, False, 90, "Target"),
        (1, SOURCE_ENDPOINT, True, 47, "Adapted QK\n+ source\nVO + MLP"),
        (1, TARGET_ENDPOINT, True, 100, "Source QK\n+ adapted\nVO + MLP"),
    ]:
        cx = PANEL_X[panel] + offset + TILE / 2
        ax.add_patch(
            Rectangle(
                (cx - 5, y),
                10,
                7,
                fill=False,
                edgecolor=color,
                lw=1.0,
                linestyle=(0, (2.6, 1.7)) if dashed else "solid",
            )
        )
        text(cx, y + (22 if dashed else 16), label, 6.0, color)
    for x, label, color, fill in [
        (12, "Source parameters", SOURCE_GRAY, "#F2F4F5"),
        (109, "Adapted QK", BLUE, "#EAF4FA"),
        (190, "Adapted VO/MLP", RED, "#FFF2E8"),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, H - 10),
                6,
                6,
                boxstyle="round,pad=0,rounding_size=1",
                edgecolor=color,
                facecolor=fill,
                lw=0.55,
            )
        )
        text(x + 10, H - 7, label, 6.2, ha="left")
    icon(ax, "\uf0ad", 294, H - 7, GRAY)
    text(301, H - 7, "Transplanted", 6.2, ha="left")
    icon(ax, "\uf06d", 390, H - 7, "#E65C35")
    text(397, H - 7, "Target model", 6.2, ha="left")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, overlaps = [], []
    for i, a in enumerate(ax.texts):
        bounds = a.get_window_extent(renderer)
        if (
            bounds.x0 < 0
            or bounds.y0 < 0
            or bounds.x1 > fig.bbox.x1
            or bounds.y1 > fig.bbox.y1
        ):
            outside.append(a.get_text())
        for b in ax.texts[i + 1 :]:
            if bounds.overlaps(b.get_window_extent(renderer)):
                overlaps.append([a.get_text(), b.get_text()])
    assert not outside and not overlaps, (outside, overlaps)
    stem = f"{case['key']}_draw{case['draw']:02d}"
    for suffix in ["pdf"]:
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=450, facecolor="white")
    plt.close(fig)
