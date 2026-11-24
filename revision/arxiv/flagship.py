"""Verified face-transfer examples plus three overlaid DINO endpoint maps."""

from pathlib import Path
import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np

from settings import DATA, INPUTS, FONTS, OUT
import sys
import csv

EPOCHS = [0, 25, 50, 100]
CODES = ["100", "001", "011", "111"]
BLUE, RED, INK, GRAY = "#0072B2", "#D55E00", "#203040", "#657484"
SOURCE_ENDPOINT, TARGET_ENDPOINT = "#008B5A", "#B23A8A"
SOURCE_GRAY = "#8C969F"
ROLE_COLORS = {"QK": BLUE, "VO": RED, "MLP": RED}
JOINT_GAP = 5
ROLE_HEADER = 14
EXAMPLE_DISTANCE_HEIGHT = 7
FRAME_LEGEND_HEIGHT = 20
W, H = (
    480,
    216 + JOINT_GAP + ROLE_HEADER + 4 * EXAMPLE_DISTANCE_HEIGHT + FRAME_LEGEND_HEIGHT,
)
TILE, PITCH, TOP = 41, 44, 32 + ROLE_HEADER
PANEL_X = [35, 218]
TITLES = ["(a) U-ViT CelebA64 → AFHQ-v2-64", "(b) LFM CelebA-HQ256 → AAHQ256"]

norm_matched = sys.argv[2] == "norm_matched"
name = "norm_matched" if norm_matched else "flagship"
if norm_matched:
    H = 263
example = json.loads((INPUTS / "flagship.json").read_text())[name]
tiles = example["tiles"]
with np.load(INPUTS / f"{name}.npz") as images:
    for tile in tiles:
        tile["image"] = images[tile["model"]][tile["index"]]
distance_lookup = {(r["family"], r["code"]): r for r in example["distances"]}
with (DATA / "spatial_dataset_mmd.csv").open() as stream:
    rows = list(csv.DictReader(stream))
SPATIAL_MMD = {
    r["view"]: float(r["mmd2"])
    for r in rows
    if r["source"] == "source"
    and (r["view"], r["target"]) in {("uvit64", "afhq"), ("lfm256", "aahq")}
}
ICON_FONT = FONTS / "FontAwesome.ttf"
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
        "font.size": 7.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")


def text(x, y, s, size=7.5, color=INK, ha="center", weight="normal"):
    return ax.text(
        x,
        y,
        s,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        weight=weight,
        zorder=12,
    )


def row_y(row):
    return (
        TOP + row * (PITCH + EXAMPLE_DISTANCE_HEIGHT) + (JOINT_GAP if row == 3 else 0)
    )


def wrench(cx, cy, w=7.2, h=7.2):
    # Convert the local FontAwesome wrench glyph to a vector path.
    path = TextPath(
        (0, 0), "\uf0ad", size=1, prop=font_manager.FontProperties(fname=str(ICON_FONT))
    )
    b = path.get_extents()
    transform = (
        Affine2D()
        .translate(-b.x0, -b.y0)
        .scale(w / b.width, -h / b.height)
        .translate(cx - w / 2, cy + h / 2)
        + ax.transData
    )
    ax.add_patch(
        PathPatch(
            path, transform=transform, facecolor=GRAY, edgecolor="none", zorder=11
        )
    )


def flame(cx, cy, w=7.5, h=10):
    points = [
        (0.52, 1),
        (0.10, 1),
        (0.04, 0.75),
        (0.19, 0.52),
        (0.23, 0.65),
        (0.30, 0.65),
        (0.33, 0.66),
        (0.21, 0.35),
        (0.55, 0.28),
        (0.50, 0),
        (0.84, 0.18),
        (0.86, 0.42),
        (0.77, 0.56),
        (0.93, 0.46),
        (1, 0.72),
        (0.88, 0.86),
        (0.78, 1),
        (0.68, 1),
        (0.52, 1),
        (0.52, 1),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 18 + [MplPath.CLOSEPOLY]
    coords = [(cx + (x - 0.5) * w, cy + (y - 0.5) * h) for x, y in points]
    ax.add_patch(
        PathPatch(
            MplPath(coords, codes),
            facecolor="#E65C35",
            edgecolor="#B7362F",
            lw=0.3,
            zorder=10,
        )
    )
    inner = [
        (cx, cy + h * 0.36),
        (cx - w * 0.23, cy + h * 0.27),
        (cx - w * 0.10, cy + h * 0.02),
        (cx + w * 0.05, cy - h * 0.12),
        (cx + w * 0.30, cy + h * 0.20),
        (cx + w * 0.21, cy + h * 0.36),
        (cx, cy + h * 0.36),
    ]
    ax.add_patch(
        PathPatch(
            MplPath(inner, [MplPath.MOVETO] + [MplPath.CURVE4] * 6),
            facecolor="#FFE5A0",
            edgecolor="none",
            zorder=11,
        )
    )


for x, label, color in [(2.3, "QK", BLUE), (175, "VO", RED), (198, "MLP", RED)]:
    ax.add_patch(
        FancyBboxPatch(
            (x, 2.5),
            20.7,
            10.2,
            boxstyle="round,pad=0,rounding_size=1.4",
            facecolor="#EAF4FA" if color == BLUE else "#FFF2E8",
            edgecolor=color,
            lw=0.6,
            zorder=8,
        )
    )
    text(x + 10.35, 7.6, label, 6.7, color, weight="bold")
text(27, 7.6, "Select context for denoising", 7.0, BLUE, ha="left")
text(223, 7.6, "Reconstruct patches using learned parameters", 7.0, RED, ha="left")

for model, x, title in zip(["uvit64", "lfm256"], PANEL_X, TITLES):
    text(
        x + (4 * TILE + 3 * (PITCH - TILE)) / 2,
        6 + ROLE_HEADER,
        title,
        8.1,
        weight="bold",
    )
    text(
        x + (4 * TILE + 3 * (PITCH - TILE)) / 2,
        16 + ROLE_HEADER,
        rf"Spatial MMD$^2$ = {SPATIAL_MMD[model]:.4f}",
        6.8,
        GRAY,
    )
    for col, name in enumerate(["Source", "Epoch 25", "Epoch 50", "Epoch 100"]):
        text(x + col * PITCH + TILE / 2, 25 + ROLE_HEADER, name, 7.3)
text(15, 25 + ROLE_HEADER, "Weights", 7.0, GRAY)
badge_records = []
for row, code in enumerate(CODES):
    y0 = row_y(row)
    for index, (name, bit) in enumerate(zip(["QK", "VO", "MLP"], code)):
        color = ROLE_COLORS[name] if bit == "1" else SOURCE_GRAY
        fill = ("#EAF4FA" if name == "QK" else "#FFF2E8") if bit == "1" else "#F2F4F5"
        y = y0 + 3.3 + index * 11.6
        ax.add_patch(
            FancyBboxPatch(
                (2.3, y),
                20.7,
                10.2,
                boxstyle="round,pad=0,rounding_size=1.4",
                facecolor=fill,
                edgecolor=color,
                lw=0.6,
                linestyle="solid",
                zorder=8,
            )
        )
        text(12.65, y + 5.1, name, 6.7, color, weight="bold")
        icon = "fire" if code == "111" else "wrench" if bit == "1" else None
        badge_records.append(
            dict(
                row=row,
                group=name,
                source_or_adapted="adapted" if bit == "1" else "source",
                color=color,
                icon=icon,
                dashed=False,
            )
        )
        if icon == "fire":
            flame(28.2, y + 5.1, 7.2, 9.5)
        elif icon == "wrench":
            wrench(28.2, y + 5.1)
separator_y = (row_y(2) + TILE + EXAMPLE_DISTANCE_HEIGHT + row_y(3)) / 2
ax.plot([2, 393], [separator_y, separator_y], color="#CDD3DA", lw=0.65, zorder=2)
image_frame_records = []
for tile in tiles:
    panel = 0 if tile["model"] == "uvit64" else 1
    x = PANEL_X[panel] + tile["col"] * PITCH
    y = row_y(tile["row"])
    transplant = tile["row"] < 3
    style = (0, (2.6, 1.7)) if transplant else "solid"
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
            lw=0.65 if transplant else 0.4,
            linestyle=style,
            zorder=5,
        )
    )
    if tile["frame"]:
        ax.add_patch(
            Rectangle(
                (x - 0.7, y - 0.7),
                TILE + 1.4,
                TILE + 1.4,
                fill=False,
                edgecolor=tile["frame"],
                lw=1.55,
                linestyle=style,
                zorder=7,
            )
        )
    image_frame_records.append(
        dict(
            model=tile["model"],
            row=tile["row"],
            epoch=tile["epoch"],
            code=tile["code"],
            dashed=transplant,
        )
    )
    if tile["epoch"] == 100:
        distance = distance_lookup[(tile["model"], tile["code"])]
        text(
            x + TILE * 0.24,
            y + TILE + 5,
            f"S {distance['source_normalized']:.2f}",
            6.1,
            SOURCE_ENDPOINT,
        )
        text(
            x + TILE * 0.76,
            y + TILE + 5,
            f"T {distance['target_normalized']:.2f}",
            6.1,
            TARGET_ENDPOINT,
        )

if norm_matched:
    border_legend = []
    for offset, color, dashed, label, label_offset in [
        (2, SOURCE_ENDPOINT, False, "Source\ngeneration", 23),
        (34, TARGET_ENDPOINT, False, "Target\ngeneration", 23),
        (66, SOURCE_ENDPOINT, True, "Adapted QK\n+ source\nVO + MLP", 28),
        (108, TARGET_ENDPOINT, True, "Source QK\n+ adapted\nVO + MLP", 28),
    ]:
        cx = PANEL_X[0] + TILE / 2
        y = TOP + offset
        ax.add_patch(
            Rectangle(
                (cx - 6, y),
                12,
                9,
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                linestyle=(0, (2.6, 1.7)) if dashed else "solid",
                zorder=8,
            )
        )
        text(cx, y + label_offset, label, 6.1, color)
        border_legend.append(
            dict(label=label.replace("\n", " "), color=color, dashed=dashed)
        )

    legend_y = H - 5.5
    for x, label, color, fill in [
        (14, "Source parameters", SOURCE_GRAY, "#F2F4F5"),
        (105, "Adapted QK", BLUE, "#EAF4FA"),
        (174, "Adapted VO/MLP", RED, "#FFF2E8"),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, legend_y - 3.5),
                7,
                7,
                boxstyle="round,pad=0,rounding_size=1",
                edgecolor=color,
                facecolor=fill,
                lw=0.6,
                zorder=8,
            )
        )
        text(x + 11, legend_y, label, 6.5, ha="left")
    wrench(260, legend_y)
    text(268, legend_y, "Transplanted", 6.5, ha="left")
    flame(333, legend_y, 7.3, 10)
    text(341, legend_y, "Target model", 6.5, ha="left")

else:
    composition_annotations = []
    for row, color, label in [
        (0, SOURCE_ENDPOINT, "Human face\ncomposed with\nanimal-face\ncontext rule"),
        (2, TARGET_ENDPOINT, "Animal face\ncomposed with\nhuman-face\ncontext rule"),
    ]:
        y = row_y(row) + TILE / 2
        text(PANEL_X[1] + TILE / 2, y, label, 6.1, color)
        ax.annotate(
            "",
            xy=(PANEL_X[0] + 3 * PITCH + TILE + 1, y),
            xytext=(PANEL_X[1] - 3, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.65),
        )
        composition_annotations.append(
            dict(
                model="uvit64",
                epoch=100,
                row=row,
                label=label.replace("\n", " "),
                color=color,
                location="Second panel empty Source column",
            )
        )

    legend_y = H - FRAME_LEGEND_HEIGHT - 5.5
    for x, label, color, fill in [
        (14, "Source parameters", SOURCE_GRAY, "#F2F4F5"),
        (105, "Adapted QK", BLUE, "#EAF4FA"),
        (174, "Adapted VO/MLP", RED, "#FFF2E8"),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, legend_y - 3.5),
                7,
                7,
                boxstyle="round,pad=0,rounding_size=1",
                edgecolor=color,
                facecolor=fill,
                lw=0.6,
                zorder=8,
            )
        )
        text(x + 11, legend_y, label, 6.5, ha="left")
    wrench(260, legend_y)
    text(268, legend_y, "Transplanted", 6.5, ha="left")
    flame(333, legend_y, 7.3, 10)
    text(341, legend_y, "Target model", 6.5, ha="left")

    border_legend = []
    frame_legend_y = legend_y + 17
    for x, color, dashed, label in [
        (14, SOURCE_ENDPOINT, False, "Source generation"),
        (105, TARGET_ENDPOINT, False, "Target generation"),
        (196, SOURCE_ENDPOINT, True, "Adapted QK + source\nVO + MLP"),
        (338, TARGET_ENDPOINT, True, "Source QK + adapted\nVO + MLP"),
    ]:
        ax.add_patch(
            Rectangle(
                (x, frame_legend_y - 4),
                10,
                8,
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                linestyle=(0, (2.6, 1.7)) if dashed else "solid",
                zorder=8,
            )
        )
        text(x + 15, frame_legend_y, label, 6.5, color, ha="left")
        border_legend.append(
            dict(label=label.replace("\n", " "), color=color, dashed=dashed)
        )

from dino_panels import draw

distances = "norm_matched_dino_distances.csv" if norm_matched else "dino_distances.csv"
with (DATA / distances).open() as stream:
    points = list(csv.DictReader(stream))
for point in points:
    for key in ("source_mean", "joint_mean"):
        point[key] = float(point[key])
draw(fig, points, norm_matched, W, H, header=ROLE_HEADER)
filename = "norm_matched_transplants" if norm_matched else "aahq_draw09"
fig.savefig(OUT / f"{filename}.pdf", dpi=600, facecolor="white")
plt.close(fig)
