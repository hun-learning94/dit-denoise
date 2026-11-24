"""Paper tables and the first saved LFM draw; no inference or image selection."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from settings import INPUTS, OUT

ARMS = ["qk", "vo", "mlp", "vo_mlp", "qk_vo_mlp", "full"]
LABELS = ["QK", "VO", "MLP", "VO+MLP", "QK+VO+MLP", "Full"]
HIGHLIGHTS = {"qk": ("metfacesQK", "#0072B2"), "vo_mlp": ("metfacesVOMLP", "#D55E00")}
display_arms = ["source"] + ARMS[:-1]
display_labels = ["Source"] + LABELS[:-1]
side, gap, header, bottom, margin, dpi = 256, 16, 38, 8, 8, 320
width = len(display_arms) * side + (len(display_arms) - 1) * gap + 2 * margin
height = side + header + bottom
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "pdf.compression": 9}
)
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
pixels = []
for i, (arm, label) in enumerate(zip(display_arms, display_labels)):
    with np.load(INPUTS / "metfaces.npz") as bank:
        tile = bank[arm]
    x = margin + i * (side + gap)
    ax = fig.add_axes([x / width, bottom / height, side / width, side / height])
    ax.imshow(tile, interpolation="none")
    ax.set_axis_off()
    color = HIGHLIGHTS[arm][1] if arm in HIGHLIGHTS else "black"
    fig.text(
        (x + side / 2) / width,
        1 - 18 / height,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color=color,
        fontweight="bold" if arm in HIGHLIGHTS else "normal",
    )
    if arm in HIGHLIGHTS:
        # Border lies outside the saved image, preserving every original pixel.
        fig.add_artist(
            Rectangle(
                ((x - 4) / width, (bottom - 4) / height),
                (side + 8) / width,
                (side + 8) / height,
                transform=fig.transFigure,
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                clip_on=False,
            )
        )
    pixels.append((x, tile))
fig.savefig(OUT / "lfm_draw01.pdf")
fig.savefig(OUT / "lfm_draw01.png", dpi=dpi)
plt.close(fig)
