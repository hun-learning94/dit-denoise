"""Page-width attention figure from the existing, verified gallery and curves."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np

from settings import DATA, INPUTS, OUT


class ScientificTicks(Formatter):
    def __init__(self, exponent):
        self.exponent = exponent

    def __call__(self, value, pos=None):
        scaled = value / 10.0**self.exponent
        return self.fix_minus(f"{0.0 if abs(scaled) < 0.05 else scaled:.1f}")

    def get_offset(self):
        return rf"$\times 10^{{{self.exponent}}}$"


def outline(ax, chosen):
    """The draft's exposed-edge boundaries with ticks pointing into the set."""
    mask = np.zeros((16, 16), bool)
    mask.flat[chosen] = True
    edges, ticks = [], []
    for r, c in np.argwhere(mask):
        for dr, dc, edge in [
            (-1, 0, [(c - 0.5, r - 0.5), (c + 0.5, r - 0.5)]),
            (1, 0, [(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)]),
            (0, -1, [(c - 0.5, r - 0.5), (c - 0.5, r + 0.5)]),
            (0, 1, [(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)]),
        ]:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < 16 and 0 <= cc < 16) or not mask[rr, cc]:
                edges.append(edge)
                mid = np.mean(edge, axis=0)
                ticks.append([mid, mid - 0.18 * np.array([dc, dr])])
    lines = LineCollection(edges + ticks, colors="#FFD700", linewidths=0.85, zorder=3)
    lines.set_path_effects(
        [pe.Stroke(linewidth=1.5, foreground="black", alpha=0.75), pe.Normal()]
    )
    ax.add_collection(lines)


DATASETS = json.loads((INPUTS / "gallery.json").read_text())
all_gallery = []
for spec in DATASETS:
    maps = []
    for index in range(len(spec["images"])):
        with np.load(INPUTS / f"{spec['key']}_map{index}.npz") as fields:
            maps.append(
                tuple(fields[k] for k in ("image", "distance", "scores", "top32"))
            )
    all_gallery.append(maps)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
WIDTH, HEIGHT = 11, 5.9
fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
panel_width = 0.108
panel_height = panel_width * WIDTH / HEIGHT
map_bottoms = [[0.720, 0.500], [0.190]]
titles = ["Input", "Attention", "Gradient", "Covariance", "Distance"]
display_order = [3, 1, 2, 0]
map_axes, curve_axes = [], []
visible, axis_limits, legend_handles = [], [], []
curve_left, curve_right = 0.650, 0.990
curve_top = 0.940
curve_column_gap, curve_row_gap = 0.050, 0.050
curve_width = (curve_right - curve_left - curve_column_gap) / 2
curve_height = curve_width * WIDTH / HEIGHT
curve_bottom = curve_top - 2 * curve_height - curve_row_gap

for dataset_index, (spec, maps) in enumerate(zip(DATASETS, all_gallery)):
    fig.text(
        0.283,
        0.985 if dataset_index == 0 else 0.450,
        ("CelebA128" if dataset_index == 0 else "ImageNet256")
        + rf", $t={spec['gallery_time']:g}$",
        fontsize=13.5,
        ha="center",
        va="top",
    )
    for local_row, (image, distance, score_values, chosen) in enumerate(maps):
        for col, title in enumerate(titles):
            ax = fig.add_axes(
                [
                    0.006 + col * 0.1115,
                    map_bottoms[dataset_index][local_row],
                    panel_width,
                    panel_height,
                ]
            )
            map_axes.append(ax)
            if col == 0:
                ax.imshow(image, interpolation="nearest")
                patch = image.shape[0] // 16
                ax.scatter(
                    5 * patch + (patch - 1) / 2,
                    5 * patch + (patch - 1) / 2,
                    s=14,
                    c="cyan",
                    edgecolors="black",
                    linewidths=0.55,
                    zorder=4,
                )
            else:
                field_index = display_order[col - 1]
                if field_index == 0:
                    ax.imshow(
                        distance,
                        cmap="viridis_r",
                        norm=Normalize(0, 15),
                        interpolation="nearest",
                    )
                else:
                    value = score_values[field_index - 1]
                    assert (
                        np.isfinite(value).all()
                        and value.min() >= 0
                        and value.sum() > 0
                    )
                    ax.imshow(
                        (256 * value / value.sum()).reshape(16, 16),
                        cmap="magma",
                        norm=LogNorm(0.03, 30),
                        interpolation="nearest",
                    )
                if spec["key"] == "imagenet256":
                    outline(ax, chosen[field_index])
                ax.scatter(
                    5, 5, s=14, c="cyan", edgecolors="black", linewidths=0.55, zorder=4
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if local_row == 0:
                heading = title
                if spec["key"] == "imagenet256" and col in [2, 3]:
                    heading += r"$^c$"
                ax.set_title(heading, pad=3, fontsize=11.5)

    with (DATA / "attention_curves.csv").open() as stream:
        curves = [r for r in csv.DictReader(stream) if r.pop("dataset") == spec["key"]]
    styles = [
        ("kamb_geometry", "Distance", "#333333", "-", 1.4),
        ("niedoba_donor", "Gradient", "#E69F00", "-", 1.4),
        ("lukoianov_covariance", "Covariance", "#009E73", "-", 1.4),
    ]
    for j, method in enumerate(spec["curve_methods"]):
        recipient_supplied = method.startswith("self_attention")
        label = (
            ("Recipient" if recipient_supplied else "Donor")
            + " Attn."
            + (" (average)" if j % 2 else "")
        )
        styles.append(
            (
                method,
                label,
                "#AA4499" if recipient_supplied else "#0072B2",
                "--" if j % 2 else "-",
                1.7 if j % 2 else 1.95,
            )
        )
    for col, t in enumerate([0.1, 0.9]):
        ax = fig.add_axes(
            [
                curve_left + col * (curve_width + curve_column_gap),
                curve_top
                - curve_height
                - dataset_index * (curve_height + curve_row_gap),
                curve_width,
                curve_height,
            ]
        )
        curve_axes.append(ax)
        values_all = []
        for method, label, color, linestyle, width in styles:
            rows = sorted(
                [
                    r
                    for r in curves
                    if r["recipient"] == spec["recipient"]
                    and np.isclose(float(r["t"]), t)
                    and r["method"] == method
                    and 12 <= int(r["budget"]) <= 48
                ],
                key=lambda r: int(r["budget"]),
            )
            assert [int(r["budget"]) for r in rows] == [12, 16, 24, 32, 48]
            assert all(r["donor"] == spec["donor"] for r in rows)
            values = [float(r["excess"]) for r in rows]
            (line,) = ax.plot(
                [int(r["budget"]) for r in rows],
                values,
                color=color,
                linestyle=linestyle,
                linewidth=width,
                label=label,
                zorder=4 if "attention" in method else 2,
            )
            if dataset_index == 0 and col == 0:
                legend_handles.append(line)
            values_all += values
            visible += [dict(dataset=spec["key"], **r) for r in rows]
        lo, hi = min(values_all), max(values_all)
        pad = 0.065 * (hi - lo)
        locator = MaxNLocator(nbins=3)
        exponent = int(np.floor(np.log10(max(abs(lo), abs(hi)))))
        while (
            np.diff(locator.tick_values(lo - pad, hi + pad)).min() / 10.0**exponent
            < 0.1
        ):
            exponent -= 1
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(ScientificTicks(exponent))
        ax.yaxis.get_offset_text().set_visible(False)
        ax.text(
            0.97,
            0.97,
            ScientificTicks(exponent).get_offset(),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
        )
        ax.set(xlim=(12, 48), ylim=(lo - pad, hi + pad), xticks=[12, 24, 32, 48])
        if col == 0:
            ax.set_ylabel("Excess MSE", labelpad=2)
        ax.set_title(
            rf"{spec['curve_title']}, $t={t:g}$", loc="right", y=1.0, pad=4, fontsize=12
        )
        if spec["recipient"] == "jit_l":
            ax.axvline(
                32, color="#666666", linewidth=1.05, linestyle=(0, (2, 2)), zorder=1
            )
        ax.grid(alpha=0.14, linewidth=0.55)
        if dataset_index == 0:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Retained patches", labelpad=2)
        axis_limits.append(
            dict(
                dataset=spec["key"],
                t=t,
                lower=lo - pad,
                upper=hi + pad,
                exponent=exponent,
            )
        )

legend_by_label = {h.get_label(): h for h in legend_handles}
legend_order = [
    "Distance",
    "Covariance",
    "Gradient",
    "Donor Attn.",
    "Donor Attn. (average)",
    "Recipient Attn.",
    "Recipient Attn. (average)",
]
legend = fig.legend(
    handles=[legend_by_label[k] for k in legend_order],
    loc="lower center",
    bbox_to_anchor=(0.820, 0.105),
    ncol=2,
    frameon=False,
    fontsize=10.5,
    handlelength=1.8,
    columnspacing=0.8,
    handletextpad=0.4,
    borderaxespad=0,
    labelspacing=0.4,
)
distance_scale = fig.colorbar(
    plt.cm.ScalarMappable(norm=Normalize(0, 15), cmap="viridis_r"),
    cax=fig.add_axes([0.452, 0.155, 0.108, 0.012]),
    orientation="horizontal",
    ticks=[0, 5, 10, 15],
)
distance_scale.set_label("Distance (patches)", fontsize=11, labelpad=1.5)
score_scale = fig.colorbar(
    plt.cm.ScalarMappable(norm=LogNorm(0.03, 30), cmap="magma"),
    cax=fig.add_axes([0.1175, 0.155, 0.310, 0.012]),
    orientation="horizontal",
    ticks=[0.03, 0.1, 1, 10, 30],
    extend="both",
)
score_scale.set_ticklabels(["0.03", "0.1", "1", "10", "30"])
score_scale.set_label("Score / spatial mean", fontsize=11, labelpad=1.5)
for scale in [distance_scale, score_scale]:
    scale.minorticks_off()
    scale.ax.tick_params(labelsize=10.5, pad=1, length=2, width=0.5)
    scale.outline.set_linewidth(0.5)

fig.canvas.draw()
legend_box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(
    fig.transFigure.inverted()
)
assert legend_box.x0 >= 0 and legend_box.x1 <= 1
for ext in ["png", "pdf"]:
    fig.savefig(
        OUT / f"attention_context.{ext}", dpi=360, bbox_inches="tight", pad_inches=0.02
    )
plt.close(fig)
