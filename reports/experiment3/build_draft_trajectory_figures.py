"""Build compact page-width Experiment 3 trajectory figures from saved data.

Each source gets four target rows and three metric columns. The loss axes use
the same 10^-2 display units as draft Figure 8; metric values stay unchanged.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "celeba64": ("CelebA", ("aahq64", "ffhq64", "afhq_v2_64", "stl10_64")),
    "stl10_64": ("STL-10", ("aahq64", "ffhq64", "afhq_v2_64", "celeba64")),
    "imagenet64": ("ImageNet64", ("aahq64", "ffhq64", "afhq_v2_64", "celeba64")),
}
NAMES = {
    "celeba64": "CelebA", "stl10_64": "STL-10", "imagenet64": "ImageNet64",
    "aahq64": "AAHQ", "ffhq64": "FFHQ", "afhq_v2_64": "AFHQ-v2",
}
METRICS = (
    ("denoising", "Denoising AUC ($10^{-2}$)"),
    ("vp_velocity_loss", "VP loss ($10^{-2}$)"),
    ("fid50k", "FID-50k"),
)
SERIES = (
    ("joint", "#1768ac", "Joint (111)"),
    ("qk_weights", "#a05b19", "QK only"),
    ("attn_weights", "#28936d", "QK+VO only"),
    ("scratch", "#666666", "Scratch"),
)
EPOCHS = tuple(range(25, 201, 25))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", type=Path,
                        help="Also save PNG previews in this directory")
    parser.add_argument("--source", choices=SOURCES,
                        help="Rebuild only one source figure")
    args = parser.parse_args()
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    with (HERE / "tables/trajectories_20m.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    endpoints = {
        (row["source"], row["target"]): row["reference_epoch"]
        for row in json.loads((HERE / "tables/endpoint_selections.json").read_text())
    }
    values = {}
    for row in rows:
        if row["source"] not in SOURCES:
            continue
        key = (row["source"], row["target"], row["metric"], row["series"], int(row["seed"]))
        epoch = int(float(row["reference_epoch"]))
        value = float(row["value"])
        assert epoch in EPOCHS and math.isfinite(value), row
        values.setdefault(key, {})[epoch] = value

    for source, (_, targets) in SOURCES.items():
        if args.source and source != args.source:
            continue
        fig = plt.figure(figsize=(5.5, 3.75))
        grid = fig.add_gridspec(
            4, 3, left=0.12, right=0.815, top=0.90, bottom=0.09,
            hspace=0.15, wspace=0.26,
        )
        for target_index, target in enumerate(targets):
            for metric_index, (metric, metric_name) in enumerate(METRICS):
                ax = fig.add_subplot(grid[target_index, metric_index])
                for series, color, _ in SERIES:
                    for seed in ((0,) if series == "scratch" else (0, 1)):
                        measured = values.get((source, target, metric, series, seed), {})
                        ax.plot(
                            EPOCHS, [measured.get(epoch, np.nan) for epoch in EPOCHS],
                            color=color,
                            linestyle=":" if series == "scratch" else "-" if seed == 0 else "--",
                            marker="^" if series == "scratch" else "o" if seed == 0 else "s",
                            markersize=1.7, linewidth=0.85, clip_on=False,
                        )
                endpoint = endpoints[source, target]
                if endpoint is not None:
                    ax.axvline(endpoint, color="#9a2b64", linestyle="-.", linewidth=0.7)
                ax.set_xlim(23, 202)
                ax.set_xticks((25, 100, 200))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{100 * value:.2f}")
                    if metric != "fid50k" else FuncFormatter(lambda value, _: f"{value:g}")
                )
                ax.tick_params(axis="both", labelsize=6.5, length=1.7, pad=1.3)
                ax.grid(alpha=0.18, linewidth=0.4)
                ax.spines[["top", "right"]].set_visible(False)
                if target_index == 0:
                    ax.set_title(metric_name, fontsize=6.8, pad=2)
                if target_index != len(targets) - 1:
                    ax.tick_params(labelbottom=False)
                if metric_index == 0:
                    ax.set_ylabel(NAMES[target], fontsize=7.3, labelpad=3)
        handles = [Line2D([], [], color=color, linewidth=1.2,
                          linestyle=":" if series == "scratch" else "-",
                          marker="^" if series == "scratch" else None,
                          markersize=3, label=label)
                   for series, color, label in SERIES]
        handles.extend((
            Line2D([], [], color="black", linestyle="-", marker="o",
                   markersize=3, linewidth=1.0, label="Seed 0"),
            Line2D([], [], color="black", linestyle="--", marker="s",
                   markersize=3, linewidth=1.0, label="Seed 1"),
        ))
        handles.append(Line2D([], [], color="#9a2b64", linestyle="-.",
                              linewidth=1.0, label="Endpoint"))
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.823, 0.50),
                   ncol=1, frameon=False, fontsize=6.8, labelspacing=0.55,
                   handlelength=1.6, handletextpad=0.4, borderaxespad=0)
        output = ROOT / "draft/figs" / f"experiment3_trajectories_{source}.pdf"
        fig.savefig(output)
        if args.preview_dir:
            fig.savefig(args.preview_dir / f"experiment3_trajectories_{source}.png", dpi=180)
        plt.close(fig)
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
