"""Restore the three-metric appendix layout, with two-training-seed FID ranges."""

import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from settings import DATA, OUT
from predictor_style import (
    COLORS,
    HEAD_MARKERS,
    EFFICIENCY_METRICS,
    efficiency_layout,
    add_efficiency_legend,
)

with (DATA / "small_predictors.csv").open() as stream:
    rows = list(csv.DictReader(stream))
for row in rows:
    row["recovery"] = row["recovery"] == "True"
    for key in ("heads", "parameters"):
        row[key] = int(row[key])
    for key in ("vp_velocity_loss", "denoising_auc", "fid", "fid_min", "fid_max"):
        row[key] = float(row[key])
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = efficiency_layout()
for ax, (key, _) in zip(axes, EFFICIENCY_METRICS):
    for family in ["Learned PLD", "DPD5", "DPD6", "DiT"]:
        for heads in [1] if family == "DiT" else [1, 8]:
            pts = sorted(
                [r for r in rows if r["plot_family"] == family and r["heads"] == heads],
                key=lambda r: r["parameters"],
            )
            xx = [r["parameters"] / 1e6 for r in pts]
            yy = [r[key] for r in pts]
            ax.plot(
                xx,
                yy,
                color=COLORS[family],
                lw=1.4,
                ls=":" if heads == 1 else "-",
                zorder=2,
            )
            if key == "fid":
                ax.errorbar(
                    xx,
                    yy,
                    yerr=[
                        [r[key] - r[key + "_min"] for r in pts],
                        [r[key + "_max"] - r[key] for r in pts],
                    ],
                    fmt="none",
                    ecolor=COLORS[family],
                    capsize=3,
                    elinewidth=1.1,
                    alpha=0.85,
                    zorder=1,
                )
            ax.scatter(
                xx, yy, color=COLORS[family], marker=HEAD_MARKERS[heads], s=58, zorder=3
            )
            if key == "fid":
                for r in pts:
                    if r["recovery"]:
                        ax.annotate(
                            "†",
                            (r["parameters"] / 1e6, r[key + "_max"]),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=10,
                            color=COLORS[family],
                        )
    ax.set_xticks([9, 11, 13])
    ax.margins(x=0.08, y=0.10)
add_efficiency_legend(fig, ["Learned PLD", "DPD5", "DPD6", "DiT"])
for legend in fig.legends:
    for label in legend.get_texts():
        label.set_text(label.get_text().replace("$H=", r"$\widetilde{H}="))
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"comparison_1x3.{ext}", dpi=220, bbox_inches="tight")
