#!/usr/bin/env python3
"""Build the Experiment 1 64px trajectory figure with corrected Learned PLD runs."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORTING_STATUS = json.loads((HERE / "pld_reporting_status.json").read_text())
OUT = HERE / "figures64"
PLD_EVAL = ROOT / "evals" / "experiment1_celeba64_pld"

import sys
sys.path.insert(0, str(ROOT))
from reports.experiment1.build_combined_report import inventory, display_name

PLD_MODELS = [
    ("h1_w128", "Learned PLD H1 W128"),
    ("h1_w256", "Learned PLD H1 W256"),
    ("h1_w512", "Learned PLD H1 W512"),
    ("h8_w128", "Learned PLD H8 W128"),
    ("h8_w256", "Learned PLD H8 W256"),
    ("h8_w512", "Learned PLD H8 W512"),
]

PLD_MODELS = [(track, display_name(label)) for track, label in PLD_MODELS
              if track not in REPORTING_STATUS["excluded_tracks"]]
PLD_MODELS += [(track, display_name(spec["label"]))
               for track, spec in REPORTING_STATUS.get("recovery_tracks", {}).items()]

from reports.experiment1.plot_style import (COLORS, HEAD_MARKERS, head_count, width_linestyle,
                                            trajectory_layout, trajectory_legend, trajectory_order,
                                            format_loss_axes)


def pld_run(track):
    recovery = REPORTING_STATUS.get("recovery_tracks", {}).get(track)
    if recovery:
        return ROOT / recovery["run_dir"]
    return ROOT / ("runs/experiment1_pld_width" if track == "h8_w512" else "runs/experiment1_pld_width_ddp") / f"celeba64_dpd4_raw_r32_{track}_g4_seed0"


def pld_records(track: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    recovery = REPORTING_STATUS.get("recovery_tracks", {}).get(track)
    prefix = recovery["label_prefix"] if recovery else f"celeba64_pld_{track}_g4_epoch"
    evaluation_root = ROOT / recovery["evaluation_root"] if recovery else PLD_EVAL
    for kind, filename in (("denoising", "summary.json"),
                           ("vp_velocity_loss", "summary.json"),
                           ("fid50k", "result.json")):
        for path in sorted((evaluation_root / kind).glob(f"{prefix}*/{filename}")):
            match = re.search(r"epoch(\d+)", path.parent.name)
            if match:
                epoch = int(match.group(1))
                obj = json.loads(path.read_text())
                key = {"denoising": "log_sigma_auc_mse_raw", "vp_velocity_loss": "vp_velocity_loss", "fid50k": "fid"}[kind]
                assert math.isfinite(float(obj[key])), path
                checkpoint = ROOT / obj["checkpoint"]
                assert checkpoint.is_file() and int(checkpoint.stem[9:]) == epoch * 5086, path
                expected_root = pld_run(track)
                assert checkpoint.parent.parent.resolve() == expected_root.resolve(), path
                assert obj["used_ema"], path
                if recovery:
                    assert epoch > recovery["recovery_from_epoch"], path
                if kind == "fid50k":
                    assert obj["generated_images"] == 50000 and obj["sampler_steps"] == 50, path
                assert kind not in out.setdefault(epoch, {}), path
                out[epoch][kind] = obj
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline = inventory()
    # Retain the established baseline curves.
    for model in baseline:
        label, family = display_name(model['model']), model['family']
        for epoch, rec in model['data'].items():
            for metric, key in (("vp_velocity_loss", "vp_velocity_loss"),
                                ("denoising", "log_sigma_auc_mse_raw"),
                                ("fid50k", "fid")):
                if metric in rec:
                    rows.append(dict(model=label, family=family, epoch=epoch,
                                     metric=metric, value=float(rec[metric][key])))
    # Add whichever corrected PLD snapshots have completed.
    pld_count = 0
    for track, label in PLD_MODELS:
        recs = pld_records(track)
        pld_count += sum(1 for e in recs for m in ("denoising", "vp_velocity_loss", "fid50k") if m in recs[e])
        for epoch, rec in recs.items():
            for metric, key in (("vp_velocity_loss", "vp_velocity_loss"),
                                ("denoising", "log_sigma_auc_mse_raw"),
                                ("fid50k", "fid")):
                if metric in rec:
                    rows.append(dict(model=label, family="Learned PLD", epoch=epoch,
                                     metric=metric, value=float(rec[metric][key])))
    with (HERE / "metric_trajectories_pld.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "family", "epoch", "metric", "value"], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    render(rows)
    print(f"wrote {pld_count} Learned PLD metric points and {len(rows)} total trajectory points")


def render(rows):
    """Render saved metric rows without discovering new evaluation results."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = trajectory_layout(wspace=.24)
    models = sorted(set((r['model'], r['family']) for r in rows), key=trajectory_order)
    for label, family in models:
        for ax, metric, title, ylabel in zip(
            axes,
            ("vp_velocity_loss", "denoising", "fid50k"),
            ("Held-out VP velocity loss", "Denoising trajectory", "Generative trajectory"),
            ("VP velocity loss", "Log-$\\sigma$ AUC MSE", "FID-50k"),
        ):
            values = sorted((r["epoch"], r["value"]) for r in rows
                            if r["model"] == label and r["metric"] == metric)
            if not values:
                continue
            ax.plot(*zip(*values), label=label, color=COLORS[family],
                    marker=HEAD_MARKERS[head_count(label)], ms=4,
                    lw=1.6,
                    linestyle=width_linestyle(label, family))
            ax.set_ylabel(ylabel, fontsize=15)
    format_loss_axes(axes)
    trajectory_legend(fig, models)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"metric_trajectories_pld.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
