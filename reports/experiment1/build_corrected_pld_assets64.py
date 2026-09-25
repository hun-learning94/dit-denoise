#!/usr/bin/env python3
"""Build active Learned-PLD trajectories and latest-epoch efficiency assets."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORTING_STATUS = json.loads((HERE / "pld_reporting_status.json").read_text())
EVAL = ROOT / "evals/experiment1_celeba64_pld"
FIG = HERE / "figures64"
TRACKS = ["h1_w128", "h1_w256", "h1_w512", "h8_w128", "h8_w256", "h8_w512"]
TRACKS = [track for track in TRACKS if track not in REPORTING_STATUS["excluded_tracks"]]
TRACKS += list(REPORTING_STATUS.get("recovery_tracks", {}))
PARAMETERS = {"h1_w128": 8_935_472, "h1_w256": 10_213_424, "h1_w512": 12_965_936,
              "h8_w128": 8_935_472, "h8_w256": 10_213_424, "h8_w512": 12_965_936}
for track, spec in REPORTING_STATUS.get("recovery_tracks", {}).items():
    PARAMETERS[track] = PARAMETERS[spec["base_track"]]


def read_pld(track: str) -> dict[int, dict]:
    from reports.experiment1.build_pld_trajectory import pld_records
    return pld_records(track)


def mse_at(obj, sigma):
    i = min(range(len(obj["sampler_sigmas"])), key=lambda j: abs(obj["sampler_sigmas"][j] - sigma))
    return float(obj["sampler_raw_mse"][i])


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from reports.experiment1.build_combined_report import inventory, display_name
    from reports.experiment1.build_pld_trajectory import PLD_MODELS, pld_run
    from reports.experiment1.plot_style import COLORS, HEAD_MARKERS, head_count, efficiency_legend
    labels = dict(PLD_MODELS)

    baseline = []
    for model in inventory():
        label, family, params = display_name(model['model']), model['family'], model['parameters']
        recs = model['data']
        if 400 not in recs or not all(k in recs[400] for k in ("denoising", "vp_velocity_loss", "fid50k")):
            continue
        d, v, f = recs[400]["denoising"], recs[400]["vp_velocity_loss"], recs[400]["fid50k"]
        baseline.append(dict(model=label, family=family, parameters=params, epoch=400,
                             mse_low=mse_at(d, .1), mse_mid=mse_at(d, 1), mse_high=mse_at(d, 4),
                             denoising_auc=d["log_sigma_auc_mse_raw"],
                             vp_velocity_loss=v["vp_velocity_loss"], fid=f["fid"]))

    pld = []
    for track in TRACKS:
        recs = read_pld(track)
        epochs = {k: max((e for e, r in recs.items() if k in r), default=None)
                  for k in ("denoising", "vp_velocity_loss", "fid50k")}
        if any(e is None for e in epochs.values()):
            continue
        epoch = max(epochs.values())
        d, v, f = (recs[epochs[k]][k] for k in ("denoising", "vp_velocity_loss", "fid50k"))
        pld.append(dict(model=labels[track], family="Learned PLD",
                         track=track, parameters=PARAMETERS[track], epoch=epoch,
                         run_dir=str(pld_run(track).relative_to(ROOT)),
                         learning_rate=REPORTING_STATUS.get("recovery_tracks", {}).get(track, {}).get("learning_rate", .0002),
                         recovery_from_epoch=REPORTING_STATUS.get("recovery_tracks", {}).get(track, {}).get("recovery_from_epoch"),
                         denoising_checkpoint=d["checkpoint"], vp_checkpoint=v["checkpoint"], fid_checkpoint=f["checkpoint"],
                         denoising_epoch=epochs['denoising'], vp_epoch=epochs['vp_velocity_loss'], fid_epoch=epochs['fid50k'],
                         mse_low=mse_at(d, .1), mse_mid=mse_at(d, 1), mse_high=mse_at(d, 4),
                         denoising_auc=d["log_sigma_auc_mse_raw"],
                         vp_velocity_loss=v["vp_velocity_loss"], fid=f["fid"]))

    rows = baseline + pld
    if not rows:
        raise RuntimeError("No complete Experiment 1 metric triplets are available")
    render(rows)
    with (HERE / "corrected_pld_latest_metrics64.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(pld)} corrected PLD latest records and {len(rows)} efficiency points")


def render(rows):
    """Render the saved endpoints, preserving each metric's selected epoch."""
    from reports.experiment1.plot_style import (COLORS, HEAD_MARKERS, head_count,
        EFFICIENCY_METRICS, efficiency_layout, add_efficiency_legend)
    FIG.mkdir(exist_ok=True)
    fig, axes = efficiency_layout()
    groups = {}
    for row in rows:
        groups.setdefault((row['family'], head_count(row['model'])), []).append(row)
    for ax, (key, _) in zip(axes, EFFICIENCY_METRICS):
        for (family, heads), points in groups.items():
            points = sorted(points, key=lambda row: row['parameters'])
            ax.plot([row['parameters'] / 1e6 for row in points],
                    [row[key] for row in points], color=COLORS[family],
                    linewidth=1.4, linestyle=":" if heads == 1 else "-", zorder=1)
        for row in rows:
            epoch_key = 'fid_epoch' if key == 'fid' else 'vp_epoch' if key == 'vp_velocity_loss' else 'denoising_epoch'
            ax.scatter(row["parameters"] / 1e6, row[key], s=65, color=COLORS[row["family"]],
                       marker=HEAD_MARKERS[head_count(row["model"])])
            epoch = row.get(epoch_key) or row['epoch']
            if epoch != 400:
                ax.annotate(f"e{epoch}", (row['parameters']/1e6, row[key]),
                            xytext=(5, 5), textcoords='offset points', fontsize=11)
    add_efficiency_legend(fig, list(dict.fromkeys(row['family'] for row in rows)))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"parameter_efficiency_latest.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
