"""Render figures and result tables from the included, small numeric summaries."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path: str) -> list[dict]:
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("efficiency", "pld-trajectories", "transfer-trajectories", "transfer-statistics", "transfer-images", "all"))
    args = parser.parse_args()
    assets = ("efficiency", "pld-trajectories", "transfer-trajectories", "transfer-statistics") if args.asset == "all" else (args.asset,)
    if "efficiency" in assets:
        from reports.experiment1.build_corrected_pld_assets64 import render
        source = rows("reports/experiment1/corrected_pld_latest_metrics64.csv")
        for row in source:
            for key in ("parameters", "epoch", "denoising_auc", "vp_velocity_loss", "fid", "denoising_epoch", "vp_epoch", "fid_epoch"):
                if row.get(key):
                    row[key] = float(row[key]) if key in ("denoising_auc", "vp_velocity_loss", "fid") else int(row[key])
        render(source)
    if "pld-trajectories" in assets:
        from reports.experiment1.build_pld_trajectory import render
        source = rows("reports/experiment1/metric_trajectories_pld.csv")
        for row in source:
            row["epoch"] = int(row["epoch"])
            row["value"] = float(row["value"])
        render(source)
    if "transfer-trajectories" in assets:
        import subprocess
        import sys
        (ROOT / "draft/figs").mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "reports/experiment3/build_draft_trajectory_figures.py"], cwd=ROOT, check=True)
    if "transfer-statistics" in assets:
        import sys
        sys.path.insert(0, str(ROOT / "reports/experiment3"))
        from reports.experiment3.association_analysis import CubeAssociations
        from reports.experiment3 import paper_summary_assets as paper
        out = ROOT / "reports/experiment3/figures_summary"
        out.mkdir(parents=True, exist_ok=True)
        (ROOT / "draft/figs").mkdir(parents=True, exist_ok=True)
        model = CubeAssociations()
        paper.figures(model, out)
        from reports.experiment3.build_summary_assets import difference_name
        paper.tables(model, out, difference_name)
        from reports.experiment3.build_hybrid_metrics import paper_dataset_geometry_figure
        import json
        geometry = json.loads((ROOT / "reports/experiment3/metrics/dataset_geometry.json").read_text())
        paper_dataset_geometry_figure(geometry)
    if "transfer-images" in assets:
        import subprocess
        import sys
        sys.path.insert(0, str(ROOT / "reports/experiment3"))
        from reports.experiment3.build_summary_assets import Summary, OUT
        OUT.mkdir(parents=True, exist_ok=True)
        (ROOT / "draft/figs").mkdir(parents=True, exist_ok=True)
        Summary().image_overview()
        for command in (
            ["reports/experiment3/build_flagship_figure.py", "--source", "celeba64"],
            ["reports/experiment3/build_flagship_figure.py", "--source", "imagenet64"],
            ["reports/experiment3/build_combined_appendix_cubes.py"],
        ):
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
