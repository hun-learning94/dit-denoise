"""Print the six manuscript tables from protocols and saved measurements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def markdown(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


def csv_rows(path):
    with (ROOT / path).open(newline="") as stream:
        reader = csv.reader(stream)
        return next(reader), list(reader)


def main():
    names = ("predictor-formulas", "datasets", "predictor-grid", "endpoints", "nested-mmd", "vo-nonadditivity")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=(*names, "all"))
    args = parser.parse_args()
    selected = names if args.table == "all" else (args.table,)
    for name in selected:
        print(f"\n## {name}\n")
        if name == "predictor-formulas":
            markdown(("Predictor", "Bank parameters", "Layerwise parameters"), (
                ("LPLD", "R M d_h'", "3 d_h² + d_h d_h' + 4 d_h"),
                ("SCA", "2 R d_h'", "2 d_h(d_h+d_h') + 5 d_h + H"),
                ("SMLP", "2 R d_h'", "2 d_h(d_h+d_h') + H d_h' + 2 d_h"),
                ("MLP", "2 L R d_h", "0"),
            ))
        elif name in ("datasets", "predictor-grid"):
            headers, data = csv_rows(f"tables/{name}.csv")
            markdown(headers, data)
        elif name == "endpoints":
            data = json.loads((ROOT / "reports/experiment3/tables/endpoint_selections.json").read_text())
            protocol = json.loads((ROOT / "reports/experiment3/metrics/selected_protocol.json").read_text())
            selected = {tuple(pair) for pair in protocol["directions"]}
            data = [row for row in data if (row["source"], row["target"]) in selected]
            markdown(("Source", "Target", "Reference epoch", "Mean FID"), (
                (r["source"], r["target"], f'{r["reference_epoch"]:g}', f'{r["mean_fid"]:.3f}') for r in data
            ))
        else:
            filename = "mmd_comparison_correlations.csv" if name == "nested-mmd" else "vo_nonadditivity_correlations.csv"
            headers, data = csv_rows(f"reports/experiment3/figures_summary/{filename}")
            markdown(headers, data)


if __name__ == "__main__":
    main()
