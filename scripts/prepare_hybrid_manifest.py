"""Freeze an eight-state metric manifest from a user-supplied checkpoint map.

The checkpoint map is a JSON array with one row per transfer and adaptation seed.
Each row gives source, target, seed, source_checkpoint, adapted_checkpoint.
The selected step and epoch come from tables/endpoint_selections.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.experiment3_primary_metrics import pipeline as metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-map", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/experiment3_primary_metrics/protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new campaign path")
    config = json.loads(args.config.read_text())
    metrics.approved(config)
    endpoints = json.loads((ROOT / "reports/experiment3/tables/endpoint_selections.json").read_text())
    selected = json.loads((ROOT / "reports/experiment3/metrics/selected_protocol.json").read_text())
    pairs = {tuple(x) for x in selected["directions"]}
    endpoint = {(r["source"], r["target"]): r for r in endpoints if (r["source"], r["target"]) in pairs}
    supplied = json.loads(args.checkpoint_map.read_text())
    key = lambda row: (row["source"], row["target"], int(row["seed"]))
    mapped = {key(r): r for r in supplied}
    expected = {(a, b, seed) for a, b in pairs for seed in (0, 1)}
    if set(mapped) != expected:
        parser.error(f"checkpoint map needs {len(expected)} unique direction/seed rows; missing={sorted(expected-set(mapped))}; extra={sorted(set(mapped)-expected)}")
    rows = []
    for source, target, seed in sorted(expected):
        spec = mapped[source, target, seed]
        source_path = Path(spec["source_checkpoint"]).resolve()
        adapted_path = Path(spec["adapted_checkpoint"]).resolve()
        for path in (source_path, adapted_path):
            if not path.is_file():
                parser.error(f"checkpoint does not exist: {path}")
        selected_endpoint = endpoint[source, target]
        step = int(selected_endpoint["step"])
        if adapted_path.stem != f"ckpt_step{step}":
            parser.error(f"{source}->{target} seed {seed}: expected selected step {step}, got {adapted_path.name}")
        ident = f"20m_{source}_to_{target}_seed{seed}_step{step}"
        cube_dir = args.output.parent / "cube_checkpoints" / ident
        checkpoints = {"000": {"path": str(source_path), "sha256": metrics.digest(source_path)}}
        for code in metrics.CODES:
            if code == "000":
                continue
            output = cube_dir / f"ckpt_{code}.pt"
            if not output.exists():
                subprocess.run([sys.executable, str(ROOT / "scripts/build_experiment3_primary_cube.py"),
                                "--source", str(source_path), "--adapted", str(adapted_path),
                                "--code", code, "--output", str(output)], check=True)
            checkpoints[code] = {"path": str(output.resolve()), "sha256": metrics.digest(output)}
        rows.append(dict(id=ident, source=source, target=target, scale="20m",
                         adaptation_seed=seed, selected_epoch=selected_endpoint["reference_epoch"],
                         selection=selected_endpoint, checkpoints=checkpoints))
    metrics.write_json(args.output, dict(protocol=config, protocol_hash=metrics.signature(config),
                                         implementation=metrics.implementation_stamp(), rows=rows))
    print(f"Wrote {len(rows)} paired hybrid rows to {args.output}")


if __name__ == "__main__":
    main()
