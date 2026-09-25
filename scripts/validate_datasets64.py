#!/usr/bin/env python3
"""Validate 64px exports against the established 32px split identities."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "celeba": (ROOT / "datasets/celeba32", ROOT / "datasets/celeba64"),
    "ffhq": (ROOT / "datasets/ffhq32", ROOT / "datasets/ffhq64"),
    "aahq": (ROOT / "datasets/aahq32", ROOT / "datasets/aahq64"),
    "cifar10": (ROOT / "datasets/cifar10_32", ROOT / "datasets/cifar10_64"),
}


def png_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.png"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    for dataset, (old, new) in PAIRS.items():
        for split in ("train", "valid", "test"):
            old_names, new_names = png_names(old / split), png_names(new / split)
            if old_names != new_names:
                raise AssertionError(f"{dataset}/{split}: 32px and 64px file identities differ")
            for name in (new_names[:1] + new_names[-1:]):
                with Image.open(new / split / name) as image:
                    if image.mode != "RGB" or image.size != (64, 64):
                        raise AssertionError(f"{dataset}/{split}/{name}: {image.mode} {image.size}")

            if dataset in {"ffhq", "aahq"}:
                old_rows = csv_rows(old / f"{split}_manifest.csv")
                new_rows = csv_rows(new / f"{split}_manifest.csv")
                keys = ("output_file", "source_file") + (("raw_name",) if dataset == "aahq" else ())
                old_identity = [tuple(row[key] for key in keys) for row in old_rows]
                new_identity = [tuple(row[key] for key in keys) for row in new_rows]
                if old_identity != new_identity:
                    raise AssertionError(f"{dataset}/{split}: source manifest differs")
            if dataset == "cifar10":
                if (old / f"{split}_labels.csv").read_bytes() != (new / f"{split}_labels.csv").read_bytes():
                    raise AssertionError(f"{dataset}/{split}: labels differ")
        print(f"[validate64] {dataset}: splits, identities, and RGB dimensions verified")


if __name__ == "__main__":
    main()
