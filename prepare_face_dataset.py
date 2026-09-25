"""Convert an aligned face-image directory to flat 32x32 train/valid/test folders.

The image transform matches ``load_celeba.py``: RGB conversion, square center
crop, and bilinear resize. Split assignments are written to CSV manifests so a
prepared dataset can be audited and reproduced exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def discover_images(source: Path) -> list[Path]:
    images = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no supported images found under {source}")
    return images


def split_images(
    images: list[Path],
    dataset: str,
    seed: int,
) -> dict[str, list[Path]]:
    if dataset == "ffhq":
        if len(images) != 70_000:
            raise ValueError(
                f"FFHQ should contain exactly 70,000 images, found {len(images)}. "
                "Use --allow_incomplete only for preprocessing tests."
            )
        # Preserve FFHQ's official 60k training partition. Divide the official
        # final 10k validation partition evenly for this repository's valid/test use.
        return {
            "train": images[:60_000],
            "valid": images[60_000:65_000],
            "test": images[65_000:70_000],
        }

    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_valid = round(0.1 * n)
    n_test = round(0.1 * n)
    n_train = n - n_valid - n_test
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }


def split_aahq_by_raw_image(
    images: list[Path], metadata_path: Path, seed: int
) -> tuple[dict[str, list[Path]], dict[Path, str]]:
    """Split AAHQ crops while keeping every source artwork in one split."""
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    groups: dict[str, list[Path]] = defaultdict(list)
    raw_name_by_path: dict[Path, str] = {}
    for path in images:
        item = metadata.get(path.name)
        if item is None or not item.get("raw_name"):
            raise ValueError(f"AAHQ metadata has no raw_name for aligned crop: {path.name}")
        raw_name = str(item["raw_name"])
        groups[raw_name].append(path)
        raw_name_by_path[path] = raw_name

    grouped = sorted(groups.items())
    random.Random(seed).shuffle(grouped)
    n = len(images)
    targets = {"train": n - round(0.1 * n) - round(0.1 * n),
               "valid": round(0.1 * n), "test": round(0.1 * n)}
    splits = {"train": [], "valid": [], "test": []}
    # Fill the deterministic shuffled group stream toward the crop-count
    # targets. A group may cross a target boundary, but it is never divided.
    split_order = ("train", "valid", "test")
    split_i = 0
    for _raw_name, paths in grouped:
        while split_i < len(split_order) - 1 and len(splits[split_order[split_i]]) >= targets[split_order[split_i]]:
            split_i += 1
        splits[split_order[split_i]].extend(paths)

    raw_sets = {
        split: {raw_name_by_path[path] for path in paths}
        for split, paths in splits.items()
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = raw_sets[left] & raw_sets[right]
        if overlap:
            raise AssertionError(f"AAHQ raw-image leakage between {left}/{right}: {len(overlap)}")
    if sum(map(len, splits.values())) != len(images):
        raise AssertionError("AAHQ grouped split lost or duplicated crops")
    print(
        "[prepare_faces] AAHQ grouped split "
        + " ".join(
            f"{split}={len(splits[split])} crops/{len(raw_sets[split])} raw"
            for split in split_order
        )
    )
    return splits, raw_name_by_path


def center_crop_resize(image: Image.Image, resolution: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((resolution, resolution), Image.Resampling.BILINEAR)


def prepare_split(
    paths: list[Path],
    source: Path,
    output: Path,
    split: str,
    resolution: int,
    group_by_path: dict[Path, str] | None = None,
) -> None:
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{split}_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["output_file", "source_file"]
        if group_by_path is not None:
            fieldnames.append("raw_name")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, path in enumerate(paths):
            output_name = f"{index:06d}.png"
            with Image.open(path) as image:
                processed = center_crop_resize(image, resolution)
                processed.save(split_dir / output_name, format="PNG", optimize=False)
            row = {
                "output_file": output_name,
                "source_file": path.relative_to(source).as_posix(),
            }
            if group_by_path is not None:
                row["raw_name"] = group_by_path[path]
            writer.writerow(row)
            if (index + 1) % 1000 == 0 or index + 1 == len(paths):
                print(f"[prepare_faces] {split}: {index + 1}/{len(paths)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["ffhq", "metfaces", "wikiart", "aahq"], required=True
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--aahq_metadata", type=Path, default=None,
        help="AAHQ-dataset.json; when provided, keep all crops from each raw_name in one split.",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Allow a partial FFHQ source and use an 80/10/10 split (testing only).",
    )
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error("--resolution must be positive")

    source = args.source.resolve()
    output = args.output.resolve()
    images = discover_images(source)
    print(f"[prepare_faces] dataset={args.dataset} found={len(images)} source={source}")
    group_by_path = None
    if args.aahq_metadata is not None and args.dataset != "aahq":
        parser.error("--aahq_metadata is valid only with --dataset aahq")
    if args.dataset == "aahq" and args.aahq_metadata is None:
        parser.error("--dataset aahq requires --aahq_metadata to prevent raw-image split leakage")
    if args.dataset == "aahq" and args.aahq_metadata is not None:
        splits, group_by_path = split_aahq_by_raw_image(
            images, args.aahq_metadata.resolve(), args.seed
        )
    elif args.dataset == "ffhq" and args.allow_incomplete and len(images) != 70_000:
        splits = split_images(images, "metfaces", args.seed)
    else:
        splits = split_images(images, args.dataset, args.seed)

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        prepare_split(
            splits[split], source, output, split, args.resolution,
            group_by_path=group_by_path,
        )
    print(
        "[prepare_faces] done "
        + " ".join(f"{name}={len(paths)}" for name, paths in splits.items())
        + f" output={output}"
    )


if __name__ == "__main__":
    main()
