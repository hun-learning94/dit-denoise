"""Download STL-10 and export deterministic folder-based splits as PNG files.

The training split combines the official unlabeled set with 4,500 labeled
training images.  The remaining 500 labeled training images form validation;
the official 8,000-image test split is kept intact.
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms


def split_labeled_train(valid_size: int, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < valid_size < 5000:
        raise ValueError("--valid_size must be between 1 and 4999")
    permutation = torch.randperm(5000, generator=torch.Generator().manual_seed(seed)).tolist()
    valid = set(permutation[:valid_size])
    train = [index for index in range(5000) if index not in valid]
    return train, sorted(valid)


def source_dataset(root: Path, split: str, download: bool):
    return datasets.STL10(root=str(root / "raw"), split=split, download=download)


def export_indices(split: str, valid_size: int, seed: int) -> tuple[str, list[int]]:
    labeled_train, valid = split_labeled_train(valid_size, seed)
    if split == "train":
        # torchvision orders train+unlabeled as 5,000 train, then 100,000 unlabeled.
        return "train+unlabeled", labeled_train + list(range(5000, 105000))
    if split == "valid":
        return "train", valid
    return "test", list(range(8000))


def export_split(root: Path, split: str, resolution: int, valid_size: int, seed: int, download: bool):
    source_split, indices = export_indices(split, valid_size, seed)
    dataset = source_dataset(root, source_split, download)
    output_root = root / f"stl10_{resolution}"
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    resize = transforms.Resize(
        (resolution, resolution),
        interpolation=transforms.InterpolationMode.LANCZOS,
        antialias=True,
    )
    label_file = output_root / f"{split}_labels.csv"
    print(f"[stl10] {split}: exporting {len(indices)} images to {output_dir}", flush=True)
    with label_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label", "class_name", "source_split", "source_index"])
        writer.writeheader()
        for output_index, source_index in enumerate(indices):
            image, label = dataset[source_index]
            image = resize(image.convert("RGB"))
            if not isinstance(image, Image.Image):
                raise TypeError("resize did not return a PIL image")
            filename = f"{output_index:06d}.png"
            path = output_dir / filename
            if not path.exists():
                image.save(path)
            writer.writerow({
                "filename": filename,
                "label": int(label),
                "class_name": dataset.classes[int(label)] if int(label) >= 0 else "unlabeled",
                "source_split": "train" if source_index < 5000 and source_split == "train+unlabeled" else source_split,
                "source_index": source_index,
            })
            if (output_index + 1) % 10000 == 0 or output_index + 1 == len(indices):
                print(f"[stl10] {split}: {output_index + 1}/{len(indices)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    parser.add_argument("--resolution", type=int, choices=[32, 64], required=True)
    parser.add_argument("--valid_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for split in splits:
        export_split(args.root, split, args.resolution, args.valid_size, args.seed, args.download)


if __name__ == "__main__":
    main()
