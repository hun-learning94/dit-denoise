# load_celeba.py
# Download / prepare CelebA faces and export them as plain image files for the
# existing folder-based diffusion training pipeline.
#
# Example usage:
#   python load_celeba.py --root ./datasets --split train --resolution 32
#   python load_celeba.py --root ./datasets --split valid --resolution 32
#   python load_celeba.py --root ./datasets --split test  --resolution 32
#
# This creates:
#   ./datasets/celeba32/train/000000.png, 000001.png, ...
#   ./datasets/celeba32/valid/...
#   ./datasets/celeba32/test/...
#
# Why this script:
# - Your current train.py already supports --dataset folder.
# - CelebA images are 178x218, so for faces it is better to center-crop to a
#   square before resizing to 32x32.
# - We export a clean RGB folder once, then all architectures can use the exact
#   same dataset and preprocessing.
#
# Notes:
# - torchvision.datasets.CelebA may need to download metadata / images on first run.
# - If automatic download fails in your environment, you can manually place the
#   CelebA files under <root>/celeba and rerun this script with --download false.

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity, download_url, extract_archive
from torchvision.utils import make_grid, save_image
from PIL import Image

CELEBA_MIRROR = "https://ftp.mi.fu-berlin.de/pub/cmb-data/celeba"


def ensure_celeba_available(root: Path, download: bool):
    """
    Prepare CelebA locally without relying on torchvision's Google Drive
    downloader, which frequently hits quota errors for the public files.
    """
    if not download:
        return

    base_dir = root / datasets.CelebA.base_folder
    image_dir = base_dir / "img_align_celeba"
    base_dir.mkdir(parents=True, exist_ok=True)

    for _, md5, filename in datasets.CelebA.file_list:
        path = base_dir / filename
        suffix = path.suffix.lower()

        if suffix in (".zip", ".7z") and image_dir.is_dir():
            continue
        if suffix not in (".zip", ".7z") and check_integrity(str(path), md5):
            continue

        url = f"{CELEBA_MIRROR}/{filename}"
        print(f"[load_celeba] downloading {filename} from mirror")
        download_url(url, str(base_dir), filename=filename, md5=md5)

    if not image_dir.is_dir():
        archive = base_dir / "img_align_celeba.zip"
        print(f"[load_celeba] extracting {archive}")
        extract_archive(str(archive), str(base_dir))


def build_celeba_dataset(root: Path, split: str, transform, download: bool):
    ensure_celeba_available(root, download)
    return datasets.CelebA(
        root=str(root),
        split=split,
        target_type="identity",
        transform=transform,
        download=False,
    )


def build_export_transform(resolution: int):
    """
    CelebA aligned images are typically 178x218.
    Standard face preprocessing is:
      1) center crop to square
      2) resize to target resolution
      3) keep RGB
    """
    return transforms.Compose([
        transforms.CenterCrop(178),  # crop to 178x178 square around the face
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.Lambda(lambda img: img.convert("RGB")),
    ])


def build_preview_transform(resolution: int, normalize: str):
    """
    Transform for preview batches only.
    """
    t = [
        transforms.CenterCrop(178),
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),  # [0,1]
    ]
    if normalize == "neg_one_to_one":
        t.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))
    elif normalize == "zero_to_one":
        pass
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")
    return transforms.Compose(t)


def export_split(
    root: Path,
    split: str,
    resolution: int,
    download: bool,
    limit: int | None = None,
    overwrite: bool = False,
):
    """
    Export CelebA split to a flat image folder:
      <root>/celeba{resolution}/{split}/000000.png, ...
    """
    out_dir = root / f"celeba{resolution}" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT:
    # We instantiate with transform=None so we can save processed PIL images.
    ds = build_celeba_dataset(root=root, split=split, transform=None, download=download)

    export_tfm = build_export_transform(resolution)

    n_total = len(ds) if limit is None else min(len(ds), limit)
    print(f"[load_celeba] split={split} total={len(ds)} exporting={n_total} -> {out_dir}")

    for i in range(n_total):
        out_path = out_dir / f"{i:06d}.png"
        if out_path.exists() and not overwrite:
            continue

        img, _ = ds[i]  # PIL image, target ignored
        img = export_tfm(img)  # still PIL
        if not isinstance(img, Image.Image):
            raise TypeError("Export transform must return a PIL.Image")

        img.save(out_path)

        if (i + 1) % 5000 == 0 or (i + 1) == n_total:
            print(f"[load_celeba] exported {i + 1}/{n_total}")

    print(f"[load_celeba] done: {out_dir}")


@torch.no_grad()
def save_preview_grid(
    root: Path,
    split: str,
    resolution: int,
    batch_size: int,
    normalize: str,
    out_path: str,
    download: bool,
):
    """
    Load a preview batch directly from torchvision CelebA and save a debug grid.
    This is just to sanity-check crop/resize/color.
    """
    tfm = build_preview_transform(resolution, normalize)

    ds = build_celeba_dataset(root=root, split=split, transform=tfm, download=download)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    x, y = next(iter(dl))  # y ignored
    print("[load_celeba] preview batch x shape:", tuple(x.shape), "dtype:", x.dtype)

    if normalize == "neg_one_to_one":
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)

    grid = make_grid(x[:64], nrow=8, padding=2)
    save_image(grid, out_path)
    print(f"[load_celeba] saved preview grid to: {out_path}")


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets",
                        help="Base dataset directory. torchvision CelebA will live under this root.")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "valid", "test", "all"],
                        help="CelebA split to export.")
    parser.add_argument("--resolution", type=int, default=32,
                        help="Target output resolution.")
    parser.add_argument("--download", type=str2bool, default=True,
                        help="Whether to let torchvision try downloading CelebA.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of images to export (for quick debugging).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite already-exported images.")
    parser.add_argument("--preview_batch", type=int, default=64,
                        help="Batch size for debug preview.")
    parser.add_argument("--preview_out", type=str, default="celeba_preview.png",
                        help="Path to save preview grid.")
    parser.add_argument("--normalize", type=str, default="neg_one_to_one",
                        choices=["zero_to_one", "neg_one_to_one"],
                        help="Normalization used only for preview grid.")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # Save a preview first for sanity check.
    preview_split = "train" if args.split == "all" else args.split
    save_preview_grid(
        root=root,
        split=preview_split,
        resolution=args.resolution,
        batch_size=args.preview_batch,
        normalize=args.normalize,
        out_path=args.preview_out,
        download=args.download,
    )

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for split in splits:
        export_split(
            root=root,
            split=split,
            resolution=args.resolution,
            download=args.download,
            limit=args.limit,
            overwrite=args.overwrite,
        )

    print()
    print("Done. You can now train with:")
    print(f"  python train.py --dataset folder --data_root {root}/celeba{args.resolution}/train "
          f"--resolution {args.resolution} --channels 3")


if __name__ == "__main__":
    main()
