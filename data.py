# data.py
# Minimal image data utilities for diffusion-style training.
#
# What this file provides:
#   1) Dataset loader: recursively scans a directory for image files.
#   2) Preprocessing transform: converts images to float tensors in [-1, 1].
#      - No cropping
#      - Optional resize (only if it is required to enforce resolution).
#      - Optional horizontal flip augmentation.
#   3) Sanity check: save a small grid of images from a DataLoader.

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Literal

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

NormalizeTo = Literal["zero_to_one", "neg_one_to_one"]

# Common image extensions we’ll accept when scanning a folder.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TENSOR_CACHE_REGISTRY = {}


def _list_images(root: str) -> List[Path]:
    """
    Recursively find all image files under `root`.

    This is intentionally simple and “filesystem based”:
    - Works for a flat folder of images.
    - Works for nested folders (e.g., train/0001.jpg, train/subdir/....jpg).
    - Does NOT assume class subfolders, labels, CSV files, etc.

    Returns:
        Sorted list of Paths to images.
    """
    root = Path(root)
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)

    paths.sort()
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {root}")
    return paths


class ImageFolderFlat(Dataset):
    """
    A minimal Dataset that returns images from a directory.

    __getitem__ returns:
        (image_tensor, label)

    Notes:
    - We always return label=0 because you said you’re using face datasets and
      don’t need class labels for now.
    - If later you do class-conditional generation (ImageNet-style), we can swap
      this dataset for torchvision.datasets.ImageFolder.
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.paths = _list_images(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        # PIL opens images as HxWxC (or other modes); convert to RGB explicitly.
        # img = Image.open(path).convert("RGB")
        img = Image.open(path)  # do NOT convert here; let transform decide RGB vs grayscale

        # Apply preprocessing (ToTensor + normalization, etc.)
        if self.transform is not None:
            img = self.transform(img)

        # Unconditional training => dummy label.
        label = 0
        return img, label


class TensorImageDataset(Dataset):
    """Tensor-backed image dataset.

    Images are stored as uint8 [N,C,H,W] and converted to [-1,1] floats on read.
    This avoids repeated directory scans and PIL image decoding after the cache
    has been built once.
    """

    def __init__(self, images: torch.Tensor, *, hflip: bool = False):
        if images.ndim != 4:
            raise ValueError(f"Expected images [N,C,H,W], got {tuple(images.shape)}")
        if images.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 cached images, got {images.dtype}")

        self.images = images.contiguous()
        self.hflip = bool(hflip)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        x = self.images[idx]
        if self.hflip and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-1,))
        x = x.float().div(127.5).sub(1.0)
        return x, 0


def _default_tensor_cache_path(root: str, resolution: int, channels: int) -> Path:
    return Path(root) / f".tensor_cache_res{resolution}_ch{channels}_uint8.pt"


def make_tensor_image_dataset(
    root: str,
    *,
    resolution: int,
    channels: int,
    hflip: bool = False,
    cache_path: Optional[str] = None,
    rebuild_cache: bool = False,
) -> TensorImageDataset:
    """Create a tensor-backed folder dataset, building a cache on first use."""
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")

    cache_file = Path(cache_path) if cache_path else _default_tensor_cache_path(root, resolution, channels)
    cache_key = str(cache_file.resolve())

    if cache_file.exists() and not rebuild_cache:
        if cache_key in _TENSOR_CACHE_REGISTRY:
            images = _TENSOR_CACHE_REGISTRY[cache_key]
            print(f"[data.py] Reusing tensor image cache in memory: {cache_file} shape={tuple(images.shape)}")
        else:
            payload = torch.load(cache_file, map_location="cpu", weights_only=True)
            images = payload["images"] if isinstance(payload, dict) and "images" in payload else payload
            _TENSOR_CACHE_REGISTRY[cache_key] = images
            print(f"[data.py] Loaded tensor image cache: {cache_file} shape={tuple(images.shape)}")
        if images.ndim != 4 or images.shape[1] != channels:
            raise ValueError(
                f"Tensor cache shape mismatch in {cache_file}: got {tuple(images.shape)}, "
                f"expected [N,{channels},{resolution},{resolution}]"
            )
        if images.shape[2] != resolution or images.shape[3] != resolution:
            raise ValueError(
                f"Tensor cache resolution mismatch in {cache_file}: got {tuple(images.shape[-2:])}, "
                f"expected ({resolution}, {resolution})"
            )
        return TensorImageDataset(images, hflip=hflip)

    paths = _list_images(root)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    tfm = []
    tfm.append(
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
    )
    if channels == 1:
        tfm.append(transforms.Grayscale(num_output_channels=1))
    else:
        tfm.append(transforms.Lambda(lambda img: img.convert("RGB")))
    tfm.append(transforms.PILToTensor())
    tfm = transforms.Compose(tfm)

    print(f"[data.py] Building tensor image cache from {len(paths)} files: {cache_file}")
    images = []
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            images.append(tfm(img))
        if (i + 1) % 10000 == 0 or (i + 1) == len(paths):
            print(f"[data.py] cached {i + 1}/{len(paths)} images")

    images = torch.stack(images, dim=0).contiguous()
    payload = {
        "images": images,
        "root": str(root),
        "resolution": int(resolution),
        "channels": int(channels),
        "num_images": int(images.shape[0]),
    }

    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(cache_file)
    _TENSOR_CACHE_REGISTRY[cache_key] = images
    print(f"[data.py] Saved tensor image cache: {cache_file} shape={tuple(images.shape)}")
    return TensorImageDataset(images, hflip=hflip)


def build_transform(
    resolution: Optional[int] = None,
    hflip: bool = False,
    channels: int = 3,                       # NEW: 1 for grayscale, 3 for RGB
    normalize_to: NormalizeTo = "neg_one_to_one",  # NEW: choose scale
) -> transforms.Compose:
    """
    Build the preprocessing transform.

    Key features:
    - No cropping (per your request).
    - Optional resize to enforce square resolution.
    - Supports grayscale (channels=1) or RGB (channels=3).
    - Output can be [0,1] or [-1,1].

    Args:
        resolution: If not None, resize to (resolution, resolution).
        hflip: Optional horizontal flip augmentation.
        channels: 1 -> grayscale, 3 -> RGB.
        normalize_to: "zero_to_one" or "neg_one_to_one".

    Returns:
        torchvision.transforms.Compose
    """
    t = []

    # Optional: enforce fixed resolution by resizing (no cropping).
    if resolution is not None:
        t.append(
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    if hflip:
        t.append(transforms.RandomHorizontalFlip(p=0.5))

    # Ensure correct channel mode for PIL images.
    if channels == 1:
        t.append(transforms.Grayscale(num_output_channels=1))
    elif channels == 3:
        # If the source is already RGB this is a no-op; otherwise it converts.
        t.append(transforms.Lambda(lambda img: img.convert("RGB")))
    else:
        raise ValueError(f"channels must be 1 or 3, got {channels}")

    # PIL -> float tensor in [0,1], shape (C,H,W)
    t.append(transforms.ToTensor())

    # Normalize range
    if normalize_to == "neg_one_to_one":
        t.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))
    elif normalize_to == "zero_to_one":
        pass
    else:
        raise ValueError(f"normalize_to must be 'zero_to_one' or 'neg_one_to_one', got {normalize_to}")

    return transforms.Compose(t)


def make_dataloader(
    root: str,
    batch_size: int,
    resolution: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    hflip: bool = False,
    drop_last: bool = True,
) -> Tuple[DataLoader, Dataset]:
    """
    Construct a PyTorch DataLoader for training.

    Args:
        root: dataset directory (recursively scanned for images).
        batch_size: training batch size.
        resolution: optional resize to fixed square resolution.
        shuffle: shuffle each epoch.
        num_workers: data loading workers (tune per machine).
        pin_memory: set True when training on GPU (faster host->device transfer).
        hflip: optional horizontal flip augmentation.
        drop_last: drop incomplete final batch (often convenient for DDP).

    Returns:
        (dataloader, dataset)
    """
    transform = build_transform(
        resolution=resolution,
        hflip=hflip,
        channels=3,
        normalize_to="neg_one_to_one",
    )
    ds = ImageFolderFlat(root=root, transform=transform)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dl, ds


@torch.no_grad()
def save_batch_grid(
    dl: DataLoader,
    out_path: str = "debug_batch.png",
    n: int = 16,
    nrow: Optional[int] = None,
) -> None:
    """
    Save a small image grid from the dataloader to visually sanity-check preprocessing.

    This answers: “Am I loading images correctly? Are they the right orientation?
    Are colors correct? Is normalization correct?”

    Assumptions:
    - Images from the dataset are in [-1,1].
    - We map them back to [0,1] before saving.

    Args:
        dl: a DataLoader created by make_dataloader().
        out_path: output filename.
        n: number of images to include from the first batch.
        nrow: number of images per row in the saved grid.
    """
    batch = next(iter(dl))
    x = batch[0]  # (B,C,H,W)
    x = x[:n]

    # Map [-1,1] -> [0,1] for saving.
    x01 = (x + 1.0) / 2.0
    x01 = x01.clamp(0.0, 1.0)

    # Choose a sensible grid width if not provided.
    if nrow is None:
        nrow = int(max(1, round(n ** 0.5)))

    grid = make_grid(x01, nrow=nrow, padding=2)
    save_image(grid, out_path)
    print(f"[data.py] Saved grid to: {out_path}")


@torch.no_grad()
def peek_batch_stats(dl: DataLoader) -> None:
    """
    Quick numerical sanity check on one batch:
    - shapes
    - dtype
    - range/min/max/mean

    Useful for confirming images are in [-1,1] and properly normalized.
    """
    batch = next(iter(dl))
    x = batch[0]
    print("[data.py] batch x shape:", tuple(x.shape))
    print("[data.py] batch x dtype:", x.dtype)
    print("[data.py] min/max:", float(x.min()), float(x.max()))
    print("[data.py] mean/std:", float(x.mean()), float(x.std()))


def make_mnist_dataloader(
    root: str,
    batch_size: int,
    train: bool = True,
    normalize_to: NormalizeTo = "neg_one_to_one",
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, Dataset]:
    """
    MNIST dataloader.
    MNIST images are 1-channel 28x28.
    """
    t = [transforms.ToTensor()]
    if normalize_to == "neg_one_to_one":
        t.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))
    tfm = transforms.Compose(t)

    ds = datasets.MNIST(root=root, train=train, download=True, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dl, ds

def make_fashionmnist_dataloader(
    root: str,
    batch_size: int,
    train: bool = True,
    normalize_to: NormalizeTo = "neg_one_to_one",
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, Dataset]:
    """Fashion-MNIST dataloader.

    Fashion-MNIST images are 1-channel 28x28, identical tensor shape conventions
    to MNIST.

    Args mirror make_mnist_dataloader.
    """
    t = [transforms.ToTensor()]
    if normalize_to == "neg_one_to_one":
        t.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))
    tfm = transforms.Compose(t)

    ds = datasets.FashionMNIST(root=root, train=train, download=True, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dl, ds
