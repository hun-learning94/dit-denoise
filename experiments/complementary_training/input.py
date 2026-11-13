"""Lossless allocation-local crops and deterministic resumable epoch order."""
from __future__ import annotations
from revision.runtime import path as _release_path

from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

from data import ImageFolderFlat, _prepare_view_image, build_transform
from experiments.complementary_benchmarks.celeba128_data import publish


def dataset_source(project, name, limit=0):
    root = project / "datasets" / name / "train"
    dataset = ImageFolderFlat(str(root), build_transform(int(name[-3:]), hflip=False))
    paths = dataset.paths
    indices = list(range(len(paths)))
    if limit and limit < len(paths):
        # A validation subset contains every ImageNet class, plus spread samples.
        required = {}
        if name.startswith("imagenet"):
            for i, path in enumerate(paths):
                required.setdefault(path.parent.name, i)
        chosen = set(required.values())
        chosen.update(np.linspace(0, len(paths)-1, limit, dtype=int).tolist())
        indices = sorted(chosen)[:limit]
        if required:
            # Never trim a mandatory class when the union exceeds the limit.
            remaining = sorted(chosen - set(required.values()))
            indices = sorted(list(required.values()) + remaining[:limit-len(required)])
        paths = [paths[i] for i in indices]
    view = dataset.image_view
    identity_payload = dict(dataset=name, root=str(root.resolve()), count=len(paths),
        source_count=len(dataset), view=view, subset_indices=indices if limit else None)
    if not view:
        identity_payload["celeba_ready"] = json.loads((root.parent / "READY.json").read_text())
        identity_payload["source_index_sha256"] = hashlib.sha256(
            (root.parent / "train_source_filenames.txt").read_bytes()).hexdigest()
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    return paths, view, identity, dict(identity_payload, identity=identity)


def local_root(needed, component):
    candidates = [os.environ.get("SLURM_TMPDIR"), f"/scratch/{os.environ['USER']}",
                  f"/tmp/{os.environ['USER']}"]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            if str(path.resolve()).startswith((_release_path('assets/external/hpc'), "/home/")):
                continue
            filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(path)], text=True).strip()
            if filesystem in {"gpfs", "nfs", "nfs4", "lustre", "beegfs", "cifs"}:
                continue
            existing = path / "pixel-dit-training" / component / "images.npy"
            allocated = existing.stat().st_size if existing.exists() else 0
            if shutil.disk_usage(path).free >= max(2*2**30, needed-allocated):
                return path / "pixel-dit-training"
        except OSError:
            continue
    raise RuntimeError(f"No allocation-local disk with {needed/2**30:.1f} GiB free")


def pixels(path, view):
    with Image.open(path) as original:
        return np.asarray(_prepare_view_image(original, view).convert("RGB"))


def build_cache(project, name, workers=16, limit=0):
    paths, view, identity, source = dataset_source(project, name, limit)
    resolution = int(name[-3:])
    count = len(paths)
    needed = int(count * resolution**2 * 3 * 1.08) + 2*2**30
    component = f"{name}-{identity[:16]}"
    destination = local_root(needed, component) / component
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / "images.npy"
    started = time.monotonic()
    with (destination / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = destination / "READY.json"
        if not ready.exists():
            from experiments.complementary_training.cache_staging import hydrate_cache
            hydrate_cache(project, destination, identity)
        reused = ready.exists()
        if reused:
            record = json.loads(ready.read_text())
            assert record["identity"] == identity and cache.stat().st_size == record["file_bytes"]
        else:
            chunks = destination / "chunks"
            chunks.mkdir(exist_ok=True)
            if not cache.exists():
                array = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8,
                                                 shape=(count, resolution, resolution, 3))
                array.flush()
                del array
            array = np.load(cache, mmap_mode="r+")
            assert array.dtype == np.uint8 and array.shape == (count, resolution, resolution, 3)
            offset = array.offset
            del array

            def chunk(start):
                stop = min(count, start+256)
                marker = chunks / f"{start:08d}.json"
                block = np.memmap(cache, mode="r+", dtype=np.uint8,
                    offset=offset+start*resolution*resolution*3,
                    shape=(stop-start, resolution, resolution, 3))
                if marker.exists():
                    assert json.loads(marker.read_text())["sha256"] == hashlib.sha256(block).hexdigest()
                else:
                    for j, path in enumerate(paths[start:stop]):
                        block[j] = pixels(path, view)
                    block.flush()
                    publish(marker, dict(start=start, stop=stop, sha256=hashlib.sha256(block).hexdigest()))
                block._mmap.close()
                return stop-start
            completed, last = 0, started
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for future in as_completed([pool.submit(chunk, i) for i in range(0, count, 256)]):
                    completed += future.result()
                    if time.monotonic()-last > 30 or completed == count:
                        print(f"CACHE {name} {completed}/{count} {time.monotonic()-started:.1f}s", flush=True)
                        last = time.monotonic()
            record = dict(identity=identity, count=count, resolution=resolution,
                file_bytes=cache.stat().st_size, source=source, host=socket.gethostname(),
                build_seconds=time.monotonic()-started, storage="allocation-local uint8 RGB HWC mmap")
        array = np.load(cache, mmap_mode="r")
        transform = build_transform(resolution, hflip=False)
        for index in np.linspace(0, count-1, min(64, count), dtype=int).tolist():
            expected = pixels(paths[index], view)
            assert np.array_equal(array[index], expected), index
            assert torch.equal(transform(Image.fromarray(array[index])), transform(Image.fromarray(expected))), index
        del array
        record["exact_pixel_and_normalized_tensor_checks"] = min(64, count)
        if not reused:
            publish(ready, record)
    return CachedImages(cache), dict(record, path=str(cache), reused=reused,
                                    this_allocation_seconds=time.monotonic()-started)


class CachedImages(Dataset):
    def __init__(self, path):
        self.path = str(path)
        array = np.load(path, mmap_mode="r")
        self.count, self.resolution = array.shape[:2]
        self.array = None
        self.transform = build_transform(self.resolution, hflip=False)

    def __getstate__(self):
        # A diagnostic can open the mmap in the parent. Send only metadata to
        # spawned workers, then recreate the lambda-containing transform there.
        return dict(self.__dict__, array=None, transform=None)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.array = None
        self.transform = build_transform(self.resolution, hflip=False)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        return self.transform(Image.fromarray(self.array[index])), 0


class SpawnImageFolder(ImageFolderFlat):
    """The same RGB evaluation transform, recreated in spawned workers."""
    def __init__(self, root, resolution):
        super().__init__(str(root), build_transform(resolution, hflip=False))
        self.resolution = resolution

    def __getstate__(self):
        return dict(self.__dict__, transform=None)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.transform = build_transform(self.resolution, hflip=False)


class EpochBatches(Sampler):
    """Prefetch does not advance state; optimizer step determines resume offset."""
    def __init__(self, count, batch, seed, epoch, start_batch=0):
        self.count, self.batch, self.seed = count, batch, seed
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 1000003*self.epoch)
        permutation = torch.randperm(self.count, generator=generator)
        for i in range(self.start_batch, self.count//self.batch):
            yield permutation[i*self.batch:(i+1)*self.batch].tolist()

    def __len__(self):
        return self.count//self.batch - self.start_batch
