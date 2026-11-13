"""Lossless node-local uint8 mmap cache for the complete CelebA128 train split."""
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
from torch.utils.data import Dataset


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_inventory(project):
    root = project / "datasets/celeba128"
    original = (root / "train_source_filenames.txt").read_bytes()
    count = len(original.splitlines())
    assert count == 162770, count
    identity = hashlib.sha256(original + (root / "READY.json").read_bytes()).hexdigest()
    paths = [root / "train" / f"{i:06d}.png" for i in range(count)]
    return paths, identity


def local_directory(identity):
    roots = [os.environ.get("SLURM_TMPDIR"), f"/scratch/{os.environ['USER']}",
             f"/tmp/{os.environ['USER']}"]
    for candidate in roots:
        if not candidate:
            continue
        root = Path(candidate)
        try:
            root.mkdir(parents=True, exist_ok=True)
            # Never silently place the 'local' cache on shared project storage.
            if str(root.resolve()).startswith((_release_path('assets/external/hpc'), "/home/")):
                continue
            filesystem = subprocess.check_output(
                ["findmnt", "-n", "-o", "FSTYPE", "-T", str(root)], text=True).strip()
            if filesystem in {"gpfs", "nfs", "nfs4", "lustre", "beegfs", "cifs"}:
                continue
            if shutil.disk_usage(root).free < 12 * 1024**3:
                continue
            destination = root / "pixel-dit-cache" / f"celeba128-{identity[:16]}"
            destination.mkdir(parents=True, exist_ok=True)
            return destination
        except OSError:
            continue
    raise RuntimeError("No writable node-local cache root with at least 12 GiB free")


def build_cache(paths, identity, workers):
    destination = local_directory(identity)
    started = time.monotonic()
    with (destination / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock_wait = time.monotonic() - started
        ready = destination / "READY.json"
        cache_file = destination / "images.npy"
        reused = ready.exists()
        if reused:
            metadata = json.loads(ready.read_text())
            assert metadata["identity"] == identity and metadata["count"] == len(paths)
            assert cache_file.stat().st_size == metadata["file_bytes"]
        else:
            chunks = destination / "chunks"
            chunks.mkdir(exist_ok=True)
            if not cache_file.exists():
                array = np.lib.format.open_memmap(cache_file, mode="w+", dtype=np.uint8,
                                                 shape=(len(paths), 128, 128, 3))
                array.flush()
                del array
            array = np.load(cache_file, mmap_mode="r+")
            assert array.shape == (len(paths), 128, 128, 3) and array.dtype == np.uint8
            offset = array.offset
            del array
            def chunk(start):
                end = min(start + 256, len(paths))
                marker = chunks / f"{start:07d}.json"
                block = np.memmap(cache_file, mode="r+", dtype=np.uint8,
                                  offset=offset + start * 128 * 128 * 3,
                                  shape=(end-start, 128, 128, 3))
                if marker.exists():
                    assert json.loads(marker.read_text())["sha256"] == hashlib.sha256(block).hexdigest()
                else:
                    for i in range(start, end):
                        with Image.open(paths[i]) as image:
                            image.load()
                            assert image.mode == "RGB" and image.size == (128, 128)
                            block[i-start] = np.asarray(image)
                    # Persist pixels before publishing the resumable chunk marker.
                    block.flush()
                    publish(marker, {"start": start, "end": end,
                                     "sha256": hashlib.sha256(block).hexdigest()})
                block._mmap.close()
                return end - start
            done, last = 0, time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(chunk, start) for start in range(0, len(paths), 256)]
                for future in as_completed(futures):
                    done += future.result()
                    if time.monotonic() - last > 20 or done == len(paths):
                        print(f"CACHE {done}/{len(paths)} {time.monotonic()-started:.1f}s", flush=True)
                        last = time.monotonic()
            metadata = {"identity": identity, "count": len(paths), "shape": [len(paths), 128, 128, 3],
                        "dtype": "uint8", "host": socket.gethostname(), "workers": workers,
                        "file_bytes": cache_file.stat().st_size,
                        "original_build_seconds": time.monotonic() - started - lock_wait}
        # Compare actual pixels, not only image sizes or metadata.
        array = np.load(cache_file, mmap_mode="r")
        checks = np.linspace(0, len(paths) - 1, 64, dtype=int).tolist()
        for i in checks:
            with Image.open(paths[i]) as image:
                assert np.array_equal(array[i], np.asarray(image)), i
        del array
        metadata["pixel_checks"] = len(checks)
        if not reused:
            publish(ready, metadata)
    return cache_file, dict(metadata, reused=reused, lock_wait_seconds=lock_wait,
                           this_job_cache_seconds=time.monotonic()-started, path=str(cache_file))


class SharedCelebA128(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image), 0


class CachedCelebA128(Dataset):
    def __init__(self, path, count, transform):
        self.path, self.count, self.transform = path, count, transform
        self.array = None

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        return self.transform(Image.fromarray(self.array[index])), 0
