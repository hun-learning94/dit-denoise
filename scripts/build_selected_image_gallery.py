"""Generate the 64-latent image banks used by the paper's qualitative figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_experiment3_secondary_examples import sample

CODES = ("000", "100", "010", "001", "110", "101", "011", "111")
SOURCE_EPOCH = {"celeba64": 400, "stl10_64": 800, "imagenet64": 800}
LATENT_SEED = {"celeba64": 20260904, "stl10_64": 20260904, "imagenet64": 20260911}
PICKS = {"celeba64": (0,), "stl10_64": (0, 1), "imagenet64": (0, 1)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest from prepare_hybrid_manifest.py")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/experiment3_selection64")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required for qualitative image generation")
    endpoint = {(r["source"], r["target"]): r for r in json.loads(
        (ROOT / "reports/experiment3/tables/endpoint_selections.json").read_text())}
    rows = json.loads(args.manifest.read_text())["rows"]
    catalogs = {source: [] for source in SOURCE_EPOCH}
    for row in rows:
        source, target, seed = row["source"], row["target"], int(row["adaptation_seed"])
        if source not in PICKS or seed not in PICKS[source]:
            continue
        rng_seed = LATENT_SEED[source]
        latents = torch.randn(64, 3, 64, 64, generator=torch.Generator().manual_seed(rng_seed))
        images = []
        for code in CODES:
            checkpoint = Path(row["checkpoints"][code]["path"])
            if digest(checkpoint) != row["checkpoints"][code]["sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
            chunks = [sample(checkpoint, latents[start:start + 16].cuda(), 50).numpy()
                      for start in range(0, 64, 16)]
            images.append(np.rint(np.concatenate(chunks) * 255).clip(0, 255).astype(np.uint8))
        bank_dir = args.output / source / "banks" / row["id"]
        bank_dir.mkdir(parents=True, exist_ok=True)
        path = bank_dir / "images.npz"
        np.savez_compressed(path, images=np.stack(images), codes=np.array(CODES),
                            latents=latents.numpy(), latent_seed=rng_seed)
        choice = endpoint[source, target]
        record = dict(bank=row["id"], source=source, target=target, seed=seed,
                      images_sha256=digest(path), raw_bank=str(bank_dir.relative_to(ROOT)),
                      source_training_epoch=SOURCE_EPOCH[source],
                      reference_epoch=choice["reference_epoch"], adaptation_step=choice["step"],
                      latent_rng_seed=rng_seed)
        (bank_dir / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        catalogs[source].append(record)
        print(source, target, seed, path, flush=True)
    for source, banks in catalogs.items():
        if not banks:
            raise ValueError(f"no image banks for {source}")
        catalog = args.output / source / "provenance.json"
        catalog.write_text(json.dumps(dict(source=source, banks=banks), indent=2) + "\n")
    print("Qualitative image gallery complete:", args.output)


if __name__ == "__main__":
    main()
