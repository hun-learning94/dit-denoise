"""Apply the official AAHQ landmarks while exporting aligned square crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_repo", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args()

    official_repo = args.official_repo.resolve()
    sys.path.insert(0, str(official_repo))
    from face_alignment import image_align

    # The reference implementation uses Pillow's removed ANTIALIAS alias.
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS

    with (official_repo / "AAHQ-dataset.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    args.output.mkdir(parents=True, exist_ok=True)
    saved = 0
    missing = 0
    failed = 0
    for index, (output_name, item) in enumerate(metadata.items(), start=1):
        source = args.raw_dir / item["raw_name"]
        destination = args.output / output_name
        if destination.is_file():
            saved += 1
            continue
        if not source.is_file():
            missing += 1
            continue
        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                aligned = image_align(
                    image,
                    item["landmarks"],
                    output_size=args.resolution,
                    transform_size=max(256, args.resolution * 8),
                    enable_padding=True,
                )
                aligned.save(destination, format="PNG")
            saved += 1
        except Exception as exc:
            failed += 1
            print(f"[aahq] failed {source}: {exc}", file=sys.stderr)
        if index % 1000 == 0 or index == len(metadata):
            print(
                f"[aahq] aligned {index}/{len(metadata)} "
                f"saved={saved} missing={missing} failed={failed}"
            )

    if saved == 0:
        raise SystemExit("No AAHQ images were aligned")
    print(f"[aahq] alignment complete saved={saved} missing={missing} failed={failed}")


if __name__ == "__main__":
    main()
