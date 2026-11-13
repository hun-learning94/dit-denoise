"""Repository-relative paths for the exported scientific implementations."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path(relative):
    return str(Path(os.environ.get('DIT_DENOISE_ROOT', ROOT)) / relative)
