"""Paths for the standalone arXiv figure commands."""

from pathlib import Path
import sys

DATA = Path(__file__).resolve().parents[1] / "data"
INPUTS = DATA / "arxiv"
FONTS = DATA.parent / "fonts"
OUT = Path(sys.argv[1])
