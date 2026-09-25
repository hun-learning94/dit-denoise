"""Train DPD6 with standard per-image EDM noise.

DPD6 retains DiT's GELU MLP and 4D hidden width while generating one or both
MLP weight matrices from depth-shared libraries through layerwise projections.
"""

from train_dpd45 import run


if __name__ == "__main__":
    run("dpd6")
