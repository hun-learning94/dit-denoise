"""Train DPD5 with standard per-image EDM noise.

Compressed CelebA32 prototype-count experiments (all widths held fixed):

  python train_dpd5.py \
    --dataset folder \
    --data_root ./datasets/celeba32/train \
    --resolution 32 --channels 3 \
    --patch_size 2 --embed_dim 256 --depth 8 --num_heads 8 \
    --prototype_count 512 --prototype_attention_heads 8 \
    --prototype_key_width 128 --prototype_value_width 160 \
    --prototype_qk_head_dim 16 --prototype_value_head_dim 24 \
    --prototype_gain_mode per_head \
    --batch 32 --steps 200000 --lr 2e-4 \
    --precision bf16 --seed 0 \
    --outdir runs/celeba32_dpd5_r512_kc128_vc160_qh16_vh24

For R=1024, change only ``--prototype_count`` and ``--outdir``. These models
must be trained natively; old full-width checkpoints are not conversion seeds.

Noise levels are sampled independently for every image, exactly as in DiT.
"""

from train_dpd45 import run


if __name__ == "__main__":
    run("dpd5")
