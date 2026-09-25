"""Train DPD4 with standard per-image EDM noise.

Initial CelebA32 experiment (R=32):

  python train_dpd4.py \
    --dataset folder \
    --data_root ./datasets/celeba32/train \
    --resolution 32 --channels 3 \
    --patch_size 2 --embed_dim 256 --depth 8 --num_heads 8 \
    --memory_size 32 --memory_attention_heads 8 \
    --noise_groups 8 \
    --batch 32 --steps 200000 --lr 2e-4 \
    --precision bf16 --seed 0 \
    --outdir runs/celeba32_dpd4_r32_d256_qh32_vh32_g8 \
    --resume runs/celeba32_dpd4_r32_d256_qh32_vh32_g8/checkpoints/ckpt_step0140000.pt

Eight noise levels are sampled independently and assigned evenly at random
across each minibatch by default.

New direct and factorized runs may additionally set ``--memory_mode``,
``--memory_code_dim``, ``--memory_qk_head_dim``, and
``--memory_value_head_dim``. Old checkpoints without these fields reconstruct
the original direct, full-width model.
"""

from train_dpd45 import run


if __name__ == "__main__":
    run("dpd4")
