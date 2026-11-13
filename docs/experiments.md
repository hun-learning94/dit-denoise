# Revised experiment protocols

## Setup and assets

CPU rendering requires only `requirements-plot.txt`. GPU model code additionally needs `requirements.txt`, a CUDA-compatible PyTorch build, datasets, source weights and upstream model packages. The recorded numerical environment used Python3.11, PyTorch2.11.0+cu126 and native BF16. CPU numerical plots do not import the training stack.

Upstream packages are separately acquired and pinned:

```bash
git clone https://github.com/baofff/U-ViT.git external/U-ViT
git -C external/U-ViT checkout ce551708dc9cde9818d2af7d84dfadfeb7bd9034
git clone https://github.com/VinAIResearch/LFM.git external/LFM
git -C external/LFM checkout 50d263b0a4471384c0ab073c8f904b73c024c6c0
```

Use the released U-ViT-S/4 CelebA64 noise predictor and LFM DiT-L/2 CelebA-HQ256 `model_475.pth`. The latter has457,062,416 denoiser parameters and a separate frozen VAE. The pixel DiT source is the ordinary CelebA128 MLP model at epoch400. Source hashes and constructor parameters are in the templates. Download the VAE and torchvision Inception weights separately. JiT/ImageNet context experiments require the released JiT-B/16 and JiT-L/16 packages/checkpoints referenced in the functional-context design.

`revision.configure` expands `@ROOT@` paths, recomputes the exported code/data identities, and verifies supplied source weights. These portable identities differ from local execution-snapshot identities. `--source-checkpoint` explicitly selects a reader-trained source with a new recorded hash; it does not relabel that source as the original checkpoint. Edit asset paths in a copied protocol template as needed. Source weights, dataset membership and prepared pixels are validated before GPU work; failed CUDA initialization does not fall back to CPU.

## Local context and attention

`experiments/functional_context/worker.py` is the class-token-safe retained-context sweep. It uses nearest patches ordered by Chebyshev distance, squared Euclidean distance and index; the target is first. All integer counts1–256 are evaluated. Context-restricted models cannot read excluded image tokens through protected class states. Full-mask equivalence, perturbation/gradient closure outside the mask, and batched/single agreement gate production.

Materialize `functional_context` to `experiments/functional_context/manifest.json`, after configuring JiT and image paths. Invoke the worker's `--help` for the model/shard interface. Numerical inputs retain14 noise levels, while the compact paper rendering showsσ≤32 on a reversed linear t-axis. The plotted count is the smallest count for which excess clean-target MSE and every larger count's excess are below0.003. No monotonic fit or smoothing is applied across noise.

`experiments/attention_context_transfer/donor_full.py` supplies attention fields, sensitivity and recipient masking. `wiener_refresh.py` and `pooled_wiener.py` supply the covariance controls. Scores average attention over all heads/layers; current-image and calibration-average attention are separate selectors. The recipient is evaluated with its own native denoising objective, paired images/noise and the same retained counts. `revision/data/attention_curves.csv` contains the current plotted means and pointwise intervals. The reproduction plot omits confidence shading as in the paper. Representative image-score maps require the corresponding image/score banks.

## Patch-predictor comparisons

Small models use CelebA64, patch4, width256, depth8, eight spatial-attention heads and global batch32. MLP hidden widths640/1024/1792 approximately match the tested shared-predictor budgets. LPLD, SCA and SMLP use their recorded prototype/head/width grids. Train400 epochs using the documented direct-image VP velocity loss, LR2e-4, zero weight decay, clipping1 and EMA0.999. Numerical-recovery variants retain their changed recipes. Training programs and model definitions are at the repository root. The21-row small-model CSV includes both training-seed endpoints, their arithmetic means and FID ranges. `small_predictor_recipes.json` and `small_predictor_seed1_endpoints.csv` retain configuration and recovery records. Original seed0 FID precision is not uniformly recorded; seed1 endpoint evaluation uses FP32. The plot uses the two original training-seed endpoints, with minimum/maximum intervals rather than confidence intervals.

Larger MLP/SMLP models use depth12, width768,12 spatial-attention heads, and256 tokens: patch8 at128px, patch16 at256px. MLP width is3072. SMLP uses eight predictor heads of width96,2,304 shared memory entries, stored width1,840, layer-specific maps and balanced initialization. Common tensors are explicitly paired across architectures. The compiled fused kernel is in `fused_grouped_gelu_wide.py`; FP32 projection/accumulation corrections and the FP32 output head remain part of the recipe.

The JiT-based recipe uses batch1024, AdamW betas(0.9,0.95), LR2e-4 with five-epoch warmup, no clipping, EMA0.9999, random horizontal flips and logit-normal time(-0.8,0.8). Pixel noise scale is resolution/256. The source campaign starts with200 epochs and extends the horizon to400 while validating unchanged recipe fields in `epoch_extension.py`. `scripts/train_jit_complementary.py --help` gives the manifest-driven entry point, and `scripts/evaluate_complementary.py --help` gives the saved-checkpoint evaluator.

The released FID table uses epoch400 with one training seed and generation seeds54321/154321/254321. ImageNet128 MLP used an FP32/lower-LR recovery, reaching LR2.5e-5 from epoch384. FID uses50,000 samples and the recorded torchvision Inception convention. Parameter matching is not computation matching, and epoch matching does not erase recovery-recipe differences.

## Related-domain component transplantation

The source families are U-ViT64, pixel DiT128 and latent LFM256. Targets are source-framed AAHQ, AFHQ-v2 and FFHQ. The jointly learned groups are dense QK, dense VO, and budget-matched LoRA on both MLP matrices. Original biases, AdaLN/conditioning, embeddings and output/global machinery are frozen. MLP ranks102/153/205 respectively nearly match the attention-group budgets; the exact counts remain in the template. The original pretrained MLP weights are frozen. Maintain EMA of effective weights, save online factors/optimizer/RNG separately, and apply each effective update once.

Use the appropriate `transfer_*` template with `experiments.component_transfer_scale.runner` for its `smoke` and `train` stages. After materializing local data/checkpoint paths, for example:

```bash
python -m revision.configure transfer_uvit64 --output outputs/transfer_uvit64.json
python -m experiments.component_transfer_scale.runner smoke \
  --manifest outputs/transfer_uvit64.json --family uvit
python -m experiments.component_transfer_scale.runner train \
  --manifest outputs/transfer_uvit64.json --run-index 0
```

Keep family-native objectives and schedules. Images are paired across Source, QK, MLP, VO+MLP and Target using shared initial noise; U-ViT additionally pairs the full stochastic reverse-SDE path. The native transplantation sampler remains the1,000-step SDE for U-ViT, VP Heun50 for pixel DiT and latent Heun50 with FP32 VAE decoding for LFM. The faster DPM protocol below applies to the independent U-ViT MetFaces study.

AAHQ/AFHQ quantitative comparisons use epoch100. FFHQ uses epoch50, chosen after visual inspection; do not relabel it epoch100. Grids stop at200 for AAHQ/AFHQ and100 for FFHQ, with the compact paper examples through100/50. `experiments/transplant_epoch100/` supplies paired sampling, DINO extraction, spatial descriptors, dual-reference FID-10k and native denoising loss. Recreate Source exactly with state000 and the adapted endpoint with111. The first1,024 paired trajectories supply the DINO analysis; each quantitative state supplies10,000 images for FID.

Global DINO distance uses unit-normalized DINOv2 ViT-B/14 class-token features. Spatial DINO distance uses standardized patch features and their upper-triangular cosine self-similarities. Divide each mean squared distance by the setting's mean Source–Target squared separation. The ratio can exceed one. These measurements describe endpoint resemblance; transplantation is a diagnostic intervention on coadapted weights.

The norm-matched control rescales each group's effective change to the same setting's global QK update norm; Source and Target remain unscaled. `experiments/norm_matched/worker.py` contains the implementation. Equal total update norms and equal trainable-factor counts control different quantities. VO+MLP has two groups' budgets, and larger total displacement does not establish a larger displacement per parameter.

## Independently optimized MetFaces allocations

Use prepared views from [data.md](data.md). Materialize `metfaces_uvit64`, `metfaces_dit128` or `metfaces_lfm256` with the chosen recorded training seed. Each restricted allocation has equal factor counts within its backbone:

| Allocation | QK rank | VO rank | MLP rank |
|---|---:|---:|---:|
| QK |60|0|0|
| VO |0|60|0|
| MLP |0|0|24|
| VO+MLP |0|30|12|
| All groups |20|20|8|

Counts are1,597,440 factors for U-ViT64,2,211,840 for DiT128, and5,898,240 for LFM256. Full fine-tuning updates the whole denoiser as a separate larger-budget reference. LoRA starts with a zero update, alpha/r=1, frozen original biases and native global conditioning. The VAE is frozen.

All methods train1,000 target epochs /33,000 updates at batch32, with family-native microbatches, optimizer/objective and effective-weight EMA. The templates preserve those differences. Save final and rolling online state separately; fixed-period EMA exports are retained without recreating deleted validation-selected checkpoints.

```bash
python -m revision.configure metfaces_lfm256 --training-seed 0 \
  --output outputs/lfm_seed0.json
python -m revision.run_metfaces cache --manifest outputs/lfm_seed0.json
python -m revision.run_metfaces smoke --manifest outputs/lfm_seed0.json
python -m revision.run_metfaces train --manifest outputs/lfm_seed0.json --arm vo
# Repeat train for each of the six arms, then:
python -m revision.run_metfaces real --manifest outputs/lfm_seed0.json
python -m revision.run_metfaces generate --manifest outputs/lfm_seed0.json --task 0
python -m revision.run_metfaces score --manifest outputs/lfm_seed0.json --task 0
```

Task0 is Source; tasks1–6 are the six final methods. U-ViT/DiT generation completes its exact KID/FID scoring internally. LFM provides a separate CPU score stage. Use the same generation seed across independently trained allocations. U-ViT uses official DPM-Solver-fast50 NFE; DiT uses native VP Heun50; LFM uses latent Heun50 with100 denoiser calls and FP32 VAE decoding. KID is exact unbiased cubic MMD; FID is the empirical small-reference Bures calculation. The143-image test set makes KID the primary measure, with FID secondary.

Reported U-ViT/DiT mean±SD is over training seeds0/1/2 at generation seed202611091. Reported LFM mean±SD is over generation seeds202611091–202611093 from training seed0. The unfinished LFM training replicas are excluded. SD uses ddof=1; image banks are scored separately, not pooled. No test KID/FID selects the final checkpoint.

## Reproducibility limits

Saved-summary commands have been checked on CPU. Full GPU retraining and new image-bank inference are not run by the summary command. Exact numerical trajectories depend on hardware, software and prepared pixels. The exported arithmetic follows the recorded methods, while path constants and hardware-only admission restrictions are made portable. Reader-created sources/data receive new identities. No runtime silently substitutes a checkpoint, upsamples inadequate pixels, merges different sampling protocols or counts generation repeats as training seeds.
