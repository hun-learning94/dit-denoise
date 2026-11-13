# Dataset views and prepared data

Acquire datasets under their own licenses. The repository supplies numerical summaries and artwork membership lists, without redistributing image pixels or pretrained checkpoints.

## Model-native views

| Backbone | Source view | Transfer and MetFaces view |
|---|---|---|
| U-ViT64 | Official CelebA crop: rows57–185, columns25–153 of the178×218 aligned image, then resize64 | Source-matched eye/mouth framing, using bounded translation/zoom before64px resizing |
| Pixel DiT128 | Centered178×178 CelebA crop, resized128 | Original full aligned target view resized128 |
| LFM256 | Official CelebA-HQ256 preprocessing; frozen SD VAE | Original full aligned256px target view and the same VAE |

U-ViT targets require the source-matched framing used in the revision. `scripts/uvit_facecrop_utils.py` contains the bounded geometry operations. Fit and apply crops using real-image landmarks; fitting does not use generation quality or test metrics. Record the crop/landmark configuration. Raw images and effective crop extents must support the requested output resolution. Exclusions and artwork-disjoint splits precede training.

The MetFaces split is1,057 train /136 validation /143 test portraits, with artwork groups separated. The corresponding per-backbone filename memberships are in `revision/data/metfaces_*_membership.json`. The source-validation membership used for LFM is also included as filenames only.

Validate already prepared native-view RGB PNGs and construct input manifests:

```bash
python -m revision.prepare_metfaces --family uvit64 --images assets/metfaces64
python -m revision.prepare_metfaces --family dit128 --images assets/metfaces128 \
  --source-images assets/celeba128_validation
python -m revision.prepare_metfaces --family lfm256 --images assets/metfaces256 \
  --source-images assets/celebahq256_validation
```

The helper verifies membership, RGB mode and exact native-view dimensions. It builds text memberships for U-ViT/LFM and the mmap representation used by the pixel DiT evaluator. Inputs are preprocessed pixels: the helper performs no crop fitting, resizing or upscaling. Paths written to a newly prepared data manifest belong to the reader's filesystem, and hashes are recomputed for those paths/files.

LFM caches FP32 VAE posterior means and log-variances for original and independently encoded flipped pixels. Every training visit draws fresh posterior noise. The cache does not freeze one sampled latent per image. The native cache preparation verifies saved posterior arrays against direct VAE re-encoding.

## Larger-model training

CelebA128 uses the ordinary aligned-image view. ImageNet128/256 use the recorded center-crop/resizing pipeline and full eligible training population. The scale code expects1,281,167 ImageNet training images and162,770 CelebA training images; keep actual counts and class coverage in the data identity. `scripts/imagenet_center_crop.py` supplies the center-crop implementation, and `experiments/complementary_training/input.py` supplies the pixel-cache/data-loader implementation. Prepared images and mmap caches stay outside Git.

## Attention/context evaluation

Functional-context inputs preserve image IDs, classes, target-patch indices and paired corruption seeds in the protocol template. ImageNet evaluation uses256 held-out images,32 classes×8 images, eight target patches per image and all256 integer context counts. Acquire the corresponding ImageNet validation images and point the template records to them. Calibration and evaluation memberships must remain disjoint.

For attention-versus-control evaluation, use the recorded calibration/evaluation populations and noise grids in the design manifest. Fit covariance/Wiener selection using the specified training/calibration images, including pooled ImageNet covariance where appropriate. Full-input predictions and excluded-token closure checks are retained as reference identities.
