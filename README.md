# Aggregation and prediction in diffusion transformers

Reproduction code and numerical inputs for the revised manuscript. The active workflow covers population local denoising, transferable attention-based context selection, alternative patch predictors, related-domain component transplantation, and matched-budget MetFaces fine-tuning.

## Reproduce the arXiv figures

With the plotting dependencies installed, run `python reproduce_revision.py arxiv --output outputs/arxiv`.
The command builds the twelve experimental figures on CPU from saved measurements, score fields, and the exact displayed image draws.

| Figure | Output PDF stem |
|---|---|
| 1 | aahq_draw09 |
| 2 | locality_summary |
| 4 | attention_context |
| 6 | comparison_1x3 |
| 7 | lfm_draw01 |
| 8 | denoising_excess_context |
| 9 | attention_pooled |
| 10 | spatial_dino_1x3 |
| 11–13 | aahq_draw04, afhq_v2_draw04, ffhq_draw12 |
| 14 | norm_matched_transplants |

Figures 3 and 5 are architectural drawings available with `python reproduce_revision.py schematics`.
The transfer grids use the displayed draws from the original sampling banks. The score gallery uses the three noisy observations and the fields measured for each. The remaining panels use saved numerical results. Full sampling and training protocols are described below.

## Regenerate the numerical figures and tables

Python 3.11 is recommended. The saved-result workflow runs on CPU and needs no datasets, model weights, or cloud account:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-plot.txt
python reproduce_revision.py all
```

Outputs are written to `outputs/revision/`, including `tables.md`, locality/context curves, the three-metric small-predictor comparison, DINO and spatial DINO distance panels, MetFaces KID comparisons, and both architectural schematics. `python reproduce_revision.py check` independently checks numerical coverage and recomputes seed means and sample SDs. Individual commands are `locality`, `attention`, `predictors`, `transplants`, `metfaces`, `tables`, and `schematics`.

These commands render measured summaries. Recomputing model outputs requires the datasets and source checkpoints described in [the experiment guide](docs/experiments.md). The arXiv figure release includes selected display inputs. Complete datasets, sample banks and checkpoints are acquired separately.

## Revised manuscript map

Figure numbers can change during editing; LaTeX labels identify the corresponding content.

| Manuscript content | Included numerical reproduction | Measurement implementation |
|---|---|---|
| Locality/context, `fig:imagenet-locality` | `python reproduce_revision.py locality` | `experiments/functional_context/`, `experiments/attention_context_transfer/` |
| PLD/DiT schematic, `fig:aggregator-predictor-overview` | `python reproduce_revision.py schematics` | `revision/figures/overview.py`; schematic cartoons with gold outlines marking the selected context |
| Context-selection comparison, `fig:attention-context` | `python reproduce_revision.py attention` | Donor/recipient masking, current/average attention, sensitivity, covariance and distance selectors in `experiments/attention_context_transfer/` |
| MLP/SMLP schematic, `fig:mlp-smlp-architecture` | `python reproduce_revision.py schematics` | `models.py`, `model_grouped_dpd6.py` and `revision/figures/architecture.py` |
| Larger MLP/SMLP comparison, `tab:predictor-scale` | `python reproduce_revision.py tables` | `scripts/train_jit_complementary.py`, `experiments/complementary_training/`, `scripts/evaluate_complementary.py` |
| Small-model comparison, `fig:predictor-fid` | `python reproduce_revision.py predictors` | `train.py`, `train_dpd4.py`, `train_dpd5.py`, `train_grouped_dpd6.py` |
| Transplant DINO panel, `fig:component-transplant-flagship`(c) | `python reproduce_revision.py transplants` | `experiments/component_transfer_scale/`, `experiments/transplant_epoch100/` |
| Complementary spatial distances, `fig:transplant-spatial-dino` | Same command; separate 1×3 plot | Spatially standardized DINO patch self-similarity descriptors |
| Transplant denoising/FID appendix tables | `python reproduce_revision.py tables` | Native-objective denoising and dual-reference FID in `experiments/transplant_epoch100/` |
| Norm-matched control | `revision/data/update_norms.csv`; measurement code included | `experiments/norm_matched/worker.py` |
| MetFaces allocations/results, `tab:metfaces-allocations`, `tab:metfaces-results` | `python reproduce_revision.py metfaces` and `tables` | `experiments/{uvit_metfaces,celeba128_metfaces,lfm_metfaces}/`, `revision/run_metfaces.py` |



## Results and reporting conventions

- **ImageNet locality:** all 14 measured noise levels remain in the numerical inputs. The compact plot shows σ≤32 on a reversed linear time axis, `t=1/(1+σ)`, from 1 to 1/33. Curves have no point markers. The vertical axis includes all 256 image patches.
- **Small predictors:** 21 configurations, two training seeds, epoch 400. FID intervals are the two-seed minimum/maximum, with numerical recoveries recorded in the input CSV.
- **Larger predictors:** CelebA128 and ImageNet128/256, epoch 400, one training seed per architecture and three generation seeds. ImageNet128 MLP's FP32/lower-learning-rate recovery is disclosed. Parameter counts are closely matched.
- **Transplants:** U-ViT64, pixel DiT128 and latent LFM256 to AAHQ/AFHQ at epoch 100 and FFHQ at epoch 50. Each setting has one adaptation seed and 1,024 paired generations for DINO distances. DINO and spatial DINO distances use separate descriptors and the same Source–Target normalization principle.
- **MetFaces:** epoch 1,000 / 33,000 optimizer updates, six independently trained allocation methods. All three backbones report means and sample SDs across training seeds 0/1/2, evaluated with the same generation seed (202611091). Each KID/FID bank has 5,000 generated images and 143 held-out real portraits. U-ViT uses the official 50-evaluation DPM-Solver-fast sampler.

Generation seeds do not count as training replication. The included FID/KID implementations use the project's frozen torchvision Inception feature convention; scores need that same convention for comparison. Norm matching controls total effective-weight displacement, while equal LoRA counts control trainable factor budgets. See [protocols and limitations](docs/experiments.md).

## Recompute experiments

Install `requirements.txt` with a CUDA-compatible PyTorch build for GPU work. The recorded environment used PyTorch 2.11.0+cu126. Native BF16 support and enough memory for the recorded physical batch are required for the revised large-model and transfer recipes. No particular cloud provider or GPU model is required by the exported implementation.

Protocol templates are under `revision/configs/`. Materialize one against your local assets:

```bash
python -m revision.configure metfaces_uvit64 \
  --data assets/manifests/metfaces_uvit64.json \
  --output outputs/uvit_seed0.json
python -m revision.run_metfaces smoke --manifest outputs/uvit_seed0.json
python -m revision.run_metfaces train --manifest outputs/uvit_seed0.json --arm vo
```

Download pretrained backbones and licensed datasets separately; supply the paths and verify source hashes before running. `--training-seed 1` or `2` selects the recorded training-corruption namespace. [The experiment guide](docs/experiments.md) describes the full sequence, sampler differences, checkpoint retention, and the inputs needed for each stage. `--dry-run` on the configure command inspects a template without requiring external assets.

Only the revised-paper workflow is maintained in this tree. [Export provenance](revision/export_manifest.json) records the originating scientific source hashes and the portability changes. Third-party code and fonts retain their [licenses and attribution](docs/third_party.md).
