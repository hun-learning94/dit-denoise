# Third-party components

- U-ViT is loaded from the separately acquired, pinned [official implementation](https://github.com/baofff/U-ViT). The vendored official DPM-Solver implementation is byte-identical to the recorded sampler source, SHA-256`9e57a29b6469e60be2bdf01f9086b1e5969e8f15e06ea856cfe3e80df8060cff`. Its upstream MIT license is preserved in`experiments/uvit_metfaces/vendor/LICENSE`.
- LFM backbones are loaded from the separately acquired [official LFM implementation](https://github.com/VinAIResearch/LFM), pinned at`50d263b0a4471384c0ab073c8f904b73c024c6c0`, including`models/DiT.py`. Keep that repository's licenses and notices.
- JiT and DINOv2 packages/checkpoints are reader-provided dependencies. Keep their upstream licenses and use the revisions/checkpoint hashes recorded in the configured experiment designs.
- Cantarell fonts are bundled unmodified for the explanatory schematic. Copyright and the SIL Open Font License1.1 are preserved in`revision/fonts/OFL.txt`. Other plots use Matplotlib's DejaVu Sans defaults.

Datasets and pretrained weights are not redistributed by this artifact. Their original acquisition and licensing conditions apply independently.
