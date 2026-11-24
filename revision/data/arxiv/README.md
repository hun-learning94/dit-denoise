# Inputs for the arXiv figures

The image archives contain only the examples displayed in the paper. NumPy loads every archive with its default numeric-array reader.

- Figures 1 and 14 use the U-ViT AFHQ-v2 draw 06 and LFM AAHQ draw 09. The two archives store the original pixels for the unscaled and norm-matched updates respectively. The JSON manifest identifies the model, epoch, group, latent index, and pixel hash of each tile. It includes the paired distances printed below the epoch-100 examples. Panel (c) reads the 27 corresponding endpoint means from the numerical release.
- Figure 4 uses two noisy CelebA observations at time 0.9 and one noisy ImageNet observation at time 0.1. The three map archives contain the displayed inputs, distance fields, measured scores, and selected patch indices. The curve panels read the published attention CSV.
- Figure 7 uses the first LFM MetFaces generation from seed 202611091 for each displayed allocation, with training seed 0 and epoch 1000.
- Figures 8 and 9 use the original nine-noise sweep and the pooled-Wiener controls for both JiT models. The latter includes both the plotting arrays and the corresponding CSV measurements. Figure 2 uses the extended noise grid already included one directory above.
- Figures 11–13 use draw 04 for AAHQ and AFHQ-v2 and draw 12 for FFHQ. The examples manifest supplies the backbone, panel positions, epochs and draw for each grid.

Rendering uses NumPy, Matplotlib and Pillow. The manuscript PDFs were originally produced with Matplotlib 3.10.8, except Figure 4 which used 3.11.0. Font metrics and rasterization can vary with the rendering environment.
