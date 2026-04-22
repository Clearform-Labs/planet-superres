# Planet Super-Resolution

Train a small CNN to super-resolve 4x-downsampled PlanetScope satellite imagery, evaluated against a bicubic baseline using PSNR and SSIM.

## Setup

```bash
uv venv --python 3.13
uv pip install rasterio numpy Pillow matplotlib scikit-image tqdm jupyter ipykernel requests
```

## Repository Layout

```
.
├── data/
│   ├── raw/          # PlanetScope GeoTIFF scenes (gitignored)
│   ├── tiles_hr/     # 128×128 HR patches (gitignored)
│   └── tiles_lr/     # 32×32 LR patches, bicubic 4× downsample (gitignored)
├── src/
│   ├── tiling.py     # Load scenes, cut into 128×128 patches
│   └── degrade.py    # Bicubic downsample / upsample for LR/HR pairs
├── notebooks/
│   └── 01_dataset.ipynb   # End-to-end dataset pipeline walkthrough
└── figures/          # Saved plots for slides
```

## Status

- [x] Dataset pipeline (`tiling.py`, `degrade.py`)
- [x] Notebook demo (`01_dataset.ipynb`)
- [ ] Finalize 4 AOIs (urban, agricultural, coastal, natural)
- [ ] Hold-out one scene per AOI as test set
- [ ] Build small CNN (sub-pixel conv, residual blocks, L1 loss)
- [ ] Train in PyTorch
- [ ] Evaluate PSNR/SSIM per terrain type with side-by-side visuals
