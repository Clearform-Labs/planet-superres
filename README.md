# Planet Super-Resolution

Train a small CNN to super-resolve 4x-downsampled PlanetScope satellite imagery, evaluated against a bicubic baseline using PSNR and SSIM.

## Setup

```bash
uv venv --python 3.10
uv sync
```

## Repository Layout

```
.
├── pyproject.toml
├── data/
│   ├── raw/          # PlanetScope GeoTIFF scenes (gitignored)
│   ├── tiles_hr/     # 128×128 HR patches (gitignored)
│   ├── tiles_lr/     # 32×32 LR patches, bicubic 4× downsample (gitignored)
│   └── manifest.csv  # train/val/test split index (gitignored)
├── src/
│   ├── tiling.py        # Load scenes, crop nodata, cut into patches
│   ├── degrade.py       # Bicubic downsample / upsample
│   ├── augment.py       # 8× D4 symmetry augmentation for training
│   └── build_dataset.py # Process all raw TIFs → tiles_hr / tiles_lr
├── notebooks/
│   ├── 01_dataset.ipynb # Dataset pipeline walkthrough
│   └── 02_splits.ipynb  # Train/val/test split and manifest
└── figures/             # Saved plots
```

## Building the dataset

```bash
uv run python src/build_dataset.py
```

Then run `notebooks/02_splits.ipynb` to generate `data/manifest.csv`.

## Status

- [x] Dataset pipeline (`tiling.py`, `degrade.py`, `build_dataset.py`)
- [x] 4 AOIs collected (urban-philly, long-beach-port, iowa-farmland, fishbone-deforestation)
- [x] Train/val/test split manifest (`02_splits.ipynb`)
- [x] Augmentation utility (`augment.py`)
- [ ] Build small CNN (sub-pixel conv, residual blocks, L1 loss)
- [ ] Train in PyTorch
- [ ] Evaluate PSNR/SSIM per terrain type with side-by-side visuals
