# Planet Super-Resolution

4× super-resolution on PlanetScope satellite imagery using a lightweight CNN. The model operates at LR resolution with residual blocks and sub-pixel convolution (PixelShuffle), trained on synthetically degraded satellite tiles and evaluated against a bicubic baseline.

**Best model (v3): +0.90 dB PSNR over bicubic, 0.616 SSIM, 0.470 LPIPS** on held-out test scenes.

## Results

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Bicubic (baseline) | 22.24 | 0.501 | 0.622 |
| v1 — L1 only | 22.90 | 0.597 | 0.507 |
| v2 — + perceptual loss | 22.92 | 0.600 | 0.487 |
| v3 — more data & params | **23.14** | **0.616** | **0.470** |

Evaluated on 282 held-out test tiles across 2 AOIs (chicago-urban, sd-terrain-and-river).

## Architecture

ESPCN-style fully-convolutional CNN — all computation happens at LR resolution, with PixelShuffle at the end to upscale 4×. This means it can run on **any input size** at inference time, not just the 32×32 training tiles.

- Residual blocks (6 blocks, 96 features in v3)
- Global skip connection
- L1 + perceptual (VGG-16) loss
- ~1.5 MB trained weights

## Dataset

17 PlanetScope scenes covering urban, farmland, port, forest, and terrain/river landscapes. Each scene is tiled into 128×128 HR patches, then bicubic-downsampled 4× to 32×32 LR inputs. D4 symmetry augmentation (rotations + flips) expands the training set 8×.

Train/val/test splits are at the **scene level** — entire AOIs are held out, so the model is always tested on geography it has never seen.

## Repository layout

```
├── src/
│   ├── model.py          # SuperResCNN architecture + perceptual loss
│   ├── dataset.py         # PyTorch dataset for LR/HR tile pairs
│   ├── inference.py       # Full-image super-res utility (any resolution)
│   ├── tiling.py          # Scene → tile extraction
│   ├── degrade.py         # Bicubic degradation & upscale
│   ├── augment.py         # D4 symmetry augmentation
│   └── build_dataset.py   # End-to-end dataset builder
├── notebooks/
│   ├── 01_dataset.ipynb   # Dataset pipeline walkthrough
│   ├── 02_splits.ipynb    # Train/val/test split & manifest
│   ├── 03_train.ipynb     # Model training
│   └── 04_compare.ipynb   # Evaluation, visual comparisons, full-image demos
├── experiments/           # Saved models & configs (v1, v2, v3)
└── data/                  # Raw scenes, tiles, manifest (gitignored)
```

## Setup

```bash
uv venv --python 3.10
uv sync
```

### Building the dataset from raw scenes

```bash
uv run python src/build_dataset.py
```

Then run `notebooks/02_splits.ipynb` to generate `data/manifest.csv`.

### Training

Open `notebooks/03_train.ipynb`. Training configs are in each experiment's `config.json`.

### Evaluation

Open `notebooks/04_compare.ipynb` — runs all models on the test set, computes metrics, and shows side-by-side visual comparisons with zoomed-in crop regions.

## Experiments

| Experiment | Blocks | Features | Loss | Epochs |
|---|---|---|---|---|
| v1_l1_only_4x | 4 | 64 | L1 | 200 |
| v2_perceptual_4x | 4 | 64 | L1 + VGG perceptual | 200 |
| v3_more_data_4x | 6 | 96 | L1 + VGG perceptual | 200 |
