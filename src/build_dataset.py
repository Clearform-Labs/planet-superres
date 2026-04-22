"""
build_dataset.py — Process all TIF scenes in data/raw/ into HR/LR tile pairs.

Usage:
    python src/build_dataset.py
    python src/build_dataset.py --raw_dir data/raw --tile_size 128 --scale 4

For each <scene>.tif found in --raw_dir:
  - Tiles into (tile_size x tile_size) HR patches
  - Bicubic-downsamples each to LR
  - Saves to data/tiles_hr/<scene>_NNNNN.png
             data/tiles_lr/<scene>_NNNNN.png

Augmentation is NOT done here — it happens at training time in the DataLoader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tiling import load_image, crop_valid, tile_image, save_tiles
from degrade import make_pair


def build(
    raw_dir: Path,
    hr_dir: Path,
    lr_dir: Path,
    tile_size: int,
    scale: int,
    min_variance: float,
    max_nodata_frac: float,
) -> None:
    tif_files = sorted(raw_dir.glob("*.tif")) + sorted(raw_dir.glob("*.tiff"))
    if not tif_files:
        print(f"No .tif files found in {raw_dir}")
        return

    total_tiles = 0

    for scene_path in tif_files:
        prefix = scene_path.stem
        print(f"\n--- {scene_path.name} ---")

        img = load_image(scene_path)
        img = crop_valid(img)
        print(f"  Loaded: {img.shape}  dtype={img.dtype}")

        tiles = tile_image(
            img,
            tile_size=tile_size,
            min_variance=min_variance,
            max_nodata_frac=max_nodata_frac,
        )
        print(f"  Tiles kept: {len(tiles)}")

        if not tiles:
            print("  Skipping — no valid tiles.")
            continue

        hr_tiles = tiles
        lr_tiles = [make_pair(t, scale=scale)[0] for t in hr_tiles]

        save_tiles(hr_tiles, hr_dir, prefix=prefix)
        save_tiles(lr_tiles, lr_dir, prefix=prefix)
        total_tiles += len(hr_tiles)

    print(f"\nDone. Total tile pairs saved: {total_tiles}")
    print(f"  HR → {hr_dir}")
    print(f"  LR → {lr_dir}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Build HR/LR tile dataset from raw scenes.")
    parser.add_argument("--raw_dir",       type=Path, default=repo_root / "data/raw")
    parser.add_argument("--hr_dir",        type=Path, default=repo_root / "data/tiles_hr")
    parser.add_argument("--lr_dir",        type=Path, default=repo_root / "data/tiles_lr")
    parser.add_argument("--tile_size",     type=int,  default=128)
    parser.add_argument("--scale",         type=int,  default=4)
    parser.add_argument("--min_variance",  type=float, default=50.0)
    parser.add_argument("--max_nodata_frac", type=float, default=0.0)
    args = parser.parse_args()

    build(
        raw_dir=args.raw_dir,
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        tile_size=args.tile_size,
        scale=args.scale,
        min_variance=args.min_variance,
        max_nodata_frac=args.max_nodata_frac,
    )


if __name__ == "__main__":
    main()
