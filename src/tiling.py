"""
tiling.py — Load satellite scenes and cut them into fixed-size patches.

Supports:
  - PlanetScope GeoTIFFs (3- or 4-band, uint16 → uint8 normalisation)
  - Standard images (JPG / PNG) via PIL
Always returns (H, W, 3) uint8 RGB arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return a (H, W, 3) uint8 RGB numpy array.

    Handles:
      - GeoTIFF / TIFF: opened with rasterio; uint16 normalised to uint8.
        4-band (RGBN) → first 3 bands kept as RGB.
      - JPG / PNG: opened with PIL, converted to RGB.

    Args:
        path: Path to the image file.

    Returns:
        numpy array of shape (H, W, 3), dtype uint8, channel order RGB.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        return _load_geotiff(path)
    else:
        return _load_pil(path)


def _load_geotiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        # Read up to the first 3 bands (RGB or first 3 of RGBN)
        n_bands = min(src.count, 3)
        data = src.read(list(range(1, n_bands + 1)))  # (C, H, W)

        # Normalise uint16 → uint8
        if data.dtype == np.uint16:
            # Use 2nd–98th percentile stretch per-band for visibility
            out = np.zeros_like(data, dtype=np.uint8)
            for i in range(n_bands):
                band = data[i].astype(np.float32)
                p2, p98 = np.percentile(band, (2, 98))
                p2 = max(p2, 0.0)
                if p98 > p2:
                    band = (band - p2) / (p98 - p2)
                else:
                    band = band / 65535.0
                out[i] = (np.clip(band, 0, 1) * 255).astype(np.uint8)
            data = out
        elif data.dtype != np.uint8:
            # Generic float or int → scale to uint8
            data = data.astype(np.float32)
            for i in range(n_bands):
                mn, mx = data[i].min(), data[i].max()
                if mx > mn:
                    data[i] = (data[i] - mn) / (mx - mn) * 255
            data = data.astype(np.uint8)

        # (C, H, W) → (H, W, C)
        image = np.transpose(data, (1, 2, 0))

        # Ensure exactly 3 channels
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] >= 3:
            image = image[:, :, :3]

    return image


def _load_pil(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def crop_valid(image: np.ndarray) -> np.ndarray:
    """Crop a scene to the bounding box of its non-black pixels.

    Planet GeoTIFFs have nodata (pure black) fill along scene edges where the
    satellite swath doesn't cover the full rectangular extent. Cropping to the
    valid region before tiling ensures no tile ever contains edge artifacts.

    Args:
        image: (H, W, 3) uint8 numpy array.

    Returns:
        Cropped (H', W', 3) array containing only the valid data region.
    """
    mask = np.any(image > 0, axis=-1)   # True wherever any channel is non-zero
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return image  # all black — return as-is, tiling will filter it out
    return image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_image(
    image: np.ndarray,
    tile_size: int = 128,
    stride: Optional[int] = None,
    min_variance: float = 50.0,
    max_nodata_frac: float = 0.0,
) -> List[np.ndarray]:
    """Slide a window across an image and return a list of patches.

    Args:
        image:           (H, W, 3) uint8 numpy array.
        tile_size:       Height and width of each square tile.
        stride:          Step between tile origins. Defaults to tile_size
                         (non-overlapping). Use stride < tile_size for overlap.
        min_variance:    Tiles whose per-pixel variance (across all channels)
                         is below this threshold are skipped — blank, cloudy,
                         or water pixels rarely help training.
        max_nodata_frac: Skip tiles where more than this fraction of pixels are
                         pure black (0, 0, 0) — nodata fill from scene edges.
                         Default 0.0: any single black pixel discards the tile,
                         since valid satellite data is almost never exactly black.

    Returns:
        List of (tile_size, tile_size, 3) uint8 numpy arrays.
    """
    if stride is None:
        stride = tile_size

    H, W, _ = image.shape
    tiles: List[np.ndarray] = []

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile = image[y : y + tile_size, x : x + tile_size]
            if tile.shape[:2] != (tile_size, tile_size):
                continue
            # Nodata filter: skip tiles with too many pure-black pixels
            black_pixels = np.all(tile == 0, axis=-1).sum()
            if black_pixels / (tile_size * tile_size) > max_nodata_frac:
                continue
            # Variance filter: skip near-uniform regions
            if float(tile.astype(np.float32).var()) < min_variance:
                continue
            tiles.append(tile.copy())

    return tiles


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_tiles(
    tiles: List[np.ndarray],
    out_dir: str | Path,
    prefix: str = "tile",
) -> None:
    """Save tiles as PNG files.

    Args:
        tiles:   List of (H, W, 3) uint8 arrays.
        out_dir: Directory to write files into (created if absent).
        prefix:  Filename prefix; files are named ``{prefix}_{i:05d}.png``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, tile in enumerate(tqdm(tiles, desc=f"Saving to {out_dir.name}", unit="tile")):
        img = Image.fromarray(tile.astype(np.uint8), mode="RGB")
        img.save(out_dir / f"{prefix}_{i:05d}.png")
