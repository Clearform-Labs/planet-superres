"""
degrade.py — Create LR/HR pairs and a bicubic upscale baseline.

All functions operate on numpy uint8 arrays of shape (H, W, 3).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def degrade(
    tile: np.ndarray,
    scale: int = 4,
    method: str = "bicubic",
) -> np.ndarray:
    """Downsample an HR tile to create a low-resolution input.

    Args:
        tile:   (H, W, 3) uint8 numpy array (HR patch).
        scale:  Downsampling factor. Output will be (H/scale, W/scale, 3).
        method: Resampling method. Supported: ``'bicubic'``, ``'bilinear'``,
                ``'nearest'``.

    Returns:
        (H//scale, W//scale, 3) uint8 numpy array.
    """
    resample = _get_resample(method)
    h, w = tile.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    img = Image.fromarray(tile.astype(np.uint8), mode="RGB")
    lr_img = img.resize((lr_w, lr_h), resample=resample)
    return np.array(lr_img, dtype=np.uint8)


def make_pair(
    hr_tile: np.ndarray,
    scale: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an (LR, HR) training pair from a high-resolution tile.

    Args:
        hr_tile: (H, W, 3) uint8 HR patch.
        scale:   Downsampling factor.

    Returns:
        Tuple of (lr, hr) where lr is (H//scale, W//scale, 3) and
        hr is the original array (unchanged).
    """
    lr = degrade(hr_tile, scale=scale, method="bicubic")
    return lr, hr_tile


def bicubic_upscale(
    lr_tile: np.ndarray,
    scale: int = 4,
) -> np.ndarray:
    """Upscale an LR tile using bicubic interpolation (baseline comparator).

    This is the performance floor the trained CNN must beat on PSNR/SSIM.

    Args:
        lr_tile: (H, W, 3) uint8 LR patch.
        scale:   Upsampling factor. Output will be (H*scale, W*scale, 3).

    Returns:
        (H*scale, W*scale, 3) uint8 numpy array.
    """
    h, w = lr_tile.shape[:2]
    img = Image.fromarray(lr_tile.astype(np.uint8), mode="RGB")
    up_img = img.resize((w * scale, h * scale), resample=Image.BICUBIC)
    return np.array(up_img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_resample(method: str) -> int:
    mapping = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
    }
    key = method.lower()
    if key not in mapping:
        raise ValueError(f"Unknown resampling method '{method}'. Choose from: {list(mapping)}")
    return mapping[key]
