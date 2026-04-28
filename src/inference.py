"""
inference.py — Run super-resolution on full images of any size.

Handles padding, degradation, model inference, bicubic baseline,
and optional metrics when ground truth is available.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from degrade import degrade, bicubic_upscale


def superres_full_image(
    image: np.ndarray,
    model: torch.nn.Module,
    scale: int = 4,
    device: torch.device | str = "cpu",
) -> dict:
    """Run super-resolution on an arbitrarily-sized image.

    The image is treated as the HR ground truth.  It is cropped to be
    divisible by ``scale``, degraded to create a synthetic LR input,
    then reconstructed by both bicubic upscaling and the CNN.

    Args:
        image:  (H, W, 3) uint8 numpy array (any resolution).
        model:  A trained ``SuperResCNN`` (or any fully-conv model).
        scale:  The upsampling factor the model was trained for.
        device: Torch device to run inference on.

    Returns:
        dict with keys ``'hr'``, ``'lr'``, ``'bicubic'``, ``'pred'`` —
        all uint8 numpy arrays.  ``hr`` is the (possibly cropped) ground
        truth at full resolution.
    """
    h, w = image.shape[:2]

    # Crop to nearest multiple of scale so degrade/upscale round-trips cleanly
    h_crop = h - (h % scale)
    w_crop = w - (w % scale)
    hr = image[:h_crop, :w_crop]

    lr = degrade(hr, scale=scale)
    bic = bicubic_upscale(lr, scale=scale)

    # Model inference
    model.eval()
    lr_t = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        pred_t = model(lr_t.to(device)).cpu()
    pred = (pred_t[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    return {"hr": hr, "lr": lr, "bicubic": bic, "pred": pred}


def load_any_image(path: str | object) -> np.ndarray:
    """Load an image from any path, converting to RGB uint8."""
    img = Image.open(str(path)).convert("RGB")
    return np.array(img, dtype=np.uint8)
