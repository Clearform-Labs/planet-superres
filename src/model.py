"""
model.py -- Small ESPCN-style super-resolution CNN.

Architecture:
  - Operates at LR resolution (32x32) to keep computation cheap.
  - A few residual blocks for feature extraction.
  - Sub-pixel convolution (PixelShuffle) to upscale 4x to HR (128x128).
  - L1 loss is recommended (less blurry than MSE).
"""

import torch
import torch.nn as nn
import torchvision


class PerceptualLoss(nn.Module):
    """Compare VGG features instead of raw pixels — produces sharper outputs."""

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return nn.functional.l1_loss(self.vgg(pred), self.vgg(target))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class SuperResCNN(nn.Module):
    def __init__(self, scale=4, n_resblocks=4, n_feats=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        # Sub-pixel upsampling: n_feats -> 3 * scale^2 channels, then PixelShuffle
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, 3 * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x  # global residual
        x = self.tail(x)
        return x.clamp(0.0, 1.0)
