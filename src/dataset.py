"""
dataset.py -- PyTorch Dataset that reads LR/HR pairs from manifest.csv.

Augmentation (D4 flips/rotations) is applied on-the-fly during training.
Pixel values are normalized from uint8 [0,255] to float32 [0,1].
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from augment import augment_pair


class SRDataset(Dataset):
    def __init__(self, manifest_path, split="train", augment=True):
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lr = np.array(Image.open(row["lr_path"]))
        hr = np.array(Image.open(row["hr_path"]))

        if self.augment:
            lr, hr = augment_pair(lr, hr)

        # uint8 [0,255] -> float32 [0,1], HWC -> CHW
        lr = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        hr = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        return lr, hr
