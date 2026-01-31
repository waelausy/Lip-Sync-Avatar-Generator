"""
Dataset PyTorch pour HygieSync
Split 80/20 sans fuite de données
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .config import MEL_WIN, TRAIN_SPLIT


class MouthDataset(Dataset):
    def __init__(self, ds_dir: str, mode: str = "train"):
        """
        Args:
            ds_dir: Répertoire du dataset préparé
            mode: "train" ou "val"
        """
        self.ds_dir = ds_dir
        self.mode = mode

        meta_path = os.path.join(ds_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        self.n = meta["n_frames"]
        self.mel = np.load(os.path.join(ds_dir, "mel.npy"))  # [80, T]

        split = int(self.n * TRAIN_SPLIT)
        if mode == "train":
            self.i0, self.i1 = 0, split
        else:
            self.i0, self.i1 = split, self.n

    def __len__(self):
        return max(0, (self.i1 - self.i0) - MEL_WIN)

    def __getitem__(self, k):
        t = self.i0 + k + MEL_WIN // 2

        x_path = os.path.join(self.ds_dir, "X_masked", f"{t:06d}.png")
        y_path = os.path.join(self.ds_dir, "Y", f"{t:06d}.png")
        m_path = os.path.join(self.ds_dir, "M", f"{t:06d}.png")

        x = cv2.imread(x_path)
        y = cv2.imread(y_path)
        m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

        if x is None or y is None or m is None:
            raise FileNotFoundError(f"Missing file at index {t}")

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m = (m.astype(np.float32) / 255.0)[None, ...]  # [1,H,W]

        a0 = t - MEL_WIN // 2
        a1 = a0 + MEL_WIN
        mel_chunk = self.mel[:, a0:a1]   # [80, MEL_WIN]
        mel_chunk = mel_chunk[None, ...]  # [1,80,MEL_WIN]

        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y).permute(2, 0, 1)
        m = torch.from_numpy(m)
        mel_chunk = torch.from_numpy(mel_chunk.copy())

        return x, mel_chunk, y, m
