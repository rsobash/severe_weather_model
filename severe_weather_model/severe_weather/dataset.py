"""
PyTorch Dataset that reads matched (features, labels) pairs from the zarr store.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .features import normalize


class SevereWindDataset(Dataset):
    """
    Each sample is one NWP valid time at one lead hour:
      features : float32 tensor (C, NY, NX)  — normalised NWP fields
      label    : float32 tensor (1, NY, NX)  — 0/1 per grid cell
    """

    def __init__(
        self,
        zarr_store: str | Path,
        years: list[int],
        lead_hours: list[int],
        norm_stats_path: str | Path,
        positive_oversample_ratio: float = 0.5,
        feature_names: list[str] | None = None,
        hazard_channels: list[int] | None = None,
    ):
        import zarr

        self.root = zarr.open(str(zarr_store), mode="r")
        stats = np.load(norm_stats_path)
        self.mean = stats["mean"]
        self.std = stats["std"]

        stored_names = list(self.root.attrs.get("feature_names", []))
        if feature_names:
            unknown = [n for n in feature_names if n not in stored_names]
            if unknown:
                raise ValueError(f"Unknown feature(s) not in zarr store: {unknown}. "
                                 f"Available: {stored_names}")
            self._feat_idx = np.array([stored_names.index(n) for n in feature_names])
            self.mean = self.mean[self._feat_idx]
            self.std = self.std[self._feat_idx]
        else:
            self._feat_idx = None

        # Build index of keys present in the store for the requested years/leads
        all_feature_keys = list(self.root["features"].keys())
        self.samples: list[tuple[str, str]] = []  # (feature_key, label_key)

        for fk in all_feature_keys:
            # key format: YYYYMMDDHH_f{lead:03d}
            try:
                date_part, lead_part = fk.split("_f")
                year = int(date_part[:4])
                lead = int(lead_part)
            except ValueError:
                continue
            if year not in years or lead not in lead_hours:
                continue
            label_key = date_part  # YYYYMMDDHH
            if label_key in self.root.get("labels", {}):
                self.samples.append((fk, label_key))

        # Partition into positive and negative samples for oversampling
        self._pos_idx: list[int] = []
        self._neg_idx: list[int] = []
        for i, (fk, lk) in enumerate(self.samples):
            label = self.root["labels"][lk][:]
            if label.max() > 0:
                self._pos_idx.append(i)
            else:
                self._neg_idx.append(i)

        self._pos_ratio = positive_oversample_ratio
        self._hazard_channels = list(hazard_channels) if hazard_channels is not None else [0, 1, 2, 3]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Oversample positives: with probability _pos_ratio, pick from positive pool
        if self._pos_idx and random.random() < self._pos_ratio:
            idx = random.choice(self._pos_idx)

        fk, lk = self.samples[idx]
        features = self.root["features"][fk][:]          # (C, H, W)
        if self._feat_idx is not None:
            features = features[self._feat_idx]
        label = self.root["labels"][lk][:]               # (3, H, W)
        any_ch = label.max(axis=0, keepdims=True)        # (1, H, W)
        label = np.concatenate([label, any_ch], axis=0)  # (4, H, W)
        label = label[self._hazard_channels]              # (C_out, H, W)

        features = normalize(features, self.mean, self.std)

        return (
            torch.from_numpy(features),
            torch.from_numpy(label),
        )



def make_dataloaders(cfg, norm_stats_path: str | Path) -> tuple[DataLoader, DataLoader]:
    feature_names = list(cfg.dataset.get("feature_names", [])) or None
    hazard_channels = list(cfg.model.get("hazard_channels", [0, 1, 2, 3]))
    train_ds = SevereWindDataset(
        zarr_store=cfg.dataset.zarr_store,
        years=cfg.dataset.train_years,
        lead_hours=list(range(6, 49, 6)),
        norm_stats_path=norm_stats_path,
        positive_oversample_ratio=cfg.dataset.positive_only_ratio,
        feature_names=feature_names,
        hazard_channels=hazard_channels,
    )
    val_ds = SevereWindDataset(
        zarr_store=cfg.dataset.zarr_store,
        years=cfg.dataset.val_years,
        lead_hours=list(range(6, 49, 6)),
        norm_stats_path=norm_stats_path,
        positive_oversample_ratio=0.0,
        feature_names=feature_names,
        hazard_channels=hazard_channels,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl
