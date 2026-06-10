"""
PyTorch Dataset that reads matched (features, labels) pairs from the zarr store.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .features import normalize


class SevereWindDataset(Dataset):
    """
    Each sample is one convective day from one 00Z NWP init, covering 4 consecutive
    6-hourly leads stacked along the channel axis:
      features : float32 tensor (4*C, NY, NX)  — normalised NWP fields
      label    : float32 tensor (C_out, NY, NX) — 0/1 per grid cell
    """

    def __init__(
        self,
        features_store: str | Path,
        labels_store: str | Path,
        start: str,                   # YYYYMMDDHH, inclusive
        end: str,                     # YYYYMMDDHH, inclusive
        forecast_days: list[int],
        norm_stats_path: str | Path,
        conus_mask_path: str | Path,
        positive_oversample_ratio: float = 0.5,
        feature_names: list[str] | None = None,
        hazard_channels: list[int] | None = None,
    ):
        import zarr

        mask_path = Path(conus_mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(
                f"CONUS mask not found at {mask_path}. "
                "Run: python scripts/build_conus_mask.py --config config.yaml"
            )
        self.domain_mask: np.ndarray = np.load(mask_path)

        self.features_root = zarr.open(str(features_store), mode="r")
        self.labels_root = zarr.open(str(labels_store), mode="r")
        stats = np.load(norm_stats_path)
        self.mean = stats["mean"]
        self.std = stats["std"]

        stored_names = list(self.features_root.attrs.get("feature_names", []))
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

        start_dt = datetime.strptime(start, "%Y%m%d%H")
        end_dt = datetime.strptime(end, "%Y%m%d%H")

        # Day N → leads (N-1)*24 + [12, 18, 24, 30]
        forecast_periods = [
            [(day - 1) * 24 + 12 + i * 6 for i in range(4)]
            for day in forecast_days
        ]

        all_feature_keys = list(self.features_root["features"].keys())
        feat_set = set(all_feature_keys)
        self.samples: list[tuple[list[str], str]] = []  # (feat_keys, label_key)

        # Collect unique 00Z init dates within [start_dt, end_dt]
        init_dates: set[str] = set()
        for fk in all_feature_keys:
            try:
                date_part, _ = fk.split("_f")
            except ValueError:
                continue
            if date_part[-2:] != "00":
                continue
            init_dt = datetime.strptime(date_part, "%Y%m%d%H")
            if not (start_dt <= init_dt <= end_dt):
                continue
            init_dates.add(date_part)

        label_keys = set(self.labels_root.get("labels", {}).keys())
        for date_part in sorted(init_dates):
            init_dt = datetime.strptime(date_part, "%Y%m%d%H")
            for period_leads in forecast_periods:
                feat_keys = [f"{date_part}_f{lead:03d}" for lead in period_leads]
                if not all(k in feat_set for k in feat_keys):
                    continue
                label_dt = init_dt + timedelta(hours=period_leads[0])
                label_key = label_dt.strftime("%Y%m%d%H")
                if label_key in label_keys:
                    self.samples.append((feat_keys, label_key))

        # Partition into positive and negative samples for oversampling
        self._pos_idx: list[int] = []
        self._neg_idx: list[int] = []
        for i, (_, lk) in enumerate(self.samples):
            if self.labels_root["labels"][lk][:].max() > 0:
                self._pos_idx.append(i)
            else:
                self._neg_idx.append(i)

        self._pos_ratio = positive_oversample_ratio
        self._hazard_channels = list(hazard_channels) if hazard_channels is not None else [0, 1, 2, 3]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pos_idx and random.random() < self._pos_ratio:
            idx = random.choice(self._pos_idx)

        feat_keys, lk = self.samples[idx]
        stacked = []
        for fk in feat_keys:
            arr = self.features_root["features"][fk][:]        # (C, H, W)
            if self._feat_idx is not None:
                arr = arr[self._feat_idx]             # (C_sel, H, W)
            arr = normalize(arr, self.mean, self.std)
            stacked.append(arr)
        features = np.nan_to_num(np.concatenate(stacked, axis=0), nan=0.0)  # (4*C_sel, H, W)

        label = self.labels_root["labels"][lk][:]               # (3, H, W)
        any_ch = label.max(axis=0, keepdims=True)        # (1, H, W)
        label = np.concatenate([label, any_ch], axis=0)  # (4, H, W)
        label = label[self._hazard_channels]              # (C_out, H, W)

        return (
            torch.from_numpy(features),
            torch.from_numpy(label),
        )


def make_dataloaders(
    cfg,
    norm_stats_path: str | Path,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
) -> tuple[DataLoader, DataLoader]:
    feature_names = list(cfg.dataset.get("feature_names", [])) or None
    hazard_channels = list(cfg.model.get("hazard_channels", [0, 1, 2, 3]))
    forecast_days = list(cfg.nwp.forecast_days)

    conus_mask_path = cfg.domain.conus_mask_path
    train_ds = SevereWindDataset(
        features_store=cfg.dataset.features_store,
        labels_store=cfg.dataset.labels_store,
        start=train_start,
        end=train_end,
        forecast_days=forecast_days,
        norm_stats_path=norm_stats_path,
        conus_mask_path=conus_mask_path,
        positive_oversample_ratio=cfg.dataset.positive_only_ratio,
        feature_names=feature_names,
        hazard_channels=hazard_channels,
    )
    val_ds = SevereWindDataset(
        features_store=cfg.dataset.features_store,
        labels_store=cfg.dataset.labels_store,
        start=val_start,
        end=val_end,
        forecast_days=forecast_days,
        norm_stats_path=norm_stats_path,
        conus_mask_path=conus_mask_path,
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
