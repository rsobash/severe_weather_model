"""
Calibrate and evaluate per-day trained models on the test set.

Usage:
    python scripts/evaluate.py --config config.yaml --checkpoint-dir models/checkpoints \\
        --val-start 2022010100 --val-end 2022123118 \\
        --test-start 2023010100 --test-end 2023123118
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from severe_weather.dataset import SevereWindDataset
from severe_weather.labels import _HAZARD_ORDER
from severe_weather.evaluate import (
    TemperatureScaler,
    compute_metrics,
    plot_reliability,
    _collect_preds,
)
from severe_weather.model import build_model
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Base checkpoint directory; expects <dir>/day{N}/best.pt per forecast day")
    p.add_argument("--val-start", required=True, metavar="YYYYMMDDHH",
                   help="First init time for validation / calibration (inclusive)")
    p.add_argument("--val-end", required=True, metavar="YYYYMMDDHH",
                   help="Last init time for validation / calibration (inclusive)")
    p.add_argument("--test-start", required=True, metavar="YYYYMMDDHH",
                   help="First init time for evaluation (inclusive)")
    p.add_argument("--test-end", required=True, metavar="YYYYMMDDHH",
                   help="Last init time for evaluation (inclusive)")
    p.add_argument("--skip-calibration", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_stats = cfg.dataset.normalization_stats
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = Path(cfg.calibration.calibration_dir)
    cal_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(cfg.dataset.get("feature_names", [])) or None
    hazard_channels = list(cfg.model.get("hazard_channels", [0, 1, 2, 3]))
    forecast_days = list(cfg.graphcast.forecast_days)
    ckpt_base = Path(args.checkpoint_dir)

    _HAZARD_NAMES = [*_HAZARD_ORDER, "any"]

    # Infer in_channels from first day's val data
    first_ds = SevereWindDataset(
        zarr_store=cfg.dataset.zarr_store,
        start=args.val_start,
        end=args.val_end,
        forecast_days=[forecast_days[0]],
        norm_stats_path=norm_stats,
        conus_mask_path=cfg.domain.conus_mask_path,
        positive_oversample_ratio=0.0,
        feature_names=feature_names,
        hazard_channels=hazard_channels,
    )
    sample_feats, _ = first_ds[0]
    in_channels = sample_feats.shape[0]
    domain_mask = first_ds.domain_mask  # (NY, NX) bool

    def _metrics_for(p, l):
        return {
            _HAZARD_NAMES[global_ch]: compute_metrics(
                p[local_ch], l[local_ch],
                thresholds=cfg.evaluation.prob_thresholds,
            )
            for local_ch, global_ch in enumerate(hazard_channels)
        }

    def _serialize(m: dict) -> dict:
        return {n: {k: float(v) for k, v in ch.items()} for n, ch in m.items()}

    # ── Per-day loop ──────────────────────────────────────────────────────────
    all_cal_probs: list[np.ndarray] = []
    all_raw_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    by_day: dict = {}

    for day in forecast_days:
        ckpt_path = ckpt_base / f"day{day}" / "best.pt"
        if not ckpt_path.exists():
            log.warning(f"No checkpoint for day {day} at {ckpt_path}, skipping.")
            continue

        model = build_model(cfg, in_channels=in_channels)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        model.to(device)
        log.info(f"Day {day}: loaded checkpoint from {ckpt_path}")

        val_ds = SevereWindDataset(
            zarr_store=cfg.dataset.zarr_store,
            start=args.val_start,
            end=args.val_end,
            forecast_days=[day],
            norm_stats_path=norm_stats,
            conus_mask_path=cfg.domain.conus_mask_path,
            positive_oversample_ratio=0.0,
            feature_names=feature_names,
            hazard_channels=hazard_channels,
        )
        test_ds = SevereWindDataset(
            zarr_store=cfg.dataset.zarr_store,
            start=args.test_start,
            end=args.test_end,
            forecast_days=[day],
            norm_stats_path=norm_stats,
            conus_mask_path=cfg.domain.conus_mask_path,
            positive_oversample_ratio=0.0,
            feature_names=feature_names,
            hazard_channels=hazard_channels,
        )
        if len(test_ds) == 0:
            log.warning(f"No test samples for day {day}, skipping.")
            continue

        val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            num_workers=cfg.training.num_workers)
        test_dl = DataLoader(test_ds, batch_size=cfg.training.batch_size,
                             num_workers=cfg.training.num_workers)

        cal_path = cal_dir / f"temperature_day{day}.pt"
        if not args.skip_calibration:
            log.info(f"Day {day}: fitting temperature scaling …")
            scaler = TemperatureScaler(model, n_channels=len(hazard_channels)).fit(
                val_dl, device, domain_mask=domain_mask
            )
            scaler.save(cal_path)
            calibrated_model = scaler
        elif cal_path.exists():
            log.info(f"Day {day}: loading saved calibration from {cal_path}")
            calibrated_model = TemperatureScaler.load(model, cal_path)
            calibrated_model.to(device)
        else:
            log.warning(f"Day {day}: no calibration applied.")
            calibrated_model = model

        log.info(f"Day {day}: collecting test predictions ({len(test_ds)} samples) …")
        day_probs, day_labels = _collect_preds(
            calibrated_model, test_dl, device, domain_mask=domain_mask
        )
        all_cal_probs.append(day_probs)
        all_labels.append(day_labels)

        day_entry: dict = {"calibrated": _serialize(_metrics_for(day_probs, day_labels))}

        if calibrated_model is not model:
            day_raw_probs, _ = _collect_preds(model, test_dl, device, domain_mask=domain_mask)
            all_raw_probs.append(day_raw_probs)
            day_entry["uncalibrated"] = _serialize(_metrics_for(day_raw_probs, day_labels))

        by_day[f"day{day}"] = day_entry
        summary_name = "any" if "any" in day_entry["calibrated"] else _HAZARD_NAMES[hazard_channels[0]]
        sm = day_entry["calibrated"][summary_name]
        log.info(f"  day{day} [{summary_name}] brier={sm['brier_score']:.4f} bss={sm['brier_skill_score']:.4f} auc_roc={sm['auc_roc']:.4f}")

    # ── Aggregate overall metrics across all days ─────────────────────────────
    output: dict = {}
    if all_cal_probs:
        probs = np.concatenate(all_cal_probs, axis=1)   # (C, N_total)
        labels = np.concatenate(all_labels, axis=1)     # (C, N_total)
        raw_probs = np.concatenate(all_raw_probs, axis=1) if all_raw_probs else None

        cal_metrics = _metrics_for(probs, labels)
        output["calibrated"] = _serialize(cal_metrics)
        for name, m in cal_metrics.items():
            log.info(f"  [{name}] brier={m['brier_score']:.4f} bss={m['brier_skill_score']:.4f} auc_roc={m['auc_roc']:.4f}")

        if raw_probs is not None:
            raw_metrics = _metrics_for(raw_probs, labels)
            output["uncalibrated"] = _serialize(raw_metrics)
            for name, m in raw_metrics.items():
                log.info(f"  [{name}] (uncal) brier={m['brier_score']:.4f} bss={m['brier_skill_score']:.4f} auc_roc={m['auc_roc']:.4f}")

        for local_ch, global_ch in enumerate(hazard_channels):
            name = _HAZARD_NAMES[global_ch]
            plot_reliability(
                probs[local_ch], labels[local_ch],
                output_dir / f"reliability_{name}.png",
                hazard_name=name,
                probs_pre=raw_probs[local_ch] if raw_probs is not None else None,
            )

    if by_day:
        output["by_forecast_day"] = by_day

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
