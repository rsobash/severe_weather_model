"""
Calibrate and evaluate a trained model on the test set.

Usage:
    python scripts/evaluate.py --config config.yaml --checkpoint models/checkpoints/best.pt \\
        --val-start 2022010100 --val-end 2022123118 \\
        --test-start 2023010100 --test-end 2023123118
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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
    p.add_argument("--checkpoint", required=True)
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

    # ── Load model ────────────────────────────────────────────────────────────
    feature_names = list(cfg.dataset.get("feature_names", [])) or None
    hazard_channels = list(cfg.model.get("hazard_channels", [0, 1, 2, 3]))
    forecast_days = list(cfg.nwp.forecast_days)
    val_ds = SevereWindDataset(
        features_store=cfg.dataset.features_store,
        labels_store=cfg.dataset.labels_store,
        start=args.val_start,
        end=args.val_end,
        forecast_days=forecast_days,
        norm_stats_path=norm_stats,
        conus_mask_path=cfg.domain.conus_mask_path,
        positive_oversample_ratio=0.0,
        feature_names=feature_names,
        hazard_channels=hazard_channels,
    )
    test_ds = SevereWindDataset(
        features_store=cfg.dataset.features_store,
        labels_store=cfg.dataset.labels_store,
        start=args.test_start,
        end=args.test_end,
        forecast_days=forecast_days,
        norm_stats_path=norm_stats,
        conus_mask_path=cfg.domain.conus_mask_path,
        positive_oversample_ratio=0.0,
        feature_names=feature_names,
        hazard_channels=hazard_channels,
    )

    # Infer in_channels
    sample_feats, _ = val_ds[0]
    in_channels = sample_feats.shape[0]

    model = build_model(cfg, in_channels=in_channels)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                        num_workers=cfg.training.num_workers)
    test_dl = DataLoader(test_ds, batch_size=cfg.training.batch_size,
                         num_workers=cfg.training.num_workers)

    domain_mask = test_ds.domain_mask  # (NY, NX) bool

    # ── Temperature scaling ───────────────────────────────────────────────────
    cal_path = cal_dir / "temperature.pt"
    if not args.skip_calibration:
        log.info("Fitting temperature scaling on validation set …")
        scaler = TemperatureScaler(model, n_channels=len(hazard_channels)).fit(
            val_dl, device, domain_mask=domain_mask
        )
        scaler.save(cal_path)
        calibrated_model = scaler
    elif cal_path.exists():
        log.info(f"Loading saved calibration from {cal_path}")
        calibrated_model = TemperatureScaler.load(model, cal_path)
        calibrated_model.to(device)
    else:
        log.warning("No calibration applied.")
        calibrated_model = model

    # ── Evaluate on test set ─────────────────────────────────────────────────
    _HAZARD_NAMES = [*_HAZARD_ORDER, "any"]
    calibration_applied = calibrated_model is not model

    log.info("Collecting test-set predictions …")
    probs, labels = _collect_preds(calibrated_model, test_dl, device, domain_mask=domain_mask)

    raw_probs = None
    if calibration_applied:
        log.info("Collecting uncalibrated test-set predictions …")
        raw_probs, _ = _collect_preds(model, test_dl, device, domain_mask=domain_mask)

    def _metrics_for(p, l):
        return {
            _HAZARD_NAMES[global_ch]: compute_metrics(
                p[local_ch], l[local_ch],
                thresholds=cfg.evaluation.prob_thresholds,
            )
            for local_ch, global_ch in enumerate(hazard_channels)
        }

    cal_metrics = _metrics_for(probs, labels)
    for name, m in cal_metrics.items():
        log.info(f"  [{name}] brier={m['brier_score']:.4f} bss={m['brier_skill_score']:.4f} auc_roc={m['auc_roc']:.4f}")

    for local_ch, global_ch in enumerate(hazard_channels):
        name = _HAZARD_NAMES[global_ch]
        plot_reliability(
            probs[local_ch], labels[local_ch],
            output_dir / f"reliability_{name}.png",
            hazard_name=name,
            probs_pre=raw_probs[local_ch] if raw_probs is not None else None,
        )

    def _serialize(m: dict) -> dict:
        return {n: {k: float(v) for k, v in ch.items()} for n, ch in m.items()}

    output: dict = {"calibrated": _serialize(cal_metrics)}
    if raw_probs is not None:
        raw_metrics = _metrics_for(raw_probs, labels)
        output["uncalibrated"] = _serialize(raw_metrics)
        for name, m in raw_metrics.items():
            log.info(f"  [{name}] (uncal) brier={m['brier_score']:.4f} bss={m['brier_skill_score']:.4f} auc_roc={m['auc_roc']:.4f}")

    # ── Per-forecast-day metrics ──────────────────────────────────────────────
    by_day: dict = {}
    for day in forecast_days:
        day_ds = SevereWindDataset(
            features_store=cfg.dataset.features_store,
            labels_store=cfg.dataset.labels_store,
            start=args.test_start,
            end=args.test_end,
            forecast_days=[day],
            norm_stats_path=norm_stats,
            conus_mask_path=cfg.domain.conus_mask_path,
            positive_oversample_ratio=0.0,
            feature_names=feature_names,
            hazard_channels=hazard_channels,
        )
        if len(day_ds) == 0:
            log.warning(f"No test samples for forecast day {day}, skipping.")
            continue
        day_dl = DataLoader(day_ds, batch_size=cfg.training.batch_size,
                            num_workers=cfg.training.num_workers)
        log.info(f"Day {day}: collecting predictions ({len(day_ds)} samples) …")
        day_probs, day_labels = _collect_preds(
            calibrated_model, day_dl, device, domain_mask=domain_mask
        )
        day_entry: dict = {"calibrated": _serialize(_metrics_for(day_probs, day_labels))}
        if calibration_applied:
            day_raw_probs, _ = _collect_preds(model, day_dl, device, domain_mask=domain_mask)
            day_entry["uncalibrated"] = _serialize(_metrics_for(day_raw_probs, day_labels))
        by_day[f"day{day}"] = day_entry
        summary_name = "any" if "any" in day_entry["calibrated"] else _HAZARD_NAMES[hazard_channels[0]]
        sm = day_entry["calibrated"][summary_name]
        log.info(f"  day{day} [{summary_name}] brier={sm['brier_score']:.4f} bss={sm['brier_skill_score']:.4f} auc_roc={sm['auc_roc']:.4f}")
    if by_day:
        output["by_forecast_day"] = by_day

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
