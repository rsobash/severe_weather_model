# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All scripts are run from `severe_weather_model/` with `config.yaml` in the working directory.

```bash
# Install dependencies (installs the severe_weather package in editable mode)
pip install -e severe_weather_model/

# 1. Build label grids from SPC LSR SQLite archive
# --start / --end are YYYYMMDD (date only; 12Z is added automatically), both inclusive:
python scripts/build_labels.py --config config.yaml --start 20160101 --end 20231231

# 2a. Build feature zarr store from GraphCast NetCDF files
# Lead-time range is set in config.yaml (graphcast.lead_start / lead_end / lead_interval)
# Restrict to a range of init times with --start / --end (YYYYMMDDHH, both inclusive, both optional):
python scripts/build_features.py --config config.yaml --start 2016010100 --end 2021123118

# 2b. Compute normalisation stats from the zarr store (run once after features are built)
# Restrict to training init times with --start / --end (YYYYMMDDHH, both inclusive, both optional):
python scripts/compute_norm_stats.py --config config.yaml --start 2016010100 --end 2021123118

# 3. Train
# --train-start/--train-end and --val-start/--val-end are YYYYMMDDHH, both inclusive:
python scripts/train.py --config config.yaml \
    --train-start 2016010100 --train-end 2021123118 \
    --val-start 2022010100 --val-end 2022123118
# Override any config key via OmegaConf dotlist, e.g.:
python scripts/train.py --config config.yaml \
    --train-start 2016010100 --train-end 2021123118 \
    --val-start 2022010100 --val-end 2022123118 \
    training.lr=5e-5 model.encoder=resnet18

# 4. Evaluate (calibrate + metrics + reliability diagram)
# --val-start/--val-end used for temperature scaling; --test-start/--test-end for metrics:
python scripts/evaluate.py --config config.yaml --checkpoint models/checkpoints/best.pt \
    --val-start 2022010100 --val-end 2022123118 \
    --test-start 2023010100 --test-end 2023123118
python scripts/evaluate.py --config config.yaml --checkpoint models/checkpoints/best.pt \
    --val-start 2022010100 --val-end 2022123118 \
    --test-start 2023010100 --test-end 2023123118 \
    --skip-calibration
```

## Architecture

The project predicts severe weather probability (wind, hail, tornado) on a CONUS grid using GraphCast NWP output as input and SPC Local Storm Reports (LSRs) as ground truth labels. Two grids are supported, selected via `domain.grid` in `config.yaml`:
- `G212` — 40km Lambert Conformal, 185×129 (default)
- `G211` — 80km Lambert Conformal, 93×65

Switching grids requires rebuilding the zarr store.

**Data pipeline (one-time preprocessing → zarr):**
- `severe_weather/grid.py` — defines both grids in `_GRID_PARAMS` and exposes `get_grid_params(name)` returning `(nx, ny, dx_metres)`. All grid functions (`get_grid_latlons`, `latlons_to_ij_bulk`, `latlon_to_ij`) accept a `grid_name` argument defaulting to `"G212"`. Both grids share the same LCC projection and SW corner.
- `severe_weather/features.py` — reads GraphCast NetCDF files (coordinate names `lat`/`lon`; filename format `weathernext_YYYYMMDDHH_FFF_mean.nc` with zero-padded 3-digit lead hours), regrids each variable from 0.25° lat/lon onto the configured grid via bilinear interpolation (`RegularGridInterpolator`), computes derived met fields (bulk shear, θe, lapse rate, MSLP gradient), appends temporal encoding planes (lead-hour, day-of-year sin/cos, hour-of-day sin/cos), and appends static `lat`/`lon` planes from the grid definition. `extract_features` returns a `(array, names)` tuple — `(C, NY, NX)` float32 array plus a list of channel names. `build_features.py` saves the names to `zarr_store.attrs["feature_names"]`.
- `severe_weather/labels.py` — queries the SQLite LSR archive for severe reports across a 12z–12z convective day and marks grid points within `radius_km` (configurable; typically matched to grid spacing — 40 km for G212, 80 km for G211) using a haversine BallTree. Three report types are combined: wind ≥ `wind_gust_threshold_kts` (50 kt), hail ≥ `hail_size_threshold_in` (1.00"), and tornado ≥ EF`tornado_ef_threshold` (EF0). DB timestamps are stored in CST; the code converts UTC window bounds accordingly. Labels are binary 0/1 per grid point.
- `scripts/build_features.py` and `scripts/build_labels.py` write features and labels into a shared zarr store (`data/processed/severe_labels.zarr`). Features are keyed `features/YYYYMMDDHH_f{lead:03d}`; labels are keyed `labels/YYYYMMDDHH` where `HH=12` (noon UTC, start of the convective day).

**Model:**
- `severe_weather/model.py` — wraps `segmentation_models_pytorch.Unet` with a ResNet encoder (default `resnet34`, random init) producing one logit channel per hazard in `model.hazard_channels` (0=wind, 1=hail, 2=tornado, 3=any; default `[0, 1, 2, 3]` → 4 channels). Loss is binary focal loss (`FocalLoss`; α=0.75, γ=2.0) or weighted BCE, controlled by `config.yaml`.

**Feature selection:**
- `dataset.feature_names` in `config.yaml` controls which extracted channels are passed to the model. Names must match those in `zarr_store.attrs["feature_names"]`. Leave the list empty to use all channels. Both training and evaluation read this key. Adding or removing features requires rebuilding the zarr store only if the new features were never extracted; otherwise it's a config-only change.

**Training:**
- `severe_weather/dataset.py` — `SevereWindDataset` produces one sample per convective day per 00Z init. Each sample stacks 4 consecutive 6-hourly lead arrays along the channel axis → `(4*C, NY, NX)` features. Forecast days are controlled by `graphcast.forecast_days` in `config.yaml` (e.g. `[1, 2, 3, 4]`); leads for Day N are derived as `(N-1)*24 + [12, 18, 24, 30]`. Init times are filtered by explicit `start`/`end` YYYYMMDDHH args (no year-list filtering). The dataset slices to the configured feature subset (if any), normalises each lead independently with the same per-channel stats, and oversamples positive samples at a configurable ratio (default 50%). Augmentation is horizontal/vertical flips (symmetrically valid for CONUS).
- `severe_weather/train.py` — training loop with AMP (`torch.cuda.amp`), AdamW + cosine LR schedule with linear warmup, gradient clipping, and early stopping. Best checkpoint saved to `models/checkpoints/best.pt`.

**Calibration and evaluation:**
- `severe_weather/evaluate.py` — temperature scaling (`TemperatureScaler`) fitted via L-BFGS on the validation set. Metrics: Brier score, BSS, AUC-ROC, AUC-PR, CSI/POD/FAR at SPC-aligned probability thresholds (0.05, 0.10, 0.15, 0.25, 0.45). Outputs `logs/eval/metrics.json` and a reliability diagram PNG.

**Config:**
- `config.yaml` uses OmegaConf. All paths are relative to the `severe_weather_model/` working directory. Any key can be overridden on the command line via dotlist syntax when calling `scripts/train.py`.
- `graphcast.lead_start / lead_end / lead_interval` — used only by `build_features.py` to control which lead files are processed into the zarr store.
- `graphcast.forecast_days` — list of forecast day indices (e.g. `[1, 2, 3, 4]`) used by the dataset at training time. Extending this list (e.g. to Day 7) requires no zarr rebuild as long as the corresponding lead files were already extracted.

**Key data dependency order:** LSR SQLite DB → `build_labels.py` → zarr labels; GraphCast NetCDF files → `build_features.py` (with `--compute-norm` on first run) → zarr features + `data/processed/norm_stats.npz`; both must exist before training. Adding a new feature to `features.py` (e.g. `lat`/`lon`) or switching grids requires rebuilding the zarr store and recomputing norm stats before training. Changing label thresholds (`wind_gust_threshold_kts`, `hail_size_threshold_in`, `tornado_ef_threshold`) or `radius_km` requires rebuilding the zarr labels.
