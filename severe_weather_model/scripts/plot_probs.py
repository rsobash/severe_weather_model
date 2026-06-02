"""
Plot model probability maps for a single 00Z initialization time.

Produces a grid of panels: rows = forecast days, columns = hazard types
(wind, hail, tornado, any). If calibration files exist they are applied
automatically unless --no-calibration is passed.

Usage (run from severe_weather_model/):
    python scripts/plot_probs.py --config config.yaml \
        --checkpoint-dir models/checkpoints \
        --init 2023060100 \
        --output plots/probs_2023060100.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from severe_weather.evaluate import TemperatureScaler
from severe_weather.features import normalize
from severe_weather.grid import (
    PROJ_STRING,
    SW_X,
    SW_Y,
    get_grid_latlons,
    get_grid_params,
)
from severe_weather.labels import _HAZARD_ORDER
from severe_weather.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_HAZARD_NAMES = [*_HAZARD_ORDER, "any"]   # wind, hail, tornado, any
_HAZARD_COLORS = {
    "wind":    "Blues",
    "hail":    "Greens",
    "tornado": "Purples",
    "any":     "Reds",
}

# Discrete probability bins; cells below the first threshold are not filled.
THRESHOLDS = [0.05, 0.15, 0.30, 0.45, 0.60]
_BOUNDARIES = THRESHOLDS + [1.01]   # sentinel keeps ≥60% in the last bin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Base dir; expects <dir>/day{N}/best.pt per forecast day")
    p.add_argument("--init", required=True, metavar="YYYYMMDDHH",
                   help="00Z initialization time to plot")
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: plots/probs_<init>.png)")
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip temperature scaling even if cal files exist")
    return p.parse_args()


def load_features_for_day(
    root,
    init_str: str,
    day: int,
    feat_idx: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor | None:
    """Stack four 6-hourly leads for day N into a (1, 4*C, NY, NX) tensor."""
    leads = [(day - 1) * 24 + 12 + i * 6 for i in range(4)]
    keys = [f"{init_str}_f{lead:03d}" for lead in leads]
    feat_group = root["features"]
    if not all(k in feat_group for k in keys):
        missing = [k for k in keys if k not in feat_group]
        log.warning(f"Day {day}: missing zarr keys: {missing}")
        return None

    stacked = []
    for k in keys:
        arr = feat_group[k][:]               # (C, NY, NX)
        if feat_idx is not None:
            arr = arr[feat_idx]
        arr = normalize(arr, mean, std)
        stacked.append(arr)

    features = np.nan_to_num(np.concatenate(stacked, axis=0), nan=0.0)
    return torch.from_numpy(features).unsqueeze(0)   # (1, 4*C, NY, NX)


def run_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Return sigmoid probabilities (C_out, NY, NX)."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))           # (1, C_out, NY, NX)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def _make_discrete_cmap_norm(base_cmap: str):
    """Return a (ListedColormap, BoundaryNorm) pair for the global THRESHOLDS."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap_obj = plt.get_cmap(base_cmap)
    # Sample from [0.3, 0.95] so the lightest color is still visible on the map background.
    colors = [cmap_obj(v) for v in np.linspace(0.3, 0.95, len(THRESHOLDS))]
    listed = ListedColormap(colors)
    norm = BoundaryNorm(_BOUNDARIES, ncolors=listed.N)
    return listed, norm


def make_map_axes(fig, nrows: int, ncols: int):
    import cartopy.crs as ccrs
    lcc = ccrs.LambertConformal(
        central_longitude=-95,
        central_latitude=25,
        standard_parallels=(25, 25),
        globe=ccrs.Globe(semimajor_axis=6371200, semiminor_axis=6371200),
    )
    axes = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1, projection=lcc)
            row.append(ax)
        axes.append(row)
    return axes


def draw_panel(ax, prob_grid, lats, lons, title, cmap_obj, norm):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent([-125, -65, 22, 52], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.LAKES,     facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.STATES,    linewidth=0.3, edgecolor="#888888", zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor="#555555", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555555", zorder=2)

    # Mask cells below the lowest threshold so the map background shows through.
    masked = np.ma.masked_where(prob_grid < THRESHOLDS[0], prob_grid)
    pc = ax.pcolormesh(
        lons, lats, masked,
        cmap=cmap_obj,
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=1,
        shading="nearest",
    )

    ax.set_title(title, fontsize=8, pad=3)
    return pc


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_str = args.init
    try:
        init_dt = datetime.strptime(init_str, "%Y%m%d%H")
    except ValueError:
        sys.exit(f"--init must be YYYYMMDDHH, got: {init_str!r}")
    if init_str[-2:] != "00":
        log.warning("--init does not end in '00'; only 00Z inits have features in the zarr store.")

    ckpt_base = Path(args.checkpoint_dir)
    cal_dir = Path(cfg.calibration.calibration_dir)
    feature_names = list(cfg.dataset.get("feature_names", [])) or None
    hazard_channels = list(cfg.model.get("hazard_channels", [0, 1, 2, 3]))
    forecast_days = list(cfg.graphcast.forecast_days)

    import zarr
    root = zarr.open(str(cfg.dataset.zarr_store), mode="r")
    stats = np.load(cfg.dataset.normalization_stats)
    mean, std = stats["mean"], stats["std"]

    stored_names = list(root.attrs.get("feature_names", []))
    feat_idx: np.ndarray | None = None
    if feature_names:
        feat_idx = np.array([stored_names.index(n) for n in feature_names])
        mean = mean[feat_idx]
        std = std[feat_idx]

    # Determine in_channels from first available day's feature shape
    in_channels: int | None = None
    for day in forecast_days:
        x = load_features_for_day(root, init_str, day, feat_idx, mean, std)
        if x is not None:
            in_channels = x.shape[1]
            break
    if in_channels is None:
        sys.exit(f"No feature data found in zarr store for init {init_str}.")

    log.info(f"in_channels={in_channels}, hazard_channels={hazard_channels}, device={device}")

    labels_grp = root.get("labels", {})

    grid_name = cfg.domain.grid
    lats, lons = get_grid_latlons(grid_name)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build per-hazard discrete cmap/norm once.
    hazard_cmaps: dict[str, tuple] = {
        _HAZARD_NAMES[ch]: _make_discrete_cmap_norm(_HAZARD_COLORS[_HAZARD_NAMES[ch]])
        for ch in hazard_channels
    }

    # Collect probability grids: probs[day_idx][local_ch] = (NY, NX)
    day_probs: list[tuple[int, np.ndarray]] = []   # (day, prob array (C_out, NY, NX))

    for day in forecast_days:
        ckpt_path = ckpt_base / f"day{day}" / "best.pt"
        if not ckpt_path.exists():
            log.warning(f"Day {day}: no checkpoint at {ckpt_path}, skipping.")
            continue

        x = load_features_for_day(root, init_str, day, feat_idx, mean, std)
        if x is None:
            log.warning(f"Day {day}: features unavailable, skipping.")
            continue

        model = build_model(cfg, in_channels=in_channels)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        model.to(device)

        cal_path = cal_dir / f"temperature_day{day}.pt"
        if not args.no_calibration and cal_path.exists():
            log.info(f"Day {day}: applying calibration from {cal_path}")
            inference_model = TemperatureScaler.load(model, cal_path)
            inference_model.to(device)
        else:
            if not args.no_calibration:
                log.info(f"Day {day}: no calibration file found, using raw logits.")
            inference_model = model

        probs = run_inference(inference_model, x, device)  # (C_out, NY, NX)
        day_probs.append((day, probs))
        log.info(f"Day {day}: prob max={probs.max():.3f} mean={probs.mean():.4f}")

    if not day_probs:
        sys.exit("No probability maps produced — check checkpoint paths and zarr features.")

    # Build figure: rows = forecast days, cols = hazard channels
    hazard_names_subset = [_HAZARD_NAMES[ch] for ch in hazard_channels]
    nrows = len(day_probs)
    ncols = len(hazard_channels)

    fig_w = max(5 * ncols, 10)
    fig_h = max(3.2 * nrows, 6)
    fig = plt.figure(figsize=(fig_w, fig_h))

    init_label = init_dt.strftime("%Y-%m-%d %HZ")
    axes_grid = make_map_axes(fig, nrows, ncols)
    pcs = {}   # hazard → (pcolormesh, cmap_obj, norm) for colorbar

    for row_idx, (day, probs) in enumerate(day_probs):
        lead_start = (day - 1) * 24 + 12
        valid_dt = init_dt + timedelta(hours=lead_start)
        valid_label = valid_dt.strftime("%Y-%m-%d %HZ")

        label_key = valid_dt.strftime("%Y%m%d%H")
        obs_labels = None
        if label_key in labels_grp:
            raw = labels_grp[label_key][:]
            any_ch = raw.max(axis=0, keepdims=True)
            obs_labels = np.concatenate([raw, any_ch], axis=0)  # (4, NY, NX)
            log.info(f"Day {day}: loaded observed labels for {label_key}")

        for col_idx, (local_ch, global_ch) in enumerate(enumerate(hazard_channels)):
            hazard = _HAZARD_NAMES[global_ch]
            cmap_obj, norm = hazard_cmaps[hazard]
            ax = axes_grid[row_idx][col_idx]
            title = f"Day {day} · {hazard.title()}\n(valid ~{valid_label})"
            pc = draw_panel(
                ax, probs[local_ch], lats, lons,
                title=title,
                cmap_obj=cmap_obj,
                norm=norm,
            )
            pcs[hazard] = (pc, cmap_obj, norm)

            # Dot observed LSR locations if available
            if obs_labels is not None and global_ch < obs_labels.shape[0]:
                obs = obs_labels[global_ch]   # (NY, NX)
                pos_rows, pos_cols = np.where(obs > 0)
                if pos_rows.size:
                    import cartopy.crs as ccrs
                    ax.scatter(
                        lons[pos_rows, pos_cols],
                        lats[pos_rows, pos_cols],
                        s=6, c="k", marker="x", linewidths=0.6,
                        transform=ccrs.PlateCarree(), zorder=4, label="LSR obs"
                    )

    # Colorbars — one per hazard, discrete ticks at threshold boundaries.
    n_cbars = len(pcs)
    cbar_h = 0.025
    cbar_y = 0.04
    cbar_w = 0.7 / n_cbars
    tick_labels = [f"{int(t * 100)}%" for t in THRESHOLDS]
    for i, (hazard, (pc, cmap_obj, norm)) in enumerate(pcs.items()):
        cax = fig.add_axes([0.15 + i * (cbar_w + 0.04), cbar_y, cbar_w, cbar_h])
        cb = fig.colorbar(pc, cax=cax, orientation="horizontal", spacing="proportional")
        cb.set_ticks(THRESHOLDS)
        cb.set_ticklabels(tick_labels)
        cb.set_label(f"{hazard.title()} probability", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Severe weather probabilities — init {init_label} — {grid_name} grid",
        fontsize=11, y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10,
                        wspace=0.04, hspace=0.25)

    out = args.output
    if out is None:
        out = f"plots/probs_{init_str}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
