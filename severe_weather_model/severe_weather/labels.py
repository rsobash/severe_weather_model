"""
Build binary label grids from SPC Local Storm Reports (LSRs).

For each 12z–12z convective day, labels[row, col] = 1 if any qualifying
severe weather report (wind/hail/tornado) fell within radius_km of that grid point.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import functools

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.neighbors import BallTree

from .grid import get_grid_latlons, get_grid_params

_EARTH_RADIUS_KM = 6371.0


@functools.lru_cache(maxsize=2)
def _grid_balltree(grid_name: str) -> BallTree:
    """Build and cache a haversine BallTree over all grid point centres."""
    lats, lons = get_grid_latlons(grid_name)
    coords = np.deg2rad(np.column_stack([lats.ravel(), lons.ravel()]))
    return BallTree(coords, metric="haversine")

log = logging.getLogger(__name__)

_CST_OFFSET = timedelta(hours=6)  # CST = UTC-6; DB timestamps are in CST

def load_lsr_window(
    t_start: datetime,
    t_end: datetime,
    db_path: Path,
    threshold_kts: float,
    hail_size_in: float,
    tornado_ef_threshold: int,
) -> dict[str, pd.DataFrame]:
    """Return per-hazard LSR DataFrames for t_start–t_end (UTC).

    Returns a dict with keys 'wind', 'hail', 'tornado', each a DataFrame
    with columns ['lat', 'lon'] (may be empty).
    """
    empty = pd.DataFrame(columns=["lat", "lon"])

    t0_str = (t_start - _CST_OFFSET).strftime("%Y-%m-%d %H:%M:%S")
    t1_str = (t_end   - _CST_OFFSET).strftime("%Y-%m-%d %H:%M:%S")

    if not db_path.exists():
        log.debug(f"LSR database not found: {db_path}")
        return {"wind": empty, "hail": empty, "tornado": empty}

    with sqlite3.connect(db_path) as con:
        wind = pd.read_sql_query(
            "SELECT slat AS lat, slon AS lon FROM reports_wind "
            "WHERE datetime BETWEEN ? AND ? AND CAST(mag AS REAL) >= ?",
            con, params=(t0_str, t1_str, threshold_kts),
        )
        hail = pd.read_sql_query(
            "SELECT slat AS lat, slon AS lon FROM reports_hail "
            "WHERE datetime BETWEEN ? AND ? AND size >= ?",
            con, params=(t0_str, t1_str, hail_size_in),
        )
        torn = pd.read_sql_query(
            "SELECT slat AS lat, slon AS lon FROM reports_torn "
            "WHERE datetime BETWEEN ? AND ? AND rating >= ?",
            con, params=(t0_str, t1_str, tornado_ef_threshold),
        )

    return {
        "wind":    wind    if not wind.empty    else empty,
        "hail":    hail    if not hail.empty    else empty,
        "tornado": torn    if not torn.empty    else empty,
    }


# ── Grid label assembly ───────────────────────────────────────────────────────

_HAZARD_ORDER = ("wind", "hail", "tornado")


def build_label_grid(
    valid_time: datetime,
    cfg: DictConfig,
    lsr_db_path: Path,
) -> np.ndarray:
    """
    Build a float32 label grid of shape (3, NY, NX).

    Channel order: 0=wind, 1=hail, 2=tornado.
    Values: 1.0 if grid point centre is within radius_km of ≥1 qualifying LSR, else 0.0.
    """
    grid_name = cfg.domain.grid
    nx, ny, _ = get_grid_params(grid_name)
    label = np.zeros((3, ny, nx), dtype=np.float32)

    # 12z–12z convective day window: valid_time is 12z, end is 12z next day
    t_end = valid_time + timedelta(hours=24)
    hazard_dfs = load_lsr_window(
        valid_time, t_end, lsr_db_path,
        cfg.labels.wind_gust_threshold_kts,
        cfg.labels.hail_size_threshold_in,
        cfg.labels.tornado_ef_threshold,
    )

    tree = _grid_balltree(grid_name)
    radius_rad = cfg.labels.radius_km / _EARTH_RADIUS_KM

    for ch, hazard in enumerate(_HAZARD_ORDER):
        df = hazard_dfs[hazard]
        if df.empty:
            continue
        report_coords = np.deg2rad(np.column_stack([
            df["lat"].values.astype(float),
            df["lon"].values.astype(float),
        ]))
        indices = tree.query_radius(report_coords, r=radius_rad)
        for idx_array in indices:
            label[ch].ravel()[idx_array] = 1.0

    return label


# ── Batch builder script helper ───────────────────────────────────────────────

def build_label_store(valid_times: list[datetime], cfg: DictConfig, zarr_root) -> None:
    """Write label grids into an open zarr store, one key per valid_time."""
    lsr_db_path = Path(cfg.labels.lsr_db)

    for vt in valid_times:
        key = "labels/" + vt.strftime("%Y%m%d%H")
        try:
            grid = build_label_grid(vt, cfg, lsr_db_path)
            if key in zarr_root:
                zarr_root[key][:] = grid
            else:
                arr = zarr_root.create_array(key, shape=grid.shape, dtype="float32", chunks=(3, 32, 32))
                arr[:] = grid
        except Exception as e:
            log.warning(f"Label build failed for {vt}: {e}")
