"""
Build and save the CONUS boolean mask for a given NCEP grid.

Run once before training:
    python scripts/build_conus_mask.py --config config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from severe_weather.grid import get_grid_latlons

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_conus_mask(grid_name: str = "G212") -> np.ndarray:
    """Return (NY, NX) bool array, True where the grid point is inside CONUS (48 states)."""
    import cartopy.io.shapereader as shpreader
    import shapely.ops
    from shapely.geometry import Point
    from shapely.prepared import prep

    lats, lons = get_grid_latlons(grid_name)
    shapefile = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_1_states_provinces"
    )
    conus_geoms = [
        rec.geometry
        for rec in shpreader.Reader(shapefile).records()
        if rec.attributes["admin"] == "United States of America"
        and rec.attributes["postal"] not in ("AK", "HI")
    ]
    conus = prep(shapely.ops.unary_union(conus_geoms))
    ny, nx = lats.shape
    mask = np.zeros((ny, nx), dtype=bool)
    for r in range(ny):
        for c in range(nx):
            mask[r, c] = conus.contains(Point(float(lons[r, c]), float(lats[r, c])))
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    grid_name = cfg.domain.grid
    out_path = Path(cfg.domain.conus_mask_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Building CONUS mask for grid {grid_name} ...")
    mask = build_conus_mask(grid_name)
    n_valid = int(mask.sum())
    n_total = mask.size
    log.info(f"  {n_valid}/{n_total} points inside CONUS ({100 * n_valid / n_total:.1f}%)")
    np.save(out_path, mask)
    log.info(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
