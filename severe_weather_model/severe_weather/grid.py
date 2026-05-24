"""CONUS grid definitions (NCEP Lambert Conformal — G211 80km, G212 40km)."""

import numpy as np
import pyproj

# Shared projection (both grids use the same LCC params and SW corner)
PROJ_STRING = (
    "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=-95 "
    "+x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
)
# South-west corner in projection coordinates
SW_X = -4226108.0
SW_Y = -832698.0

_proj = pyproj.Proj(PROJ_STRING)

# (nx, ny, dx_metres)
_GRID_PARAMS: dict[str, tuple[int, int, float]] = {
    "G212": (185, 129, 40635.0),
    "G211": (93,  65,  81270.0),
}


def get_grid_params(name: str) -> tuple[int, int, float]:
    """Return (nx, ny, dx_metres) for the named NCEP grid."""
    if name not in _GRID_PARAMS:
        raise ValueError(f"Unknown grid {name!r}. Available: {sorted(_GRID_PARAMS)}")
    return _GRID_PARAMS[name]


def get_grid_latlons(grid_name: str = "G212") -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) arrays of shape (NY, NX) for the named grid."""
    nx, ny, dx = get_grid_params(grid_name)
    xs = SW_X + np.arange(nx) * dx
    ys = SW_Y + np.arange(ny) * dx
    xx, yy = np.meshgrid(xs, ys)
    lons, lats = _proj(xx, yy, inverse=True)
    return lats.astype(np.float32), lons.astype(np.float32)


def latlon_to_ij(lat: float, lon: float, grid_name: str = "G212") -> tuple[int, int]:
    """Convert a lat/lon point to nearest (row, col) on the named grid."""
    nx, ny, dx = get_grid_params(grid_name)
    x, y = _proj(lon, lat)
    col = int(round((x - SW_X) / dx))
    row = int(round((y - SW_Y) / dx))
    col = max(0, min(nx - 1, col))
    row = max(0, min(ny - 1, row))
    return row, col


def latlons_to_ij_bulk(
    lats: np.ndarray, lons: np.ndarray, grid_name: str = "G212"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised lat/lon → (rows, cols, valid_mask) for the named grid."""
    nx, ny, dx = get_grid_params(grid_name)
    xs, ys = _proj(lons, lats)
    cols = np.round((xs - SW_X) / dx).astype(int)
    rows = np.round((ys - SW_Y) / dx).astype(int)
    valid = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
    cols = np.clip(cols, 0, nx - 1)
    rows = np.clip(rows, 0, ny - 1)
    return rows, cols, valid
