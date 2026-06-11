"""
Load NWP output (GraphCast NetCDF or GEFS GRIB), regrid to G212/G211, compute derived features.

GraphCast files follow ERA5 variable naming. GEFS GRIB files are normalised to the same
naming convention by _load_gefs before being passed to extract_features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import DictConfig
from scipy.interpolate import RegularGridInterpolator

from .grid import get_grid_latlons, get_grid_params

log = logging.getLogger(__name__)

# ── File loaders ─────────────────────────────────────────────────────────────

def load_nwp_file(path: str | Path, source: str) -> xr.Dataset:
    """Load an NWP forecast file and return an ERA5-compatible xr.Dataset."""
    if source == "graphcast":
        return _load_graphcast(path)
    if source == "gefs":
        return _load_gefs(path)
    raise ValueError(f"Unknown NWP source: {source!r}. Expected 'graphcast' or 'gefs'.")


def _load_graphcast(path: str | Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4")


# GEFS GRIB shortName → ERA5 long name used by extract_features.
# ecCodes (cfgrib) shortNames differ between NCEP and ECMWF conventions; include both.
_GEFS_SURFACE_RENAME = {
    "u10":   "10m_u_component_of_wind",   # NCEP convention
    "10u":   "10m_u_component_of_wind",   # ECMWF/ecCodes convention
    "v10":   "10m_v_component_of_wind",
    "10v":   "10m_v_component_of_wind",
    "t2m":   "2m_temperature",
    "2t":    "2m_temperature",             # ECMWF/ecCodes convention
    "d2m":   "2m_dewpoint_temperature",
    "2d":    "2m_dewpoint_temperature",
    "r2":    "2m_relative_humidity",       # NCEP convention
    "2r":    "2m_relative_humidity",       # ECMWF/ecCodes convention
    "msl":   "mean_sea_level_pressure",
    "prmsl": "mean_sea_level_pressure",
    "sp":    "surface_pressure",
    "cape":  "convective_available_potential_energy",
    "tp":    "total_precipitation_6hr",
}
_GEFS_PLEVEL_RENAME = {
    "u":  "u_component_of_wind",
    "v":  "v_component_of_wind",
    "t":  "temperature",
    "q":  "specific_humidity",
    "r":  "relative_humidity",
    "gh": "geopotential",          # converted from metres → m²/s² below
}
_G = 9.80665  # standard gravity (m/s²)


def _load_gefs(path: str | Path) -> xr.Dataset:
    """
    Load a single GEFS ensemble-mean GRIB file and return an ERA5-compatible xr.Dataset.

    Variable names, coordinate names (lat/lon), and the pressure-level dimension name
    (level) are normalised so that extract_features works without modification.
    """
    try:
        import cfgrib  # noqa: F401 — triggers a clear ImportError if eccodes missing
    except ImportError as exc:
        raise ImportError(
            "cfgrib is required for GEFS GRIB support. "
            "Install with: pip install cfgrib  (also needs eccodes: "
            "conda install -c conda-forge eccodes  or  brew install eccodes)"
        ) from exc

    # Open each variable individually to avoid cfgrib coordinate conflicts that arise
    # when variables share a typeOfLevel but have different level values (e.g. 2m vs 10m)
    # or different level sets across pressure-level variables.
    datasets: list[xr.Dataset] = []

    def _open_var(short_name: str, level_type: str, level: int | None = None,
                  aliases: tuple[str, ...] = ()) -> None:
        for name in (short_name, *aliases):
            keys: dict = {"typeOfLevel": level_type, "shortName": name}
            if level is not None:
                keys["level"] = level
            try:
                ds = xr.open_dataset(path, engine="cfgrib", filter_by_keys=keys, indexpath=None)
                if ds.data_vars:
                    datasets.append(ds)
                    return
            except Exception:
                continue

    # Surface / near-surface — try both NCEP ("u10") and ECMWF/ecCodes ("10u") shortNames
    _open_var("u10",   "heightAboveGround", level=10, aliases=("10u",))
    _open_var("v10",   "heightAboveGround", level=10, aliases=("10v",))
    _open_var("t2m",   "heightAboveGround", level=2,  aliases=("2t",))
    _open_var("d2m",   "heightAboveGround", level=2,  aliases=("2d",))
    _open_var("r2",    "heightAboveGround", level=2,  aliases=("2r",))  # 2-m RH (present in GEFS when d2m is absent)
    _open_var("msl",   "meanSea")
    _open_var("prmsl", "meanSea")
    _open_var("sp",    "surface")
    _open_var("cape",  "surface")
    _open_var("tp",    "surface")

    # Pressure-level fields — one call per variable to avoid level-set conflicts
    for short_name in ("u", "v", "t", "q", "r", "gh"):
        _open_var(short_name, "isobaricInhPa")

    if not datasets:
        raise ValueError(f"No recognisable GRIB messages found in {path}")

    merged = xr.merge(datasets, compat="override")

    # Rename GRIB shortNames → ERA5 long names
    var_rename = {k: v for k, v in {**_GEFS_SURFACE_RENAME, **_GEFS_PLEVEL_RENAME}.items() if k in merged}
    if var_rename:
        merged = merged.rename(var_rename)

    # gh (geopotential height, m) → geopotential (m²/s²) to match GraphCast/ERA5
    if "geopotential" in merged:
        merged["geopotential"] = merged["geopotential"] * _G

    # Rename pressure-level dim to "level" so extract_features .sel(level=...) works
    if "isobaricInhPa" in merged.dims:
        merged = merged.rename({"isobaricInhPa": "level"})

    # Normalise coordinate names: latitude/longitude → lat/lon
    coord_rename = {}
    if "latitude" in merged.coords:
        coord_rename["latitude"] = "lat"
    if "longitude" in merged.coords:
        coord_rename["longitude"] = "lon"
    if coord_rename:
        merged = merged.rename(coord_rename)

    return merged


# ── Regridding ───────────────────────────────────────────────────────────────

def _make_interpolator(data: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray):
    """Build a 2D RegularGridInterpolator for a single field."""
    # GraphCast data is on a regular 0.25° lat/lon grid, lats ascending
    if src_lats[0] > src_lats[-1]:
        data = data[::-1, :]
        src_lats = src_lats[::-1]
    # Wrap lons to [-180, 180] if needed
    if src_lons.max() > 180:
        src_lons = np.where(src_lons > 180, src_lons - 360, src_lons)
        order = np.argsort(src_lons)
        src_lons = src_lons[order]
        data = data[:, order]
    return RegularGridInterpolator(
        (src_lats, src_lons), data, method="linear", bounds_error=False, fill_value=np.nan
    )


def regrid_to_g212(field: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray,
                   grid_name: str = "G212") -> np.ndarray:
    """Bilinearly interpolate a (lat, lon) field onto the named grid."""
    interp = _make_interpolator(field, src_lats, src_lons)
    target_lats, target_lons = get_grid_latlons(grid_name)
    pts = np.column_stack([target_lats.ravel(), target_lons.ravel()])
    out = interp(pts).reshape(target_lats.shape)
    return out.astype(np.float32)


# ── Derived feature computation ───────────────────────────────────────────────

def _wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u**2 + v**2).astype(np.float32)


def _mslp_gradient(mslp: np.ndarray, dx: float = 40635.0) -> np.ndarray:
    """Central-difference gradient magnitude in Pa/m."""
    dy_grad = np.gradient(mslp, dx, axis=0)
    dx_grad = np.gradient(mslp, dx, axis=1)
    return np.sqrt(dy_grad**2 + dx_grad**2).astype(np.float32)


def _bulk_shear(u10: np.ndarray, v10: np.ndarray,
                u500: np.ndarray, v500: np.ndarray) -> np.ndarray:
    """Approximate 0-6km bulk shear as |V_500 - V_sfc|."""
    return _wind_speed(u500 - u10, v500 - v10)


def _theta_e(temp_k: np.ndarray, q: np.ndarray, pressure_pa: float) -> np.ndarray:
    """Bolton (1980) equivalent potential temperature (K)."""
    Rd, Rv, cpd, lv = 287.05, 461.5, 1005.7, 2.501e6
    e = q * pressure_pa / (0.622 + q)
    e = np.maximum(e, 1e-10)
    tl = 2840.0 / (3.5 * np.log(temp_k) - np.log(e / 100.0) - 4.805) + 55.0
    theta_l = temp_k * (1e5 / pressure_pa) ** (Rd / cpd) * np.exp(
        (3.376 / tl - 0.00254) * q * 1000 * (1 + 0.81e-3 * q * 1000)
    )
    return theta_l.astype(np.float32)


def _lapse_rate(t700: np.ndarray, t500: np.ndarray) -> np.ndarray:
    """700-500 hPa lapse rate in K/km (positive = unstable)."""
    z700, z500 = 3000.0, 5500.0  # approximate geopotential heights in metres
    return ((t700 - t500) / (z500 - z700) * 1000).astype(np.float32)


def _rh_to_specific_humidity(rh_pct: np.ndarray, temp_k: np.ndarray, pressure_pa: float) -> np.ndarray:
    """Convert relative humidity (%) to specific humidity (kg/kg) via Magnus formula."""
    tc = temp_k - 273.15
    e_s = 611.2 * np.exp(17.67 * tc / (tc + 243.5))          # saturation vapour pressure [Pa]
    e = np.clip(rh_pct / 100.0, 0.0, 1.0) * e_s              # actual vapour pressure [Pa]
    q = 0.622 * e / (pressure_pa - 0.378 * e)
    return np.maximum(q, 0.0).astype(np.float32)


def _rh_to_dewpoint(rh_pct: np.ndarray, temp_k: np.ndarray) -> np.ndarray:
    """Convert relative humidity (%) + temperature (K) to dewpoint temperature (K)."""
    tc = temp_k - 273.15
    e_s = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    e = np.clip(rh_pct / 100.0, 1e-6, 1.0) * e_s
    log_e = np.log(e / 611.2)
    td_c = 243.5 * log_e / (17.67 - log_e)
    return (td_c + 273.15).astype(np.float32)


# ── Main feature extraction ───────────────────────────────────────────────────

def extract_features(ds: xr.Dataset, cfg: DictConfig, lead_hour: int, valid_time=None) -> tuple[np.ndarray, list[str]]:
    """
    Extract and regrid all features from one GraphCast xr.Dataset snapshot.

    Returns (array of shape (C, NY, NX), list of feature names).
    """
    grid_name = cfg.domain.grid
    lats = ds.lat.values
    lons = ds.lon.values

    # Synthesise specific_humidity from relative_humidity when q is absent (e.g. GEFS GRIB2)
    if "specific_humidity" not in ds and "relative_humidity" in ds and "temperature" in ds:
        rh_da = ds["relative_humidity"]
        t_da = ds["temperature"]
        if "level" in rh_da.dims and "level" in t_da.dims:
            common_levels = np.intersect1d(rh_da.level.values, t_da.level.values)
            q_vals = np.stack([
                _rh_to_specific_humidity(
                    rh_da.sel(level=lvl).values.squeeze(),
                    t_da.sel(level=lvl).values.squeeze(),
                    float(lvl) * 100.0,
                )
                for lvl in common_levels
            ], axis=0)
            ds = ds.assign(specific_humidity=rh_da.sel(level=common_levels).copy(data=q_vals))
            log.debug("Synthesised specific_humidity from relative_humidity at %d levels", len(common_levels))

    # Synthesise 2m_dewpoint_temperature from 2m_relative_humidity when d2m is absent (e.g. GEFS GRIB2)
    if "2m_dewpoint_temperature" not in ds and "2m_relative_humidity" in ds and "2m_temperature" in ds:
        r2m_vals = ds["2m_relative_humidity"].values.squeeze()
        t2m_vals = ds["2m_temperature"].values.squeeze()
        ds = ds.assign(**{
            "2m_dewpoint_temperature": ds["2m_relative_humidity"].copy(
                data=_rh_to_dewpoint(r2m_vals, t2m_vals)
            )
        })
        log.debug("Synthesised 2m_dewpoint_temperature from 2m_relative_humidity")

    channels: list[np.ndarray] = []
    names: list[str] = []

    def _get(var, level=None):
        if level is not None:
            da = ds[var].sel(level=level).values.squeeze()
        else:
            da = ds[var].values.squeeze()
        return regrid_to_g212(da, lats, lons, grid_name)

    # ── Surface fields ──────────────────────────────────────────────────────
    u10 = _get("10m_u_component_of_wind")
    v10 = _get("10m_v_component_of_wind")
    mslp = _get("mean_sea_level_pressure")
    t2m = _get("2m_temperature")

    channels += [u10, v10, mslp, t2m, _wind_speed(u10, v10), _mslp_gradient(mslp)]
    names += ["u10", "v10", "mslp", "t2m", "wspd10", "mslp_grad"]

    if "2m_dewpoint_temperature" in ds:
        channels.append(_get("2m_dewpoint_temperature"))
        names.append("d2m")

    if "2m_relative_humidity" in ds:
        channels.append(_get("2m_relative_humidity"))
        names.append("r2m")

    if "convective_available_potential_energy" in ds:
        channels.append(_get("convective_available_potential_energy"))
        names.append("cape")

    # ── Pressure level fields ───────────────────────────────────────────────
    for lvl in cfg.nwp.pressure_levels:
        for var in cfg.nwp.pressure_vars:
            if var in ds:
                field = _get(var, level=lvl)
                channels.append(field)
                names.append(f"{var}_{lvl}hPa")

    # ── Derived multi-level features ────────────────────────────────────────
    if "u_component_of_wind" in ds and "v_component_of_wind" in ds:
        u500 = _get("u_component_of_wind", level=500)
        v500 = _get("v_component_of_wind", level=500)
        u850 = _get("u_component_of_wind", level=850)
        v850 = _get("v_component_of_wind", level=850)
        channels += [_bulk_shear(u10, v10, u500, v500), _wind_speed(u850, v850)]
        names += ["bulk_shear_0_6km", "wspd850"]

    if "temperature" in ds and "specific_humidity" in ds:
        t700 = _get("temperature", level=700)
        t500 = _get("temperature", level=500)
        t850 = _get("temperature", level=850)
        q850 = _get("specific_humidity", level=850)
        channels += [_theta_e(t850, q850, 85000.0), _lapse_rate(t700, t500)]
        names += ["theta_e_850", "lapse_700_500"]

    # ── Temporal metadata ────────────────────────────────────────────────────
    # Encoded as constant planes so the model can learn lead-time uncertainty
    # and diurnal/seasonal cycles without date arithmetic inside the network.
    nx, ny, _ = get_grid_params(grid_name)
    lead_norm = np.full((ny, nx), lead_hour / 240.0, dtype=np.float32)  # normalise to ~0-1 (0-240h)

    import pandas as pd
    if valid_time is not None:
        ts = pd.Timestamp(valid_time)
    else:
        vt = ds.time.values
        if hasattr(vt, "__len__"):
            vt = vt[0]
        ts = pd.Timestamp(vt)
    doy_sin = np.full((ny, nx), np.sin(2 * np.pi * ts.day_of_year / 365.25), dtype=np.float32)
    doy_cos = np.full((ny, nx), np.cos(2 * np.pi * ts.day_of_year / 365.25), dtype=np.float32)
    hod_sin = np.full((ny, nx), np.sin(2 * np.pi * ts.hour / 24.0), dtype=np.float32)
    hod_cos = np.full((ny, nx), np.cos(2 * np.pi * ts.hour / 24.0), dtype=np.float32)

    grid_lats, grid_lons = get_grid_latlons(grid_name)
    channels += [lead_norm, doy_sin, doy_cos, hod_sin, hod_cos, grid_lats, grid_lons]
    names += ["lead_norm", "doy_sin", "doy_cos", "hod_sin", "hod_cos", "lat", "lon"]

    return np.stack(channels, axis=0), names  # (C, NY, NX)

# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean[:, None, None]) / std[:, None, None]).astype(np.float32)
