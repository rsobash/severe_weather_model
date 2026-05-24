"""Four-panel map of SPC LSR labels for 16 May 2016 (any / wind / hail / tornado)."""

import sys
sys.path.insert(0, "/Users/rsobash/claude-test/severe_weather_model")

import numpy as np
import zarr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj

from severe_weather.grid import get_grid_latlons, get_grid_params, PROJ_STRING, SW_X, SW_Y

ZARR_PATH = "/Users/rsobash/claude-test/severe_weather_model/data/processed/severe_labels.zarr"
KEY = "labels/2016051612"
GRID = "G211"

store = zarr.open(ZARR_PATH)
raw = store[KEY][:]                              # (3, NY, NX): wind, hail, tornado
any_ch = raw.max(axis=0, keepdims=True)          # (1, NY, NX)
labels = np.concatenate([any_ch, raw], axis=0)  # (4, NY, NX): any, wind, hail, tornado

PANELS = [
    (0, "Any",     "#d62728"),
    (1, "Wind",    "#1f77b4"),
    (2, "Hail",    "#2ca02c"),
    (3, "Tornado", "#9467bd"),
]

lats, lons = get_grid_latlons(GRID)
nx, ny, dx = get_grid_params(GRID)
half = dx / 2.0

lcc = ccrs.LambertConformal(
    central_longitude=-95,
    central_latitude=25,
    standard_parallels=(25, 25),
    globe=ccrs.Globe(semimajor_axis=6371200, semiminor_axis=6371200),
)
proj = pyproj.Proj(PROJ_STRING)

# Pre-compute corner polygons for every grid point once
corner_lons = np.empty((ny, nx, 5))
corner_lats = np.empty((ny, nx, 5))
for row in range(ny):
    for col in range(nx):
        cx = SW_X + col * dx
        cy = SW_Y + row * dx
        xs = [cx - half, cx + half, cx + half, cx - half, cx - half]
        ys = [cy - half, cy - half, cy + half, cy + half, cy - half]
        ln, lt = proj(xs, ys, inverse=True)
        corner_lons[row, col] = ln
        corner_lats[row, col] = lt

fig, axes = plt.subplots(
    2, 2, figsize=(16, 10),
    subplot_kw={"projection": lcc},
)
axes = axes.flat

for ax, (ch, title, color) in zip(axes, PANELS):
    ax.set_extent([-125, -65, 22, 52], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.LAKES,     facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor="#888888", zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.7, edgecolor="#555555", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#555555", zorder=2)

    n_pos = 0
    for row in range(ny):
        for col in range(nx):
            if labels[ch, row, col] > 0:
                ax.fill(
                    corner_lons[row, col], corner_lats[row, col],
                    color=color, alpha=0.7,
                    transform=ccrs.PlateCarree(), zorder=3,
                )
                n_pos += 1

    patch = mpatches.Patch(color=color, alpha=0.7, label=f"≥1 LSR ({n_pos} cells)")
    ax.legend(handles=[patch], loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=12, fontweight="bold")

fig.suptitle(
    "SPC LSR Labels — 16 May 2016 (convective day 12z–12z) · G211 80km grid",
    fontsize=13, y=1.01,
)
fig.tight_layout()

out = "/Users/rsobash/claude-test/labels_20160516.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
