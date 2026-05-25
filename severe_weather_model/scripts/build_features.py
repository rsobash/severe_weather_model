"""
Build the feature zarr store from raw GraphCast output.

Lead-time range is controlled by graphcast.lead_start / lead_end / lead_interval in config.yaml.

Usage:
    # All initialisations in the local_dir
    python scripts/build_features.py --config config.yaml

    # Restrict to a range of init times (YYYYMMDDHH, both inclusive, both optional)
    python scripts/build_features.py --config config.yaml --start 2016010100 --end 2021123118
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from omegaconf import OmegaConf
from tqdm import tqdm

from severe_weather.features import (
    FEATURE_NAMES,
    extract_features,
    load_graphcast_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", metavar="YYYYMMDDHH",
                   help="Earliest init time to include (inclusive)")
    p.add_argument("--end", metavar="YYYYMMDDHH",
                   help="Latest init time to include (inclusive)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    out_dir = Path(cfg.dataset.zarr_store).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    store_path = cfg.dataset.zarr_store
    root = zarr.open(store_path, mode="a")

    lead_hours = list(range(cfg.graphcast.lead_start, cfg.graphcast.lead_end + 1, cfg.graphcast.lead_interval))

    local_dir = Path(cfg.graphcast.local_dir)
    files = sorted(local_dir.glob("*.nc"))

    if args.start or args.end:
        lo = args.start or "0000000000"
        hi = args.end   or "9999999999"
        files = [f for f in files if lo <= f.name[:10] <= hi]

    if not files:
        log.error("No GraphCast files matched the requested range.")
        return

    log.info(f"Processing {len(files)} init times")

    for fpath in tqdm(files, desc="inits"):
        try:
            ds = load_graphcast_file(fpath)
        except Exception as e:
            log.warning(f"Failed to load {fpath}: {e}")
            continue

        for lead_h in lead_hours:
            try:
                if "prediction_timedelta" in ds.dims:
                    step = ds.sel(prediction_timedelta=np.timedelta64(lead_h, "h"))
                else:
                    step = ds.isel(time=0)

                feats = extract_features(step, cfg, lead_hour=lead_h)

                init_ts = pd.Timestamp(ds.time.values.flat[0])
                key = init_ts.strftime("%Y%m%d%H") + f"_f{lead_h:03d}"

                feat_key = f"features/{key}"
                if feat_key not in root:
                    root.create_array(feat_key, shape=feats.shape, dtype="float32", chunks=(feats.shape[0], 32, 32))
                root[feat_key][:] = feats

            except Exception as e:
                log.warning(f"  lead={lead_h}h error: {e}")

    # Save feature names
    if FEATURE_NAMES:
        root.attrs["feature_names"] = FEATURE_NAMES
        log.info(f"Feature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    log.info(f"Feature store → {store_path}")


if __name__ == "__main__":
    main()
