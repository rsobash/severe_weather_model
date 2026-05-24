"""
Build the feature zarr store from raw GraphCast output.

Lead-time range is controlled by graphcast.lead_start / lead_end / lead_interval in config.yaml.

Usage:
    # Single init time
    python scripts/build_features.py --config config.yaml --init-time 2016050112

    # All initialisations for one or more years
    python scripts/build_features.py --config config.yaml --years 2016 2017 2018
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
    find_graphcast_file,
    list_graphcast_files,
    load_graphcast_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--init-time", metavar="YYYYMMDDHH",
                     help="Process a single forecast initialisation time")
    grp.add_argument("--years", nargs="+", type=int,
                     help="Process all initialisations for the given years")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    out_dir = Path(cfg.dataset.zarr_store).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    store_path = cfg.dataset.zarr_store
    root = zarr.open(store_path, mode="a")

    lead_hours = list(range(cfg.graphcast.lead_start, cfg.graphcast.lead_end + 1, cfg.graphcast.lead_interval))

    if args.init_time:
        try:
            files = [find_graphcast_file(cfg, args.init_time)]
        except FileNotFoundError as e:
            log.error(e)
            return
        log.info(f"Processing single init time {args.init_time}")
    else:
        files = []
        for year in args.years:
            log.info(f"Collecting files for year {year}")
            files.extend(list_graphcast_files(cfg, year))

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
