"""
Build the feature zarr store from raw NWP forecast files.

Source, filename pattern, and lead-time range are controlled by the nwp section of config.yaml.
Init times are enumerated at 24-hourly frequency between --start and --end.

Usage:
    # Single init time
    python scripts/build_features.py --config config.yaml --start 2024050100

    # Range of init times (24-hourly, both inclusive)
    python scripts/build_features.py --config config.yaml --start 2016010100 --end 2021123100
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
    extract_features,
    load_nwp_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", required=True, metavar="YYYYMMDDHH",
                   help="First init time to process (inclusive)")
    p.add_argument("--end", metavar="YYYYMMDDHH",
                   help="Last init time to process (inclusive); defaults to --start")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    out_dir = Path(cfg.dataset.zarr_store).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    store_path = cfg.dataset.zarr_store
    root = zarr.open(store_path, mode="a")

    lead_hours = list(range(cfg.nwp.lead_start, cfg.nwp.lead_end + 1, cfg.nwp.lead_interval))
    local_dir = Path(cfg.nwp.local_dir)

    end = args.end or args.start
    init_range = pd.date_range(
        pd.to_datetime(args.start, format="%Y%m%d%H"),
        pd.to_datetime(end, format="%Y%m%d%H"),
        freq="24h",
    )

    log.info(f"Source: {cfg.nwp.source} | {len(init_range)} init time(s), {len(lead_hours)} lead hour(s) each")

    feature_names: list[str] = []
    for init_dt in tqdm(init_range, desc="inits"):
        init_str = init_dt.strftime("%Y%m%d%H")
        for lead_h in lead_hours:
            fname = cfg.nwp.filename_pattern.format(
                init_str=init_str, HH=init_str[-2:], lead=lead_h
            )
            fpath = local_dir / fname
            if not fpath.exists():
                log.warning(f"Missing {fpath.name}, skipping")
                continue
            try:
                ds = load_nwp_file(fpath, cfg.nwp.source)
                step = ds.isel(prediction_timedelta=0) if "prediction_timedelta" in ds.dims else ds.isel(time=0) if "time" in ds.dims else ds
                valid_dt = init_dt + pd.Timedelta(hours=lead_h)
                feats, names = extract_features(step, cfg, lead_hour=lead_h, valid_time=valid_dt)
                if not feature_names:
                    feature_names = names

                feat_key = f"features/{init_str}_f{lead_h:03d}"
                if feat_key not in root:
                    root.create_array(feat_key, shape=feats.shape, dtype="float32", chunks=(feats.shape[0], 32, 32))
                root[feat_key][:] = feats

            except Exception as e:
                log.warning(f"  {fpath.name} error: {e}")

    if feature_names:
        root.attrs["feature_names"] = feature_names
        log.info(f"Feature names ({len(feature_names)}): {feature_names}")

    log.info(f"Feature store → {store_path}")


if __name__ == "__main__":
    main()
