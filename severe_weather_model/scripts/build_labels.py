"""
Build the label zarr store from SPC LSR archive.

Labels are keyed by convective day start (noon UTC), so --start / --end are
interpreted as YYYYMMDD; the 12Z time is added automatically.

Usage:
    python scripts/build_labels.py --config config.yaml --start 20160101 --end 20231231
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import zarr
from omegaconf import OmegaConf

from severe_weather.labels import build_label_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", required=True, metavar="YYYYMMDD",
                   help="First convective day to process (inclusive)")
    p.add_argument("--end", metavar="YYYYMMDD",
                   help="Last convective day to process (inclusive); defaults to --start")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    root = zarr.open(cfg.dataset.labels_store, mode="a")

    end = args.end or args.start
    dates = pd.date_range(
        pd.to_datetime(args.start, format="%Y%m%d"),
        pd.to_datetime(end, format="%Y%m%d"),
        freq="D",
    )
    valid_times = [d.to_pydatetime().replace(hour=12) for d in dates]

    log.info(f"Building labels for {len(valid_times)} convective day(s)")
    build_label_store(valid_times, cfg, root)
    log.info(f"Label store → {cfg.dataset.labels_store}")


if __name__ == "__main__":
    main()
