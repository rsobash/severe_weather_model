"""
Build the label zarr store from SPC LSR archive.

Usage:
    python scripts/build_labels.py --config config.yaml --years 2016 2017 2018
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import zarr
from omegaconf import OmegaConf
from tqdm import tqdm

from severe_weather.labels import build_label_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--hour-step", type=int, default=24,
                   help="Generate one label grid every N hours")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    root = zarr.open(cfg.dataset.zarr_store, mode="a")

    for year in args.years:
        log.info(f"Building labels for {year}")
        t = datetime(year, 1, 1, 12, 0)  # noon start so ±12h spans exactly one UTC calendar day
        valid_times = []
        while t.year == year:
            valid_times.append(t)
            t += timedelta(hours=args.hour_step)

        build_label_store(valid_times, cfg, root)
        log.info(f"  {len(valid_times)} time steps written")

    log.info(f"Label store → {cfg.dataset.zarr_store}")


if __name__ == "__main__":
    main()
