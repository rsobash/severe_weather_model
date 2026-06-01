"""
Compute per-channel normalisation stats from the feature zarr store.

Reads raw feature arrays already written by build_features.py and produces
norm_stats.npz (mean, std per channel) without loading everything into RAM.

Usage:
    # All keys in store
    python scripts/compute_norm_stats.py --config config.yaml

    # Restrict to a range of init times (inclusive)
    python scripts/compute_norm_stats.py --config config.yaml --start 2016010100 --end 2021123118
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import zarr
from omegaconf import OmegaConf
from tqdm import tqdm

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

    mask_path = Path(cfg.domain.conus_mask_path)
    if not mask_path.exists():
        log.error(f"CONUS mask not found at {mask_path}. Run build_conus_mask.py first.")
        return
    domain_mask = np.load(mask_path).ravel()  # (H*W,) boolean

    store_path = cfg.dataset.zarr_store
    root = zarr.open(store_path, mode="r")

    if "features" not in root:
        log.error("No 'features' group found in zarr store.")
        return

    all_keys = sorted(root["features"].keys())
    if args.start or args.end:
        lo = args.start or "0000000000"
        hi = args.end   or "9999999999"
        all_keys = [k for k in all_keys if lo <= k.split("_f")[0] <= hi]

    if not all_keys:
        log.error("No feature keys matched the requested range.")
        return

    log.info(f"Computing stats over {len(all_keys)} feature arrays ({domain_mask.sum()} CONUS pixels each)")

    n_sum = None
    x_sum = None
    x2_sum = None

    for key in tqdm(all_keys, desc="accumulating"):
        arr = root["features"][key][:]   # (C, H, W)
        c = arr.shape[0]

        if n_sum is None:
            n_sum  = np.zeros(c, dtype=np.float64)
            x_sum  = np.zeros(c, dtype=np.float64)
            x2_sum = np.zeros(c, dtype=np.float64)

        masked = arr.reshape(c, -1)[:, domain_mask]  # (C, n_conus)
        n_sum  += masked.shape[1]
        x_sum  += masked.sum(axis=1)
        x2_sum += (masked ** 2).sum(axis=1)

    mean = (x_sum / n_sum).astype(np.float32)
    var  = x2_sum / n_sum - (x_sum / n_sum) ** 2
    std  = (np.sqrt(np.maximum(var, 0)) + 1e-8).astype(np.float32)

    out_path = cfg.dataset.normalization_stats
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mean=mean, std=std)
    log.info(f"Saved norm stats ({len(mean)} channels) → {out_path}")


if __name__ == "__main__":
    main()
