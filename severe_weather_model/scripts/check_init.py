"""
Inspect features and labels in the zarr store for a specific forecast initialization.
Checks for NaNs in feature arrays across CONUS pixels, prints per-hazard label box
counts, and optionally prints per-feature min/mean/max stats.

Usage:
    python scripts/check_init.py --config config.yaml --init 2023060100
    python scripts/check_init.py --config config.yaml --init 2023060100 --stats
    python scripts/check_init.py --config config.yaml --init 2023060100 --stats --per-lead
"""

import argparse

import numpy as np
import zarr
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--init", required=True, metavar="YYYYMMDDHH")
    p.add_argument("--stats", action="store_true", help="Print min/mean/max per feature")
    p.add_argument("--per-lead", action="store_true", help="Print stats per lead time (implies --stats)")
    return p.parse_args()


def print_stats_table(arrays_by_lead, feature_names, conus_mask, per_lead):
    """Print min/mean/max per feature, either per lead or aggregated across leads."""
    mask_flat = conus_mask.ravel()
    col_w = max(len(n) for n in feature_names)

    if per_lead:
        for lead_h, arr in arrays_by_lead:
            print(f"\n  {'Feature':<{col_w}}   {'Min':>12}   {'Mean':>12}   {'Max':>12}")
            print(f"  {'-'*col_w}   {'-'*12}   {'-'*12}   {'-'*12}")
            conus_arr = arr.reshape(arr.shape[0], -1)[:, mask_flat]  # (C, N_conus)
            for i, name in enumerate(feature_names):
                ch = conus_arr[i]
                valid = ch[~np.isnan(ch)]
                if valid.size == 0:
                    print(f"  f{lead_h:03d} {name:<{col_w}}   {'all-NaN':>12}")
                    continue
                print(f"  f{lead_h:03d} {name:<{col_w}}   {valid.min():>12.4g}   {valid.mean():>12.4g}   {valid.max():>12.4g}")
    else:
        # Aggregate across all lead times
        all_data = {}  # feature_idx -> list of conus-masked 1-D arrays
        for _, arr in arrays_by_lead:
            conus_arr = arr.reshape(arr.shape[0], -1)[:, mask_flat]
            for i in range(arr.shape[0]):
                all_data.setdefault(i, []).append(conus_arr[i])

        print(f"\n  {'Feature':<{col_w}}   {'Min':>12}   {'Mean':>12}   {'Max':>12}")
        print(f"  {'-'*col_w}   {'-'*12}   {'-'*12}   {'-'*12}")
        for i, name in enumerate(feature_names):
            chunks = all_data.get(i, [])
            if not chunks:
                print(f"  {name:<{col_w}}   {'no data':>12}")
                continue
            combined = np.concatenate(chunks)
            valid = combined[~np.isnan(combined)]
            if valid.size == 0:
                print(f"  {name:<{col_w}}   {'all-NaN':>12}")
                continue
            print(f"  {name:<{col_w}}   {valid.min():>12.4g}   {valid.mean():>12.4g}   {valid.max():>12.4g}")


def main():
    args = parse_args()
    show_stats = args.stats or args.per_lead
    cfg = OmegaConf.load(args.config)
    root = zarr.open(cfg.dataset.zarr_store, mode="r")

    feature_names = list(root.attrs.get("feature_names", []))
    lead_hours = range(cfg.nwp.lead_start, cfg.nwp.lead_end + 1, cfg.nwp.lead_interval)

    print(f"\nFeature channels ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  [{i:3d}] {name}")

    conus_mask = np.load(cfg.domain.conus_mask_path)  # (NY, NX) bool

    print(f"\nNaN check for init {args.init} (CONUS domain only):")
    found_any = False
    arrays_by_lead = []
    for lead_h in lead_hours:
        key = f"features/{args.init}_f{lead_h:03d}"
        if key not in root:
            print(f"  f{lead_h:03d}  MISSING")
            continue
        arr = root[key][:]  # (C, NY, NX)
        if show_stats:
            arrays_by_lead.append((lead_h, arr))
        nan_in_conus = np.isnan(arr) & conus_mask[np.newaxis, :, :]
        if nan_in_conus.any():
            found_any = True
            nan_counts = nan_in_conus.reshape(arr.shape[0], -1).sum(axis=1)
            flagged = [(feature_names[i] if i < len(feature_names) else f"ch{i}", int(nan_counts[i]))
                       for i in range(len(nan_counts)) if nan_counts[i] > 0]
            print(f"  f{lead_h:03d}  NaNs in: " + ", ".join(f"{n}={c}" for n, c in flagged))
        else:
            print(f"  f{lead_h:03d}  OK")

    if not found_any:
        print("\nNo NaNs found.")

    if show_stats and arrays_by_lead:
        label = "per lead" if args.per_lead else "aggregated across all leads"
        print(f"\nFeature stats for init {args.init} ({label}, CONUS only):")
        print_stats_table(arrays_by_lead, feature_names, conus_mask, args.per_lead)

    # Labels: keyed by convective day (12Z), derived from the init date
    label_key = f"labels/{args.init[:8]}12"
    print(f"\nNaN check for labels {label_key} (CONUS domain only):")
    hazard_names = ["wind", "hail", "tornado"]
    if label_key not in root:
        print("  MISSING")
    else:
        arr = root[label_key][:]  # (3, NY, NX)
        nan_in_conus = np.isnan(arr) & conus_mask[np.newaxis, :, :]
        if nan_in_conus.any():
            nan_counts = nan_in_conus.reshape(arr.shape[0], -1).sum(axis=1)
            flagged = [(hazard_names[i] if i < len(hazard_names) else f"ch{i}", int(nan_counts[i]))
                       for i in range(len(nan_counts)) if nan_counts[i] > 0]
            print("  NaNs in: " + ", ".join(f"{n}={c}" for n, c in flagged))
        else:
            print("  OK")
        conus_arr = arr * conus_mask[np.newaxis, :, :]
        wind_boxes    = int(conus_arr[0].sum())
        hail_boxes    = int(conus_arr[1].sum())
        tornado_boxes = int(conus_arr[2].sum())
        any_boxes     = int(((conus_arr[0] + conus_arr[1] + conus_arr[2]) > 0).sum())
        print(f"  Label counts — wind: {wind_boxes}, hail: {hail_boxes}, tornado: {tornado_boxes}, any: {any_boxes}")


if __name__ == "__main__":
    main()
