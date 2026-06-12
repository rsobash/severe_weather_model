"""
Main training entry point.

Usage:
    python scripts/train.py --config config.yaml \\
        --train-start 2016010100 --train-end 2021123118 \\
        --val-start 2022010100 --val-end 2022123118
    # Override any config key via OmegaConf dotlist:
    python scripts/train.py --config config.yaml \\
        --train-start 2016010100 --train-end 2021123118 \\
        --val-start 2022010100 --val-end 2022123118 \\
        training.lr=5e-5 model.encoder=resnet18
    # Fine-tune from a pretrained checkpoint:
    python scripts/train.py --config config.yaml \\
        --train-start 2016010100 --train-end 2021123118 \\
        --val-start 2022010100 --val-end 2022123118 \\
        --pretrained-checkpoint models/checkpoints/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from severe_weather.dataset import make_dataloaders
from severe_weather.model import build_loss, build_model
from severe_weather.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--train-start", required=True, metavar="YYYYMMDDHH",
                   help="First init time for training (inclusive)")
    p.add_argument("--train-end", required=True, metavar="YYYYMMDDHH",
                   help="Last init time for training (inclusive)")
    p.add_argument("--val-start", required=True, metavar="YYYYMMDDHH",
                   help="First init time for validation (inclusive)")
    p.add_argument("--val-end", required=True, metavar="YYYYMMDDHH",
                   help="Last init time for validation (inclusive)")
    p.add_argument("--pretrained-checkpoint", metavar="PATH",
                   help="Path to a checkpoint (best.pt or final.pt) to initialise weights from before training")
    p.add_argument("overrides", nargs="*", help="OmegaConf dot-path overrides, e.g. training.lr=1e-4")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    norm_stats = cfg.dataset.normalization_stats
    if not Path(norm_stats).exists():
        log.error(
            f"Normalisation stats not found at {norm_stats}. "
            "Run: python scripts/compute_norm_stats.py"
        )
        sys.exit(1)

    log.info("Building dataloaders …")
    train_dl, val_dl = make_dataloaders(
        cfg, norm_stats,
        train_start=args.train_start, train_end=args.train_end,
        val_start=args.val_start, val_end=args.val_end,
    )
    log.info(f"  train={len(train_dl.dataset)} val={len(val_dl.dataset)} samples")

    # Infer in_channels from the first batch
    sample_feats, _ = next(iter(train_dl))
    in_channels = sample_feats.shape[1]
    log.info(f"  in_channels={in_channels}")

    log.info("Building model …")
    model = build_model(cfg, in_channels=in_channels)
    if args.pretrained_checkpoint:
        ckpt_path = Path(args.pretrained_checkpoint)
        if not ckpt_path.exists():
            log.error(f"Pretrained checkpoint not found: {ckpt_path}")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)
        log.info(f"  Loaded pretrained weights from {ckpt_path}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  {n_params:,} trainable parameters")

    loss_fn = build_loss(cfg)

    domain_mask = torch.from_numpy(
        train_dl.dataset.domain_mask.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, NY, NX)

    log.info("Training …")
    model = train(model, loss_fn, train_dl, val_dl, cfg, device, domain_mask=domain_mask)

    final_path = Path(cfg.training.checkpoint_dir) / "final.pt"
    torch.save(model.state_dict(), final_path)
    log.info(f"Final model saved → {final_path}")


if __name__ == "__main__":
    main()
