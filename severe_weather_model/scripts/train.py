"""
Main training entry point.

Usage:
    python scripts/train.py --config config.yaml
    python scripts/train.py --config config.yaml training.lr=5e-5 model.encoder=resnet18
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from severe_weather.dataset import make_dataloaders
from severe_weather.features import FEATURE_NAMES
from severe_weather.model import build_loss, build_model
from severe_weather.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
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
            "Run: python scripts/build_features.py --compute-norm"
        )
        sys.exit(1)

    log.info("Building dataloaders …")
    train_dl, val_dl = make_dataloaders(cfg, norm_stats)
    log.info(f"  train={len(train_dl.dataset)} val={len(val_dl.dataset)} samples")

    # Infer in_channels from the first batch
    sample_feats, _ = next(iter(train_dl))
    in_channels = sample_feats.shape[1]
    log.info(f"  in_channels={in_channels}")

    log.info("Building model …")
    model = build_model(cfg, in_channels=in_channels)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  {n_params:,} trainable parameters")

    loss_fn = build_loss(cfg)

    log.info("Training …")
    model = train(model, loss_fn, train_dl, val_dl, cfg, device)

    # Save final model alongside checkpoints
    final_path = Path(cfg.training.checkpoint_dir) / "final.pt"
    torch.save(model.state_dict(), final_path)
    log.info(f"Final model saved → {final_path}")


if __name__ == "__main__":
    main()
