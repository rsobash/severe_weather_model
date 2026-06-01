"""
Training loop with mixed precision, LR scheduling, early stopping, and W&B logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


def _make_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    tc = cfg.training
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    if tc.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=tc.lr)
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


def _make_scheduler(optimizer, cfg: DictConfig, steps_per_epoch: int):
    tc = cfg.training
    warmup_steps = tc.warmup_epochs * steps_per_epoch
    total_steps = tc.max_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if tc.scheduler == "cosine":
            import math
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0  # constant after warmup if scheduler=plateau (handled separately)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    domain_mask: torch.Tensor | None = None,
) -> nn.Module:
    tc = cfg.training
    ckpt_dir = Path(tc.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg, len(train_dl))
    scaler = GradScaler(device.type, enabled=tc.amp and device.type == "cuda")

    model.to(device)
    loss_fn.to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, tc.max_epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for features, labels in tqdm(train_dl, desc=f"Epoch {epoch} train", leave=False):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=tc.amp and device.type == "cuda"):
                logits = model(features)
                loss = loss_fn(logits, labels, domain_mask=domain_mask)

            scaler.scale(loss).backward()
            if tc.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in tqdm(val_dl, desc=f"Epoch {epoch} val", leave=False):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device.type, enabled=tc.amp and device.type == "cuda"):
                    logits = model(features)
                    loss = loss_fn(logits, labels, domain_mask=domain_mask)
                val_loss += loss.item()
        val_loss /= len(val_dl)

        lr = optimizer.param_groups[0]["lr"]
        log.info(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss}, ckpt_path)
            log.info(f"  Saved best checkpoint → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= tc.early_stopping_patience:
                log.info(f"Early stopping at epoch {epoch} (patience={tc.early_stopping_patience})")
                break

    # Load best weights before returning
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        log.warning("No checkpoint was saved (val_loss may have been NaN every epoch). Returning current model weights.")
        return model
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    log.info(f"Loaded best weights from epoch {best['epoch']} (val_loss={best['val_loss']:.4f})")

    return model
