"""
U-Net model and loss functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from omegaconf import DictConfig


def build_model(cfg: DictConfig, in_channels: int | None = None) -> nn.Module:
    """
    Instantiate a U-Net from segmentation_models_pytorch.

    in_channels overrides cfg.model.in_channels so callers can pass the
    actual channel count derived from the feature extraction pipeline.
    """
    c = cfg.model
    n_ch = in_channels if in_channels is not None else c.in_channels
    out_channels = len(list(c.hazard_channels))
    model = smp.Unet(
        encoder_name=c.encoder,
        encoder_weights=c.encoder_weights,
        in_channels=n_ch,
        classes=out_channels,
        decoder_channels=list(c.decoder_channels),
        decoder_use_batchnorm=True,
        activation=None,            # raw logits; sigmoid applied at inference/loss
    )
    return model


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for heavily imbalanced grids.
    alpha: weight for positive class; gamma: focusing parameter.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        domain_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        if domain_mask is not None:
            loss = torch.where(domain_mask.bool(), loss, torch.zeros_like(loss))
            return loss.sum() / (domain_mask.sum() * logits.shape[0] * logits.shape[1])
        return loss.mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        domain_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=logits.device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        if domain_mask is not None:
            loss = torch.where(domain_mask.bool(), loss, torch.zeros_like(loss))
            return loss.sum() / (domain_mask.sum() * logits.shape[0] * logits.shape[1])
        return loss.mean()


def build_loss(cfg: DictConfig) -> nn.Module:
    lc = cfg.loss
    if lc.type == "focal":
        return FocalLoss(alpha=lc.focal_alpha, gamma=lc.focal_gamma)
    if lc.type == "bce_weighted":
        return WeightedBCELoss(pos_weight=lc.bce_pos_weight)
    raise ValueError(f"Unknown loss type: {lc.type}")
