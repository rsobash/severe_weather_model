"""
U-Net model and loss functions.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Temporal channels (per lead) that FiLM conditions on. These are the constant
# spatial planes appended by features.extract_features; any single pixel carries
# the per-sample value, so the wrapper reads them at (0, 0).
_DEFAULT_FILM_CHANNELS = ["lead_norm", "doy_sin", "doy_cos", "hod_sin", "hod_cos"]


class TemporalConditionedUNet(nn.Module):
    """
    Wraps an SMP U-Net with FiLM conditioning at the encoder bottleneck.

    The conditioning vector is read directly out of the input tensor's temporal
    channels (constant planes), so the public forward signature stays ``forward(x)``
    and no caller needs to thread conditioning through the pipeline. An MLP maps the
    temporal vector to per-channel (gamma, beta) that modulate the deepest encoder
    feature map: ``feat = gamma * feat + beta``.
    """

    def __init__(self, unet: smp.Unet, temporal_channel_indices: list[int]):
        super().__init__()
        self.unet = unet
        self.register_buffer(
            "temporal_ch",
            torch.tensor(temporal_channel_indices, dtype=torch.long),
            persistent=False,
        )
        cond_dim = len(temporal_channel_indices)
        bottleneck_channels = unet.encoder.out_channels[-1]
        self.film_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim * 4, 2 * bottleneck_channels),
        )
        # Per-call (gamma, beta), set in forward and consumed by the encoder hook.
        self._film_params: tuple[torch.Tensor, torch.Tensor] | None = None
        # Intercept the encoder output rather than reimplementing SMP's
        # encoder→decoder→head wiring (which differs across SMP versions).
        self.unet.encoder.register_forward_hook(self._apply_film)

    def _apply_film(self, module, inputs, output):
        if self._film_params is None:
            return output
        gamma, beta = self._film_params
        output = list(output)
        output[-1] = gamma * output[-1] + beta
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = x[:, self.temporal_ch, 0, 0]              # (B, cond_dim)
        gamma, beta = self.film_mlp(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None] + 1.0            # residual init → identity at start
        beta = beta[:, :, None, None]
        self._film_params = (gamma, beta)
        try:
            return self.unet(x)                          # SMP wires encoder→decoder→head
        finally:
            self._film_params = None


def build_model(
    cfg: DictConfig,
    in_channels: int | None = None,
    feature_names: list[str] | None = None,
) -> nn.Module:
    """
    Instantiate a U-Net from segmentation_models_pytorch.

    in_channels overrides cfg.model.in_channels so callers can pass the
    actual channel count derived from the feature extraction pipeline.

    feature_names is the per-lead channel name list (one lead's worth, in input
    order). It is required when FiLM conditioning is enabled so the temporal
    channel indices can be located within the stacked input.
    """
    c = cfg.model
    n_ch = in_channels if in_channels is not None else c.in_channels
    out_channels = len(list(c.hazard_channels))
    decoder_channels = list(c.decoder_channels)
    model = smp.Unet(
        encoder_name=c.encoder,
        encoder_weights=c.encoder_weights,
        in_channels=n_ch,
        classes=out_channels,
        encoder_depth=len(decoder_channels),
        decoder_channels=decoder_channels,
        decoder_use_batchnorm=True,
        decoder_dropout=c.dropout,
        activation=None,            # raw logits; sigmoid applied at inference/loss
    )

    if not bool(c.get("film_conditioning", False)):
        return model

    if feature_names is None:
        raise ValueError(
            "model.film_conditioning is enabled but build_model was called without "
            "feature_names; pass the per-lead channel name list so the temporal "
            "channels can be located."
        )

    film_names = list(c.get("film_channels", _DEFAULT_FILM_CHANNELS))
    temporal_idx = [feature_names.index(n) for n in film_names if n in feature_names]
    if not temporal_idx:
        raise ValueError(
            f"FiLM conditioning is enabled but none of {film_names} are present in "
            f"the active feature set {feature_names}. Add them to dataset.feature_names "
            "(or clear the list) or set model.film_conditioning=false."
        )
    missing = [n for n in film_names if n not in feature_names]
    if missing:
        log.warning(f"FiLM conditioning: skipping channels not in feature set: {missing}")
    log.info(f"FiLM conditioning enabled on channels {[feature_names[i] for i in temporal_idx]} "
             f"(indices {temporal_idx})")
    return TemporalConditionedUNet(model, temporal_idx)


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
