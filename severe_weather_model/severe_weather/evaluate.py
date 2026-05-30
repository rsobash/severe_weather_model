"""
Calibration (temperature scaling) and evaluation metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ── Temperature scaling ───────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """Wraps a trained model and learns one temperature parameter T per output channel."""

    def __init__(self, model: nn.Module, n_channels: int = 1):
        super().__init__()
        self.model = model
        # shape (n_channels, 1, 1) for broadcast over (B, C, H, W)
        self.temperature = nn.Parameter(torch.ones(n_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature

    def fit(
        self,
        val_dl: DataLoader,
        device: torch.device,
        max_iter: int = 50,
        domain_mask: np.ndarray | None = None,
    ):
        """Find T per channel that minimises NLL on the validation set (logits fixed)."""
        self.model.eval()
        self.to(device)

        # Collect all logits and labels first (no gradient through model)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for feats, labels in val_dl:
                feats = feats.to(device)
                all_logits.append(self.model(feats).cpu())
                all_labels.append(labels.cpu())

        logits = torch.cat(all_logits)   # (N, C, H, W)
        labels = torch.cat(all_labels)   # (N, C, H, W)

        mask_t = torch.from_numpy(domain_mask) if domain_mask is not None else None

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss_elem = nn.functional.binary_cross_entropy_with_logits(
                scaled, labels, reduction="none"
            )
            if mask_t is not None:
                loss = (loss_elem * mask_t).sum() / (
                    mask_t.sum() * scaled.shape[0] * scaled.shape[1]
                )
            else:
                loss = loss_elem.mean()
            loss.backward()
            return loss

        optimizer.step(eval_step)
        t_vals = self.temperature.data.squeeze().tolist()
        if isinstance(t_vals, float):
            t_vals = [t_vals]
        log.info(f"Temperature scaling: T = {[f'{t:.4f}' for t in t_vals]}")
        return self

    def save(self, path: str | Path):
        torch.save({"temperature": self.temperature.data.cpu()}, path)

    @classmethod
    def load(cls, model: nn.Module, path: str | Path) -> "TemperatureScaler":
        state = torch.load(path, map_location="cpu")
        t = state["temperature"]
        scaler = cls(model, n_channels=t.shape[0])
        scaler.temperature = nn.Parameter(t)
        return scaler


# ── Metric helpers ────────────────────────────────────────────────────────────

def _collect_preds(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    domain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel arrays (probs, labels) of shape (C, N) over the full dataloader."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for feats, labels in dl:
            feats = feats.to(device)
            logits = model(feats)                        # (B, C, H, W)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, C, H, W)
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    probs_arr  = np.concatenate(all_probs,  axis=0)  # (N, C, H, W)
    labels_arr = np.concatenate(all_labels, axis=0)  # (N, C, H, W)
    n_channels = probs_arr.shape[1]
    if domain_mask is not None:
        # (N, C, H, W) → (N, C, N_valid) → (C, N*N_valid)
        probs_arr  = probs_arr[:, :, domain_mask].transpose(1, 0, 2).reshape(n_channels, -1)
        labels_arr = labels_arr[:, :, domain_mask].transpose(1, 0, 2).reshape(n_channels, -1)
    else:
        probs_arr  = probs_arr.transpose(1, 0, 2, 3).reshape(n_channels, -1)
        labels_arr = labels_arr.transpose(1, 0, 2, 3).reshape(n_channels, -1)
    return probs_arr, labels_arr


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> dict:
    """Compute Brier score, BSS, AUC-ROC, AUC-PR, and CSI at each threshold."""
    climo_rate = labels.mean()
    brier = brier_score_loss(labels, probs)
    brier_climo = brier_score_loss(labels, np.full_like(probs, climo_rate))
    bss = 1.0 - brier / (brier_climo + 1e-10)

    metrics = {
        "brier_score": brier,
        "brier_skill_score": bss,
        "auc_roc": roc_auc_score(labels, probs),
        "auc_pr": average_precision_score(labels, probs),
        "climatology_rate": climo_rate,
    }

    for t in thresholds:
        preds_bin = (probs >= t).astype(int)
        tp = ((preds_bin == 1) & (labels == 1)).sum()
        fp = ((preds_bin == 1) & (labels == 0)).sum()
        fn = ((preds_bin == 0) & (labels == 1)).sum()
        csi = tp / (tp + fp + fn + 1e-10)
        pod = tp / (tp + fn + 1e-10)
        far = fp / (tp + fp + 1e-10)
        metrics[f"csi_{t:.2f}"] = csi
        metrics[f"pod_{t:.2f}"] = pod
        metrics[f"far_{t:.2f}"] = far

    return metrics


def plot_reliability(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    hazard_name: str = "",
    n_bins: int = 10,
):
    """Save a reliability diagram to output_path."""
    import matplotlib.pyplot as plt

    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="uniform")

    title = f"Reliability Diagram — {hazard_name}" if hazard_name else "Reliability Diagram"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Reliability diagram → {output_path}")
