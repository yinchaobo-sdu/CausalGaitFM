"""Deep CORAL (CORrelation ALignment) baseline.

Reference: Sun & Saenko, "Deep CORAL: Correlation alignment for deep domain adaptation",
ECCV Workshops 2016.
Minimizes the difference in second-order statistics between source and target features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone_shared import SharedBackbone


def coral_loss(source_features: Tensor, target_features: Tensor) -> Tensor:
    """Compute CORAL loss: ||C_s - C_t||_F^2 / (4 * d^2).

    Where C is the covariance matrix of features.
    """
    d = source_features.size(1)

    # Center features
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)

    # Covariance matrices
    ns = source_features.size(0)
    nt = target_features.size(0)
    cov_source = (source_centered.T @ source_centered) / max(ns - 1, 1)
    cov_target = (target_centered.T @ target_centered) / max(nt - 1, 1)

    # Frobenius norm of difference
    diff = cov_source - cov_target
    loss = (diff * diff).sum() / (4.0 * d * d)
    return loss


class CORALModel(nn.Module):
    """Deep CORAL baseline: Mamba backbone + covariance alignment."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        d_model: int = 128,
        coral_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.coral_weight = coral_weight
        self.backbone = SharedBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            **kwargs,
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        return self.backbone(x, lengths=lengths)

    def compute_loss(
        self,
        x: Tensor,
        targets: Tensor,
        domain_ids: Tensor,
        lengths: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        features = self.backbone.extract_features(x, lengths)
        logits = self.backbone.classifier(features)
        class_loss = F.cross_entropy(logits, targets)

        # CORAL: align features across all domain pairs
        unique_domains = domain_ids.unique()
        coral_total = torch.tensor(0.0, device=x.device)
        n_pairs = 0
        for i in range(len(unique_domains)):
            for j in range(i + 1, len(unique_domains)):
                mask_i = domain_ids == unique_domains[i]
                mask_j = domain_ids == unique_domains[j]
                if mask_i.sum() >= 2 and mask_j.sum() >= 2:
                    coral_total = coral_total + coral_loss(features[mask_i], features[mask_j])
                    n_pairs += 1

        if n_pairs > 0:
            coral_total = coral_total / n_pairs

        total = class_loss + self.coral_weight * coral_total
        return {"loss": total, "class_loss": class_loss, "coral_loss": coral_total, "logits": logits}


__all__ = ["CORALModel"]
