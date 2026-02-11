"""MIRO (Mutual-Information Regularization with pre-trained models) baseline.

Reference: Cha et al., "Domain generalization by mutual-information regularization
with pre-trained models", ECCV 2022.

Simplified version: regularize learned features to stay close to an exponential
moving average (EMA) teacher, encouraging domain-invariant representations.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone_shared import SharedBackbone


class MIROModel(nn.Module):
    """MIRO baseline: Mamba backbone + mutual-information regularization via EMA teacher."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        d_model: int = 128,
        mi_weight: float = 0.1,
        ema_decay: float = 0.999,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mi_weight = mi_weight
        self.ema_decay = ema_decay

        self.backbone = SharedBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            **kwargs,
        )

        # EMA teacher (frozen copy)
        self.teacher = copy.deepcopy(self.backbone)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Variational projector for MI estimation
        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)

    @torch.no_grad()
    def _update_teacher(self) -> None:
        """Exponential moving average update of teacher parameters."""
        for t_param, s_param in zip(self.teacher.parameters(), self.backbone.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        return self.backbone(x, lengths=lengths)

    def compute_loss(
        self,
        x: Tensor,
        targets: Tensor,
        lengths: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        # Student forward
        student_features = self.backbone.extract_features(x, lengths)
        logits = self.backbone.classifier(student_features)
        class_loss = F.cross_entropy(logits, targets)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_features = self.teacher.extract_features(x, lengths)

        # Mutual information regularization via variational bound:
        # Encourage student features to match teacher features distribution
        mu = self.mu_proj(student_features)
        logvar = self.logvar_proj(student_features)

        # KL(q(z|x_student) || p(z|x_teacher))
        # Treat teacher features as the mean of a unit-variance Gaussian
        kl_loss = 0.5 * (
            logvar.exp() + (mu - teacher_features).pow(2) - logvar - 1.0
        ).sum(dim=-1).mean()

        total = class_loss + self.mi_weight * kl_loss

        # Update teacher after loss computation
        self._update_teacher()

        return {"loss": total, "class_loss": class_loss, "mi_loss": kl_loss, "logits": logits}


__all__ = ["MIROModel"]
