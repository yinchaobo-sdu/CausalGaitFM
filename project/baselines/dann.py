"""DANN (Domain-Adversarial Neural Network) baseline.

Reference: Ganin et al., "Domain-adversarial training of neural networks", JMLR 2016.
Uses a gradient reversal layer to learn domain-invariant features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function

from .backbone_shared import SharedBackbone


class _GradientReversalFn(Function):
    """Gradient reversal layer (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: Tensor, alpha: float = 1.0) -> Tensor:
    return _GradientReversalFn.apply(x, alpha)


class DANNModel(nn.Module):
    """DANN baseline: Mamba backbone + domain discriminator with GRL."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        num_domains: int = 6,
        d_model: int = 128,
        grl_alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.grl_alpha = grl_alpha
        self.backbone = SharedBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            **kwargs,
        )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_domains),
        )

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        features = self.backbone.extract_features(x, lengths)
        class_logits = self.backbone.classifier(features)
        reversed_features = gradient_reversal(features, self.grl_alpha)
        domain_logits = self.domain_discriminator(reversed_features)
        return class_logits, domain_logits

    def compute_loss(
        self,
        x: Tensor,
        targets: Tensor,
        domain_ids: Tensor,
        lengths: Tensor | None = None,
        domain_weight: float = 1.0,
        **kwargs,
    ) -> dict[str, Tensor]:
        class_logits, domain_logits = self.forward(x, lengths=lengths)
        class_loss = F.cross_entropy(class_logits, targets)
        domain_loss = F.cross_entropy(domain_logits, domain_ids)
        total = class_loss + domain_weight * domain_loss
        return {"loss": total, "class_loss": class_loss, "domain_loss": domain_loss, "logits": class_logits}


__all__ = ["DANNModel"]
