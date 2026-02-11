"""ERM (Empirical Risk Minimization) baseline.

Standard supervised learning without any domain generalization strategy.
This is the simplest baseline in Table 2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone_shared import SharedBackbone


class ERMModel(nn.Module):
    """ERM baseline: Mamba backbone + standard cross-entropy."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        d_model: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        self.net = SharedBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            **kwargs,
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        return self.net(x, lengths=lengths)

    def compute_loss(
        self,
        x: Tensor,
        targets: Tensor,
        lengths: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        logits = self.forward(x, lengths=lengths)
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss, "logits": logits}


__all__ = ["ERMModel"]
