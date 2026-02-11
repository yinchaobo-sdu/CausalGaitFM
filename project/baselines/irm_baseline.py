"""IRM (Invariant Risk Minimization) standalone baseline.

Reference: Arjovsky et al., "Invariant risk minimization", 2019.
Uses the IRM penalty without the SCM / causal disentanglement of CausalGaitFM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone_shared import SharedBackbone

try:
    from project.utils.losses import irm_penalty
except ModuleNotFoundError:
    from utils.losses import irm_penalty


class IRMBaselineModel(nn.Module):
    """IRM baseline: Mamba backbone + IRM penalty (no SCM)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        d_model: int = 128,
        irm_weight: float = 1.0,
        irm_warmup: int = 500,
        **kwargs,
    ) -> None:
        super().__init__()
        self.irm_weight = irm_weight
        self.irm_warmup = irm_warmup
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
        iter_idx: int = 0,
        **kwargs,
    ) -> dict[str, Tensor]:
        logits = self.forward(x, lengths=lengths)
        class_loss = F.cross_entropy(logits, targets)

        # Per-domain IRM penalty
        penalties = []
        for d in domain_ids.unique():
            mask = domain_ids == d
            if mask.sum() >= 2:
                penalties.append(irm_penalty(logits[mask], targets[mask]))

        if penalties:
            irm_loss = torch.stack(penalties).mean()
        else:
            irm_loss = torch.tensor(0.0, device=x.device)

        # Anneal IRM weight
        if self.irm_warmup > 0 and iter_idx < self.irm_warmup:
            weight = self.irm_weight * (iter_idx / self.irm_warmup)
        else:
            weight = self.irm_weight

        total = class_loss + weight * irm_loss
        return {"loss": total, "class_loss": class_loss, "irm_loss": irm_loss, "logits": logits}


__all__ = ["IRMBaselineModel"]
