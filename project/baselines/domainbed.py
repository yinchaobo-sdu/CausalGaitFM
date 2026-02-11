"""DomainBed-style baseline.

Reference: Gulrajani & Lopez-Paz, "In search of lost domain generalization", ICLR 2021.

Implements the DomainBed training protocol: ERM with domain-balanced sampling
and model selection using a validation set from training domains.
This matches the DomainBed evaluation framework without requiring the full library.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone_shared import SharedBackbone


class DomainBedModel(nn.Module):
    """DomainBed baseline: ERM with domain-balanced training.

    Key difference from vanilla ERM: averages losses across domains rather
    than across samples, ensuring each domain contributes equally.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        num_domains: int = 6,
        d_model: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
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
        logits = self.forward(x, lengths=lengths)

        # Domain-balanced loss: average per-domain losses (DomainBed protocol)
        unique_domains = domain_ids.unique(sorted=True)
        domain_losses = []
        for d in unique_domains:
            mask = domain_ids == d
            if mask.sum() > 0:
                domain_losses.append(F.cross_entropy(logits[mask], targets[mask]))

        if domain_losses:
            loss = torch.stack(domain_losses).mean()
        else:
            loss = F.cross_entropy(logits, targets)

        return {"loss": loss, "logits": logits}


__all__ = ["DomainBedModel"]
