"""GroupDRO (Group Distributionally Robust Optimization) baseline.

Reference: Sagawa et al., "Distributionally robust neural networks for group shifts",
ICLR 2020.
Upweights domains with highest loss to improve worst-group performance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone_shared import SharedBackbone


class GroupDROModel(nn.Module):
    """GroupDRO baseline: Mamba backbone + robust domain-wise loss weighting."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        num_domains: int = 6,
        d_model: int = 128,
        eta: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        self.eta = eta
        self.num_domains = num_domains
        self.backbone = SharedBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            **kwargs,
        )
        # Domain weights (log-space for stability)
        self.register_buffer(
            "domain_log_weights",
            torch.zeros(num_domains),
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

        # Per-domain losses
        domain_losses = []
        unique_domains = domain_ids.unique(sorted=True)
        for d in range(self.num_domains):
            mask = domain_ids == d
            if mask.sum() > 0:
                domain_losses.append(F.cross_entropy(logits[mask], targets[mask]))
            else:
                domain_losses.append(torch.tensor(0.0, device=x.device))

        domain_losses_t = torch.stack(domain_losses)

        # Update domain weights: upweight domains with higher loss
        with torch.no_grad():
            self.domain_log_weights = self.domain_log_weights + self.eta * domain_losses_t
            # Normalize weights via log-softmax
            self.domain_log_weights = self.domain_log_weights - self.domain_log_weights.max()

        weights = torch.exp(self.domain_log_weights)
        weights = weights / weights.sum()

        # Weighted sum of domain losses
        total = (weights * domain_losses_t).sum()
        return {"loss": total, "logits": logits, "domain_losses": domain_losses_t}


__all__ = ["GroupDROModel"]
