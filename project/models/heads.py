from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


class MultiTaskHeads(nn.Module):
    """Three-task heads with homoscedastic uncertainty weighting."""

    def __init__(
        self,
        input_dim: int,
        num_fall_classes: int = 3,
        num_frailty_classes: int = 5,
        num_disease_classes: int = 4,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
        )
        self.fall_head = nn.Linear(input_dim, num_fall_classes)
        self.frailty_head = nn.Linear(input_dim, num_frailty_classes)
        self.disease_head = nn.Linear(input_dim, num_disease_classes)

        # log(sigma_t), where sigma_t controls per-task loss weight:
        # L_mt = sum( 1/(2*sigma_t^2) * L_t + log(sigma_t) )
        self.log_sigma_disease = nn.Parameter(torch.zeros(1))
        self.log_sigma_fall = nn.Parameter(torch.zeros(1))
        self.log_sigma_frailty = nn.Parameter(torch.zeros(1))

    def forward(self, features: Tensor | Mapping[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(features, Mapping):
            if "z_c" in features:
                x = features["z_c"]
            elif "pooled" in features:
                x = features["pooled"]
            else:
                raise KeyError("Expected key `z_c` or `pooled` in features mapping.")
        else:
            x = features

        x = self.shared(x)

        # NOTE: frailty is modeled as standard classification for now.
        # A Cumulative Link / ordinal formulation can replace this head later.
        return {
            "fall_logits": self.fall_head(x),
            "frailty_logits": self.frailty_head(x),
            "disease_logits": self.disease_head(x),
            "sigma_disease": torch.exp(self.log_sigma_disease),
            "sigma_fall": torch.exp(self.log_sigma_fall),
            "sigma_frailty": torch.exp(self.log_sigma_frailty),
        }

    def multi_task_uncertainty_loss(
        self,
        disease_loss: Tensor,
        fall_loss: Tensor,
        frailty_loss: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        def weighted(loss_value: Tensor, log_sigma: Tensor) -> Tensor:
            sigma_sq = torch.exp(2.0 * log_sigma)
            return 0.5 * loss_value / sigma_sq + log_sigma

        loss_d = weighted(disease_loss, self.log_sigma_disease)
        loss_f = weighted(fall_loss, self.log_sigma_fall)
        loss_r = weighted(frailty_loss, self.log_sigma_frailty)
        total = (loss_d + loss_f + loss_r).sum()
        return total, {
            "mt_disease": loss_d.sum(),
            "mt_fall": loss_f.sum(),
            "mt_frailty": loss_r.sum(),
        }


__all__ = ["MultiTaskHeads"]
