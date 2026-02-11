from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Cumulative Link Ordinal Head  (paper Section 4.6, Ref [23])
# ---------------------------------------------------------------------------

class OrdinalHead(nn.Module):
    """Cumulative-link ordinal regression head.

    For K ordered classes the head predicts K-1 cumulative log-odds.
    P(Y <= k | x) = sigma(theta_k - f(x))   for k = 1, ..., K-1

    The class probabilities are:
      p_0 = P(Y <= 0)
      p_k = P(Y <= k) - P(Y <= k-1)   for k = 1, ..., K-2
      p_{K-1} = 1 - P(Y <= K-2)

    Loss is negative log-likelihood of the true ordinal class.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, 1)  # shared scalar latent
        # K-1 ordered thresholds (initialized as ascending)
        init_thresholds = torch.linspace(-1.0, 1.0, num_classes - 1)
        self.thresholds = nn.Parameter(init_thresholds)

    def forward(self, x: Tensor) -> Tensor:
        """Return class log-probabilities [B, K] compatible with ``F.nll_loss``.

        Internally computes ordinal probabilities via the cumulative link
        model, then returns their log for use with negative log-likelihood.

        .. note::
            Use ``F.nll_loss(output, target)`` -- **not** ``F.cross_entropy``.
        """
        f_x = self.linear(x)  # [B, 1]
        # Cumulative probabilities: sigma(theta_k - f(x))
        cum_probs = torch.sigmoid(self.thresholds.unsqueeze(0) - f_x)  # [B, K-1]

        # Class probabilities
        # p_0 = cum_probs[:, 0]
        # p_k = cum_probs[:, k] - cum_probs[:, k-1]
        # p_{K-1} = 1 - cum_probs[:, K-2]
        ones = torch.ones(f_x.size(0), 1, device=f_x.device, dtype=f_x.dtype)
        zeros = torch.zeros(f_x.size(0), 1, device=f_x.device, dtype=f_x.dtype)
        cum_extended = torch.cat([zeros, cum_probs, ones], dim=1)  # [B, K+1]
        probs = cum_extended[:, 1:] - cum_extended[:, :-1]  # [B, K]
        probs = probs.clamp(min=1e-7)
        # Convert to log-logits for compatibility with cross_entropy
        logits = torch.log(probs)
        return logits

    def ordinal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Negative log-likelihood loss for ordinal regression."""
        return F.nll_loss(logits, targets)


# ---------------------------------------------------------------------------
# Multi-task heads with uncertainty weighting
# ---------------------------------------------------------------------------

class MultiTaskHeads(nn.Module):
    """Three-task heads with homoscedastic uncertainty weighting (paper Eq. 10).

    Tasks:
      - Disease classification: standard softmax (4 classes)
      - Fall risk: standard softmax (3 classes)
      - Frailty: ordinal regression via Cumulative Link Model (5 levels)
    """

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
        self.disease_head = nn.Linear(input_dim, num_disease_classes)

        # Frailty: ordinal regression head (Cumulative Link, paper Sec 4.6)
        self.frailty_head = OrdinalHead(input_dim, num_frailty_classes)

        # log(sigma_t), where sigma_t controls per-task loss weight:
        # L_mt = sum( 1/(2*sigma_t^2) * L_t + log(sigma_t) )   [Eq. 10]
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


__all__ = ["MultiTaskHeads", "OrdinalHead"]
