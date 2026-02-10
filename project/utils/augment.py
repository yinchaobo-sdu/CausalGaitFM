from __future__ import annotations

import torch
from torch import Tensor


def counterfactual_domain_swap(
    z_c: Tensor,
    z_d: Tensor,
    indices: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Minimal domain intervention placeholder:
    keeps causal factors and swaps domain factors across the batch.
    """
    if z_c.size(0) != z_d.size(0):
        raise ValueError("z_c and z_d must have the same batch size.")

    if indices is None:
        indices = torch.randperm(z_d.size(0), device=z_d.device)
    z_d_cf = z_d[indices]
    return z_c, z_d_cf, indices


def counterfactual_signal_mix(x: Tensor, alpha: float = 0.1) -> Tensor:
    """Simple signal-space counterfactual augmentation placeholder."""
    if alpha <= 0:
        return x

    perm = torch.randperm(x.size(0), device=x.device)
    mixed = 0.5 * x + 0.5 * x[perm]
    noise = alpha * torch.randn_like(x)
    return mixed + noise


__all__ = ["counterfactual_domain_swap", "counterfactual_signal_mix"]

