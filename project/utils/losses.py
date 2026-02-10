from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _rbf_kernel(x: Tensor, sigma: float | None = None) -> Tensor:
    if x.dim() != 2:
        raise ValueError(f"Expected [N,D], got shape {tuple(x.shape)}")

    n = x.size(0)
    x_norm = (x * x).sum(dim=1, keepdim=True)
    dist_sq = (x_norm + x_norm.t() - 2.0 * (x @ x.t())).clamp_min(0.0)

    if sigma is None:
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)
        upper = dist_sq[idx_i, idx_j]
        positive = upper[upper > 0]
        if positive.numel() == 0:
            sigma_val = x.new_tensor(1.0)
        else:
            sigma_val = positive.median().sqrt().clamp_min(1e-6)
    else:
        sigma_val = x.new_tensor(float(sigma)).clamp_min(1e-6)

    gamma = 1.0 / (2.0 * sigma_val * sigma_val)
    return torch.exp(-gamma * dist_sq)


def hsic_loss(
    z_c: Tensor,
    z_d: Tensor,
    sigma_c: float | None = None,
    sigma_d: float | None = None,
) -> Tensor:
    """
    Eq. (8) HSIC penalty:
        (1/(n-1)^2) * tr(Kc H Kd H)
    """
    if z_c.dim() != 2 or z_d.dim() != 2:
        raise ValueError("hsic_loss expects 2D tensors [N,D].")
    if z_c.size(0) != z_d.size(0):
        raise ValueError("z_c and z_d must have the same batch size.")
    if z_c.size(0) < 2:
        return z_c.new_tensor(0.0)

    n = z_c.size(0)
    k_c = _rbf_kernel(z_c, sigma=sigma_c)
    k_d = _rbf_kernel(z_d, sigma=sigma_d)

    h = torch.eye(n, device=z_c.device, dtype=z_c.dtype) - (1.0 / n) * torch.ones(
        n, n, device=z_c.device, dtype=z_c.dtype
    )
    return torch.trace(k_c @ h @ k_d @ h) / ((n - 1) ** 2)


def irm_penalty(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Simple IRM penalty using scalar scale parameter gradient norm.
    """
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    if logits.dim() > 1 and logits.size(-1) > 1:
        loss = F.cross_entropy(logits * scale, targets)
    else:
        loss = F.mse_loss(logits.view_as(targets).float() * scale, targets.float())

    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad * grad).sum()


__all__ = ["hsic_loss", "irm_penalty"]
