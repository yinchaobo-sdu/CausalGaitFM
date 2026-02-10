from __future__ import annotations

import torch
from torch import Tensor, nn


def dag_acyclicity_loss(adjacency: Tensor) -> Tensor:
    """
    Eq. (6):
        L_DAG = tr(exp(A * A)) - K
    """
    if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
        raise ValueError(f"`adjacency` must be square [K,K], got {tuple(adjacency.shape)}")

    k = adjacency.size(0)
    squared = adjacency * adjacency
    return torch.trace(torch.matrix_exp(squared)) - k


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


def hsic_independence_loss(
    z_c: Tensor,
    z_d: Tensor,
    sigma_c: float | None = None,
    sigma_d: float | None = None,
) -> Tensor:
    """
    Eq. (8):
        L_HSIC = 1/(n-1)^2 * tr(Kc H Kd H)
    """
    if z_c.dim() != 2 or z_d.dim() != 2:
        raise ValueError("hsic_independence_loss expects 2D tensors [N,D].")
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


class SCM_Layer(nn.Module):
    """
    Structural Causal Module:
      - z_c: causal factors (Eq. 5) with learned DAG adjacency A
      - z_d: domain factors (Eq. 7) as Gaussian latent
    """

    def __init__(
        self,
        input_dim: int,
        causal_dim: int = 32,
        domain_dim: int = 16,
        scm_hidden_dim: int = 64,
        n_graph_iters: int = 2,
    ) -> None:
        super().__init__()
        if causal_dim <= 0 or domain_dim <= 0:
            raise ValueError("`causal_dim` and `domain_dim` must be positive.")

        self.input_dim = input_dim
        self.causal_dim = causal_dim
        self.domain_dim = domain_dim
        self.n_graph_iters = n_graph_iters

        self.shared_norm = nn.LayerNorm(input_dim)

        # Exogenous noise uk for Eq. (5)
        self.u_proj = nn.Linear(input_dim, causal_dim)

        # Node-wise causal mechanisms fk(Pa(z_k), u_k)
        self.node_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, scm_hidden_dim),
                    nn.GELU(),
                    nn.Linear(scm_hidden_dim, 1),
                )
                for _ in range(causal_dim)
            ]
        )

        # Learnable adjacency logits. A in [0,1] after sigmoid.
        # Bias toward sparse graph at initialization to keep DAG penalty stable.
        self.adjacency_logits = nn.Parameter(torch.full((causal_dim, causal_dim), -5.0))

        # Domain branch (Gaussian latent, Eq. 7)
        self.domain_backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, scm_hidden_dim),
            nn.GELU(),
        )
        self.domain_mu = nn.Linear(scm_hidden_dim, domain_dim)
        self.domain_logvar = nn.Linear(scm_hidden_dim, domain_dim)

    def adjacency_matrix(self) -> Tensor:
        eye = torch.eye(
            self.causal_dim,
            device=self.adjacency_logits.device,
            dtype=self.adjacency_logits.dtype,
        )
        return torch.sigmoid(self.adjacency_logits) * (1.0 - eye)

    def _causal_structural_forward(self, u: Tensor, adjacency: Tensor) -> Tensor:
        z = u
        for _ in range(self.n_graph_iters):
            parents = z @ adjacency
            next_nodes = []
            for node_idx, node_fn in enumerate(self.node_functions):
                node_input = torch.stack((parents[:, node_idx], u[:, node_idx]), dim=-1)
                next_nodes.append(node_fn(node_input))
            z = torch.cat(next_nodes, dim=-1)
        return z

    @staticmethod
    def _sample_gaussian(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features: Tensor, sample_domain: bool = True) -> dict[str, Tensor]:
        if features.dim() == 3:
            pooled = features.mean(dim=1)
        elif features.dim() == 2:
            pooled = features
        else:
            raise ValueError(
                f"SCM_Layer expects [B,T,D] or [B,D], got shape {tuple(features.shape)}"
            )

        pooled = self.shared_norm(pooled)
        adjacency = self.adjacency_matrix()

        u = self.u_proj(pooled)
        z_c = self._causal_structural_forward(u=u, adjacency=adjacency)

        d_hidden = self.domain_backbone(pooled)
        z_d_mu = self.domain_mu(d_hidden)
        z_d_logvar = self.domain_logvar(d_hidden)
        z_d = self._sample_gaussian(z_d_mu, z_d_logvar) if sample_domain else z_d_mu

        domain_kl_loss = -0.5 * (1.0 + z_d_logvar - z_d_mu.pow(2) - torch.exp(z_d_logvar)).sum(dim=-1).mean()
        dag_loss = dag_acyclicity_loss(adjacency)
        hsic_loss = hsic_independence_loss(z_c, z_d)

        return {
            "z_c": z_c,
            "z_d": z_d,
            "z_d_mu": z_d_mu,
            "z_d_logvar": z_d_logvar,
            "adjacency": adjacency,
            "dag_loss": dag_loss,
            "hsic_loss": hsic_loss,
            "domain_kl_loss": domain_kl_loss,
        }


class SCMEncoder(SCM_Layer):
    """Backward-compatible alias of SCM_Layer."""

    def __init__(
        self,
        input_dim: int,
        causal_dim: int = 32,
        domain_dim: int = 16,
        scm_hidden_dim: int = 64,
        n_graph_iters: int = 2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            causal_dim=causal_dim,
            domain_dim=domain_dim,
            scm_hidden_dim=scm_hidden_dim,
            n_graph_iters=n_graph_iters,
        )


__all__ = [
    "SCM_Layer",
    "SCMEncoder",
    "dag_acyclicity_loss",
    "hsic_independence_loss",
]
