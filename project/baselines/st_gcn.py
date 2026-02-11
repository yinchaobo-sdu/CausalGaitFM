"""ST-GCN (Spatial Temporal Graph Convolutional Networks) baseline for Table 3.

Reference: Yan et al., "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition", AAAI 2018.

Adapted for 1D gait sensor data: treats sensor channels as graph nodes with
a learnable adjacency matrix (since we don't have fixed skeleton topology).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GraphConvolution(nn.Module):
    """Spatial graph convolution: aggregates features across adjacent channels."""

    def __init__(self, in_features: int, out_features: int, num_nodes: int) -> None:
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        # Learnable adjacency + identity (self-loop)
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, N, C] -> [B, N, C_out].

        N = num nodes (channels), C = features per node.
        """
        # Normalize adjacency
        A = torch.softmax(self.adj, dim=-1) + torch.eye(
            self.adj.size(0), device=x.device
        )
        # Spatial aggregation: x_agg[b, i] = sum_j A[i,j] * x[b, j]
        x = torch.einsum("ij,bjc->bic", A, x)
        x = self.weight(x) + self.bias
        return x


class STGCNBlock(nn.Module):
    """ST-GCN block: spatial GCN + temporal Conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        temporal_kernel: int = 9,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, num_nodes)
        self.tcn = nn.Sequential(
            nn.BatchNorm1d(out_channels * num_nodes),
            nn.ReLU(),
            nn.Conv1d(
                out_channels * num_nodes,
                out_channels * num_nodes,
                kernel_size=temporal_kernel,
                stride=stride,
                padding=temporal_kernel // 2,
                groups=num_nodes,
            ),
            nn.BatchNorm1d(out_channels * num_nodes),
        )
        self.relu = nn.ReLU()

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels * num_nodes, out_channels * num_nodes,
                          kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels * num_nodes),
            )
        else:
            self.residual = nn.Identity()

        self.num_nodes = num_nodes
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, N, C] -> [B, T', N, C_out]."""
        B, T, N, C = x.shape

        # Spatial: GCN over each timestep
        x_flat = x.reshape(B * T, N, C)
        gcn_out = self.gcn(x_flat)  # [B*T, N, C_out]
        gcn_out = gcn_out.reshape(B, T, N, self.out_channels)

        # Temporal: Conv1d over time
        # Reshape to [B, C_out*N, T]
        h = gcn_out.permute(0, 3, 2, 1).reshape(B, self.out_channels * N, T)
        res = x.permute(0, 3, 2, 1).reshape(B, C * N, T)

        h = self.tcn(h)
        res = self.residual(res)

        out = self.relu(h + res)
        # Back to [B, T', N, C_out]
        T_out = out.size(-1)
        out = out.reshape(B, self.out_channels, N, T_out).permute(0, 3, 2, 1)
        return out


class STGCNModel(nn.Module):
    """ST-GCN for sensor-based gait analysis.

    Treats each sensor channel as a graph node with learnable adjacency.
    Input: [B, T, D] -- D channels are treated as D graph nodes with 1 feature each.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_channels: tuple[int, ...] = (64, 64, 128, 128),
        temporal_kernel: int = 9,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Build ST-GCN blocks
        layers = []
        in_ch = 1  # each node starts with 1 feature (its sensor value)
        for i, out_ch in enumerate(hidden_channels):
            stride = 2 if i >= 2 else 1
            layers.append(STGCNBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                num_nodes=input_dim,
                temporal_kernel=temporal_kernel,
                stride=stride,
            ))
            in_ch = out_ch
        self.st_gcn = nn.Sequential(*layers)

        # Global pooling + classifier
        final_ch = hidden_channels[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_ch * input_dim),
            nn.Linear(final_ch * input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """x: [B, T, D] -> features: [B, final_ch * D]."""
        B, T, D = x.shape
        # Reshape to [B, T, N, 1] where N=D (each channel is a node)
        h = x.unsqueeze(-1)  # [B, T, D, 1]
        h = self.st_gcn(h)  # [B, T', D, C_out]
        B2, T2, N2, C2 = h.shape
        # Pool over time
        h = h.permute(0, 2, 3, 1).reshape(B2, N2 * C2, T2)  # [B, N*C, T']
        h = self.pool(h).squeeze(-1)  # [B, N*C]
        return h

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        features = self.extract_features(x, lengths)
        return self.classifier(features)

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


__all__ = ["STGCNModel"]
