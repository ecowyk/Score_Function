"""Residual convolutions along trajectory time."""

from __future__ import annotations

import torch
from torch import nn


class TemporalResidualBlock(nn.Module):
    """Two noncausal, same-length convolutions along candidate trajectory time."""

    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        if dim < 1 or dilation < 1:
            raise ValueError("dim and dilation must be positive")
        self.dilation = int(dilation)
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, 3, dilation=dilation, padding=dilation)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(dim, dim, 3, dilation=dilation, padding=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = self.norm(tokens).transpose(1, 2)
        residual = self.conv2(self.activation(self.conv1(residual)))
        return tokens + self.dropout(residual.transpose(1, 2))
