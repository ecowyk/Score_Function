"""Cross-attention from trajectory queries to frozen scene tokens."""

from __future__ import annotations

import torch
from torch import nn


class SceneCrossAttentionBlock(nn.Module):
    """Residual scene attention followed by a residual point-wise feed-forward."""

    def __init__(self, dim: int, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout_attention = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        attention, _ = self.attention(self.norm_attention(tokens), scene, scene, need_weights=False)
        tokens = tokens + self.dropout_attention(attention)
        return tokens + self.ffn(self.norm_ffn(tokens))
