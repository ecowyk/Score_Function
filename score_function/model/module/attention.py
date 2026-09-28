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

    def forward(
        self,
        tokens: torch.Tensor,
        scene: torch.Tensor,
        *,
        use_math_attention: bool = False,
    ) -> torch.Tensor:
        # Energy-based scores differentiate attention twice during DSM training.
        # need_weights=True uses the explicit attention operations, whose double
        # backward is supported by the production PyTorch 2.0 environment.
        attention, _ = self.attention(
            self.norm_attention(tokens), scene, scene, need_weights=use_math_attention
        )
        tokens = tokens + self.dropout_attention(attention)
        return tokens + self.ffn(self.norm_ffn(tokens))
