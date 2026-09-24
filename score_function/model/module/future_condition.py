"""Optional full-horizon ego attention and predicted-neighbor conditioning."""

import torch
from torch import nn


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            4 * dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, tokens):
        return self.layer(tokens)


class NeighborFutureAttention(nn.Module):
    """One token per predicted neighbor, with explicit padding and a null token.

    Full neighbor futures are conditioning only; no ego/neighbor state is changed.
    The null token keeps attention finite when every neighbor is absent.
    """

    def __init__(self, future_len, dim, heads, dropout):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(future_len * 4, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.null_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.future_len = future_len

    def forward(self, tokens, future, valid):
        batch = tokens.shape[0]
        if future is None or valid is None:
            raise ValueError("This branch requires predicted neighbor futures and their valid mask")
        if future.shape != (batch, 10, self.future_len, 4) or valid.shape != (batch, 10):
            raise ValueError("Expected neighbor_future [B,10,80,4] and neighbor_valid [B,10]")
        if valid.dtype != torch.bool:
            raise ValueError("neighbor_valid must be boolean")
        future = future.masked_fill(~valid[:, :, None, None], 0)
        condition = self.embedding(future.flatten(2))
        condition = torch.cat((self.null_token.expand(batch, -1, -1), condition), dim=1)
        padding = torch.cat((valid.new_zeros(batch, 1), ~valid), dim=1)
        value, _ = self.attention(
            self.norm(tokens), condition, condition, key_padding_mask=padding, need_weights=False
        )
        return tokens + self.dropout(value)
