"""Time-independent ego trajectory score network conditioned on frozen scene and route features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from score_function.model.module.attention import SceneCrossAttentionBlock
from score_function.model.module.temporal import TemporalResidualBlock


class ScoreFunctionBranch(nn.Module):
    """S4: temporal blocks, scene cross-attention, route, and point score head."""

    def __init__(
        self,
        future_len: int = 80,
        input_dim: int = 4,
        hidden_dim: int = 192,
        num_heads: int = 6,
        dropout: float = 0.1,
        pre_dilations: Sequence[int] = (1, 2),
        post_dilations: Sequence[int] = (2, 4),
        context_dim: int = 192,
    ):
        super().__init__()
        if future_len < 1 or input_dim != 4:
            raise ValueError("Expected a positive future_len and input_dim=4")
        if min(hidden_dim, context_dim, num_heads) < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        pre_dilations, post_dilations = tuple(pre_dilations), tuple(post_dilations)
        if (
            not pre_dilations
            or not post_dilations
            or any(
                not isinstance(d, int) or isinstance(d, bool) or d < 1
                for d in (*pre_dilations, *post_dilations)
            )
        ):
            raise ValueError("Both dilation stages need positive integer dilations")
        self.future_len = int(future_len)
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        # Scene attention mixes each query with frozen scene tokens, not with
        # other trajectory queries.  Thus this is the candidate receptive field.
        self.temporal_receptive_field = 1 + 4 * sum((*pre_dilations, *post_dilations))
        self.point_embedding = nn.Linear(input_dim, hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, future_len, hidden_dim))
        self.scene_projection = (
            nn.Identity() if context_dim == hidden_dim else nn.Linear(context_dim, hidden_dim)
        )
        self.route_projection = (
            nn.Identity() if context_dim == hidden_dim else nn.Linear(context_dim, hidden_dim)
        )
        self.pre_blocks = nn.ModuleList(
            TemporalResidualBlock(hidden_dim, dilation, dropout) for dilation in pre_dilations
        )
        self.scene_attention = SceneCrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.post_blocks = nn.ModuleList(
            TemporalResidualBlock(hidden_dim, dilation, dropout) for dilation in post_dilations
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        ego_traj_norm: torch.Tensor,
        scene_context: torch.Tensor,
        route_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if ego_traj_norm.ndim != 3 or ego_traj_norm.shape[1:] != (
            self.future_len,
            self.input_dim,
        ):
            raise ValueError(
                f"ego_traj_norm must have shape [B,{self.future_len},{self.input_dim}]"
            )
        batch = ego_traj_norm.shape[0]
        if (
            scene_context.ndim != 3
            or scene_context.shape[0] != batch
            or scene_context.shape[1] < 1
            or scene_context.shape[2] != self.context_dim
        ):
            raise ValueError(f"scene_context must have shape [B,N,{self.context_dim}]")
        if route_embedding.shape != (batch, self.context_dim):
            raise ValueError(f"route_embedding must have shape [B,{self.context_dim}]")
        tokens = (
            self.point_embedding(ego_traj_norm)
            + self.temporal_pos
            + self.route_projection(route_embedding)[:, None, :]
        )
        for block in self.pre_blocks:
            tokens = block(tokens)
        tokens = self.scene_attention(tokens, self.scene_projection(scene_context))
        for block in self.post_blocks:
            tokens = block(tokens)
        return self.score_head(tokens)


def build_model(config: dict, device: torch.device | str) -> ScoreFunctionBranch:
    """Construct only the trainable branch; conditioning belongs to the adapter."""
    settings = dict(config["model"])
    if settings.pop("architecture", None) != "temporal_score_function":
        raise ValueError("Expected model.architecture='temporal_score_function'")
    return ScoreFunctionBranch(**settings).to(device)
