"""Time-independent ego trajectory score network conditioned on frozen scene and route features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from score_function.model.module.attention import SceneCrossAttentionBlock
from score_function.model.module.future_condition import (
    NeighborFutureAttention,
    TemporalSelfAttentionBlock,
)
from score_function.model.module.temporal import TemporalResidualBlock


class ScoreFunctionBranch(nn.Module):
    """A direct score or negative energy gradient, with identical conditioning.

    Both parameterizations return a [B,T,4] score from ``forward``.  The energy
    model defines E as a sum of learned scalar point contributions, and returns
    -grad_x E.  Its DSM objective and fixed-step refinement remain unchanged.
    """

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
        temporal_attention: bool = False,
        neighbor_future: bool = False,
        parameterization: str = "score",
    ):
        super().__init__()
        if future_len < 1 or input_dim != 4:
            raise ValueError("Expected a positive future_len and input_dim=4")
        if min(hidden_dim, context_dim, num_heads) < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if parameterization not in ("score", "energy"):
            raise ValueError("model.parameterization must be 'score' or 'energy'")
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
        self.uses_neighbor_future = neighbor_future
        self.parameterization = parameterization
        # Scene attention mixes each query with frozen scene tokens, not with
        # other trajectory queries.  This is the backbone feature receptive-field
        # upper bound; boundary clipping/dilation gaps can reduce actual support.
        self.feature_receptive_field = 1 + 4 * sum((*pre_dilations, *post_dilations))
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
        self.global_attention = (
            TemporalSelfAttentionBlock(hidden_dim, num_heads, dropout)
            if temporal_attention
            else nn.Identity()
        )
        self.neighbor_attention = (
            NeighborFutureAttention(future_len, hidden_dim, num_heads, dropout)
            if neighbor_future
            else None
        )
        if temporal_attention:
            self.feature_receptive_field = future_len
        # E is a sum of local scalar contributions.  Its derivative at point i
        # combines every contribution whose feature window includes i, so the
        # score can depend on a window of up to 2*R-1 points.  Retain the original
        # direct-score metadata convention (unclipped backbone upper bound).
        self.temporal_receptive_field = (
            min(future_len, 2 * self.feature_receptive_field - 1)
            if parameterization == "energy"
            else self.feature_receptive_field
        )
        self.post_blocks = nn.ModuleList(
            TemporalResidualBlock(hidden_dim, dilation, dropout) for dilation in post_dilations
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            # An energy-only additive bias has identically zero DSM gradient.
            # Omit it rather than leave an unused parameter in DDP training.
            nn.Linear(
                hidden_dim,
                input_dim if parameterization == "score" else 1,
                bias=parameterization == "score",
            ),
        )

    def forward(
        self,
        ego_traj_norm: torch.Tensor,
        scene_context: torch.Tensor,
        route_embedding: torch.Tensor,
        neighbor_future: torch.Tensor | None = None,
        neighbor_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_score(
            ego_traj_norm, scene_context, route_embedding, neighbor_future, neighbor_valid
        )

    def predict_score(
        self,
        ego_traj_norm: torch.Tensor,
        scene_context: torch.Tensor,
        route_embedding: torch.Tensor,
        neighbor_future: torch.Tensor | None = None,
        neighbor_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the score, including when called under no_grad/inference_mode.

        Capture the caller's gradient mode before temporarily enabling it to
        evaluate an energy derivative.  Normal training retains the derivative
        graph for DSM backward; no-grad inference returns a detached score.
        An input already requiring gradients keeps its existing graph connection.
        """
        if self.parameterization == "score":
            return self.score_head(
                self._trajectory_features(
                    ego_traj_norm,
                    scene_context,
                    route_embedding,
                    neighbor_future,
                    neighbor_valid,
                )
            )

        create_graph = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            # Inference tensors cannot be saved for backward, including frozen
            # conditioning.  Clone only those tensors; keep ordinary inputs and
            # their gradient connections intact.
            trajectory = self._autograd_compatible(ego_traj_norm)
            if not trajectory.requires_grad:
                trajectory = trajectory.detach().requires_grad_(True)
            energy = self.energy_value(
                trajectory,
                self._autograd_compatible(scene_context),
                self._autograd_compatible(route_embedding),
                self._autograd_compatible(neighbor_future),
                self._autograd_compatible(neighbor_valid),
            )
            score = -torch.autograd.grad(energy.sum(), trajectory, create_graph=create_graph)[0]
        return score if create_graph else score.detach()

    @staticmethod
    def _autograd_compatible(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is not None and torch.is_inference(tensor):
            return tensor.clone()
        return tensor

    def energy_value(
        self,
        ego_traj_norm: torch.Tensor,
        scene_context: torch.Tensor,
        route_embedding: torch.Tensor,
        neighbor_future: torch.Tensor | None = None,
        neighbor_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return E(x,C) as [B]; its absolute additive offset is unidentifiable.

        This method follows the caller's gradient mode, like an ordinary module
        forward.  Use ``predict_score`` for energy gradients under no_grad.
        """
        if self.parameterization != "energy":
            raise ValueError("energy_value requires model.parameterization='energy'")
        tokens = self._trajectory_features(
            ego_traj_norm, scene_context, route_embedding, neighbor_future, neighbor_valid
        )
        return self.score_head(tokens).sum(dim=(1, 2))

    def _trajectory_features(
        self,
        ego_traj_norm: torch.Tensor,
        scene_context: torch.Tensor,
        route_embedding: torch.Tensor,
        neighbor_future: torch.Tensor | None = None,
        neighbor_valid: torch.Tensor | None = None,
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
        use_math_attention = self.parameterization == "energy"
        tokens = self.scene_attention(
            tokens,
            self.scene_projection(scene_context),
            use_math_attention=use_math_attention,
        )
        if self.neighbor_attention is not None:
            tokens = self.neighbor_attention(
                tokens,
                neighbor_future,
                neighbor_valid,
                use_math_attention=use_math_attention,
            )
        if isinstance(self.global_attention, TemporalSelfAttentionBlock):
            tokens = self.global_attention(tokens, use_math_attention=use_math_attention)
        else:
            tokens = self.global_attention(tokens)
        for block in self.post_blocks:
            tokens = block(tokens)
        return tokens


def build_model(config: dict, device: torch.device | str) -> ScoreFunctionBranch:
    """Construct only the trainable branch; conditioning belongs to the adapter."""
    settings = dict(config["model"])
    if settings.pop("architecture", None) != "temporal_score_function":
        raise ValueError("Expected model.architecture='temporal_score_function'")
    return ScoreFunctionBranch(**settings).to(device)
