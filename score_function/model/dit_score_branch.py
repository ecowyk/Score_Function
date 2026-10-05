"""Fixed-sigma temporal-token score branch aligned with the DP decoder DiT."""

import math

import torch
from torch import nn

from score_function.model.module.dit import DiTBlock, FinalLayer, Mlp, initialize_dit_weights


class DiTScoreFunctionBranch(nn.Module):
    """Predict normalized-coordinate scores using one token per future time point.

    Scene cross-attention, route-conditioned adaLN, block ordering and output
    initialization follow DP. Fixed-sigma DSM needs neither a diffusion timestep
    nor DP's SDE output rescaling. Scene and route features still come from the
    existing frozen encoder/route encoder; this module never loads a planner.
    """

    def __init__(
        self,
        future_len=80,
        input_dim=4,
        hidden_dim=192,
        num_heads=6,
        dropout=0.1,
        context_dim=192,
        depth=3,
        mlp_ratio=4.0,
        preproj_hidden_dim=512,
        parameterization="score",
    ):
        super().__init__()
        for name, value in (
            ("future_len", future_len),
            ("hidden_dim", hidden_dim),
            ("num_heads", num_heads),
            ("context_dim", context_dim),
            ("depth", depth),
            ("preproj_hidden_dim", preproj_hidden_dim),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if input_dim != 4 or hidden_dim % num_heads or not 0 <= dropout < 1:
            raise ValueError(
                "Require input_dim=4, divisible hidden width/heads and dropout in [0,1)"
            )
        if (
            isinstance(mlp_ratio, bool)
            or not isinstance(mlp_ratio, (int, float))
            or not math.isfinite(mlp_ratio)
            or mlp_ratio <= 0
            or int(hidden_dim * mlp_ratio) < 1
        ):
            raise ValueError("mlp_ratio must give a positive MLP width")
        if parameterization != "score":
            raise ValueError("dit_score_function supports only parameterization='score'")
        self.future_len = future_len
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.parameterization = parameterization
        self.uses_neighbor_future = False
        self.feature_receptive_field = future_len
        self.temporal_receptive_field = future_len

        # Same two-linear preprojection as DP, applied to a single point rather
        # than a flattened whole-agent trajectory. Time replaces agent identity.
        self.preproj = Mlp(input_dim, preproj_hidden_dim, hidden_dim)
        self.temporal_embedding = nn.Embedding(future_len, hidden_dim)
        self.scene_projection = (
            nn.Identity() if context_dim == hidden_dim else nn.Linear(context_dim, hidden_dim)
        )
        self.route_projection = (
            nn.Identity() if context_dim == hidden_dim else nn.Linear(context_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, num_heads, dropout, mlp_ratio) for _ in range(depth)
        )
        self.final_layer = FinalLayer(hidden_dim, input_dim)
        initialize_dit_weights(self)

    def forward(
        self,
        ego_traj_norm,
        scene_context,
        route_embedding,
        neighbor_future=None,
        neighbor_valid=None,
    ):
        return self.predict_score(
            ego_traj_norm, scene_context, route_embedding, neighbor_future, neighbor_valid
        )

    def predict_score(
        self,
        ego_traj_norm,
        scene_context,
        route_embedding,
        neighbor_future=None,
        neighbor_valid=None,
    ):
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
        if neighbor_future is not None or neighbor_valid is not None:
            raise ValueError("dit_score_function does not condition on neighbor futures")
        positions = torch.arange(self.future_len, device=ego_traj_norm.device)
        tokens = self.preproj(ego_traj_norm) + self.temporal_embedding(positions)[None]
        scene = self.scene_projection(scene_context)
        condition = self.route_projection(route_embedding)
        # All 80 ego future tokens are valid. As in DP, scene cross-attention
        # consumes the encoder output directly without adding a padding mask.
        for block in self.blocks:
            tokens = block(tokens, scene, condition, attn_mask=None)
        return self.final_layer(tokens, condition)
