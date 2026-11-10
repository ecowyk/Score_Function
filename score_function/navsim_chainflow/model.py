"""Conditional density score using frozen ChainFlow visual and VLM features.

The PDM utility scorer and residual diffusion planner are not trained here.
Only the semantic adapter and this direct score network receive DSM gradients.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from score_function.model.score_branch import ScoreFunctionBranch

ARCHITECTURE = "chainflow_scene_score"


class ChainFlowScore(nn.Module):
    parameterization = "score"
    uses_neighbor_future = False

    def __init__(self, model_cfg):
        super().__init__()
        settings = copy.deepcopy(model_cfg)
        if settings.pop("architecture", None) != ARCHITECTURE:
            raise ValueError(f"Expected architecture={ARCHITECTURE}")
        if settings.get("parameterization", "score") != "score":
            raise ValueError("ChainFlow conditioner supports direct score only")
        if (settings.get("future_len"), settings.get("input_dim"), settings.get("context_dim")) != (8, 4, 256):
            raise ValueError("Require eight NAVSIM poses, four channels and 256-wide scene tokens")
        if settings.get("neighbor_future", False):
            raise ValueError("ChainFlow score does not use neighbor futures")
        self.vlm_dim = int(settings.pop("vlm_dim", 1536))
        self.semantic_tokens = int(settings.pop("semantic_tokens", 16))
        semantic_heads = int(settings.pop("semantic_heads", 8))
        if self.vlm_dim <= 0 or self.semantic_tokens <= 0 or semantic_heads <= 0 or 256 % semantic_heads:
            raise ValueError("Invalid VLM dimensions, semantic token count or attention heads")
        dropout = settings.get("dropout", .1)
        self.vlm_projection = nn.Sequential(nn.LayerNorm(self.vlm_dim), nn.Linear(self.vlm_dim, 256))
        self.semantic_queries = nn.Parameter(torch.randn(1, self.semantic_tokens, 256) * .02)
        self.semantic_attention = nn.MultiheadAttention(256, semantic_heads, dropout=dropout, batch_first=True)
        self.semantic_norm = nn.LayerNorm(256)
        self.semantic_ffn = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 256))
        self.semantic_output_norm = nn.LayerNorm(256)
        self.modality_embedding = nn.Parameter(torch.zeros(3, 256))
        self.score = ScoreFunctionBranch(**settings)

    def prepare_condition(self, scene, ego, vlm, vlm_mask=None):
        """Read every valid VLM token through learned queries, without truncating it.

        Pooling is trainable and is performed after the frozen feature cache.
        The prepared condition can be reused across refinement steps at inference.
        ``vlm_mask=True`` denotes a real (non-padding) token.
        """
        if scene.ndim != 3 or tuple(scene.shape[1:]) != (64, 256):
            raise ValueError("Expected scene [B,64,256]")
        if ego.shape != (len(scene), 256):
            raise ValueError("Expected ego [B,256]")
        if vlm.ndim != 3 or len(vlm) != len(scene) or vlm.shape[-1] != self.vlm_dim or vlm.shape[1] < 1:
            raise ValueError(f"Expected nonempty VLM [B,L,{self.vlm_dim}]")
        if vlm_mask is None:
            vlm_mask = torch.ones(vlm.shape[:2], dtype=torch.bool, device=vlm.device)
        if vlm_mask.shape != vlm.shape[:2] or vlm_mask.dtype != torch.bool:
            raise ValueError("VLM mask must be boolean [B,L], True for valid tokens")
        if not vlm_mask.any(dim=1).all():
            raise ValueError("Every scene needs at least one valid VLM token")
        # Padding must affect neither the projection nor attention, even when
        # collators use arbitrary values outside each sequence's valid extent.
        values = self.vlm_projection(vlm.masked_fill(~vlm_mask[..., None], 0.))
        queries = self.semantic_queries.expand(len(scene), -1, -1)
        attended = self.semantic_attention(queries, values, values, key_padding_mask=~vlm_mask, need_weights=False)[0]
        semantics = self.semantic_norm(queries + attended)
        semantics = self.semantic_output_norm(semantics + self.semantic_ffn(semantics))
        context = torch.cat((scene + self.modality_embedding[0],
                             ego[:, None] + self.modality_embedding[1],
                             semantics + self.modality_embedding[2]), dim=1)
        return {"context": context, "ego": ego}

    def score_prepared(self, noisy, prepared):
        if noisy.ndim != 3 or tuple(noisy.shape[1:]) != (8, 4):
            raise ValueError("Expected normalized noisy trajectory [B,8,4]")
        if len(noisy) != len(prepared["context"]):
            raise ValueError("Trajectory and condition batch sizes differ")
        return self.score(noisy, prepared["context"], prepared["ego"])

    def forward(self, noisy, scene, ego, vlm, vlm_mask=None):
        return self.score_prepared(noisy, self.prepare_condition(scene, ego, vlm, vlm_mask))
