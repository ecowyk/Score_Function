"""Deterministic density-score refinement after native ChainFlow top-1 selection."""
from __future__ import annotations

import math

import torch

from score_function.navsim_chainflow.model import ChainFlowScore
from score_function.navsim_dd.runtime import make_normalizer, poses_to_channels
from score_function.utils.normalizer import normalize_ego_future, denormalize_ego_future, project_heading
from score_function.utils.train_utils import load_tensor

CHECKPOINT_FORMAT = "navsim-chainflow-score-v1"


class ChainFlowRefiner:
    def __init__(self, model, normalizer, sigma, gamma, conditioner_identity, device):
        self.device = torch.device(device)
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.normalizer = make_normalizer(normalizer)
        self.sigma, self.gamma = float(sigma), float(gamma)
        self.conditioner_identity = dict(conditioner_identity)
        if not self.conditioner_identity:
            raise ValueError("Missing frozen ChainFlow condition identity")
        if not math.isfinite(self.sigma) or self.sigma <= 0 or not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("Invalid sigma or gamma")

    @classmethod
    def from_checkpoint(cls, path, device):
        saved = load_tensor(path)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Expected portable {CHECKPOINT_FORMAT}")
        model = ChainFlowScore(saved["model"])
        model.load_state_dict(saved["state_dict"], strict=True)
        return cls(model, saved["normalizer"], saved["sigma"], saved["gamma"], saved["conditioner_identity"], device)

    def require_conditioner(self, identity):
        if identity != self.conditioner_identity:
            raise ValueError("ChainFlow conditioner identity differs from score training")

    @torch.no_grad()
    def refine_many(self, rawposes, condition, ks, gamma=None):
        steps = list(ks)
        if not steps or any(type(k) is not int or k < 0 for k in steps):
            raise ValueError("ks must be a nonempty sequence of nonnegative integers")
        steps = sorted(set(steps))
        gamma = self.gamma if gamma is None else float(gamma)
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if rawposes.ndim != 3 or tuple(rawposes.shape[1:]) != (8, 3) or not torch.isfinite(rawposes).all():
            raise ValueError("Expected finite raw poses [B,8,3]")
        # Exact disabled baseline: no coordinate conversion, feature reading or
        # yaw wrapping. Native K0 bytes are shared by every refinement arm.
        if steps[-1] == 0 or gamma == 0:
            return {k: rawposes for k in steps}
        features = {}
        for name in ("scene", "ego", "vlm"):
            value = condition[name]
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite/missing condition: {name}")
            # Match the persisted FP16 frozen features used during training.
            features[name] = value.to(self.device, torch.float16).float()
            if not torch.isfinite(features[name]).all():
                raise ValueError(f"Condition overflows FP16 cache precision: {name}")
        mask = condition.get("vlm_mask")
        if mask is not None:
            features["vlm_mask"] = mask.to(self.device)
        prepared = self.model.prepare_condition(**features)
        current = normalize_ego_future(poses_to_channels(rawposes.to(self.device).float()), self.normalizer)
        result = {0: rawposes} if 0 in steps else {}
        for iteration in range(1, steps[-1] + 1):
            score = self.model.score_prepared(current, prepared)
            if score.shape != current.shape or not torch.isfinite(score).all():
                raise ValueError(f"Invalid score at refinement step {iteration}")
            updated = current + gamma * self.sigma**2 * score
            if not torch.isfinite(updated).all():
                raise ValueError(f"Nonfinite refinement step {iteration}")
            current = project_heading(updated, self.normalizer, previous_norm=current)
            if iteration in steps:
                physical = denormalize_ego_future(current, self.normalizer)
                yaw = torch.atan2(physical[..., 3:4], physical[..., 2:3])
                result[iteration] = torch.cat((physical[..., :2], yaw), dim=-1).to(rawposes)
        return result
