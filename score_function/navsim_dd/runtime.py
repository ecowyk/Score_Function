"""Bridge physical NAVSIM poses to a trained fixed-sigma score in normalized space."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch

from score_function.model.refinement import refine_ego
from score_function.model.score_branch import build_model
from score_function.utils.normalizer import denormalize_ego_future, normalize_ego_future

CHECKPOINT_FORMAT = "navsim-dd-score-v1"


def poses_to_channels(poses: torch.Tensor) -> torch.Tensor:
    """Accept one or a batch of trajectories, in metres and radians, excluding t=0."""
    if poses.ndim not in (2, 3) or poses.shape[-2:] != (8, 3):
        raise ValueError(f"Expected [...,8,3] NAVSIM poses, got {tuple(poses.shape)}")
    if not poses.is_floating_point() or not torch.isfinite(poses).all():
        raise ValueError("NAVSIM poses must be finite floating-point values")
    yaw = poses[..., 2:3]
    return torch.cat((poses[..., :2], yaw.cos(), yaw.sin()), dim=-1)


def make_normalizer(stats):
    """Use one set of train-only xy statistics for all timesteps, preserving angle scale."""
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32)
    if mean.shape != (4,) or std.shape != (4,):
        raise ValueError("Expected four scalar mean/std values")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Nonfinite statistics or nonpositive standard deviation")
    if (mean[2:] != 0).any() or (std[2:] != 1).any():
        raise ValueError("NAVSIM v1 expects unscaled cos/sin channels")
    return SimpleNamespace(mean=mean[None], std=std[None])


class NavsimRefiner:
    """Reusable postprocessor for any planner with the NAVSIM 4s / 2Hz trajectory contract."""

    def __init__(self, model, stats, sigma, gamma, conditioner_identity, device):
        self.device = torch.device(device)
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.normalizer = make_normalizer(stats)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Checkpoint sigma must be positive and finite")
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("Checkpoint gamma must be nonnegative and finite")
        if not conditioner_identity:
            raise ValueError("Missing frozen conditioner identity")
        self.conditioner_identity = dict(conditioner_identity)

    @classmethod
    def from_checkpoint(cls, path: Path, device):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Expected a portable {CHECKPOINT_FORMAT} EMA checkpoint")
        settings = saved["model"]
        if (settings.get("future_len"), settings.get("input_dim"), settings.get("context_dim")) != (8, 4, 256):
            raise ValueError("Incompatible NAVSIM trajectory/condition dimensions")
        if settings.get("parameterization") != "score" or settings.get("neighbor_future"):
            raise ValueError("Expected an ego-only direct score")
        model = build_model({"model": settings}, torch.device("cpu"))
        model.load_state_dict(saved["state_dict"], strict=True)
        return cls(
            model, saved["normalizer"], saved["sigma"], saved["gamma"],
            saved["conditioner_identity"], device,
        )

    def require_conditioner(self, identity):
        if self.conditioner_identity != identity:
            raise ValueError("Frozen conditioner assets/schema differ from score training")

    @torch.no_grad()
    def refine_many(self, poses, context, global_context, ks, gamma=None):
        """Return matched physical trajectories at each K, advancing the same path only once."""
        steps = tuple(ks)
        if not steps or any(type(k) is not int or k < 0 for k in steps):
            raise ValueError("ks must contain nonnegative integers")
        steps = sorted(set(steps))
        gamma = self.gamma if gamma is None else float(gamma)
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be nonnegative and finite")
        if poses.ndim != 3 or poses.shape[1:] != (8, 3):
            raise ValueError("Expected raw NAVSIM poses [B,8,3]")
        # Preserve original dtype, yaw branch and every input bit on the disabled path.
        if steps[-1] == 0 or gamma == 0:
            return {k: poses for k in steps}
        if context.shape != (len(poses), 64, 256) or global_context.shape != (len(poses), 256):
            raise ValueError("Expected DD scene [B,64,256] and status [B,256]")
        # Match the FP16 feature storage used in the training cache. The score runs in FP32.
        scene = context.to(self.device, torch.float16).float()
        status = global_context.to(self.device, torch.float16).float()
        current = normalize_ego_future(poses_to_channels(poses.to(self.device).float()), self.normalizer)
        result = {}
        previous = 0
        for k in steps:
            if k == 0:
                result[k] = poses
                continue
            current, _ = refine_ego(
                self.model, current, scene, status, self.normalizer, self.sigma, gamma,
                k - previous, heading_projection=True, record_trace=False,
            )
            previous = k
            physical = denormalize_ego_future(current, self.normalizer)
            yaw = torch.atan2(physical[..., 3:4], physical[..., 2:3])
            result[k] = torch.cat((physical[..., :2], yaw), dim=-1).to(poses)
        return result

    def refine(self, poses, context, global_context, steps, gamma=None):
        return self.refine_many(poses, context, global_context, [steps], gamma=gamma)[steps]
