"""Physical-pose inference with a frozen DD scene reader and a learned conditional score."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from score_function.model.refinement import refine_ego
from score_function.navsim_dd.runtime import NavsimRefiner, poses_to_channels
from score_function.utils.normalizer import denormalize_ego_future, normalize_ego_future
from score_function.utils.train_utils import load_tensor

CHECKPOINT_FORMAT = "navsim-dd-reader-score-v1"


class _PreparedScore(nn.Module):
    """The scene is fixed; score_prepared re-reads it at every current trajectory."""

    parameterization = "score"

    def __init__(self, model, scene):
        super().__init__()
        self.model = model
        self.scene = scene
        self.eval()

    def forward(self, trajectory, latent, status):
        return self.model.score_prepared(trajectory, self.scene)


class DDReaderRefiner(NavsimRefiner):
    """Refine arbitrary NAVSIM 8-pose trajectories from one fixed scene encoding."""

    def require_native_conditioner(self, identity):
        """Verify freshly encoded native features against the exported official assets.

        Unlike require_conditioner, this does not compare a cache-specific reader
        asset-file digest: the frozen reader is already embedded in this checkpoint.
        """
        if self.conditioner_identity.get("condition") != "dd-backbone512-status-reader-v1":
            raise ValueError("Checkpoint does not declare full DD backbone conditioning")
        keys = ("dd_commit", "checkpoint_sha256", "anchor_sha256", "backbone_sha256")
        if any(not self.conditioner_identity.get(key)
               or identity.get(key) != self.conditioner_identity[key] for key in keys):
            raise ValueError("Native DD conditioning assets differ from reader score training")

    @classmethod
    def from_checkpoint(cls, path: Path, device):
        from score_function.navsim_dd.reader import DDReaderScore

        saved = load_tensor(path)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Expected a portable full DD reader score checkpoint")
        settings = saved["model"]
        required = {
            "architecture": "dd_scene_reader_score", "parameterization": "score",
            "future_len": 8, "input_dim": 4, "context_dim": 256,
            "temporal_attention": True, "neighbor_future": False,
        }
        if any(settings.get(key) != value for key, value in required.items()):
            raise ValueError("Incompatible full DD reader score configuration")
        model = DDReaderScore(settings, saved["normalizer"], saved["reader_spec"])
        model.load_state_dict(saved["state_dict"], strict=True)
        for name, tensor in (("mean", model.normalizer_mean), ("std", model.normalizer_std)):
            if not torch.equal(tensor.cpu(), torch.as_tensor(saved["normalizer"][name], dtype=tensor.dtype)):
                raise ValueError("Checkpoint normalizer metadata differs from the reader buffers")
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Nonfinite tensor in reader score checkpoint")
        return cls(model, saved["normalizer"], saved["sigma"], saved["gamma"],
                   saved["conditioner_identity"], device)

    @torch.no_grad()
    def refine_many(self, poses, context, global_context, ks, gamma=None):
        steps = tuple(ks)
        if not steps or any(type(k) is not int or k < 0 for k in steps):
            raise ValueError("ks must contain nonnegative integers")
        steps = sorted(set(steps))
        gamma = self.gamma if gamma is None else float(gamma)
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if poses.ndim != 3 or poses.shape[1:] != (8, 3):
            raise ValueError("Expected raw NAVSIM poses [B,8,3]")
        # Preserve the literal starting trajectory and avoid encoding a disabled arm.
        if steps[-1] == 0 or gamma == 0:
            return {k: poses for k in steps}
        if (context.shape != (len(poses), 512, 8, 8)
                or global_context.shape != (len(poses), 256)):
            raise ValueError("Expected DD backbone [B,512,8,8] and status [B,256]")
        latent = context.to(self.device, torch.float16).float()
        status = global_context.to(self.device, torch.float16).float()
        if not torch.isfinite(latent).all() or not torch.isfinite(status).all():
            raise ValueError("Nonfinite frozen DD conditioning")
        prepared = self.model.prepare_scene(latent, status)
        branch = _PreparedScore(self.model, prepared)
        current = normalize_ego_future(poses_to_channels(poses.to(self.device).float()), self.normalizer)
        result = {}
        previous = 0
        for k in steps:
            if k == 0:
                result[k] = poses
                continue
            current, _ = refine_ego(
                branch, current, latent, status, self.normalizer, self.sigma, gamma,
                k - previous, heading_projection=True, record_trace=False,
            )
            previous = k
            physical = denormalize_ego_future(current, self.normalizer)
            yaw = torch.atan2(physical[..., 3:4], physical[..., 2:3])
            result[k] = torch.cat((physical[..., :2], yaw), dim=-1).to(poses)
        return result
