"""Ego-only normalization and heading representation validity helpers."""

from __future__ import annotations

import math

import torch


def _validate_trajectory(trajectory: torch.Tensor, name: str) -> None:
    if trajectory.ndim != 3 or trajectory.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [B,T,4]")
    if not trajectory.is_floating_point():
        raise TypeError(f"{name} must be floating-point")
    if not torch.isfinite(trajectory).all():
        raise ValueError(f"{name} contains nonfinite values")


def ego_statistics(trajectory: torch.Tensor, normalizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Select agent zero explicitly, avoiding accidental multi-agent broadcasting."""
    _validate_trajectory(trajectory, "trajectory")
    mean = torch.as_tensor(normalizer.mean, device=trajectory.device, dtype=trajectory.dtype)
    std = torch.as_tensor(normalizer.std, device=trajectory.device, dtype=trajectory.dtype)
    if (
        mean.ndim not in (2, 3)
        or mean.shape != std.shape
        or mean.shape[0] < 1
        or mean.shape[-1] != 4
        or mean[0].numel() != 4
    ):
        raise ValueError("Expected normalizer mean/std [agents,1,4] or [agents,4]")
    mean, std = mean[0].reshape(1, 1, 4), std[0].reshape(1, 1, 4)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Ego normalization requires finite mean and positive finite std")
    return mean, std


def normalize_ego_future(trajectory: torch.Tensor, normalizer) -> torch.Tensor:
    """Map physical relative ego future [B,T,4] into official ego coordinates."""
    mean, std = ego_statistics(trajectory, normalizer)
    result = (trajectory - mean) / std
    if not torch.isfinite(result).all():
        raise ValueError("Ego normalization produced nonfinite values")
    return result


def denormalize_ego_future(trajectory_norm: torch.Tensor, normalizer) -> torch.Tensor:
    """Map normalized ego future [B,T,4] back to physical relative coordinates."""
    mean, std = ego_statistics(trajectory_norm, normalizer)
    result = trajectory_norm * std + mean
    if not torch.isfinite(result).all():
        raise ValueError("Ego denormalization produced nonfinite values")
    return result


def heading_norm_deviation(trajectory_norm: torch.Tensor, normalizer) -> torch.Tensor:
    """Return per-point absolute heading unit-circle error in physical space."""
    physical = denormalize_ego_future(trajectory_norm, normalizer)
    return (torch.linalg.vector_norm(physical[..., 2:4], dim=-1) - 1).abs()


def project_heading(
    trajectory_norm: torch.Tensor,
    normalizer,
    previous_norm: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project physical (cos,sin), using previous heading if the update is zero.

    If both headings are degenerate, the valid fallback is (1,0).  x/y remain
    bitwise unchanged in normalized coordinates.  Inputs are never mutated.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    physical = denormalize_ego_future(trajectory_norm, normalizer)
    heading = physical[..., 2:4]
    norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    fallback = torch.zeros_like(heading)
    fallback[..., 0] = 1
    if previous_norm is not None:
        if previous_norm.shape != trajectory_norm.shape:
            raise ValueError("previous_norm must have the same shape as trajectory_norm")
        previous = denormalize_ego_future(previous_norm, normalizer)[..., 2:4]
        previous_length = torch.linalg.vector_norm(previous, dim=-1, keepdim=True)
        fallback = torch.where(
            previous_length > eps, previous / previous_length.clamp_min(eps), fallback
        )
    projected_heading = torch.where(norm > eps, heading / norm.clamp_min(eps), fallback)
    mean, std = ego_statistics(trajectory_norm, normalizer)
    result = trajectory_norm.clone()
    result[..., 2:4] = (projected_heading - mean[..., 2:4]) / std[..., 2:4]
    if not torch.isfinite(result).all():
        raise ValueError("Heading projection produced nonfinite values")
    return result
