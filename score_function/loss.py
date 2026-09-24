"""Fixed-scale DSM, independent of diffusion schedules and timestep inputs."""

import math

import torch


def fixed_sigma_dsm_loss(score: torch.Tensor, epsilon: torch.Tensor, sigma: float) -> torch.Tensor:
    """Mean over samples, future points, and coordinates of (sigma*s + eps)^2."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if score.shape != epsilon.shape or score.ndim != 3 or score.shape[-1] != 4:
        raise ValueError("score and epsilon must have the same [B,T,4] shape")
    return (sigma * score + epsilon).square().mean()
