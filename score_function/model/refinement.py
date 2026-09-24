"""Fixed-budget ego score refinement with an exact disabled baseline path."""

from __future__ import annotations

import math
import time

import torch

from score_function.utils.normalizer import (
    _validate_trajectory,
    heading_norm_deviation,
    project_heading,
)


def _synchronize(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _norm(trajectory: torch.Tensor) -> list[float]:
    return torch.linalg.vector_norm(trajectory.flatten(1), dim=1).cpu().tolist()


@torch.no_grad()
def refine_ego(
    score_branch,
    initial_norm: torch.Tensor,
    scene: torch.Tensor,
    route: torch.Tensor,
    normalizer,
    sigma: float,
    gamma: float,
    steps: int,
    heading_projection: bool = True,
    record_trace: bool = True,
    neighbor_future: torch.Tensor | None = None,
    neighbor_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Apply exactly K updates x += gamma*sigma^2*s(x,C,R), without adding noise.

    The branch must already be in evaluation mode.  The returned trace is JSON
    serializable; full per-step trajectories/scores are optional.  CUDA timing
    explicitly synchronizes.  No stopping/peak/density claim is inferred from K.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    _validate_trajectory(initial_norm, "initial_norm")
    disabled = gamma == 0 or steps == 0
    trace = {
        "sigma": float(sigma),
        "gamma": float(gamma),
        "requested_steps": steps,
        "completed_steps": 0,
        "disabled": disabled,
        "heading_projection": bool(heading_projection),
        "trajectories": [initial_norm.cpu().tolist()] if record_trace else [],
        "scores": [],
        "score_norm": [],
        "displacement_norm": [],
        "step_displacement_norm": [],
        "heading_norm_deviation_pre_projection": [],
        "heading_norm_max_deviation_pre_projection": [],
        "step_seconds": [],
        "total_seconds": 0.0,
        "coordinates": "normalized_ego_future",
        "timing_scope": "synchronized score and update, excluding trace serialization",
    }
    if disabled:
        # In particular, do not round-trip normalization or project headings.
        return initial_norm, trace
    if getattr(score_branch, "training", False):
        raise ValueError("score_branch must be in eval mode for deterministic refinement")
    if not torch.isfinite(scene).all() or not torch.isfinite(route).all():
        raise ValueError("Scene/route conditioning contains nonfinite values")
    _synchronize(initial_norm)
    started = time.perf_counter()
    current = initial_norm
    for step in range(steps):
        _synchronize(current)
        step_started = time.perf_counter()
        kwargs = (
            {"neighbor_future": neighbor_future, "neighbor_valid": neighbor_valid}
            if neighbor_future is not None
            else {}
        )
        score = score_branch(current, scene, route, **kwargs)
        if score.shape != current.shape or not torch.isfinite(score).all():
            raise ValueError(f"Invalid/nonfinite score at refinement step {step + 1}")
        updated = current + gamma * sigma**2 * score
        _validate_trajectory(updated, f"update at step {step + 1}")
        deviation = heading_norm_deviation(updated, normalizer)
        if heading_projection:
            updated = project_heading(updated, normalizer, previous_norm=current)
        _synchronize(updated)
        step_seconds = time.perf_counter() - step_started
        trace["score_norm"].append(_norm(score))
        trace["displacement_norm"].append(_norm(updated - initial_norm))
        trace["step_displacement_norm"].append(_norm(updated - current))
        trace["heading_norm_deviation_pre_projection"].append(deviation.mean(dim=1).cpu().tolist())
        trace["heading_norm_max_deviation_pre_projection"].append(
            deviation.amax(dim=1).cpu().tolist()
        )
        trace["step_seconds"].append(step_seconds)
        if record_trace:
            trace["scores"].append(score.cpu().tolist())
            trace["trajectories"].append(updated.cpu().tolist())
        trace["completed_steps"] = step + 1
        current = updated
    _synchronize(current)
    trace["total_seconds"] = time.perf_counter() - started
    trace["total_seconds_scope"] = "refinement including trace serialization"
    return current, trace
