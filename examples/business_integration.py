"""Physical-trajectory adapter; run ``python -m examples.business_integration --smoke``.

The smoke uses random, untrained G weights and synthetic conditions. It checks
the interface only, with no claim about driving quality. See docs/integration.md
for checkpoint, coordinate, time-grid, and frozen-encoder requirements.
"""

from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

import torch

from score_function.model.refinement import refine_ego
from score_function.model.score_branch import ScoreFunctionBranch
from score_function.utils.normalizer import (
    _validate_trajectory,
    denormalize_ego_future,
    normalize_ego_future,
)


@torch.no_grad()
def refine_physical_ego(
    initial_physical: torch.Tensor,
    branch,
    scene: torch.Tensor | None,
    route: torch.Tensor | None,
    normalizer,
    *,
    sigma: float,
    gamma: float,
    steps: int,
    heading_projection: bool = True,
    record_trace: bool = False,
    neighbor_future: torch.Tensor | None = None,
    neighbor_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Refine [B,T,4] physical (x,y,cos,sin) in the current ego frame.

    The caller loads matching weights, normalization statistics and frozen
    conditions once; this function neither loads nor runs a business planner.
    Optional neighbor conditions are passed through in their training format.
    All tensors must share the branch's device/dtype and training time grid.

    With K=0 or gamma=0, return the original tensor before normalization or any
    access to branch/conditions/normalizer. A business trajectory object should
    also be bypassed at the caller, before conversion/resampling/encoding.
    Enabled traces describe normalized trajectories; the returned tensor is
    always physical. Disabled traces describe the untouched physical input.
    """
    # Validate before the bypass: disabling refinement must not hide bad knobs.
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    _validate_trajectory(initial_physical, "initial_physical")
    if steps == 0 or gamma == 0:
        return initial_physical, {
            "sigma": float(sigma),
            "gamma": float(gamma),
            "requested_steps": steps,
            "completed_steps": 0,
            "disabled": True,
            "coordinates": "physical_ego_future",
            "trajectories": [initial_physical.cpu().tolist()] if record_trace else [],
        }
    if scene is None or route is None:
        raise ValueError("Enabled refinement requires frozen scene and route conditions")
    initial_norm = normalize_ego_future(initial_physical, normalizer)
    refined_norm, trace = refine_ego(
        branch,
        initial_norm,
        scene,
        route,
        normalizer,
        sigma=sigma,
        gamma=gamma,
        steps=steps,
        heading_projection=heading_projection,
        record_trace=record_trace,
        neighbor_future=neighbor_future,
        neighbor_valid=neighbor_valid,
    )
    return denormalize_ego_future(refined_norm, normalizer), trace


def smoke_check() -> dict:
    """Exercise the full-width G architecture on CPU without external assets."""
    torch.set_num_threads(2)
    torch.manual_seed(20261019)
    branch = ScoreFunctionBranch(
        temporal_attention=True, neighbor_future=False, parameterization="score"
    ).eval().requires_grad_(False)
    trajectory = torch.zeros(1, 80, 4)
    trajectory[0, :, 0] = torch.linspace(0.1, 8.0, 80)
    trajectory[..., 2] = 1.0
    original = trajectory.clone()
    scene = torch.randn(1, 107, 192)
    route = torch.randn(1, 192)
    # Synthetic statistics only: real weights require their saved statistics.
    normalizer = SimpleNamespace(
        mean=torch.tensor([[1.0, -2.0, 0.1, -0.2]]),
        std=torch.tensor([[10.0, 5.0, 0.5, 0.25]]),
    )
    refined, trace = refine_physical_ego(
        trajectory, branch, scene, route, normalizer, sigma=0.1, gamma=0.1, steps=20
    )
    disabled, disabled_trace = refine_physical_ego(
        trajectory, None, None, None, None, sigma=0.1, gamma=0.1, steps=0
    )
    assert refined.shape == trajectory.shape and torch.isfinite(refined).all()
    assert disabled is trajectory and disabled_trace["completed_steps"] == 0
    assert torch.equal(trajectory, original) and trace["completed_steps"] == 20
    heading_error = (refined[..., 2:].norm(dim=-1) - 1).abs().max().item()
    assert heading_error < 1e-6
    return {
        "status": "passed",
        "scope": "interface only; random untrained G weights; synthetic conditions",
        "device": "cpu",
        "shape": list(refined.shape),
        "completed_steps": trace["completed_steps"],
        "disabled_returns_original": disabled is trajectory,
        "input_unchanged": torch.equal(trajectory, original),
        "max_heading_unit_error": heading_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a CPU interface check")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Use --smoke for synthetic inputs; import refine_physical_ego for integration")
    print(json.dumps(smoke_check(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
