"""Extend the official nuPlan planner by wrapping only its neural-model output."""

from pathlib import Path
from uuid import uuid4

import torch
from diffusion_planner.planner.planner import DiffusionPlanner

from score_function.model.refined_planner import ScoreRefinedPlanner
from score_function.utils.checkpoint import load_selected
from score_function.utils.config import load_config
from score_function.utils.planner_utils import load_planner_config, planner_identity
from score_function.utils.sampling import planner_seed
from score_function.utils.train_utils import atomic_write, load_tensor, resolve_path


class ScoreFunctionPlanner(DiffusionPlanner):
    def __init__(
        self,
        score_config,
        score_checkpoint,
        past_trajectory_sampling,
        future_trajectory_sampling,
        root=None,
        device="cuda",
        gamma=None,
        steps=None,
        heading_projection=None,
        trace_dir=None,
    ):
        self.settings = load_config(score_config, root)
        for key, value in (
            ("gamma", gamma),
            ("steps", steps),
            ("heading_projection", heading_projection),
        ):
            if value is not None:
                self.settings["refinement"][key] = value
        self.score_checkpoint = score_checkpoint
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if (
            future_trajectory_sampling.num_poses != 80
            or future_trajectory_sampling.time_horizon != 8
        ):
            raise ValueError("Require official future sampling 80 poses / 8 seconds")
        super().__init__(
            config=load_planner_config(self.settings, device),
            ckpt_path=str(resolve_path(self.settings, "planner_checkpoint")),
            past_trajectory_sampling=past_trajectory_sampling,
            future_trajectory_sampling=future_trajectory_sampling,
            enable_ema=True,
            device=device,
        )

    def name(self):
        return "score_function_refined_diffusion_planner"

    def initialize(self, initialization):
        # nuPlan can reinitialize an instance for another scenario. Unwrap before
        # the official initializer restores the original model checkpoint.
        if isinstance(self._planner, ScoreRefinedPlanner):
            self._planner = self._planner.base_model
        state = load_tensor(self._ckpt_path)["ema_state_dict"]
        if state and all(key.startswith("module.") for key in state):
            del state
            super().initialize(initialization)
        else:
            # The official initializer discards every unprefixed key. Retain
            # compatibility with single-process checkpoints without patching it.
            self._planner.load_state_dict(
                {key.removeprefix("module."): value for key, value in state.items()}, strict=True
            )
            del state
            self._planner.to(self._device).eval()
            self._map_api = initialization.map_api
            self._route_roadblock_ids = initialization.route_roadblock_ids
            self._initialization = initialization
        self._planner.requires_grad_(False)
        branch, checkpoint = load_selected(self.settings, self.score_checkpoint, self._device)
        if checkpoint["planner"] != planner_identity(self.settings):
            raise ValueError("Selected branch requires its original frozen Planner and normalizers")
        self._planner = ScoreRefinedPlanner(
            self._planner,
            branch,
            self._config.state_normalizer,
            checkpoint["sigma_score"],
            **self.settings["refinement"],
        ).eval()
        self._planner.record_trace = self.trace_dir is not None
        self.scenario_trace_dir = self.trace_dir / uuid4().hex if self.trace_dir else None
        if self.scenario_trace_dir is not None:
            atomic_write(
                self.scenario_trace_dir / "scenario_context.json",
                {
                    "map_name": getattr(initialization.map_api, "map_name", None),
                    "route_roadblock_ids": list(initialization.route_roadblock_ids),
                    "score_checkpoint": str(self.score_checkpoint),
                },
            )

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input):
        timestamp = current_input.history.ego_states[-1].time_point.time_us
        device = next(self._planner.parameters()).device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(planner_seed(timestamp, self.settings["reference_seed"]))
            # Official input adapter, normalization, model call, trajectory
            # conversion and InterpolatedTrajectory construction run unchanged.
            trajectory = super().compute_planner_trajectory(current_input)
        if self.scenario_trace_dir is not None:
            atomic_write(
                self.scenario_trace_dir / f"{timestamp}.json",
                {
                    "timestamp_us": timestamp,
                    "planning_iteration": current_input.iteration.index,
                    "baseline_ego": self._planner.last_baseline_ego[0].tolist(),
                    "refined_ego": self._planner.last_refined_ego[0].tolist(),
                    "trace": self._planner.last_refinement_trace,
                },
            )
        return trajectory
