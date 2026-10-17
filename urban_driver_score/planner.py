"""Keep native UrbanDriver inference; refine only its finished ego trajectory."""

import json
import logging
from pathlib import Path
from uuid import uuid4

import torch
from omegaconf import OmegaConf

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from score_function.model.refinement import refine_ego
from score_function.utils.normalizer import denormalize_ego_future, normalize_ego_future
from urban_driver_score.geometry import physical_to_poses, sample_native

LOG = logging.getLogger(__name__)


class UrbanDriverScorePlanner(AbstractPlanner):
    def __init__(self, model_config, checkpoint_path, score_config, score_checkpoint,
                 steps=20, gamma=.1, sigma=.1, heading_projection=True,
                 audit_dir=None, device="cuda"):
        if steps not in (0, 20) or gamma != .1 or sigma != .1 or heading_projection is not True:
            raise ValueError("This adapter fixes K=20, gamma=0.1, sigma=0.1, projection=True; K=0 is check-only")
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path
        self.score_config, self.score_checkpoint = score_config, score_checkpoint
        self.steps, self.gamma, self.sigma = steps, gamma, sigma
        self.audit_dir, self.device = audit_dir, device
        self._native = None

    def name(self):
        return "UrbanDriver_S10G_K20" if self.steps else "UrbanDriver_K0_passthrough"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization):
        from nuplan.planning.script.builders.planner_builder import build_planners

        # Construct/load only after Ray assigns the worker a GPU. The driver's
        # 272 planner objects contain configuration only, not 272 GPU models.
        native_cfg = OmegaConf.create({"ml_planner": {
            "_target_": "nuplan.planning.simulation.planner.ml_planner.ml_planner.MLPlanner",
            "_convert_": "all", "model_config": self.model_config,
            "checkpoint_path": self.checkpoint_path,
        }})
        self._native = build_planners(native_cfg, None)[0]
        self._native.initialize(initialization)
        if self._native._num_output_dim != 16 or self._native._future_horizon != 8.0:
            raise ValueError("Require the native UrbanDriver 16-pose / 8-second model")
        self._initialization = initialization
        self._audited = False
        if not self.steps:
            return
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this refinement run")
        from diffusion_planner.data_process.data_processor import DataProcessor
        from urban_driver_score.runtime import load_conditions, load_score

        cfg, state, self._score, self._args = load_score(
            self.score_config, self.score_checkpoint, self.device)
        self._encoder, self._route_encoder = load_conditions(cfg, self._args, self.device)
        self._processor = DataProcessor(self._args)
        self._checkpoint_metadata = {k: state[k] for k in ("epoch", "weight_kind", "sigma_score")}
        LOG.info("UrbanDriver score ready: K=20 gamma=0.1 sigma=0.1; native16 -> score80; no DP sampling")

    def refine_native(self, native, current_input):
        # Return the *same object* before any interpolation, normalization,
        # condition encoding, score evaluation or trajectory reconstruction.
        if not self.steps:
            return native
        ego = current_input.history.ego_states[-1]
        physical = torch.as_tensor(sample_native(native, ego), device=self.device).unsqueeze(0)
        initial = normalize_ego_future(physical, self._args.state_normalizer)
        init = self._initialization
        raw = self._processor.observation_adapter(
            current_input.history, list(current_input.traffic_light_data or []),
            init.map_api, list(init.route_roadblock_ids), self.device)
        inputs = self._args.observation_normalizer(raw)
        scene = self._encoder(inputs)["encoding"]
        route = self._route_encoder(inputs["route_lanes"])
        if tuple(scene.shape) != (1, 107, 192) or tuple(route.shape) != (1, 192):
            raise ValueError("Unexpected frozen score conditioning shape")
        result, trace = refine_ego(
            self._score, initial, scene, route, self._args.state_normalizer,
            self.sigma, self.gamma, self.steps, heading_projection=True,
            record_trace=not self._audited)
        if trace["completed_steps"] != self.steps:
            raise RuntimeError("Refinement did not execute all 20 steps")
        revised = denormalize_ego_future(result, self._args.state_normalizer)
        poses = physical_to_poses(revised[0].detach().cpu().numpy())
        states = transform_predictions_to_states(poses, current_input.history.ego_states, 8., .1)
        trajectory = InterpolatedTrajectory(states)
        if not self._audited and self.audit_dir:
            directory = Path(self.audit_dir)
            directory.mkdir(parents=True, exist_ok=True)
            # One small audit per scenario, rather than tracing every frame.
            audit = dict(self._checkpoint_metadata, K=self.steps, gamma=self.gamma,
                         timestamp_us=ego.time_point.time_us,
                         native_pose_count=16, score_pose_count=80,
                         completed_steps=trace["completed_steps"],
                         finite=bool(torch.isfinite(revised).all()),
                         mean_xy_change_m=float(torch.linalg.vector_norm(
                             revised[..., :2] - physical[..., :2], dim=-1).mean()),
                         max_heading_unit_error=float((torch.linalg.vector_norm(
                             revised[..., 2:], dim=-1) - 1).abs().max()))
            (directory / (uuid4().hex + ".json")).write_text(json.dumps(audit, indent=2) + "\n")
            LOG.info("First-frame refinement audit: %s", audit)
        self._audited = True
        return trajectory

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input):
        native = self._native.compute_planner_trajectory(current_input)
        return self.refine_native(native, current_input)
