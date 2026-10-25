"""Preserve official PlanTF inference and refine its exact 80 future knots.

The model is constructed inside initialize(), after Ray assigns a GPU. K0
returns the native trajectory object untouched; K20 adds the frozen S10-G
update without changing the native model, mode selection or state conversion.
"""

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from plantf_score.rng import preserve_random_state
from score_function.model.refinement import refine_ego
from score_function.utils.normalizer import denormalize_ego_future, normalize_ego_future
from urban_driver_score.geometry import physical_to_poses, relative_features


LOG = logging.getLogger(__name__)
PLANNER_TARGET = "src.planners.imitation_planner.ImitationPlanner"
MODEL_TARGET = "src.models.planTF.planning_model.PlanningModel"
FEATURE_TARGET = "src.feature_builders.nuplan_feature_builder.NuplanFeatureBuilder"


def validated_native_config(native_planner_config):
    cfg = OmegaConf.to_container(OmegaConf.create(native_planner_config), resolve=True)
    if not isinstance(cfg, dict) or list(cfg) != ["imitation_planner"]:
        raise ValueError("Require exactly one official imitation_planner configuration")
    planner = cfg["imitation_planner"]
    model = planner.get("planner", {})
    feature = model.get("feature_builder", {})
    if (planner.get("_target_") != PLANNER_TARGET
            or planner.get("replan_interval", 1) != 1
            or model.get("_target_") != MODEL_TARGET
            or model.get("future_steps") != 80
            or model.get("history_steps") != 21
            or feature.get("_target_") != FEATURE_TARGET
            or feature.get("future_horizon") != 8
            or feature.get("history_horizon") != 2
            or feature.get("sample_interval") != .1
            or not planner.get("planner_ckpt")):
        raise ValueError("Require official PlanTF: 80 poses / 8 seconds, 21 history poses, replan_interval=1 and checkpoint")
    return cfg


def native_knot_features(trajectory, ego):
    """Use the native knots directly; do not interpolate or infer dummy sampling.

    PlanTF's TorchModuleWrapper sampling metadata is an unused 8-pose dummy.
    The actual planner returns the observed state followed by 80 future states.
    Float-second conversion may truncate a timestamp by one microsecond.
    """
    states = trajectory.get_sampled_trajectory()
    if len(states) != 81 or states[0] is not ego:
        raise ValueError("Require observed ego t0 plus exactly 80 native PlanTF future states")
    start = int(ego.time_point.time_us)
    times = np.asarray([state.time_point.time_us for state in states], dtype=np.int64)
    expected = start + np.arange(81, dtype=np.int64) * 100000
    error = np.abs(times - expected)
    if np.any(np.diff(times) <= 0) or int(error.max()) > 2:
        raise ValueError("Native PlanTF knots do not cover 0..8 seconds at 0.1-second intervals")
    if (trajectory.start_time.time_us != times[0]
            or trajectory.end_time.time_us != times[-1]):
        raise ValueError("Native trajectory endpoints disagree with its sampled knots")
    global_poses = np.asarray([state.rear_axle.serialize() for state in states[1:]], dtype=np.float64)
    physical = relative_features(global_poses, ego.rear_axle.serialize())
    audit = {
        "native_t0_is_observed_ego": True,
        "native_knot_sampling": "exact_sampled_states_no_interpolation",
        "native_first_future_offset_us": int(times[1] - start),
        "native_last_future_offset_us": int(times[-1] - start),
        "native_max_timestamp_error_us": int(error.max()),
    }
    return physical, audit


def trajectory_from_local_poses(poses, current_input, native_planner):
    """Reuse PlanTF's own global conversion and observed-current-state policy."""
    from src.planners.planner_utils import global_trajectory_to_states

    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (80, 3) or not np.isfinite(poses).all():
        raise ValueError("Require finite local rear-axle poses with shape (80, 3)")
    ego = current_input.history.ego_states[-1]
    global_poses = native_planner._get_global_trajectory(poses, ego)
    states = global_trajectory_to_states(
        global_trajectory=global_poses, ego_history=current_input.history.ego_states,
        future_horizon=8.0, step_interval=.1, include_ego_state=True,
    )
    if len(states) != 81 or states[0] is not ego:
        raise RuntimeError("Official PlanTF trajectory conversion changed its t0/80-pose convention")
    return InterpolatedTrajectory(states)


class PlanTFScorePlanner(AbstractPlanner):
    requires_scenario = False
    native_pose_count = 80

    def __init__(self, native_planner_config, score_config="unused", score_checkpoint="unused",
                 steps=20, gamma=.1, sigma=.1, heading_projection=True,
                 audit_dir=None, device="cuda"):
        if (not isinstance(steps, int) or isinstance(steps, bool) or steps not in (0, 20)
                or gamma != .1 or sigma != .1 or heading_projection is not True):
            raise ValueError("Require K20, gamma=0.1, sigma=0.1, projection=True; K0 is passthrough-only")
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("Score device must be cpu or cuda")
        self.native_planner_config = copy.deepcopy(validated_native_config(native_planner_config))
        self.score_config, self.score_checkpoint = score_config, score_checkpoint
        self.steps, self.gamma, self.sigma = steps, gamma, sigma
        self.audit_dir, self.device = audit_dir, str(torch.device(device))
        self._native = None
        self._audited = False

    def name(self):
        return "PlanTF_S10G_K20" if self.steps else "PlanTF_K0_passthrough"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization):
        # Capture before native initialization so a future native route correction
        # cannot change the condition supplied to the frozen score processor.
        original_route = tuple(initialization.route_roadblock_ids)
        native_cfg = self.native_planner_config["imitation_planner"]
        if native_cfg.get("use_gpu", True) and not torch.cuda.is_available():
            raise RuntimeError("Native PlanTF requests GPU inference, but CUDA is unavailable")
        self._native = instantiate(OmegaConf.create(native_cfg))
        self._native.initialize(initialization)
        if (self._native._future_horizon != 8.0 or self._native._step_interval != .1
                or self._native._replan_interval != 1 or self._native._planner.future_steps != 80):
            raise ValueError("Loaded native PlanTF does not have the required real 80-pose sampling")
        self._initialization = SimpleNamespace(map_api=initialization.map_api,
                                               route_roadblock_ids=original_route)
        self._audited = False
        if not self.steps:
            return
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Score requests CUDA inference, but CUDA is unavailable")
        with preserve_random_state(self.device):
            from diffusion_planner.data_process.data_processor import DataProcessor
            from urban_driver_score.runtime import load_conditions, load_score

            cfg, state, self._score, self._args = load_score(
                self.score_config, self.score_checkpoint, self.device)
            if state["epoch"] != 529:
                raise ValueError("Require frozen S10-G final EMA at epoch 529")
            self._encoder, self._route_encoder = load_conditions(cfg, self._args, self.device)
            self._processor = DataProcessor(self._args)
        self._checkpoint_metadata = {
            key: state[key] for key in ("epoch", "weight_kind", "sigma_score")
        }
        self._checkpoint_metadata.update(
            baseline="plantf", planner_name="plantf",
            native_checkpoint=native_cfg["planner_ckpt"],
            native_device=str(self._native.device), score_device=self.device,
            trajectory_conversion="official_PlanTF_include_observed_ego_state",
            selection="official_output_trajectory_argmax",
        )
        LOG.info("PlanTF score ready: native80 -> score80, K20 gamma=.1 sigma=.1; no resampling or DP generation")

    def refine_native(self, native, current_input):
        if not self.steps:
            return native
        ego = current_input.history.ego_states[-1]
        values, native_audit = native_knot_features(native, ego)
        physical = torch.as_tensor(values, device=self.device).unsqueeze(0)
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
        if trace["completed_steps"] != 20:
            raise RuntimeError("Refinement did not complete exactly 20 updates")
        revised = denormalize_ego_future(result, self._args.state_normalizer)
        poses = physical_to_poses(revised[0].detach().cpu().numpy())
        trajectory = trajectory_from_local_poses(poses, current_input, self._native)
        if not self._audited and self.audit_dir:
            directory = Path(self.audit_dir)
            directory.mkdir(parents=True, exist_ok=True)
            audit = dict(self._checkpoint_metadata, **native_audit,
                         K=self.steps, gamma=self.gamma, sigma=self.sigma,
                         timestamp_us=int(ego.time_point.time_us),
                         native_pose_count=80, score_pose_count=80,
                         completed_steps=trace["completed_steps"],
                         finite=bool(torch.isfinite(revised).all()),
                         refined_t0_is_observed_ego=trajectory.get_sampled_trajectory()[0] is ego,
                         mean_xy_change_m=float(torch.linalg.vector_norm(
                             revised[..., :2] - physical[..., :2], dim=-1).mean()),
                         max_heading_unit_error=float((torch.linalg.vector_norm(
                             revised[..., 2:], dim=-1) - 1).abs().max()))
            (directory / (uuid4().hex + ".json")).write_text(json.dumps(audit, indent=2) + "\n")
            LOG.info("First-frame PlanTF refinement audit: %s", audit)
        self._audited = True
        return trajectory

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input):
        native = self._native.compute_planner_trajectory(current_input)
        if not self.steps:
            return native
        # Preserve the native planner's own RNG consumption; isolate only added work.
        with preserve_random_state(self.device):
            return self.refine_native(native, current_input)
