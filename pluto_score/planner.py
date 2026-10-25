"""PLUTO w/o rule refinement, with an optional frozen score update.

Keep the official feature builder, reference lines, learned argmax/fallback and
trajectory conversion. No rule evaluator, candidate simulation, safety brake or
DP trajectory generator is instantiated. Model construction is deferred until
initialize(), after Ray assigns the simulation's GPU.
"""

import copy
import inspect
import logging
from types import SimpleNamespace

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from native_planner_score.planner import preserve_random_state
from urban_driver_score.planner import UrbanDriverScorePlanner


LOG = logging.getLogger(__name__)
MODEL_TARGET = "src.models.pluto.pluto_model.PlanningModel"
FEATURE_TARGET = "src.feature_builders.pluto_feature_builder.PlutoFeatureBuilder"
PLUTO_SOURCE_SHA = "b9964b649c660f1f4a971d614c66f5992e24c18a"
SCENARIO_RADIUS = 60.0  # official planner YAML: eval_dt .1 * eval_num_frames 40 * 60 / 4


def validated_model_config(model_config):
    cfg = OmegaConf.to_container(OmegaConf.create(model_config), resolve=True)
    if not isinstance(cfg, dict) or cfg.get("_target_") != MODEL_TARGET:
        raise ValueError("Require the official PLUTO PlanningModel configuration")
    feature = cfg.get("feature_builder", {})
    if (cfg.get("future_steps") != 80 or cfg.get("history_steps") != 21
            or feature.get("_target_") != FEATURE_TARGET
            or feature.get("build_reference_line") is not True
            or feature.get("future_horizon") != 8
            or feature.get("history_horizon") != 2
            or feature.get("sample_interval") != .1):
        raise ValueError("Require official PLUTO 80 future poses / 8 seconds, 21 history poses and reference lines")
    return cfg


def load_pluto_model(model_config, checkpoint_path, device="cpu"):
    """Load an official Lightning checkpoint strictly, using its original model."""
    cfg = validated_model_config(model_config)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("PLUTO device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested native PLUTO CUDA inference, but CUDA is unavailable")
    model = instantiate(OmegaConf.create(cfg))
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    checkpoint = torch.load(checkpoint_path, **options)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Require an official PLUTO Lightning checkpoint with state_dict")
    state = {}
    for name, tensor in checkpoint["state_dict"].items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Invalid PLUTO checkpoint parameter")
        # The official loader removes the Lightning model. prefix. A strict
        # load below rejects a different architecture or incomplete checkpoint.
        key = name[len("model."):] if name.startswith("model.") else name
        if key in state:
            raise ValueError(f"Duplicate PLUTO checkpoint parameter after prefix removal: {key}")
        state[key] = tensor
    model.load_state_dict(state, strict=True)
    builders = model.get_list_of_required_feature()
    if (model.future_steps != 80 or len(builders) != 1
            or builders[0].future_samples != 80 or builders[0].history_samples != 20
            or builders[0].sample_interval != .1 or not builders[0].build_reference_line):
        raise ValueError("Loaded PLUTO model/feature sampling differs from the official 80-pose configuration")
    model.to(device).eval().requires_grad_(False)
    return model


def preflight_model(model_config, checkpoint_path):
    """Check checkpoint/model compatibility without constructing a simulation."""
    with preserve_random_state("cpu"):
        model = load_pluto_model(model_config, checkpoint_path, "cpu")
        metadata = {
            "model_target": MODEL_TARGET, "native_pose_count": 80,
            "future_horizon_s": 8.0, "step_interval_s": .1,
            "reference_lines": True, "selection": "official_output_trajectory",
            "rule_postprocessing": False,
            "trajectory_conversion": "official_PLUTO_include_ego_state_false",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        del model
    return metadata


def trajectory_from_local_poses(poses, current_input):
    """Keep PLUTO's original conversion, including its synthetic state at t0.

    Predicted poses still occupy .1, .2, ..., 8 seconds. The official helper copies
    the first predicted state into t0 rather than inserting the observed EgoState.
    Use this same convention for native AND score-refined trajectories.
    """
    from src.planners.ml_planner_utils import global_trajectory_to_states

    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (80, 3) or not np.isfinite(poses).all():
        raise ValueError("Require finite PLUTO local rear-axle poses with shape (80, 3)")
    ego = current_input.history.ego_states[-1]
    angle = ego.rear_axle.heading
    rotation = np.array([[np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])
    global_poses = poses.copy()
    global_poses[:, :2] = poses[:, :2] @ rotation + ego.rear_axle.array
    global_poses[:, 2] += angle
    states = global_trajectory_to_states(
        global_trajectory=global_poses,
        ego_history=current_input.history.ego_states,
        future_horizon=8.0, step_interval=.1, include_ego_state=False,
    )
    return InterpolatedTrajectory(states)


class PlutoWithoutRefinementPlanner(AbstractPlanner):
    """Official learned argmax trajectory, without the hybrid postprocessor."""

    requires_scenario = False

    def __init__(self, model_config, checkpoint_path, device="cuda"):
        self.model_config = validated_model_config(model_config)
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self._model = None

    def name(self):
        return "PLUTO_wo_refine"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization):
        from src.scenario_manager.scenario_manager import ScenarioManager

        self._model = load_pluto_model(self.model_config, self.checkpoint_path, self.device)
        self._feature_builder = self._model.get_list_of_required_feature()[0]
        self._initialization = initialization
        # This manager also supplies reference-line INPUTS. Removing the rule
        # postprocessor must not remove reference lines or their route updates.
        self._scenario_manager = ScenarioManager(
            map_api=initialization.map_api, ego_state=None,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            radius=SCENARIO_RADIUS,
        )
        self._feature_builder.scenario_manager = self._scenario_manager

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input):
        ego = current_input.history.ego_states[-1]
        self._scenario_manager.update_ego_state(ego)
        self._scenario_manager.update_drivable_area_map()
        feature = self._feature_builder.get_features_from_simulation(
            current_input, self._initialization)
        batch = feature.collate([feature.to_feature_tensor()]).to_device(self.device)
        output = self._model.forward(batch.data)
        # Official eval forward already masks invalid reference lines, takes
        # learned-probability argmax, and uses ref-free output if no lines exist.
        selected = output["output_trajectory"]
        if selected.shape != (1, 80, 3) or not torch.isfinite(selected).all():
            raise ValueError("Invalid/nonfinite PLUTO output_trajectory")
        return trajectory_from_local_poses(selected[0].detach().cpu().numpy(), current_input)


class PlutoScorePlanner(UrbanDriverScorePlanner):
    """Same learned native baseline at K0; S10-G only changes the selected path."""

    requires_scenario = False
    native_pose_count = 80

    def __init__(self, model_config, checkpoint_path, score_config="unused",
                 score_checkpoint="unused", steps=0, gamma=.1, sigma=.1,
                 heading_projection=True, audit_dir=None, device="cuda", native_device="cuda"):
        cfg = validated_model_config(model_config)
        if native_device not in ("cpu", "cuda") or torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("Native and score devices must be cpu or cuda")
        super().__init__(copy.deepcopy(cfg), checkpoint_path, score_config, score_checkpoint,
                         steps, gamma, sigma, heading_projection, audit_dir, device)
        self.native_device = native_device

    def name(self):
        return "PLUTO_wo_refine_S10G_K20" if self.steps else "PLUTO_wo_refine"

    def initialize(self, initialization):
        self._native = PlutoWithoutRefinementPlanner(
            self.model_config, self.checkpoint_path, self.native_device)
        self._native.initialize(initialization)
        self._initialization = SimpleNamespace(
            map_api=initialization.map_api,
            route_roadblock_ids=tuple(initialization.route_roadblock_ids),
        )
        self._audited = False
        if not self.steps:
            return
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA score inference, but CUDA is unavailable")
        with preserve_random_state(self.device):
            from diffusion_planner.data_process.data_processor import DataProcessor
            from urban_driver_score.runtime import load_conditions, load_score

            cfg, state, self._score, self._args = load_score(
                self.score_config, self.score_checkpoint, self.device)
            if state["epoch"] != 529:
                raise ValueError("Require the frozen S10-G final EMA at epoch 529")
            self._encoder, self._route_encoder = load_conditions(cfg, self._args, self.device)
            self._processor = DataProcessor(self._args)
        self._checkpoint_metadata = {
            key: state[key] for key in ("epoch", "weight_kind", "sigma_score")
        }
        self._checkpoint_metadata.update(
            baseline="pluto_wo_refine", planner_name="pluto_wo_refine",
            native_device=self.native_device, score_device=self.device,
            rule_postprocessing=False, selection="official_output_trajectory",
            trajectory_conversion="official_PLUTO_include_ego_state_false",
        )
        LOG.info("PLUTO w/o refine ready: native %s, S10-G %s, K20 gamma=.1 sigma=.1",
                 self.native_device, self.device)

    def _trajectory_from_poses(self, poses, current_input):
        return trajectory_from_local_poses(poses, current_input)

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input):
        native = self._native.compute_planner_trajectory(current_input)
        if not self.steps:
            return native
        with preserve_random_state(self.device):
            return self.refine_native(native, current_input)
