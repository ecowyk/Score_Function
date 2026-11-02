"""Run the published Flow Planner unchanged before optional S10-G refinement.

Native feature processing, flow sampling, normalizers and nuPlan conversion stay
in the official implementation. Models are constructed only inside initialize,
after Ray assigns a GPU. Extra score work cannot advance the native RNG stream.
"""

import hashlib
import json
import logging
import math
from numbers import Real
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from native_planner_score.planner import preserve_random_state
from urban_driver_score.planner import UrbanDriverScorePlanner


LOG = logging.getLogger(__name__)
FLOW_SOURCE_SHA = "366df7654d7b7fa77f9d2577279da63127c6d2f8"
MODEL_TARGET = "flow_planner.model.flow_planner_model.flow_planner.FlowPlanner"
CORE_TARGET = "flow_planner.core.flow_matching_core.FlowMatchingCore"
ODE_TARGET = "flow_planner.model.flow_planner_model.flow_utils.flow_ode.FlowODE"
SEED_POLICY = "once_per_scenario_before_native_initialization"
ALLOWED_TARGETS = {
    MODEL_TARGET, CORE_TARGET, ODE_TARGET,
    "flow_matching.path.affine.AffineProbPath",
    "flow_matching.path.scheduler.CondOTScheduler",
    "flow_planner.model.model_utils.time_sampler.TimeSampler",
    "flow_planner.data.normalization.obs_normalize.ObservationNormalizer",
    "flow_planner.data.normalization.state_normalize.StateNormalizer",
    "flow_planner.model.model_utils.input_preprocess.ModelInputProcessor",
    "flow_planner.model.flow_planner_model.encoder.FlowPlannerEncoder",
    "flow_planner.model.flow_planner_model.decoder.FlowPlannerDecoder",
    "flow_planner.model.modules.encoder_modules.AgentFusionEncoder",
    "flow_planner.model.modules.encoder_modules.StaticFusionEncoder",
    "flow_planner.model.modules.encoder_modules.LaneFusionEncoder",
    "flow_planner.model.modules.encoder_modules.RouteEncoder",
    "flow_planner.model.modules.decoder_modules.TimestepEmbedder",
    "flow_planner.data.augmentation.nuplan_augmentation.NuPlanAugment",
    "flow_planner.data.augmentation.state_aug.StatePerturbation",
}


def _validate_targets(value):
    if isinstance(value, dict):
        if "_target_" in value and value["_target_"] not in ALLOWED_TARGETS:
            raise ValueError(f"Unexpected Flow Planner component: {value['_target_']}")
        if value.get("_partial_", False) or value.get("_recursive_", True) is not True:
            raise ValueError("Require fully constructed official Flow Planner components")
        for child in value.values():
            _validate_targets(child)
    elif isinstance(value, list):
        for child in value:
            _validate_targets(child)


def validated_config(config_path, device="cuda", *, cfg_weight=1.0):
    """Resolve only inference components, preserving published normalizers.

The released YAML omits its training dataset subtree, although its state
normalizer interpolates two fields from it. Fill those fields with their exact
official nuplan_data.yaml values. Unused recorder/DDP environment variables are
deliberately never resolved. The downloaded file is never modified.
    """
    device = str(torch.device(device))
    if device not in ("cpu", "cuda"):
        raise ValueError("Flow Planner device must be cpu or cuda")
    if (isinstance(cfg_weight, bool) or not isinstance(cfg_weight, Real)
            or not math.isfinite(cfg_weight) or cfg_weight not in (1.0, 1.8)):
        raise ValueError("Flow CFG weight must be the explicit supported value 1.0 or 1.8")
    cfg = OmegaConf.load(config_path)
    if not OmegaConf.is_dict(cfg) or "model" not in cfg or "core" not in cfg:
        raise ValueError("Require the published Flow Planner model/core configuration")
    for name, expected in (
        ("data.dataset.train.future_downsampling_method", "uniform"),
        ("data.dataset.train.predicted_neighbor_num", 0),
    ):
        existing = OmegaConf.select(cfg, name, default=None)
        if existing is not None and existing != expected:
            raise ValueError(f"Unexpected published Flow Planner normalizer setting: {name}")
        OmegaConf.update(cfg, name, expected, merge=False)
    cfg.device = device
    model = OmegaConf.to_container(cfg.model, resolve=True)
    core = OmegaConf.to_container(cfg.core, resolve=True)
    _validate_targets(model)
    _validate_targets(core)
    expected_model = dict(
        _target_=MODEL_TARGET, future_len=80, state_dim=4, neighbor_pred_num=0,
        neighbor_num=32, static_num=5, lane_num=70, cfg_neighbor_num=10,
        cfg_type="neighbors", cfg_weight=1., action_len=20, action_overlap=10,
        model_type="x_start", assemble_method="average",
    )
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("Require the published Flow Planner 80-pose/8-second ego-only model configuration")
    if model.get("kinematic", "waypoints") != "waypoints":
        raise ValueError("Require official Flow Planner rear-axle waypoints")
    ode = model.get("flow_ode", {})
    expected_ode = dict(_target_=ODE_TARGET, sample_steps=4, sample_method="midpoint",
                        sample_temperature=1., cfg_weight=1.)
    if any(ode.get(key) != value for key, value in expected_ode.items()):
        raise ValueError("Require published Flow sampling: 4 midpoint steps, temperature=1, CFG weight=1")
    if (ode.get("path", {}).get("_target_") != "flow_matching.path.affine.AffineProbPath"
            or ode.get("path", {}).get("scheduler", {}).get("_target_")
            != "flow_matching.path.scheduler.CondOTScheduler"
            or core.get("_target_") != CORE_TARGET):
        raise ValueError("Require official conditional-OT flow path and inference core")
    state = model.get("state_normalizer", {})
    processor = model.get("data_processor", {})
    if (state.get("_target_") != "flow_planner.data.normalization.state_normalize.StateNormalizer"
            or state.get("future_downsampling_method") != "uniform"
            or state.get("predicted_neighbor_num") != 0
            or processor.get("state_normalizer") != state
            or processor.get("obs_normalizer") != model.get("obs_normalizer")
            or processor.get("future_len") != 80
            or processor.get("neighbor_pred_num") != 0):
        raise ValueError("Require the published Flow Planner observation/state normalizers")
    ego_stats = state.get("config", {}).get("ego", {}).get("uniform", {})
    if ego_stats != {"mean": [10, 0, 0, 0], "std": [20, 20, 1, 1]}:
        raise ValueError("Unexpected published Flow Planner ego normalization")
    # The official model defaults to CUDA even when the planner device is CPU.
    # Explicitly carry the chosen device into every instantiated component.
    # For CUDA this is exactly the official default; CPU is useful for preflight.
    model["device"] = device
    # Validate the unchanged published source configuration above before an
    # explicit inference override. The official sampler reads its own stored
    # coefficient; passing planner.cfg_weight alone does not change that value.
    model["cfg_weight"] = float(cfg_weight)
    model["flow_ode"]["cfg_weight"] = float(cfg_weight)
    return {"model": model, "core": core, "device": device}


def _checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Require the published bare Flow Planner state dictionary")
    if any(not isinstance(key, str) or not key.startswith("module.")
           or not isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise ValueError("Require published Flow weights containing only module.* tensors, without an EMA wrapper")
    return {key[len("module."):]: value for key, value in checkpoint.items()}


def load_flow_model(config_path, checkpoint_path, device="cpu", *, cfg_weight=1.0):
    """Strictly load published bare weights for CPU preflight or a GPU check."""
    cfg = validated_config(config_path, device, cfg_weight=cfg_weight)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested Flow Planner CUDA inference, but CUDA is unavailable")
    model = instantiate(OmegaConf.create(cfg["model"]))
    model.load_state_dict(_checkpoint_state(checkpoint_path), strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


def _protocol_metadata(cfg):
    ode = cfg["model"]["flow_ode"]
    return {
        "model_target": MODEL_TARGET, "source_sha": FLOW_SOURCE_SHA,
        "native_pose_count": 80, "state_dim": 4, "future_horizon_s": 8.,
        "step_interval_s": .1, "past_pose_count": 20, "past_horizon_s": 2.,
        "checkpoint_format": "published_bare_module_state_dict", "enable_ema": False,
        "use_cfg": True, "cfg_weight": ode["cfg_weight"],
        "source_cfg_weight": 1.0, "effective_cfg_weight": ode["cfg_weight"],
        "sample_steps": ode["sample_steps"], "sample_method": ode["sample_method"],
        "sample_temperature": ode["sample_temperature"],
        "assemble_method": cfg["model"]["assemble_method"],
        "effective_inference_config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "effective_config_device": cfg["device"],
        "normalizer_reference_defaults": {
            "source": "official flow_planner/script/data/dataset/nuplan_data.yaml",
            "future_downsampling_method": "uniform", "predicted_neighbor_num": 0,
        },
        "trajectory_conversion": "nuplan_transform_predictions_to_states_include_ego_state_true",
        "rule_postprocessing": False,
    }


def preflight_model(config_path, checkpoint_path, device="cpu", *, cfg_weight=1.0):
    """Check real checkpoint compatibility without altering native RNG state."""
    with preserve_random_state(device):
        cfg = validated_config(config_path, device, cfg_weight=cfg_weight)
        model = load_flow_model(config_path, checkpoint_path, device, cfg_weight=cfg_weight)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Published Flow checkpoint contains nonfinite weights")
        metadata = _protocol_metadata(cfg)
        metadata["preflight_device"] = str(device)
        # Queue preflight checks weights on CPU, while simulations use CUDA.
        # Record the exact CUDA configuration separately rather than presenting
        # the CPU check configuration hash as the simulation configuration hash.
        metadata["canonical_cuda_effective_config_sha256"] = _protocol_metadata(
            validated_config(config_path, "cuda", cfg_weight=cfg_weight))["effective_inference_config_sha256"]
        metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        del model
    return metadata


class FlowScorePlanner(UrbanDriverScorePlanner):
    """Official stochastic Flow Planner; optionally refine its final ego path."""

    requires_scenario = False
    native_pose_count = 80

    def __init__(self, config_path, checkpoint_path, score_config="unused",
                 score_checkpoint="unused", steps=0, gamma=.1, sigma=.1,
                 heading_projection=True, audit_dir=None, device="cuda", native_device="cuda",
                 evaluation_seed=0, cfg_weight=1.0):
        if native_device not in ("cpu", "cuda") or str(torch.device(device)) not in ("cpu", "cuda"):
            raise ValueError("Native and score devices must be cpu or cuda")
        if type(evaluation_seed) is not int or not 0 <= evaluation_seed < 2 ** 32:
            raise ValueError("evaluation_seed must be an integer in [0, 2**32)")
        cfg = validated_config(config_path, native_device, cfg_weight=cfg_weight)
        super().__init__(cfg, checkpoint_path, score_config, score_checkpoint,
                         steps, gamma, sigma, heading_projection, audit_dir, device)
        self.config_path = str(config_path)
        self.native_device = native_device
        self.evaluation_seed = evaluation_seed
        self.cfg_weight = cfg["model"]["cfg_weight"]

    def name(self):
        return "FlowPlanner_S10G_K20" if self.steps else "FlowPlanner"

    def initialize(self, initialization):
        # nuPlan seeds its driver but not each Ray worker task. Flow samples
        # torch.randn at every frame, so worker history must not decide the
        # baseline/K20 noise sequence. Reset once per scenario, before model
        # construction; subsequent frames advance the original native stream.
        random.seed(self.evaluation_seed)
        np.random.seed(self.evaluation_seed)
        torch.manual_seed(self.evaluation_seed)
        from flow_planner.planner import FlowPlanner

        # Keep a separate immutable copy before official code handles its route.
        self._initialization = SimpleNamespace(
            map_api=initialization.map_api,
            route_roadblock_ids=tuple(initialization.route_roadblock_ids),
        )
        # The published YAML has dangling training interpolations. Pass the
        # resolved inference-only copy to the unmodified official constructor.
        # It reads the file immediately and never reopens it during inference.
        with TemporaryDirectory(prefix="flow_native_config_") as temporary:
            effective_path = Path(temporary) / "model_config.yaml"
            OmegaConf.save(OmegaConf.create(self.model_config), effective_path)
            self._native = FlowPlanner(
                config_path=str(effective_path), ckpt_path=self.checkpoint_path,
                past_trajectory_sampling=TrajectorySampling(num_poses=20, time_horizon=2.),
                future_trajectory_sampling=TrajectorySampling(num_poses=80, time_horizon=8.),
                enable_ema=False, device=self.native_device, use_cfg=True, cfg_weight=self.cfg_weight,
            )
        self._native.initialize(initialization)
        self._audited = False
        if not self.steps:
            return
        if self.device == "cuda" and not torch.cuda.is_available():
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
        self._checkpoint_metadata.update(_protocol_metadata(self.model_config))
        self._checkpoint_metadata.update(
            baseline="flow_planner", planner_name="flow_planner",
            native_device=self.native_device, score_device=self.device,
            evaluation_seed=self.evaluation_seed, seed_policy=SEED_POLICY,
        )
        LOG.info("Flow Planner ready: 80 poses, 4 midpoint steps, CFG=%s; native %s, score %s, K20",
                 self.cfg_weight, self.native_device, self.device)

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input):
        native = self._native.compute_planner_trajectory(current_input)
        if not self.steps:
            return native
        with preserve_random_state(self.device):
            return self.refine_native(native, current_input)
