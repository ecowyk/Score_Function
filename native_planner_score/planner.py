"""Preserve native PDM-Open/GC-PGP inference and refine its final trajectory.

The native device is explicit so the baseline and K20 run can use the same
inference backend. PDM-Open stays on its official CPU backend; GC-PGP supports
its official CUDA backend and a CPU option. The score device is independent.
The frozen refinement and 80-point conversion are shared with UrbanDriver.
"""

import copy
import logging
import random
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader

from urban_driver_score.planner import UrbanDriverScorePlanner

LOG = logging.getLogger(__name__)
PDM_TARGET = "tuplan_garage.planning.simulation.planner.pdm_planner.pdm_open_planner.PDMOpenPlanner"
ML_TARGET = "nuplan.planning.simulation.planner.ml_planner.ml_planner.MLPlanner"
PGP_TARGET = "tuplan_garage.planning.training.modeling.models.pgp.pgp_model.PGPModel"


@contextmanager
def preserve_random_state(device="cpu"):
    """Score loading/inference must not advance the native planner's RNG stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class CPUModelLoader(ModelLoader):
    """nuPlan 1.2.2 loader with the same feature/inference code, fixed to CPU.

    ModelLoader otherwise automatically chooses CUDA whenever it is visible.
    This instance-local override keeps the native model identical to the CPU
    baseline even when the score is explicitly placed on a GPU.
    """

    def _initialize_torch(self):
        torch.set_grad_enabled(False)
        self.device = torch.device("cpu")


class NativePlannerScorePlanner(UrbanDriverScorePlanner):
    requires_scenario = False

    def __init__(self, native_planner_config, score_config, score_checkpoint,
                 steps=20, gamma=.1, sigma=.1, heading_projection=True,
                 audit_dir=None, device="cpu", planner_name=None, native_device="cpu"):
        if native_device not in ("cpu", "cuda"):
            raise ValueError("native_device must be cpu or cuda")
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        native_cfg = OmegaConf.to_container(OmegaConf.create(native_planner_config), resolve=True)
        if not isinstance(native_cfg, dict) or len(native_cfg) != 1:
            raise ValueError("Expected exactly one resolved native planner configuration")
        native = next(iter(native_cfg.values()))
        if native.get("_target_") == PDM_TARGET:
            self.baseline_name = "pdm_open"
            if native_device != "cpu":
                raise ValueError("PDM-Open must keep its official CPU native backend")
            sampling = native["model"]["trajectory_sampling"]
        elif (native.get("_target_") == ML_TARGET
              and isinstance(native.get("model_config"), dict)
              and native["model_config"].get("_target_") == PGP_TARGET):
            self.baseline_name = "gc_pgp"
            sampling = native["model_config"]["future_trajectory_sampling"]
        else:
            raise ValueError("Supported native planners are PDM-Open and GC-PGP only")
        if planner_name is not None and planner_name != self.baseline_name:
            raise ValueError("planner_name does not match the native planner configuration")
        poses = sampling.get("num_poses")
        horizon = sampling.get("time_horizon")
        if horizon is None and poses is not None and "interval_length" in sampling:
            horizon = poses * sampling["interval_length"]
        if poses != 16 or horizon != 8.0:
            raise ValueError("Require the original 16-pose / 8-second native model")
        super().__init__(
            model_config={}, checkpoint_path=native["checkpoint_path"],
            score_config=score_config, score_checkpoint=score_checkpoint,
            steps=steps, gamma=gamma, sigma=sigma,
            heading_projection=heading_projection, audit_dir=audit_dir,
            device=device,
        )
        self.native_planner_config = copy.deepcopy(native_cfg)
        self.native_device = native_device

    def name(self):
        return (f"{self.baseline_name}_S10G_K20" if self.steps
                else f"{self.baseline_name}_K0_passthrough")

    def initialize(self, initialization):
        from nuplan.planning.script.builders.planner_builder import build_planners

        # Construct within each simulation worker, not all 272 models in the
        # launcher. PDMOpen's native constructor already fixes its model to CPU.
        if self.native_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested native CUDA inference, but CUDA is unavailable")
        self._native = build_planners(OmegaConf.create(self.native_planner_config), None)[0]
        if self.baseline_name == "gc_pgp" and self.native_device == "cpu":
            self._native._model_loader = CPUModelLoader(self._native._model_loader._model)
        self._native.initialize(initialization)

        # PDM-Open corrects its own route dictionaries. The score keeps the same
        # original nuPlan route IDs that the existing DP observation adapter uses.
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
            baseline=self.baseline_name, planner_name=self.baseline_name,
            native_device=self.native_device, score_device=self.device,
        )
        LOG.info("%s ready: native %s, S10-G %s, K=20, gamma=0.1, sigma=0.1",
                 self.baseline_name, self.native_device, self.device)

    @torch.no_grad()
    def compute_planner_trajectory(self, current_input):
        native = self._native.compute_planner_trajectory(current_input)
        if not self.steps:
            return native
        # Native GC-PGP's own random seeding and selection run unmodified above.
        # Only the additional score work is isolated from the global RNG streams.
        with preserve_random_state(self.device):
            return self.refine_native(native, current_input)
