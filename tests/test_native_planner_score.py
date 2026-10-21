"""Dataset-free PDM-Open/GC-PGP adapter checks with real nuPlan trajectories."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from native_planner_score.planner import (
    CPUModelLoader, ML_TARGET, PDM_TARGET, PGP_TARGET,
    NativePlannerScorePlanner, preserve_random_state,
)
from test_urban_driver_score import ConstantScore, sample_inputs
from urban_driver_score.geometry import sample_native


def native_config(name):
    if name == "pdm_open":
        return {"pdm_open_planner": {
            "_target_": PDM_TARGET, "checkpoint_path": "native.ckpt",
            "model": {"trajectory_sampling": {"num_poses": 16, "interval_length": .5}},
        }}
    return {"ml_planner": {
        "_target_": ML_TARGET, "checkpoint_path": "native.ckpt",
        "model_config": {
            "_target_": PGP_TARGET,
            "future_trajectory_sampling": {"num_poses": 16, "time_horizon": 8.},
        },
    }}


def draws():
    return random.random(), np.random.random(), torch.rand(1).item()


def seed():
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)


class RandomConsumingScore(ConstantScore):
    def forward(self, *args):
        draws()
        return super().forward(*args)


class NativePlannerScoreTests(unittest.TestCase):
    def test_only_native_16_pose_8_second_models_are_accepted(self):
        for name in ("pdm_open", "gc_pgp"):
            planner = NativePlannerScorePlanner(
                native_config(name), "score.json", "score.pt", planner_name=name)
            self.assertIsNone(planner._native)
            self.assertEqual(planner.name(), name + "_S10G_K20")
        wrong = native_config("pdm_open")
        wrong["pdm_open_planner"]["model"]["trajectory_sampling"]["num_poses"] = 80
        with self.assertRaisesRegex(ValueError, "16-pose"):
            NativePlannerScorePlanner(wrong, "score.json", "score.pt")
        with self.assertRaisesRegex(ValueError, "planner_name"):
            NativePlannerScorePlanner(native_config("pdm_open"), "x", "y", planner_name="gc_pgp")

    def test_k0_is_the_same_native_object_and_does_no_score_work(self):
        for name in ("pdm_open", "gc_pgp"):
            planner = NativePlannerScorePlanner(native_config(name), "unused", "unused", steps=0)
            sentinel = object()
            planner._native = SimpleNamespace(compute_planner_trajectory=lambda _: sentinel)
            with patch.object(planner, "refine_native", side_effect=AssertionError("refined K0")):
                self.assertIs(planner.compute_planner_trajectory(object()), sentinel)

    def test_cpu_native_loader_does_not_follow_visible_cuda(self):
        model = SimpleNamespace(get_list_of_required_feature=lambda: [], eval=lambda: None)
        moved = []
        model.to = lambda device: moved.append(str(device)) or model
        loader = CPUModelLoader(model)
        with torch.enable_grad(), patch("torch.cuda.is_available", return_value=True):
            loader.initialize()
        self.assertEqual(moved, ["cpu"])
        self.assertEqual(str(loader.device), "cpu")
        self.assertTrue(loader._initialized)

    def test_gc_native_cuda_keeps_official_loader_and_cpu_is_explicit(self):
        module = ModuleType("nuplan.planning.script.builders.planner_builder")
        init = SimpleNamespace(map_api=None, route_roadblock_ids=[])
        for device in ("cpu", "cuda"):
            model = SimpleNamespace(get_list_of_required_feature=lambda: [])
            original_loader = SimpleNamespace(_model=model)
            native = SimpleNamespace(_model_loader=original_loader, initialize=lambda _: None)
            module.build_planners = lambda *args: [native]
            planner = NativePlannerScorePlanner(
                native_config("gc_pgp"), "unused", "unused", steps=0,
                native_device=device, device="cuda")
            with patch.dict(sys.modules, {module.__name__: module}), patch(
                "torch.cuda.is_available", return_value=True,
            ):
                planner.initialize(init)
            if device == "cuda":
                self.assertIs(planner._native._model_loader, original_loader)
            else:
                self.assertIsInstance(planner._native._model_loader, CPUModelLoader)
        with self.assertRaisesRegex(ValueError, "official CPU"):
            NativePlannerScorePlanner(
                native_config("pdm_open"), "unused", "unused", steps=0,
                native_device="cuda")

    def test_initialization_preserves_rng_and_uses_original_route(self):
        # Only native construction/initialization may advance the native stream.
        module = ModuleType("nuplan.planning.script.builders.planner_builder")
        def build_native(*args):
            draws()
            return [SimpleNamespace(initialize=lambda _: draws())]
        module.build_planners = build_native
        processor = ModuleType("diffusion_planner.data_process.data_processor")
        processor.DataProcessor = lambda args: draws()
        init = SimpleNamespace(map_api=object(), route_roadblock_ids=["one", "two"])
        state = {"epoch": 529, "weight_kind": "ema", "sigma_score": .1}
        def load_score(*args):
            draws()
            return {}, state, object(), object()
        def load_conditions(*args):
            draws()
            return object(), object()
        planner = NativePlannerScorePlanner(native_config("pdm_open"), "x", "y")
        seed()
        build_native()[0].initialize(init)
        expected = draws()
        seed()
        with patch.dict(sys.modules, {
            module.__name__: module, processor.__name__: processor,
        }), patch("urban_driver_score.runtime.load_score", load_score), patch(
            "urban_driver_score.runtime.load_conditions", load_conditions,
        ):
            planner.initialize(init)
        self.assertEqual(draws(), expected)
        init.route_roadblock_ids.append("later")
        self.assertEqual(planner._initialization.route_roadblock_ids, ("one", "two"))
        self.assertEqual(planner._checkpoint_metadata["epoch"], 529)

    def test_twenty_updates_real_conversion_audit_and_rng_preservation(self):
        for name in ("pdm_open", "gc_pgp"):
            with self.subTest(planner=name):
                current, native, _ = sample_inputs()
                planner = NativePlannerScorePlanner(native_config(name), "x", "y")
                def native_forward(_):
                    draws()
                    return native
                planner._native = SimpleNamespace(compute_planner_trajectory=native_forward)
                planner._score = RandomConsumingScore().eval()
                planner._args = SimpleNamespace(
                    state_normalizer=SimpleNamespace(
                        mean=torch.tensor([[[5., -2., .1, -.2]]]),
                        std=torch.tensor([[[20., 10., .3, .4]]])),
                    observation_normalizer=lambda inputs: inputs,
                )
                planner._processor = SimpleNamespace(observation_adapter=lambda *args: {
                    "route_lanes": torch.zeros(1)})
                planner._encoder = lambda inputs: {"encoding": torch.zeros(1, 107, 192)}
                planner._route_encoder = lambda route: torch.zeros(1, 192)
                planner._initialization = SimpleNamespace(map_api=None, route_roadblock_ids=[])
                planner._checkpoint_metadata = dict(
                    epoch=529, weight_kind="ema", sigma_score=.1,
                    planner_name=name, native_device="cpu", score_device="cpu")
                planner._audited = False
                seed()
                native_forward(current)
                expected_rng = draws()
                seed()
                with tempfile.TemporaryDirectory() as temporary:
                    planner.audit_dir = temporary
                    refined = planner.compute_planner_trajectory(current)
                    self.assertEqual(draws(), expected_rng)
                    self.assertEqual(planner._score.calls, 20)
                    before = sample_native(native, current.history.ego_states[-1])
                    after = sample_native(refined, current.history.ego_states[-1])
                    np.testing.assert_allclose(after[:, 0] - before[:, 0], .4, atol=3e-5)
                    np.testing.assert_allclose(after[:, 1:], before[:, 1:], atol=1e-6)
                    self.assertIs(refined.get_sampled_trajectory()[0], current.history.ego_states[-1])
                    audit_files = list(Path(temporary).glob("*.json"))
                    self.assertEqual(len(audit_files), 1)
                    audit = json.loads(audit_files[0].read_text())
                    self.assertEqual(audit["completed_steps"], 20)
                    self.assertEqual(audit["planner_name"], name)
                    self.assertTrue(audit["finite"])

    def test_rng_is_restored_on_refinement_error(self):
        seed()
        expected = draws()
        seed()
        with self.assertRaisesRegex(RuntimeError, "score failure"):
            with preserve_random_state():
                draws()
                raise RuntimeError("score failure")
        self.assertEqual(draws(), expected)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
