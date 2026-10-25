"""Dataset-free PLUTO checks with real nuPlan and official PLUTO conversion."""

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
from torch import nn

from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from src.planners.ml_planner_utils import global_trajectory_to_states

from pluto_score.planner import (
    FEATURE_TARGET, MODEL_TARGET, SCENARIO_RADIUS, PlutoScorePlanner,
    PlutoWithoutRefinementPlanner, load_pluto_model, preflight_model,
    trajectory_from_local_poses,
)
from test_urban_driver_score import ConstantScore, sample_inputs
from urban_driver_score.geometry import sample_native


def model_config():
    return {"_target_": MODEL_TARGET, "future_steps": 80, "history_steps": 21,
            "feature_builder": {"_target_": FEATURE_TARGET, "future_horizon": 8.,
                                "history_horizon": 2., "sample_interval": .1,
                                "build_reference_line": True}}


def draws():
    return random.random(), np.random.random(), torch.rand(1).item()


def seed():
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)


def local_poses():
    poses = np.zeros((80, 3), dtype=np.float64)
    poses[:, 0] = np.arange(1, 81) * .5
    poses[:, 1] = np.arange(1, 81) * .02
    poses[:, 2] = np.arange(1, 81) * .002
    return poses


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 3)
        self.future_steps = 80
        self.builder = SimpleNamespace(future_samples=80, history_samples=20,
                                       sample_interval=.1, build_reference_line=True)

    def get_list_of_required_feature(self):
        return [self.builder]


class RandomScore(ConstantScore):
    def forward(self, *args):
        draws()
        return super().forward(*args)


class PlutoScoreTests(unittest.TestCase):
    def test_constructor_is_lazy_and_rejects_wrong_model_or_missing_reference_lines(self):
        with patch("pluto_score.planner.instantiate", side_effect=AssertionError("constructed in driver")):
            planner = PlutoScorePlanner(model_config(), "unused", steps=0)
        self.assertIsNone(planner._native)
        self.assertFalse(planner.requires_scenario)
        for change in (lambda c: c.update(future_steps=16),
                       lambda c: c.update(_target_="src.OtherModel"),
                       lambda c: c["feature_builder"].update(build_reference_line=False)):
            cfg = model_config()
            change(cfg)
            with self.assertRaises(ValueError):
                PlutoScorePlanner(cfg, "unused")

    def test_checkpoint_load_is_strict_cpu_and_preflight_preserves_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "pluto.ckpt"
            saved = {"model." + name: torch.full_like(value, .25)
                     for name, value in DummyModel().state_dict().items()}
            torch.save({"state_dict": saved}, checkpoint)
            with patch("pluto_score.planner.instantiate", side_effect=lambda _: DummyModel()), \
                    patch("torch.cuda.is_available", return_value=True):
                loaded = load_pluto_model(model_config(), checkpoint, "cpu")
                self.assertFalse(loaded.training)
                self.assertTrue(all(p.device.type == "cpu" for p in loaded.parameters()))
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                for value in loaded.state_dict().values():
                    torch.testing.assert_close(value, torch.full_like(value, .25))
                seed()
                expected = draws()
                seed()
                metadata = preflight_model(model_config(), checkpoint)
                self.assertEqual(draws(), expected)
                self.assertEqual(metadata["native_pose_count"], 80)
                self.assertFalse(metadata["rule_postprocessing"])
                broken = dict(saved)
                broken.pop(next(iter(broken)))
                torch.save({"state_dict": broken}, checkpoint)
                with self.assertRaises(RuntimeError):
                    load_pluto_model(model_config(), checkpoint, "cpu")

    def test_native_reuses_selected_output_and_original_feature_sequence(self):
        current, _, _ = sample_inputs()
        poses = local_poses()
        events = []
        batch_data = object()
        class Feature:
            def to_feature_tensor(self):
                events.append("tensor")
                return self
            def collate(self, values):
                self_test.assertEqual(values, [self])
                events.append("collate")
                return self
            def to_device(self, device):
                self_test.assertEqual(str(device), "cpu")
                events.append("device")
                return SimpleNamespace(data=batch_data)
        self_test = self
        feature = Feature()
        def build_feature(inp, init):
            self.assertIs(inp, current)
            self.assertEqual(init.route_roadblock_ids, ["original_route"])
            events.append("feature")
            return feature
        def forward(data):
            self.assertIs(data, batch_data)
            self.assertFalse(torch.is_grad_enabled())
            events.append("forward")
            return {"output_trajectory": torch.as_tensor(poses).unsqueeze(0),
                    # These deliberately unusable fields must never be rescored,
                    # simulated or substituted for the model-selected output.
                    "candidate_trajectories": object(), "probability": object(),
                    "output_prediction": object()}
        planner = PlutoWithoutRefinementPlanner(model_config(), "unused", "cpu")
        planner._scenario_manager = SimpleNamespace(
            update_ego_state=lambda ego: events.append("ego"),
            update_drivable_area_map=lambda: events.append("map"))
        planner._feature_builder = SimpleNamespace(get_features_from_simulation=build_feature)
        planner._model = SimpleNamespace(forward=forward)
        planner._initialization = SimpleNamespace(route_roadblock_ids=["original_route"])
        trajectory = planner.compute_planner_trajectory(current)
        self.assertEqual(events, ["ego", "map", "feature", "tensor", "collate", "device", "forward"])
        physical = sample_native(trajectory, current.history.ego_states[-1])
        np.testing.assert_allclose(physical[:, :2], poses[:, :2], atol=4e-6)
        np.testing.assert_allclose(physical[:, 2:], np.column_stack((
            np.cos(poses[:, 2]), np.sin(poses[:, 2]))), atol=1e-7)

    def test_native_initialization_preserves_official_manager_radius_and_reference_builder(self):
        created = []
        module = ModuleType("src.scenario_manager.scenario_manager")
        module.ScenarioManager = lambda **kwargs: created.append(kwargs) or object()
        model = DummyModel()
        init = SimpleNamespace(map_api=object(), route_roadblock_ids=["one", "two"])
        planner = PlutoWithoutRefinementPlanner(model_config(), "unused", "cpu")
        with patch.dict(sys.modules, {module.__name__: module}), \
                patch("pluto_score.planner.load_pluto_model", return_value=model):
            planner.initialize(init)
        self.assertEqual(created[0]["radius"], SCENARIO_RADIUS)
        self.assertEqual(created[0]["radius"], 60.)
        self.assertIs(created[0]["map_api"], init.map_api)
        self.assertIs(created[0]["route_roadblocks_ids"], init.route_roadblock_ids)
        self.assertIs(planner._feature_builder.scenario_manager, planner._scenario_manager)

    def test_conversion_matches_official_helper_and_80_future_timestamps(self):
        current, _, _ = sample_inputs(heading=np.pi - .05)
        ego = current.history.ego_states[-1]
        poses = local_poses()
        angle = ego.rear_axle.heading
        rotation = np.array([[np.cos(angle), np.sin(angle)],
                             [-np.sin(angle), np.cos(angle)]])
        expected_global = poses.copy()
        expected_global[:, :2] = poses[:, :2] @ rotation + ego.rear_axle.array
        expected_global[:, 2] += angle
        expected = InterpolatedTrajectory(global_trajectory_to_states(
            expected_global, current.history.ego_states, 8., .1, include_ego_state=False))
        actual = trajectory_from_local_poses(poses, current)
        for got, wanted in zip(actual.get_sampled_trajectory(), expected.get_sampled_trajectory()):
            np.testing.assert_allclose(list(got), list(wanted))
        states = actual.get_sampled_trajectory()
        self.assertEqual(len(states), 81)
        self.assertEqual(states[0].time_point, ego.time_point)
        self.assertIsNot(states[0], ego)
        np.testing.assert_allclose(states[0].rear_axle.serialize(), states[1].rear_axle.serialize())
        self.assertAlmostEqual((states[1].time_us - ego.time_us) / 1e6, .1, places=5)
        self.assertAlmostEqual((states[-1].time_us - ego.time_us) / 1e6, 8., places=5)

    def test_k0_returns_identical_native_object_and_never_runs_score(self):
        planner = PlutoScorePlanner(model_config(), "unused", steps=0)
        sentinel = object()
        planner._native = SimpleNamespace(compute_planner_trajectory=lambda _: sentinel)
        with patch.object(planner, "refine_native", side_effect=AssertionError("K0 refined")):
            self.assertIs(planner.compute_planner_trajectory(object()), sentinel)

    def test_score_initialization_preserves_native_rng_and_original_route(self):
        def native_initialize(instance, initialization):
            draws()
        processor = ModuleType("diffusion_planner.data_process.data_processor")
        processor.DataProcessor = lambda args: draws()
        state = {"epoch": 529, "weight_kind": "ema", "sigma_score": .1}
        def load_score(*args):
            draws()
            return {}, state, object(), object()
        def load_conditions(*args):
            draws()
            return object(), object()
        init = SimpleNamespace(map_api=object(), route_roadblock_ids=["one", "two"])
        planner = PlutoScorePlanner(model_config(), "unused", steps=20,
                                    device="cpu", native_device="cpu")
        seed()
        draws()
        expected = draws()
        seed()
        with patch.object(PlutoWithoutRefinementPlanner, "initialize", native_initialize), \
                patch.dict(sys.modules, {processor.__name__: processor}), \
                patch("urban_driver_score.runtime.load_score", load_score), \
                patch("urban_driver_score.runtime.load_conditions", load_conditions):
            planner.initialize(init)
        self.assertEqual(draws(), expected)
        init.route_roadblock_ids.append("later")
        self.assertEqual(planner._initialization.route_roadblock_ids, ("one", "two"))
        self.assertFalse(planner._checkpoint_metadata["rule_postprocessing"])

    def test_twenty_updates_preserve_rng_and_same_official_t0_policy(self):
        current, _, _ = sample_inputs()
        native = trajectory_from_local_poses(local_poses(), current)
        planner = PlutoScorePlanner(model_config(), "unused", steps=20,
                                    device="cpu", native_device="cpu")
        def native_forward(_):
            draws()
            return native
        planner._native = SimpleNamespace(compute_planner_trajectory=native_forward)
        planner._score = RandomScore().eval()
        planner._args = SimpleNamespace(
            state_normalizer=SimpleNamespace(mean=torch.tensor([[[5., -2., .1, -.2]]]),
                                             std=torch.tensor([[[20., 10., .3, .4]]])),
            observation_normalizer=lambda inputs: inputs)
        planner._processor = SimpleNamespace(observation_adapter=lambda *args: {"route_lanes": torch.zeros(1)})
        planner._encoder = lambda inputs: {"encoding": torch.zeros(1, 107, 192)}
        planner._route_encoder = lambda route: torch.zeros(1, 192)
        planner._initialization = SimpleNamespace(map_api=None, route_roadblock_ids=[])
        planner._checkpoint_metadata = dict(epoch=529, weight_kind="ema", sigma_score=.1)
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
            states = refined.get_sampled_trajectory()
            np.testing.assert_allclose(states[0].rear_axle.serialize(), states[1].rear_axle.serialize())
            self.assertIsNot(states[0], current.history.ego_states[-1])
            self.assertEqual(refined.start_time, native.start_time)
            self.assertEqual(refined.end_time, native.end_time)
            audit = json.loads(next(Path(temporary).glob("*.json")).read_text())
            self.assertEqual(audit["native_pose_count"], 80)
            self.assertEqual(audit["completed_steps"], 20)
            self.assertTrue(audit["finite"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
