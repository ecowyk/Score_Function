"""Dataset-free Flow checks, including the real official trajectory conversion."""

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
from omegaconf import OmegaConf
from torch import nn

from flow_planner.planner import FlowPlanner as OfficialFlowPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from flow_score.planner import (
    CORE_TARGET, MODEL_TARGET, ODE_TARGET, FlowScorePlanner, load_flow_model,
    preflight_model, validated_config,
)
from test_urban_driver_score import ConstantScore, sample_inputs
from urban_driver_score.geometry import physical_to_poses, sample_native


def model_config():
    stats = {"ego": {"uniform": {"mean": [10, 0, 0, 0], "std": [20, 20, 1, 1]}}}
    return {
        "device": "cuda", "normalization_stats": stats,
        "model": {
            "_target_": MODEL_TARGET, "future_len": 80, "state_dim": 4,
            "neighbor_pred_num": 0, "neighbor_num": 32, "static_num": 5,
            "lane_num": 70, "cfg_neighbor_num": 10, "cfg_type": "neighbors",
            "cfg_weight": 1., "action_len": 20, "action_overlap": 10,
            "model_type": "x_start", "assemble_method": "average",
            "flow_ode": {
                "_target_": ODE_TARGET, "sample_steps": 4, "sample_method": "midpoint",
                "sample_temperature": 1., "cfg_weight": "${..cfg_weight}",
                "path": {"_target_": "flow_matching.path.affine.AffineProbPath",
                         "scheduler": {"_target_": "flow_matching.path.scheduler.CondOTScheduler"}},
                "time_sampler": {"_target_": "flow_planner.model.model_utils.time_sampler.TimeSampler",
                                 "device": "${device}"},
            },
            "obs_normalizer": {
                "_target_": "flow_planner.data.normalization.obs_normalize.ObservationNormalizer",
                "config": "${normalization_stats}"},
            "state_normalizer": {
                "_target_": "flow_planner.data.normalization.state_normalize.StateNormalizer",
                "config": "${normalization_stats}",
                "future_downsampling_method": "${data.dataset.train.future_downsampling_method}",
                "predicted_neighbor_num": "${data.dataset.train.predicted_neighbor_num}"},
            "data_processor": {
                "_target_": "flow_planner.model.model_utils.input_preprocess.ModelInputProcessor",
                "future_len": "${..future_len}", "neighbor_pred_num": "${..neighbor_pred_num}",
                "obs_normalizer": "${..obs_normalizer}", "state_normalizer": "${..state_normalizer}"},
        },
        "core": {"_target_": CORE_TARGET, "device": "${device}"},
        # These unused training values must never be resolved by inference.
        "recorder": {"save_path": "${oc.env:FLOW_TEST_UNSET_ENV_42}"},
        "save_dir": "${now:%Y-%m-%d}",
    }


def draws():
    return random.random(), np.random.random(), torch.rand(1).item()


def seed():
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)


def native_output():
    poses = np.zeros((80, 4), dtype=np.float64)
    poses[:, 0] = np.arange(1, 81) * .5
    poses[:, 1] = np.arange(1, 81) * .02
    headings = np.arange(1, 81) * .002
    poses[:, 2] = np.cos(headings)
    poses[:, 3] = np.sin(headings)
    return torch.as_tensor(poses)[None, None]


def native_trajectory(current, output=None):
    native = object.__new__(OfficialFlowPlanner)
    native._future_horizon, native._step_interval = 8., .1
    return InterpolatedTrajectory(native.outputs_to_trajectory(
        native_output() if output is None else output, current.history.ego_states))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 3)


class RandomScore(ConstantScore):
    def forward(self, *args):
        draws()
        return super().forward(*args)


class FlowScoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config_path = Path(self.temporary.name) / "model_config.yaml"
        self.write_config(model_config())

    def write_config(self, cfg):
        OmegaConf.save(OmegaConf.create(cfg), self.config_path)

    def planner(self, **kwargs):
        return FlowScorePlanner(self.config_path, "unused", device="cpu", native_device="cpu", **kwargs)

    def test_config_resolves_only_inference_with_official_missing_defaults(self):
        before = self.config_path.read_bytes()
        cfg = validated_config(self.config_path, "cpu")
        self.assertEqual(set(cfg), {"model", "core", "device"})
        self.assertEqual(cfg["model"]["state_normalizer"]["future_downsampling_method"], "uniform")
        self.assertEqual(cfg["model"]["state_normalizer"]["predicted_neighbor_num"], 0)
        self.assertEqual(cfg["model"]["device"], "cpu")
        self.assertEqual(cfg["model"]["flow_ode"]["time_sampler"]["device"], "cpu")
        self.assertEqual(cfg["core"]["device"], "cpu")
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_constructor_is_lazy_and_requires_published_sampling_and_targets(self):
        with patch("flow_score.planner.instantiate", side_effect=AssertionError("driver constructed model")):
            planner = self.planner()
        self.assertIsNone(planner._native)
        for change in (
            lambda c: c["model"].update(future_len=16),
            lambda c: c["model"].update(cfg_weight=1.8),
            lambda c: c["model"]["flow_ode"].update(sample_steps=20),
            lambda c: c["model"]["flow_ode"].update(sample_method="euler"),
            lambda c: c["model"]["obs_normalizer"].update(_target_="os.system"),
            lambda c: c.update(data={"dataset": {"train": {"future_downsampling_method": "log"}}}),
        ):
            cfg = model_config()
            change(cfg)
            self.write_config(cfg)
            with self.assertRaises(ValueError):
                self.planner()

    def test_published_bare_checkpoint_load_is_strict_cpu_and_rng_preserved(self):
        checkpoint = Path(self.temporary.name) / "model.pth"
        saved = {"module." + key: torch.full_like(value, .25)
                 for key, value in DummyModel().state_dict().items()}
        torch.save(saved, checkpoint)
        def instantiate_cpu(cfg):
            self.assertEqual(cfg.device, "cpu")
            self.assertEqual(cfg.flow_ode.time_sampler.device, "cpu")
            return DummyModel()
        with patch("flow_score.planner.instantiate", instantiate_cpu), \
                patch("torch.cuda.is_available", return_value=True):
            loaded = load_flow_model(self.config_path, checkpoint, "cpu")
            self.assertFalse(loaded.training)
            self.assertTrue(all(p.device.type == "cpu" and not p.requires_grad for p in loaded.parameters()))
            for value in loaded.state_dict().values():
                torch.testing.assert_close(value, torch.full_like(value, .25))
            seed()
            expected = draws()
            seed()
            metadata = preflight_model(self.config_path, checkpoint)
            self.assertEqual(draws(), expected)
            self.assertEqual(metadata["native_pose_count"], 80)
            self.assertEqual(metadata["cfg_weight"], 1.)
            self.assertEqual(metadata["sample_steps"], 4)
            self.assertEqual(metadata["sample_method"], "midpoint")
            self.assertFalse(metadata["enable_ema"])
            self.assertEqual(metadata["preflight_device"], "cpu")
            self.assertEqual(metadata["effective_config_device"], "cpu")
            self.assertNotEqual(metadata["effective_inference_config_sha256"],
                                metadata["canonical_cuda_effective_config_sha256"])
            json.dumps(metadata)
            broken = dict(saved)
            broken.pop(next(iter(broken)))
            torch.save(broken, checkpoint)
            with self.assertRaises(RuntimeError):
                load_flow_model(self.config_path, checkpoint)
            torch.save({"ema_state_dict": saved}, checkpoint)
            with self.assertRaises(ValueError):
                load_flow_model(self.config_path, checkpoint)
            torch.save({"layer.weight": torch.zeros(3, 2)}, checkpoint)
            with self.assertRaises(ValueError):
                load_flow_model(self.config_path, checkpoint)

    def test_initialization_uses_official_native_loader_and_preserves_extra_rng(self):
        events = []
        self_test = self
        class Native:
            def __init__(self, **kwargs):
                draws()
                events.append(kwargs)
                effective = OmegaConf.load(kwargs["config_path"])
                self_test.assertEqual(effective.model.device, "cpu")
                self_test.assertEqual(effective.core.device, "cpu")
            def initialize(self, initialization):
                draws()
                self_test.assertIs(initialization, init)
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
        planner = self.planner(steps=20)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        draws()
        draws()
        expected = draws()
        seed()
        with patch("flow_planner.planner.FlowPlanner", Native), \
                patch.dict(sys.modules, {processor.__name__: processor}), \
                patch("urban_driver_score.runtime.load_score", load_score), \
                patch("urban_driver_score.runtime.load_conditions", load_conditions):
            planner.initialize(init)
        self.assertEqual(draws(), expected)
        self.assertFalse(events[0]["enable_ema"])
        self.assertTrue(events[0]["use_cfg"])
        self.assertEqual(events[0]["cfg_weight"], 1.)
        self.assertEqual(events[0]["future_trajectory_sampling"].num_poses, 80)
        self.assertEqual(events[0]["past_trajectory_sampling"].num_poses, 20)
        init.route_roadblock_ids.append("later")
        self.assertEqual(planner._initialization.route_roadblock_ids, ("one", "two"))
        self.assertFalse(planner._checkpoint_metadata["enable_ema"])
        self.assertEqual(planner._checkpoint_metadata["evaluation_seed"], 0)
        self.assertEqual(planner._checkpoint_metadata["seed_policy"],
                         "once_per_scenario_before_native_initialization")

    def test_worker_seed_pairs_k0_k20_independently_of_previous_tasks_and_advances_per_frame(self):
        class Native:
            def __init__(self, **kwargs):
                self.init_draws = [draws()]
            def initialize(self, initialization):
                self.init_draws.append(draws())
            def compute_planner_trajectory(self, _):
                return draws()
        processor = ModuleType("diffusion_planner.data_process.data_processor")
        processor.DataProcessor = lambda args: draws()
        state = {"epoch": 529, "weight_kind": "ema", "sigma_score": .1}
        def load_score(*args):
            draws()
            return {}, state, object(), object()
        def load_conditions(*args):
            draws()
            return object(), object()
        def extra_score_work(native, current):
            for _ in range(20):
                draws()
            return native
        init = SimpleNamespace(map_api=None, route_roadblock_ids=[])
        results = []
        with patch("flow_planner.planner.FlowPlanner", Native), \
                patch.dict(sys.modules, {processor.__name__: processor}), \
                patch("urban_driver_score.runtime.load_score", load_score), \
                patch("urban_driver_score.runtime.load_conditions", load_conditions):
            for steps, prior_seed in ((0, 111), (20, 987), (0, 333)):
                planner = self.planner(steps=steps, evaluation_seed=3402)
                random.seed(prior_seed)
                np.random.seed(prior_seed)
                torch.manual_seed(prior_seed)
                planner.initialize(init)
                with patch.object(planner, "refine_native", extra_score_work):
                    frames = [planner.compute_planner_trajectory(None) for _ in range(3)]
                results.append((planner._native.init_draws, frames, draws()))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        self.assertNotEqual(results[0][1][0], results[0][1][1])
        self.assertNotEqual(results[0][1][1], results[0][1][2])
        for bad in (-1, 2 ** 32, True, 1.5):
            with self.assertRaises(ValueError):
                self.planner(evaluation_seed=bad)

    def test_k0_returns_same_object_and_never_samples_or_refines(self):
        planner = self.planner()
        sentinel = object()
        planner._native = SimpleNamespace(compute_planner_trajectory=lambda _: sentinel)
        with patch.object(planner, "refine_native", side_effect=AssertionError("K0 refined")), \
                patch("urban_driver_score.planner.sample_native", side_effect=AssertionError("K0 sampled")):
            self.assertIs(planner.compute_planner_trajectory(object()), sentinel)

    def test_identical_inputs_match_all_official_state_fields_across_heading_wrap(self):
        current, _, _ = sample_inputs(heading=np.pi - .05)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                output = native_output().to(dtype)
                native = native_trajectory(current, output)
                # Production promotes to float64 before atan2, just like Flow's
                # official outputs_to_trajectory. Compare identical inputs;
                # this checks velocities/accelerations as well as poses/timing.
                poses = physical_to_poses(output[0, 0].numpy())
                rebuilt = self.planner()._trajectory_from_poses(poses, current)
                got_states = rebuilt.get_sampled_trajectory()
                expected_states = native.get_sampled_trajectory()
                self.assertEqual(len(got_states), 81)
                self.assertEqual(len(expected_states), 81)
                self.assertIs(got_states[0], current.history.ego_states[-1])
                for got, expected in zip(got_states, expected_states):
                    self.assertEqual(got.time_us, expected.time_us)
                    np.testing.assert_array_equal(list(got), list(expected))

    def test_native_roundtrip_preserves_80_poses_and_observed_t0_across_heading_wrap(self):
        current, _, _ = sample_inputs(heading=np.pi - .05)
        native = native_trajectory(current)
        physical = sample_native(native, current.history.ego_states[-1])
        np.testing.assert_allclose(physical, native_output()[0, 0].numpy(), rtol=0, atol=4e-6)
        poses = physical_to_poses(physical)
        rebuilt = self.planner()._trajectory_from_poses(poses, current)
        # Sampling for the score intentionally returns float32. At exactly pi,
        # tiny quantization errors can select opposite equivalent heading signs.
        # nuPlan interpolates those signed headings when projecting derivatives,
        # so derived velocities need not survive this lossy roundtrip. Their
        # official conversion is checked with identical inputs above instead.
        for got, expected in zip(rebuilt.get_sampled_trajectory(), native.get_sampled_trajectory()):
            self.assertEqual(got.time_us, expected.time_us)
            np.testing.assert_allclose(
                [got.rear_axle.x, got.rear_axle.y],
                [expected.rear_axle.x, expected.rear_axle.y], rtol=0, atol=4e-6)
            angle_error = got.rear_axle.heading - expected.rear_axle.heading
            self.assertLess(abs(np.arctan2(np.sin(angle_error), np.cos(angle_error))), 2e-7)
        ego = current.history.ego_states[-1]
        states = rebuilt.get_sampled_trajectory()
        self.assertEqual(len(states), 81)
        self.assertIs(states[0], ego)
        self.assertAlmostEqual((states[1].time_us - ego.time_us) / 1e6, .1, places=5)
        self.assertAlmostEqual((states[-1].time_us - ego.time_us) / 1e6, 8., places=5)

    def prepared_refiner(self):
        current, _, _ = sample_inputs()
        native = native_trajectory(current)
        planner = self.planner(steps=20)
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
        return planner, current, native

    def test_twenty_updates_rng_preservation_and_real_conversion_audit(self):
        planner, current, native = self.prepared_refiner()
        seed()
        planner._native.compute_planner_trajectory(current)
        expected_rng = draws()
        seed()
        planner.audit_dir = self.temporary.name
        refined = planner.compute_planner_trajectory(current)
        self.assertEqual(draws(), expected_rng)
        self.assertEqual(planner._score.calls, 20)
        ego = current.history.ego_states[-1]
        before, after = sample_native(native, ego), sample_native(refined, ego)
        np.testing.assert_allclose(after[:, 0] - before[:, 0], .4, atol=3e-5)
        np.testing.assert_allclose(after[:, 1:], before[:, 1:], atol=1e-6)
        self.assertIs(refined.get_sampled_trajectory()[0], ego)
        self.assertEqual(refined.start_time, native.start_time)
        self.assertEqual(refined.end_time, native.end_time)
        audit = json.loads(next(Path(self.temporary.name).glob("*.json")).read_text())
        self.assertEqual(audit["native_pose_count"], 80)
        self.assertEqual(audit["completed_steps"], 20)
        self.assertTrue(audit["finite"])

    def test_rng_is_restored_when_score_fails(self):
        planner, current, _ = self.prepared_refiner()
        def fail(*args):
            draws()
            raise RuntimeError("score failure")
        planner._score.forward = fail
        seed()
        planner._native.compute_planner_trajectory(current)
        expected = draws()
        seed()
        with self.assertRaisesRegex(RuntimeError, "score failure"):
            planner.compute_planner_trajectory(current)
        self.assertEqual(draws(), expected)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
