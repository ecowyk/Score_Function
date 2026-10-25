"""Dataset-free CPU tests; run with the real PlanTF and nuPlan on PYTHONPATH.

The model/score weights and maps are not needed. Geometry and state conversion
use the installed official PlanTF planner and real nuPlan EgoState objects.
"""

import json
import random
import sys
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from src.planners.imitation_planner import ImitationPlanner

from plantf_score.planner import (
    FEATURE_TARGET, MODEL_TARGET, PLANNER_TARGET, PlanTFScorePlanner,
    native_knot_features, trajectory_from_local_poses,
)
from urban_driver_score.geometry import physical_to_poses


def native_config():
    return {"imitation_planner": {
        "_target_": PLANNER_TARGET, "planner_ckpt": "test-only.ckpt",
        "replan_interval": 1, "use_gpu": False,
        "planner": {
            "_target_": MODEL_TARGET, "future_steps": 80, "history_steps": 21,
            "feature_builder": {
                "_target_": FEATURE_TARGET, "future_horizon": 8.,
                "history_horizon": 2., "sample_interval": .1,
            },
        },
    }}


def sample_inputs(heading=np.pi / 2):
    # A distant rotated rear axle detects global/local and center/axle errors.
    history = deque(EgoState.build_from_rear_axle(
        StateSE2(500000 + .5 * i * np.cos(heading),
                 400000 + .5 * i * np.sin(heading), heading),
        StateVector2D(5., 0.), StateVector2D(0., 0.), 0.,
        TimePoint(1600000000000000 + 100000 * i), get_pacifica_parameters(),
    ) for i in range(-20, 1))
    current = SimpleNamespace(history=SimpleNamespace(ego_states=history), traffic_light_data=[])
    poses = np.zeros((80, 3), dtype=np.float64)
    poses[:, 0] = np.arange(1, 81) * .5
    # The official model's wrapper sampling is deliberately a dummy 8/1s.
    model = SimpleNamespace(future_steps=80,
                            future_trajectory_sampling=SimpleNamespace(num_poses=8,
                                                                       time_horizon=8.,
                                                                       interval_length=1.),
                            get_list_of_required_feature=lambda: [None])
    native = ImitationPlanner(model, planner_ckpt=None, use_gpu=False)

    def planning(_):
        native._start_time = time.perf_counter()
        return poses.copy()

    native._planning = planning
    trajectory = native.compute_planner_trajectory(current)
    return current, native, trajectory, poses


def draw():
    return np.asarray([random.random(), np.random.random(), torch.rand(()).item()])


def seed():
    random.seed(1987)
    np.random.seed(1987)
    torch.manual_seed(1987)


class ConstantScore(nn.Module):
    def __init__(self, consume_rng=False, fail_at=None):
        super().__init__()
        self.calls = 0
        self.consume_rng, self.fail_at = consume_rng, fail_at

    def forward(self, trajectory, scene, route):
        self.calls += 1
        if self.consume_rng:
            draw()
        if self.calls == self.fail_at:
            raise RuntimeError("deliberate score failure")
        score = torch.zeros_like(trajectory)
        score[..., 0] = 1.
        return score


def attach_score(planner, native, score):
    planner._native = native
    planner._score = score.eval()
    planner._args = SimpleNamespace(
        state_normalizer=SimpleNamespace(mean=torch.tensor([[[5., -2., .1, -.2]]]),
                                         std=torch.tensor([[[20., 10., .3, .4]]])),
        observation_normalizer=lambda inputs: inputs,
    )
    planner._processor = SimpleNamespace(
        observation_adapter=lambda *args: {"route_lanes": torch.zeros(1)})
    planner._encoder = lambda inputs: {"encoding": torch.zeros(1, 107, 192)}
    planner._route_encoder = lambda route: torch.zeros(1, 192)
    planner._initialization = SimpleNamespace(map_api=None, route_roadblock_ids=("a", "b"))
    planner._checkpoint_metadata = dict(epoch=529, weight_kind="ema", sigma_score=.1)


class PlanTFScoreTests(unittest.TestCase):
    def test_constructor_is_lazy_and_rejects_changed_native_contract(self):
        with patch("plantf_score.planner.instantiate", side_effect=AssertionError("early native construction")), \
                patch("plantf_score.planner.torch.cuda.is_available", side_effect=AssertionError("early CUDA probe")):
            planner = PlanTFScorePlanner(native_config(), device="cpu")
            self.assertIsNone(planner._native)
        cfg = native_config()
        cfg["imitation_planner"]["planner"]["future_steps"] = 16
        with self.assertRaises(ValueError):
            PlanTFScorePlanner(cfg, device="cpu")
        cfg = native_config()
        cfg["imitation_planner"]["replan_interval"] = 2
        with self.assertRaises(ValueError):
            PlanTFScorePlanner(cfg, device="cpu")

    def test_k0_returns_exact_native_object_without_score_or_conversion(self):
        planner = PlanTFScorePlanner(native_config(), steps=0, device="cpu")
        sentinel = object()
        planner._native = SimpleNamespace(compute_planner_trajectory=lambda _: sentinel)
        with patch("plantf_score.planner.native_knot_features", side_effect=AssertionError("sampled K0")), \
                patch("plantf_score.planner.refine_ego", side_effect=AssertionError("refined K0")), \
                patch("plantf_score.planner.trajectory_from_local_poses", side_effect=AssertionError("converted K0")):
            self.assertIs(planner.compute_planner_trajectory(object()), sentinel)
            self.assertIs(planner.refine_native(sentinel, object()), sentinel)

    def test_exact_80_native_knots_and_official_conversion_round_trip(self):
        current, native, trajectory, poses = sample_inputs()
        ego = current.history.ego_states[-1]
        with patch.object(trajectory, "get_state_at_time", side_effect=AssertionError("interpolated a native knot")):
            values, audit = native_knot_features(trajectory, ego)
        np.testing.assert_allclose(physical_to_poses(values), poses, atol=1e-6)
        rebuilt = trajectory_from_local_poses(physical_to_poses(values), current, native)
        expected, actual = trajectory.get_sampled_trajectory(), rebuilt.get_sampled_trajectory()
        self.assertEqual(len(actual), 81)
        self.assertIs(actual[0], ego)
        self.assertEqual([s.time_us for s in actual], [s.time_us for s in expected])
        np.testing.assert_allclose([s.rear_axle.serialize() for s in actual],
                                   [s.rear_axle.serialize() for s in expected], atol=1e-6)
        self.assertEqual(audit["native_knot_sampling"], "exact_sampled_states_no_interpolation")
        self.assertLessEqual(audit["native_max_timestamp_error_us"], 2)

    def test_timestamp_rounding_tolerated_but_wrong_grid_rejected(self):
        current, _, trajectory, _ = sample_inputs()
        ego = current.history.ego_states[-1]
        states = [ego] + [SimpleNamespace(rear_axle=s.rear_axle,
                           time_point=TimePoint(ego.time_us + i * 100000 - 1))
                          for i, s in enumerate(trajectory.get_sampled_trajectory()[1:], 1)]
        fake = SimpleNamespace(get_sampled_trajectory=lambda: states,
                               start_time=states[0].time_point, end_time=states[-1].time_point)
        _, audit = native_knot_features(fake, ego)
        self.assertEqual(audit["native_max_timestamp_error_us"], 1)
        states[30].time_point = TimePoint(ego.time_us + 3000000 + 1000)
        with self.assertRaises(ValueError):
            native_knot_features(fake, ego)
        with self.assertRaises(ValueError):
            native_knot_features(SimpleNamespace(get_sampled_trajectory=lambda: states[1:]), ego)

    def test_heading_wrap_preserves_local_short_arc(self):
        current, native, _, poses = sample_inputs(heading=np.pi - .05)
        poses[:, 2] = np.arange(1, 81) * .004
        trajectory = trajectory_from_local_poses(poses, current, native)
        sampled, _ = native_knot_features(trajectory, current.history.ego_states[-1])
        np.testing.assert_allclose(physical_to_poses(sampled), poses, atol=1e-6)

    def test_exact_twenty_updates_use_official_conversion_and_audit(self):
        current, native, trajectory, _ = sample_inputs()
        planner = PlanTFScorePlanner(native_config(), device="cpu")
        score = ConstantScore()
        attach_score(planner, native, score)
        with tempfile.TemporaryDirectory() as directory:
            planner.audit_dir = directory
            refined = planner.compute_planner_trajectory(current)
            ego = current.history.ego_states[-1]
            before, _ = native_knot_features(trajectory, ego)
            after, _ = native_knot_features(refined, ego)
            self.assertEqual(score.calls, 20)
            # K * gamma * sigma^2 * physical_x_std = 20 * .1 * .01 * 20.
            np.testing.assert_allclose(after[:, 0] - before[:, 0], .4, atol=3e-5)
            np.testing.assert_allclose(after[:, 1:], before[:, 1:], atol=1e-6)
            self.assertIs(refined.get_sampled_trajectory()[0], ego)
            self.assertEqual(refined.start_time, trajectory.start_time)
            self.assertEqual(refined.end_time, trajectory.end_time)
            audit = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertEqual(audit["completed_steps"], 20)
            self.assertEqual(audit["native_pose_count"], 80)
            self.assertTrue(audit["native_t0_is_observed_ego"])
            self.assertTrue(audit["refined_t0_is_observed_ego"])
            self.assertTrue(audit["finite"])
            planner.compute_planner_trajectory(current)
            self.assertEqual(score.calls, 40)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)

    def test_extra_rng_preserved_on_success_and_failure_native_draw_kept(self):
        for fail_at in (None, 3):
            with self.subTest(fail_at=fail_at):
                current, native, trajectory, _ = sample_inputs()
                planner = PlanTFScorePlanner(native_config(), device="cpu")
                attach_score(planner, native, ConstantScore(consume_rng=True, fail_at=fail_at))

                def native_forward(_):
                    draw()
                    return trajectory

                native.compute_planner_trajectory = native_forward
                seed()
                draw()  # This one native draw must remain consumed.
                expected = draw()
                seed()
                if fail_at is None:
                    planner.compute_planner_trajectory(current)
                else:
                    with self.assertRaisesRegex(RuntimeError, "deliberate score failure"):
                        planner.compute_planner_trajectory(current)
                    self.assertFalse(planner._audited)
                np.testing.assert_array_equal(draw(), expected)

    def test_lazy_initialization_keeps_original_route_and_only_native_rng_draws(self):
        planner = PlanTFScorePlanner(native_config(), device="cpu")
        native = SimpleNamespace(_future_horizon=8., _step_interval=.1, _replan_interval=1,
                                  _planner=SimpleNamespace(future_steps=80), device="cpu")
        initialization = SimpleNamespace(map_api=object(), route_roadblock_ids=["a", "b"])

        def native_init(init):
            draw()
            init.route_roadblock_ids.append("native-correction")

        native.initialize = native_init

        def construct_native(cfg):
            draw()
            self.assertEqual(cfg["planner_ckpt"], "test-only.ckpt")
            return native

        def load_score(*args):
            draw()
            return {}, dict(epoch=529, weight_kind="ema", sigma_score=.1), None, None

        def load_conditions(*args):
            draw()
            return None, None

        def processor(*args):
            draw()
            return object()

        runtime = ModuleType("urban_driver_score.runtime")
        runtime.load_score, runtime.load_conditions = load_score, load_conditions
        processor_module = ModuleType("diffusion_planner.data_process.data_processor")
        processor_module.DataProcessor = processor
        modules = {"urban_driver_score.runtime": runtime,
                   "diffusion_planner": ModuleType("diffusion_planner"),
                   "diffusion_planner.data_process": ModuleType("diffusion_planner.data_process"),
                   "diffusion_planner.data_process.data_processor": processor_module}
        seed()
        draw()
        draw()
        expected = draw()
        seed()
        with patch.dict(sys.modules, modules), patch("plantf_score.planner.instantiate", construct_native):
            planner.initialize(initialization)
        np.testing.assert_array_equal(draw(), expected)
        self.assertEqual(planner._initialization.route_roadblock_ids, ("a", "b"))
        self.assertEqual(initialization.route_roadblock_ids, ["a", "b", "native-correction"])
        self.assertEqual(planner._checkpoint_metadata["epoch"], 529)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
