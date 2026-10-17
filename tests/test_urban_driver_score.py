"""Small, dataset-free checks run in the actual UrbanDriver environment."""

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from urban_driver_score.geometry import physical_to_poses, sample_native
from urban_driver_score.planner import UrbanDriverScorePlanner
from urban_driver_score.runtime import load_cpu


class ConstantScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, trajectory, scene, route):
        self.calls += 1
        score = torch.zeros_like(trajectory)
        score[..., 0] = 1.
        return score


def sample_inputs(heading=np.pi / 2):
    # A far-from-origin, rotated rear axle exposes center/global/local mistakes.
    history = deque(EgoState.build_from_rear_axle(
        StateSE2(500000 + .5 * i * np.cos(heading), 400000 + .5 * i * np.sin(heading), heading),
        StateVector2D(5., 0.), StateVector2D(0., 0.), 0.,
        TimePoint(1600000000000000 + 100000 * i), get_pacifica_parameters(),
    ) for i in range(-20, 1))
    current = SimpleNamespace(history=SimpleNamespace(ego_states=history), traffic_light_data=[])
    poses = np.zeros((16, 3), dtype=np.float64)
    poses[:, 0] = np.arange(1, 17) * 2.5
    native = InterpolatedTrajectory(transform_predictions_to_states(poses, history, 8., .5))
    return current, native, poses


class UrbanDriverScoreTests(unittest.TestCase):
    def test_k0_returns_exact_native_object_without_sampling_or_score(self):
        planner = UrbanDriverScorePlanner({}, "unused", "unused", "unused", steps=0)
        sentinel = object()
        planner._native = SimpleNamespace(compute_planner_trajectory=lambda _: sentinel)
        with patch("urban_driver_score.planner.sample_native", side_effect=AssertionError("sampled K0")):
            self.assertIs(planner.compute_planner_trajectory(object()), sentinel)

    def test_16_to_80_timing_frame_and_original_knots(self):
        current, native, poses = sample_inputs()
        dense = sample_native(native, current.history.ego_states[-1])
        np.testing.assert_allclose(dense[:, 0], np.arange(1, 81) * .5, atol=1e-5)
        np.testing.assert_allclose(dense[:, 1], 0., atol=1e-7)
        np.testing.assert_allclose(dense[:, 2:], np.tile([1., 0.], (80, 1)), atol=1e-7)
        np.testing.assert_allclose(physical_to_poses(dense)[4::5], poses, atol=1e-6)

    def test_native_heading_wrap_is_short_arc(self):
        current, _, poses = sample_inputs(heading=np.pi - .05)
        poses[:, 2] = np.arange(1, 17) * .02
        native = InterpolatedTrajectory(transform_predictions_to_states(
            poses, current.history.ego_states, 8., .5))
        sampled = physical_to_poses(sample_native(native, current.history.ego_states[-1]))
        np.testing.assert_allclose(sampled[:, 2], np.arange(1, 81) * .004, atol=2e-7)

    def test_exact_twenty_step_update_with_real_nuplan_conversion(self):
        current, native, _ = sample_inputs()
        branch = ConstantScore().eval()
        planner = UrbanDriverScorePlanner({}, "unused", "unused", "unused", device="cpu")
        planner._native = SimpleNamespace(compute_planner_trajectory=lambda _: native)
        planner._score = branch
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
        with tempfile.TemporaryDirectory() as directory:
            planner.audit_dir = directory
            refined = planner.compute_planner_trajectory(current)
            before = sample_native(native, current.history.ego_states[-1])
            after = sample_native(refined, current.history.ego_states[-1])
            self.assertEqual(branch.calls, 20)
            # x changes by K * gamma * sigma^2 * physical_x_std = 0.4m.
            np.testing.assert_allclose(after[:, 0] - before[:, 0], .4, atol=3e-5)
            np.testing.assert_allclose(after[:, 1:], before[:, 1:], atol=1e-6)
            self.assertEqual(refined.start_time, native.start_time)
            self.assertEqual(refined.end_time, native.end_time)
            self.assertIs(refined.get_sampled_trajectory()[0], current.history.ego_states[-1])
            audit = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertEqual(audit["completed_steps"], 20)
            self.assertTrue(audit["finite"])

    def test_old_torch_loader_has_no_unsupported_weights_only_argument(self):
        def legacy_load(path, map_location=None, **kwargs):
            self.assertEqual(map_location, "cpu")
            self.assertEqual(kwargs, {})
            return {"ok": True}
        with patch("urban_driver_score.runtime.torch.load", legacy_load):
            self.assertEqual(load_cpu("unused"), {"ok": True})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
