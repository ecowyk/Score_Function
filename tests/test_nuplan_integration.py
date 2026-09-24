"""Official nuPlan lifecycle and trajectory conversion; no dataset simulation."""

import json
import os
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from test_official_boundary import feature_record, official_arguments

OFFICIAL_ROOT = os.environ.get("SCORE_FUNCTION_OFFICIAL_ROOT")


@unittest.skipUnless(OFFICIAL_ROOT, "Set SCORE_FUNCTION_OFFICIAL_ROOT")
class NuPlanIntegrationTests(unittest.TestCase):
    def test_official_lifecycle_disabled_baseline_active_refinement_and_reinitialize(self):
        from diffusion_planner.model.diffusion_planner import Diffusion_Planner
        from diffusion_planner.planner.planner import DiffusionPlanner
        from diffusion_planner.utils.config import Config
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import (
            StateSE2,
            StateVector2D,
            TimePoint,
        )
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        from score_function.model.refined_planner import ScoreRefinedPlanner
        from score_function.model.score_branch import build_model
        from score_function.planner.planner import ScoreFunctionPlanner
        from score_function.utils.config import load_config
        from score_function.utils.planner_utils import planner_identity
        from score_function.utils.sampling import planner_seed

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = official_arguments()
            args["route_len"] = 20
            args_path = root / "args.json"
            args_path.write_text(json.dumps(args))
            original = Diffusion_Planner(Config(str(args_path), None)).eval()
            checkpoint = root / "planner.pth"
            state_dict = original.state_dict()
            torch.save(
                {"ema_state_dict": {"module." + k: v for k, v in state_dict.items()}}, checkpoint
            )
            config = load_config(
                Path(__file__).resolve().parents[1] / "configs/score_function.json", root
            )
            config["paths"].update(
                planner_dir=OFFICIAL_ROOT,
                planner_args=str(args_path),
                planner_checkpoint=str(checkpoint),
            )
            config["model"].update(hidden_dim=24, num_heads=3)
            config["refinement"].update(gamma=0.0, steps=2)
            config_path = root / "score.json"
            config_path.write_text(json.dumps(config))
            branch = build_model(config, "cpu").eval()
            with torch.no_grad():
                branch.score_head[-1].weight.zero_()
                branch.score_head[-1].bias.zero_()
                branch.score_head[-1].bias[0] = 1.0
            selected = {
                "planner": planner_identity(config),
                "sigma_score": config["training"]["sigma"],
            }
            _, arrays = feature_record(root)
            raw = {
                key: torch.from_numpy(value)[None]
                for key, value in arrays.items()
                if key != "ego_agent_future"
            }
            raw["ego_current_state"] = torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            )
            states = deque(
                EgoState.build_from_rear_axle(
                    StateSE2(0.1 * i, 0.0, 0.0),
                    StateVector2D(1.0, 0.0),
                    StateVector2D(0.0, 0.0),
                    0.0,
                    TimePoint(1000000 + 100000 * i),
                    get_pacifica_parameters(),
                )
                for i in range(21)
            )
            current = SimpleNamespace(
                history=SimpleNamespace(ego_states=states), iteration=SimpleNamespace(index=0)
            )
            init = SimpleNamespace(
                map_api=SimpleNamespace(map_name="synthetic"), route_roadblock_ids=[]
            )
            past = TrajectorySampling(num_poses=20, time_horizon=2.0)
            future = TrajectorySampling(num_poses=80, time_horizon=8.0)
            baseline = DiffusionPlanner(
                Config(str(args_path), None), str(checkpoint), past, future, True, "cpu"
            )
            baseline.initialize(init)
            baseline.planner_input_to_model_inputs = lambda value: raw
            timestamp = states[-1].time_point.time_us
            with torch.random.fork_rng():
                torch.manual_seed(planner_seed(timestamp, config["reference_seed"]))
                baseline_trajectory = baseline.compute_planner_trajectory(current)

            def trajectory_array(trajectory):
                return np.array(
                    [
                        [s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading]
                        for s in trajectory.get_sampled_trajectory()
                    ]
                )

            init_impl = DiffusionPlanner.initialize
            compute_impl = DiffusionPlanner.compute_planner_trajectory
            with (
                patch("score_function.planner.planner.load_selected", return_value=(branch, selected)),
                patch.object(
                    DiffusionPlanner, "initialize", autospec=True, side_effect=init_impl
                ) as initialize,
                patch.object(
                    DiffusionPlanner,
                    "compute_planner_trajectory",
                    autospec=True,
                    side_effect=compute_impl,
                ) as compute,
            ):
                planner = ScoreFunctionPlanner(
                    str(config_path),
                    "unused.pt",
                    past,
                    future,
                    device="cpu",
                    trace_dir=str(root / "traces"),
                )
                planner.initialize(init)
                self.assertEqual(initialize.call_count, 1)
                self.assertIsInstance(planner._planner, ScoreRefinedPlanner)
                planner.planner_input_to_model_inputs = lambda value: raw
                rng = torch.get_rng_state()
                disabled = planner.compute_planner_trajectory(current)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                np.testing.assert_array_equal(
                    trajectory_array(disabled), trajectory_array(baseline_trajectory)
                )
                self.assertEqual(compute.call_count, 1)
                self.assertTrue(
                    all(not p.requires_grad for p in planner._planner.base_model.parameters())
                )
                self.assertTrue(planner._planner.last_refinement_trace["disabled"])
                first_trace = planner.scenario_trace_dir
                self.assertTrue((first_trace / f"{timestamp}.json").is_file())

                planner._planner.gamma = 0.25
                active = planner.compute_planner_trajectory(current)
                self.assertEqual(compute.call_count, 2)
                self.assertEqual(planner._planner.last_refinement_trace["completed_steps"], 2)
                self.assertFalse(
                    np.array_equal(trajectory_array(active)[1:], trajectory_array(disabled)[1:])
                )
                np.testing.assert_array_equal(
                    trajectory_array(active)[0], trajectory_array(disabled)[0]
                )
                trace = json.loads((first_trace / f"{timestamp}.json").read_text())
                self.assertEqual(len(trace["baseline_ego"]), 80)
                self.assertEqual(len(trace["refined_ego"]), 80)

                planner.initialize(init)
                self.assertEqual(initialize.call_count, 2)
                self.assertIsInstance(planner._planner.base_model, Diffusion_Planner)
                self.assertNotEqual(planner.scenario_trace_dir, first_trace)

                # Explicit compatibility fallback for non-DDP checkpoints.
                torch.save({"ema_state_dict": state_dict}, checkpoint)
                selected["planner"] = planner_identity(config)
                planner.initialize(init)
                self.assertEqual(initialize.call_count, 2)
                np.testing.assert_array_equal(
                    trajectory_array(planner.compute_planner_trajectory(current)),
                    trajectory_array(baseline_trajectory),
                )


if __name__ == "__main__":
    unittest.main()
