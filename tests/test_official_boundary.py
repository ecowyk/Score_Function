"""Opt-in official-code integration with synthetic inputs and synthetic weights.

Set SCORE_FUNCTION_OFFICIAL_ROOT to an existing official Diffusion-Planner checkout.
This module downloads nothing and does not claim to test released trained weights,
nuPlan data, or a closed-loop simulation.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from score_function.model.refined_planner import ScoreRefinedPlanner
from score_function.model.score_branch import build_model
from score_function.utils.config import load_config
from score_function.utils.feature_dataset import read_feature_batch
from score_function.utils.normalizer import normalize_ego_future
from score_function.utils.planner_utils import (
    ego_normalizer_metadata,
    encode_conditions,
    load_frozen_planner,
)
from score_function.utils.sampling import predict_candidates
from score_function.utils.train_utils import atomic_write

OFFICIAL_ROOT = os.environ.get("SCORE_FUNCTION_OFFICIAL_ROOT")


def official_arguments():
    """Official module dimensions, with deliberately nonidentity normalization."""
    widths = {
        "ego_current_state": 10,
        "neighbor_agents_past": 11,
        "static_objects": 10,
        "lanes": 12,
        "route_lanes": 12,
        "lanes_speed_limit": 1,
    }
    return {
        "hidden_dim": 192,
        "agent_num": 32,
        "static_objects_num": 5,
        "lane_num": 70,
        "time_len": 21,
        "static_objects_state_dim": 10,
        "lane_len": 20,
        "route_num": 25,
        "num_heads": 6,
        "encoder_depth": 3,
        "decoder_depth": 3,
        "encoder_drop_path_rate": 0.1,
        "decoder_drop_path_rate": 0.1,
        "predicted_neighbor_num": 10,
        "future_len": 80,
        "diffusion_model_type": "x_start",
        "device": "cpu",
        "state_normalizer": {
            "mean": [[[4.0, -2.0, 0.3, -0.2]]] + [[[1.0, 2.0, 0.0, 0.0]]] * 10,
            "std": [[[20.0, 5.0, 0.5, 0.8]]] + [[[8.0, 7.0, 0.7, 0.9]]] * 10,
        },
        "observation_normalizer": {
            key: {"mean": [0.1] * width, "std": [1.5] * width} for key, width in widths.items()
        },
    }


def feature_record(directory):
    generator = np.random.default_rng(124)
    shapes = {
        "ego_current_state": (10,),
        "neighbor_agents_past": (32, 21, 11),
        "static_objects": (5, 10),
        "lanes": (70, 20, 12),
        "lanes_speed_limit": (70, 1),
        "route_lanes": (25, 20, 12),
        "ego_agent_future": (80, 3),
    }
    arrays = {key: generator.normal(size=shape).astype(np.float32) for key, shape in shapes.items()}
    arrays["lanes_has_speed_limit"] = np.ones((70, 1), dtype=bool)
    # Include masked/padded agents and objects as in the real preprocessing format.
    arrays["neighbor_agents_past"][20:] = 0.0
    arrays["static_objects"][3:] = 0.0
    arrays["lanes"][60:] = 0.0
    path = Path(directory) / "frame.npz"
    np.savez(path, **arrays)
    return {"feature": str(path), "token": "synthetic", "start_time_us": 1000}, arrays


@unittest.skipUnless(
    OFFICIAL_ROOT, "Set SCORE_FUNCTION_OFFICIAL_ROOT for official-code boundary tests"
)
class OfficialBoundaryTests(unittest.TestCase):
    def test_frozen_official_encoder_cache_branch_backward_and_full_prediction(self):
        source = Path(OFFICIAL_ROOT).resolve()
        self.assertTrue((source / "diffusion_planner/model/diffusion_planner.py").is_file())
        sys.path.insert(0, str(source))
        from diffusion_planner.model.diffusion_planner import Diffusion_Planner
        from diffusion_planner.utils.config import Config

        torch.set_num_threads(1)
        torch.manual_seed(141)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args_path = root / "args.json"
            checkpoint = root / "synthetic_model.pth"
            atomic_write(args_path, official_arguments())
            original = Diffusion_Planner(Config(str(args_path), None)).eval()
            # All actual official module classes are retained. Weights are synthetic.
            atomic_write(checkpoint, {"ema_state_dict": original.state_dict()}, tensor=True)
            del original
            config = load_config(
                Path(__file__).resolve().parents[1] / "configs/score_function.json", root
            )
            config["paths"].update(
                planner_dir=str(source),
                planner_args=str(args_path),
                planner_checkpoint=str(checkpoint),
            )
            config["model"].update(hidden_dim=24, num_heads=3)
            planner, planner_args = load_frozen_planner(config, "cpu")
            self.assertFalse(planner.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in planner.parameters()))
            record, arrays = feature_record(root)
            inputs, target = read_feature_batch([record], planner_args, "cpu")
            self.assertEqual(tuple(target.shape), (1, 80, 4))
            physical = torch.from_numpy(arrays["ego_agent_future"])[None]
            encoded_target = torch.cat(
                (physical[..., :2], physical[..., 2:3].cos(), physical[..., 2:3].sin()), -1
            )
            torch.testing.assert_close(
                target,
                normalize_ego_future(encoded_target, planner_args.state_normalizer),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                torch.tensor(ego_normalizer_metadata(planner_args)["mean"]),
                torch.tensor(official_arguments()["state_normalizer"]["mean"][0][0]),
                atol=0,
                rtol=0,
            )
            # Training conditioning must not execute the diffusion decoder at all.
            with patch.object(
                planner.decoder, "forward", side_effect=AssertionError("decoder called")
            ):
                context, route = encode_conditions(planner, inputs)
            self.assertEqual(tuple(context.shape), (1, 107, 192))
            self.assertEqual(tuple(route.shape), (1, 192))
            self.assertFalse(context.requires_grad)
            self.assertFalse(route.requires_grad)
            self.assertTrue(torch.isfinite(context).all())
            self.assertTrue(torch.isfinite(route).all())

            before = {key: value.clone() for key, value in planner.state_dict().items()}
            branch = build_model(config, "cpu").train()
            branch_before = {key: value.clone() for key, value in branch.state_dict().items()}
            optimizer = torch.optim.AdamW(branch.parameters(), lr=1e-3)
            noise = torch.randn_like(target)
            sigma = config["training"]["sigma"]
            score = branch(target + sigma * noise, context, route)
            loss = (sigma * score + noise).square().mean()
            loss.backward()
            self.assertTrue(
                any(
                    parameter.grad is not None and parameter.grad.abs().sum() > 0
                    for parameter in branch.parameters()
                )
            )
            self.assertTrue(all(parameter.grad is None for parameter in planner.parameters()))
            optimizer.step()
            self.assertTrue(
                any(
                    not torch.equal(value, branch_before[key])
                    for key, value in branch.state_dict().items()
                )
            )
            for key, value in planner.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

            # Real official DPM sampler is exercised, not a replacement prediction stub.
            encoding, output = predict_candidates(
                planner, planner_args, inputs, [1000], config["reference_seed"]
            )
            self.assertEqual(tuple(encoding["encoding"].shape), (1, 107, 192))
            self.assertEqual(tuple(output["prediction"].shape), (1, 11, 80, 4))
            self.assertTrue(torch.isfinite(output["prediction"]).all())
            self.assertFalse(output["prediction"].requires_grad)
            self._assert_wrapper_contract(planner, planner_args, branch, inputs, sigma)

    def _assert_wrapper_contract(self, planner, planner_args, branch, inputs, sigma):
        wrapper = ScoreRefinedPlanner(
            planner,
            branch,
            planner_args.state_normalizer,
            sigma=sigma,
            gamma=0.0,
            steps=2,
            heading_projection=True,
        ).eval()
        calls = []

        def capture_arguments(module, args):
            # The actual integration calls only trajectory, scene, and route.
            self.assertEqual(len(args), 3)
            self.assertEqual(tuple(args[0].shape), (1, 80, 4))
            self.assertEqual(tuple(args[1].shape), (1, 107, 192))
            self.assertEqual(tuple(args[2].shape), (1, 192))
            calls.append(args)

        hook = branch.register_forward_pre_hook(capture_arguments)
        try:
            generator_state = torch.get_rng_state()
            with torch.no_grad():
                encoding, baseline = planner(inputs)
            saved_prediction = baseline["prediction"].clone()
            # Synthetic official x_start initialization returns deliberately
            # nonunit physical headings; gamma=0 must not silently project them.
            headings = saved_prediction[:, 0, :, 2:4]
            self.assertGreater((headings.norm(dim=-1) - 1.0).abs().max().item(), 0.1)
            torch.set_rng_state(generator_state)
            _, disabled = wrapper(inputs)
            torch.testing.assert_close(disabled["prediction"], saved_prediction, atol=0, rtol=0)
            self.assertEqual(calls, [])
            self.assertIs(wrapper.refine_prediction(inputs, encoding, baseline), baseline)
            self.assertTrue(wrapper.last_refinement_trace["disabled"])

            wrapper.gamma = 0.25
            torch.set_rng_state(generator_state)
            _, refined = wrapper(inputs, record_trace=True)
            self.assertEqual(len(calls), 2)
            torch.testing.assert_close(
                refined["prediction"][:, 1:], saved_prediction[:, 1:], atol=0, rtol=0
            )
            torch.testing.assert_close(baseline["prediction"], saved_prediction, atol=0, rtol=0)
            self.assertFalse(torch.equal(refined["prediction"][:, 0], saved_prediction[:, 0]))
            self.assertEqual(wrapper.last_refinement_trace["completed_steps"], 2)
            for key in ("t", "diffusion_time", "sampled_trajectories"):
                self.assertNotIn(key, inputs)
            self.assertFalse(planner.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in planner.parameters()))
        finally:
            hook.remove()


if __name__ == "__main__":
    unittest.main()
