"""Scientific metric identities and offline report plumbing, with synthetic fixtures."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from score_function.evaluation.diagnostics import evaluate
from score_function.evaluation.metrics import (
    one_step_recovery,
    score_diagnostics,
    summarize_cosines,
    trajectory_metrics,
)
from score_function.evaluation.planner_evaluation import evaluate_planner
from score_function.utils.config import METHOD
from score_function.utils.dataset import write_shard
from score_function.utils.train_utils import atomic_write, file_hash, read_json

NORMALIZER = {"mean": [0.0, 0.0, 0.0, 0.0], "std": [10.0, 2.0, 0.5, 0.25]}


def clean_target():
    physical = torch.zeros(1, 80, 4)
    physical[..., 0] = torch.arange(80) * 0.3
    physical[..., 2] = 1.0
    return physical / torch.tensor(NORMALIZER["std"])


class Oracle(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, trajectory, scene, route):
        return (clean_target().to(trajectory) - trajectory) / self.sigma**2


class FailedScore(torch.nn.Module):
    def forward(self, trajectory, scene, route):
        return torch.full_like(trajectory, float("nan"))


class FakePlanner:
    def __init__(self, config, device):
        self.ego_normalizer = NORMALIZER
        self.config = SimpleNamespace(
            state_normalizer=SimpleNamespace(
                mean=torch.tensor(NORMALIZER["mean"]).reshape(1, 1, 4),
                std=torch.tensor(NORMALIZER["std"]).reshape(1, 1, 4),
            )
        )

        self.decoder = SimpleNamespace(
            decoder=SimpleNamespace(dit=SimpleNamespace(route_encoder=self.route_encoder))
        )

    def read_features(self, records):
        size = len(records)
        return {"route_lanes": torch.zeros(size, 1)}, clean_target().repeat(size, 1, 1)

    def route_encoder(self, route):
        return torch.zeros(len(route), 192)

    def predict(self, inputs, timestamps, seed):
        physical = clean_target() * torch.tensor(NORMALIZER["std"])
        physical = physical.clone()
        physical[..., 0] += 1.0
        prediction = physical[:, None].repeat(len(timestamps), 11, 1, 1)
        return {"encoding": torch.zeros(len(timestamps), 107, 192)}, {"prediction": prediction}


def fixture(directory):
    root = Path(directory)
    cache = root / "cache"
    rows = []
    for index in range(3):
        feature = root / f"feature_{index}.npz"
        np.savez(feature, lanes=np.zeros((1, 2, 7)), route_lanes=np.zeros((1, 2, 7)))
        rows.append(
            {
                "record": {
                    "token": f"v{index}",
                    "recording": "validation_drive",
                    "split": "val",
                    "start_time_us": index,
                    "feature": str(feature),
                    "feature_sha256": file_hash(feature),
                },
                "target": clean_target()[0],
                "context": torch.zeros(107, 192),
                "route": torch.zeros(192),
            }
        )
    write_shard(cache / "shard_000000", rows, "synthetic evaluation fixture")
    index = cache / "index.json"
    atomic_write(
        index,
        {
            "schema_version": 1,
            "method": METHOD,
            "metadata": {
                "identity": "synthetic evaluation fixture",
                "samples": len(rows),
                "planner": {"identity": "fixture"},
                "ego_normalizer": NORMALIZER,
            },
            "shards": [
                {
                    "path": "shard_000000",
                    "marker_sha256": file_hash(cache / "shard_000000/complete.json"),
                }
            ],
        },
    )
    config = {
        "output": str(root / "run"),
        "cache": str(index),
        "paths": {
            "root": str(root),
            "planner_dir": str(root / "official"),
            "devkit_dir": str(root / "devkit"),
        },
        "runtime": {"device": "cpu", "cpu_threads": 1, "memory_fraction": 0.5},
        "training": {
            "sigma": 0.05,
            "validation_repeats": 2,
            "validation_seed": 22,
            "num_workers": 0,
        },
        "evaluation": {
            "noise_repeats": 2,
            "seed": 22,
            "gammas": [0.5, 1.0],
            "batch_size": 2,
            "visualize_samples": 1,
        },
        "refinement": {"gamma": 0.5, "steps": 1, "heading_projection": True},
        "reference_seed": 27,
    }
    state = {"config": config, "planner": {"identity": "fixture"}, "ego_normalizer": NORMALIZER}
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(
        b"Fixture path only; actual model build_data_loader is patched in this test."
    )
    return config, state, checkpoint


class EvaluationTests(unittest.TestCase):
    def test_oracle_score_tweedie_and_temporal_metrics(self):
        sigma = 0.05
        target = clean_target()
        noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(2))
        noisy = target + sigma * noise
        score = -noise / sigma
        metrics = score_diagnostics(score, noise, sigma)
        self.assertLess(metrics["dsm"].item(), 1e-12)
        self.assertAlmostEqual(metrics["cosine"].item(), 1.0, places=6)
        recovered, after = one_step_recovery(noisy, target, score, sigma, 1, NORMALIZER)
        torch.testing.assert_close(recovered, target, rtol=0, atol=3e-7)
        self.assertLess(after["mse_normalized"].item(), 1e-12)
        ramp = torch.zeros_like(score)
        ramp[..., 0] = torch.arange(80)
        smoothness = score_diagnostics(ramp, noise, sigma)
        self.assertEqual(smoothness["score_d1"].item(), 1.0)
        self.assertEqual(smoothness["score_d2"].item(), 0.0)

    def test_zero_norm_is_undefined_and_nonfinite_is_error(self):
        noise = torch.ones(2, 80, 4)
        result = score_diagnostics(torch.zeros_like(noise), noise, 0.1)
        self.assertTrue(torch.isnan(result["cosine"]).all())
        with self.assertRaises(FloatingPointError):
            score_diagnostics(torch.full_like(noise, float("nan")), noise, 0.1)

    def test_metric_units_wrapped_heading_and_unknown_heading(self):
        target = torch.tensor([[[0.0, 0.0, -1.0, 0.0]]]).repeat(1, 80, 1)
        prediction = target.clone()
        prediction[..., :2] += torch.tensor([1.0, 2.0])
        prediction[..., 2:] = torch.tensor([-0.995004165, -0.099833417])
        # Normalize actual physical vectors using deliberately unequal scales.
        std = torch.tensor(NORMALIZER["std"])
        metrics = trajectory_metrics(prediction / std, target / std, NORMALIZER)
        self.assertAlmostEqual(metrics["ade_m"].item(), 5**0.5, places=6)
        self.assertAlmostEqual(metrics["fde_m"].item(), 5**0.5, places=6)
        self.assertAlmostEqual(metrics["heading_mae_rad"].item(), 0.1, places=6)
        prediction[..., 2:] = 0
        metrics = trajectory_metrics(prediction / std, target / std, NORMALIZER)
        self.assertEqual(metrics["heading_undefined_points"].item(), 80)
        self.assertTrue(torch.isnan(metrics["heading_mae_rad"]).all())

    def test_exact_disk_median(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cosine.float64"
            values = np.array([-1, 0, 0.25, 1], dtype=np.float64)
            values.tofile(path)
            histogram = np.histogram(values, bins=40, range=(-1, 1))[0]
            result = summarize_cosines(path, histogram, 2)
            self.assertEqual(result["median"], 0.125)
            self.assertEqual(result["mean"], 0.0625)
            self.assertEqual(result["fraction_positive"], 0.5)
            self.assertEqual(result["undefined_count"], 2)

    def test_streamed_evaluation_report_and_batch_invariant_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, checkpoint = fixture(directory)
            with patch(
                "score_function.utils.checkpoint.load_selected",
                return_value=(Oracle(0.05).eval(), state),
            ):
                summary = evaluate(config, checkpoint)
                config2 = copy.deepcopy(config)
                config2["evaluation"].update(batch_size=1, visualize_samples=0)
                second = evaluate(config2, checkpoint, output=Path(directory) / "second")
            self.assertEqual(summary["samples"], 3)
            self.assertEqual(summary["frame_noise_pairs"], 6)
            self.assertLess(summary["diagnostics"]["dsm"]["mean"], 1e-10)
            self.assertLess(summary["tweedie"]["metrics"]["mse_normalized"]["mean"], 1e-12)
            self.assertAlmostEqual(
                summary["corrupted_baseline"]["mse_normalized"]["mean"],
                second["corrupted_baseline"]["mse_normalized"]["mean"],
                places=12,
            )
            output = Path(config["output"]) / "offline_val"
            rows = [
                json.loads(line) for line in (output / "per_sample.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 12)
            self.assertTrue((output / "trajectory_000.png").is_file())
            self.assertTrue((output / "cosine_histogram.png").is_file())
            self.assertEqual(read_json(output / "status.json")["state"], "complete")
            with self.assertRaises(FileExistsError):
                evaluate(config, checkpoint)

    def test_nan_failure_and_planner_identity_mismatch_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, checkpoint = fixture(directory)
            with patch(
                "score_function.utils.checkpoint.load_selected",
                return_value=(FailedScore().eval(), state),
            ):
                with self.assertRaises(FloatingPointError):
                    evaluate(config, checkpoint)
            output = Path(config["output"]) / "offline_val"
            self.assertTrue(read_json(output / "status.json")["numerical_failure"])
            state = {**state, "planner": {"identity": "different"}}
            with patch(
                "score_function.utils.checkpoint.load_selected",
                return_value=(Oracle(0.05).eval(), state),
            ):
                with self.assertRaisesRegex(ValueError, "different frozen planners"):
                    evaluate(config, checkpoint, output=Path(directory) / "mismatch")

    def test_planner_pair_and_disabled_exact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, checkpoint = fixture(directory)
            with (
                patch(
                    "score_function.utils.checkpoint.load_selected",
                    return_value=(Oracle(0.05).eval(), state),
                ),
                patch(
                    "score_function.utils.planner_utils.load_frozen_planner",
                    side_effect=lambda config, device: (
                        FakePlanner(config, device),
                        FakePlanner(config, device).config,
                    ),
                ),
                patch(
                    "score_function.utils.feature_dataset.read_feature_batch",
                    side_effect=lambda records, args, device: FakePlanner(
                        None, device
                    ).read_features(records),
                ),
                patch(
                    "score_function.utils.sampling.predict_candidates",
                    side_effect=lambda model, args, inputs, timestamps, seed: model.predict(
                        inputs, timestamps, seed
                    ),
                ),
                patch(
                    "score_function.utils.planner_utils.planner_identity",
                    return_value=state["planner"],
                ),
            ):
                summary = evaluate_planner(config, checkpoint)
                self.assertAlmostEqual(summary["baseline"]["ade_m"]["mean"], 1.0, places=5)
                self.assertAlmostEqual(summary["refined"]["ade_m"]["mean"], 0.5, places=5)
                config["refinement"]["gamma"] = 0
                config["evaluation"]["visualize_samples"] = 0
                second = evaluate_planner(config, checkpoint, output=Path(directory) / "disabled")
            self.assertEqual(second["diagnostics"]["displacement_ade_m"]["mean"], 0)
            self.assertIsNone(second["diagnostics"]["initial_score_l2"]["mean"])
            for line in (
                (Path(directory) / "disabled/refinement_traces.jsonl").read_text().splitlines()
            ):
                record = json.loads(line)
                self.assertEqual(record["baseline_ego_physical"], record["refined_ego_physical"])
                self.assertEqual(record["trace"]["completed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
