"""Long diagnostic dynamics, sparse storage, divergence and rendering contracts."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from score_function.evaluation.long_refinement import (
    DEFAULT_SNAPSHOTS,
    SCALAR_COLUMNS,
    diagnose_refinement,
    representative_indices,
    trace_refinement,
)
from score_function.evaluation.long_refinement_plots import plot_long_refinement
from score_function.model.refinement import refine_ego
from score_function.utils.train_utils import read_json

NORMALIZER = {"mean": [0.0, 0.0, 0.0, 0.0], "std": [10.0, 2.0, 0.5, 0.25]}


def target_trajectory(length=80):
    physical = torch.zeros(1, length, 4, dtype=torch.float64)
    physical[..., 0] = torch.arange(length) * 0.3
    physical[..., 2] = 1
    return physical / torch.tensor(NORMALIZER["std"])


class GaussianScore(torch.nn.Module):
    parameterization = "score"

    def __init__(self, target, sigma=0.05):
        super().__init__()
        self.register_buffer("target", target)
        self.sigma = sigma

    def forward(self, value, scene, route, **kwargs):
        return (self.target.to(value) - value) / self.sigma**2


class QuadraticEnergy(GaussianScore):
    parameterization = "energy"

    def energy_value(self, value, scene, route, **kwargs):
        return 0.5 * ((value - self.target.to(value)) / self.sigma).square().flatten(1).sum(1)

    def forward(self, value, scene, route, **kwargs):
        with torch.enable_grad():
            value = value.detach().requires_grad_()
            return -torch.autograd.grad(
                self.energy_value(value, scene, route, **kwargs).sum(), value
            )[0].detach()


class LongRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.target = target_trajectory()
        self.initial = self.target + 0.1 * torch.randn(
            self.target.shape, generator=torch.Generator().manual_seed(17), dtype=torch.float64
        )
        self.scene, self.route = torch.zeros(1, 107, 192), torch.zeros(1, 192)

    def trace(self, model, **kwargs):
        return trace_refinement(
            model.eval(),
            self.initial,
            self.target,
            self.scene,
            self.route,
            NORMALIZER,
            0.05,
            kwargs.pop("gamma", 0.002),
            kwargs.pop("steps", 5000),
            **kwargs,
        )

    def test_5000_step_analytic_gaussian_and_energy_sparse_storage(self):
        for cls in (GaussianScore, QuadraticEnergy):
            with (
                self.subTest(parameterization=cls.parameterization),
                tempfile.TemporaryDirectory() as directory,
            ):
                result, scalars, snapshots = self.trace(
                    cls(self.target),
                    heading_projection=False,
                    scalar_path=Path(directory) / "steps.csv",
                )
                self.assertEqual(result["completed_steps"], 5000)
                self.assertIsNone(result["failure"])
                self.assertEqual(list(snapshots), list(DEFAULT_SNAPSHOTS))
                expected = self.target + (0.998**5000) * (self.initial - self.target)
                torch.testing.assert_close(
                    torch.from_numpy(snapshots[5000]), expected[0], rtol=0, atol=1e-13
                )
                self.assertEqual(scalars.shape, (5001, len(SCALAR_COLUMNS)))
                self.assertLess(sum(value.nbytes for value in snapshots.values()), 21000)
                with (Path(directory) / "steps.csv").open() as stream:
                    self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 5001)
                self.assertTrue(result["small_last_window_xy_updates"])
                if cls.parameterization == "energy":
                    energy = scalars[:, SCALAR_COLUMNS.index("energy_after")]
                    self.assertTrue(np.all(np.diff(energy) <= 1e-12))
                    self.assertLess(energy[-1], energy[0] * 1e-8)

    def test_projection_matches_legacy_and_reports_meter_units(self):
        model = GaussianScore(self.target).eval()
        state_normalizer = SimpleNamespace(
            mean=torch.tensor(NORMALIZER["mean"])[None, None],
            std=torch.tensor(NORMALIZER["std"])[None, None],
        )
        old, trace = refine_ego(
            model,
            self.initial,
            self.scene,
            self.route,
            state_normalizer,
            0.05,
            0.1,
            20,
            record_trace=False,
        )
        result, scalars, snapshots = self.trace(model, gamma=0.1, steps=20)
        torch.testing.assert_close(torch.from_numpy(snapshots[20]), old[0], rtol=0, atol=0)
        self.assertLess(scalars[-1, SCALAR_COLUMNS.index("heading_deviation_post_max")], 1e-14)
        moved = (
            (snapshots[20] - self.initial[0].numpy())[:, :2] * np.array(NORMALIZER["std"][:2])
        ).mean(0)
        self.assertGreater(
            scalars[-1, SCALAR_COLUMNS.index("net_xy_mean_m")], np.linalg.norm(moved)
        )
        self.assertEqual(result["snapshot_steps"], [0, 5, 20])
        self.assertEqual(trace["completed_steps"], result["completed_steps"])

    def test_numerical_failure_preserves_last_finite_state(self):
        class FailsAtSeventh(GaussianScore):
            calls = 0

            def forward(self, *args, **kwargs):
                self.calls += 1
                value = super().forward(*args, **kwargs)
                return value if self.calls < 7 else value * float("nan")

        result, scalars, snapshots = self.trace(FailsAtSeventh(self.target), steps=5000)
        self.assertEqual(result["failure"]["step"], 7)
        self.assertEqual(result["completed_steps"], 6)
        self.assertEqual(result["snapshot_steps"], [0, 5, 6])
        self.assertEqual(len(scalars), 7)
        self.assertTrue(np.isfinite(snapshots[6]).all())

    def test_stability_does_not_early_stop_and_final_budget_is_saved(self):
        model = GaussianScore(self.target)
        result, scalars, snapshots = self.trace(
            model, gamma=1.0, steps=123, snapshot_steps=[0, 5], heading_projection=False
        )
        self.assertTrue(result["small_last_window_xy_updates"])
        self.assertEqual(result["completed_steps"], 123)
        self.assertEqual(sorted(snapshots), [0, 5, 123])
        self.assertEqual(len(scalars), 124)
        result, _, snapshots = self.trace(model, gamma=0, steps=123)
        self.assertTrue(result["disabled"])
        np.testing.assert_array_equal(snapshots[0], self.initial[0].numpy())

    def test_selection_is_fixed_and_covers_groups_before_repeats(self):
        records = [
            {"token": str(index), "recording": "recording", "scenario_type": str(index % 4)}
            for index in range(20)
        ]
        chosen = representative_indices(records, 8, 5)
        reversed_chosen = representative_indices(list(reversed(records)), 8, 5)
        self.assertEqual(
            {records[index]["scenario_type"] for index in chosen[:4]}, {"0", "1", "2", "3"}
        )
        self.assertEqual(
            [records[index]["token"] for index in chosen],
            [list(reversed(records))[index]["token"] for index in reversed_chosen],
        )

    def test_plotting_smoke(self):
        _, scalars, snapshots = self.trace(QuadraticEnergy(self.target), steps=25)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "energy"
            plot_long_refinement(
                stem,
                self.target[0].numpy(),
                snapshots,
                scalars,
                SCALAR_COLUMNS,
                NORMALIZER,
                0.1,
                title="Synthetic Gaussian diagnostic",
            )
            self.assertGreater(Path(str(stem) + "_trajectories.png").stat().st_size, 1000)
            self.assertGreater(Path(str(stem) + "_diagnostics.png").stat().st_size, 1000)

    def test_expert_only_runner_continues_after_numerical_failure(self):
        from test_evaluation import clean_target, fixture

        class FirstPathFails(GaussianScore):
            calls = 0

            def forward(self, *args, **kwargs):
                self.calls += 1
                value = super().forward(*args, **kwargs)
                return value * float("nan") if self.calls == 2 else value

        with tempfile.TemporaryDirectory() as directory:
            config, state, checkpoint = fixture(directory)
            config["long_refinement"] = {
                "steps": 3,
                "max_samples": 2,
                "initializations": ["expert_noise"],
            }
            with patch(
                "score_function.utils.checkpoint.load_selected",
                return_value=(FirstPathFails(clean_target()).eval(), state),
            ):
                summary = diagnose_refinement(config, checkpoint)
            output = Path(config["output"]) / "long_refinement_val"
            self.assertEqual(summary["paths"], 2)
            self.assertEqual(summary["numerical_failures"], 1)
            self.assertEqual(summary["results"][1]["completed_steps"], 3)
            self.assertEqual(read_json(output / "status.json")["state"], "complete_with_failures")
            self.assertIn(
                "sample_000/expert_noise_trajectories.png", (output / "report.html").read_text()
            )
            with np.load(output / "sample_001/expert_noise.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["snapshot_steps"].tolist(), [0, 3])
                self.assertEqual(saved["trajectories_normalized"].shape, (2, 80, 4))
            self.assertTrue((output / "sample_001/condition.npz").is_file())

    def test_paired_initializations_share_condition_and_reproducible_noise(self):
        from test_evaluation import FakePlanner, clean_target, fixture

        with tempfile.TemporaryDirectory() as directory:
            config, state, checkpoint = fixture(directory)
            with (
                patch(
                    "score_function.utils.checkpoint.load_selected",
                    return_value=(GaussianScore(clean_target()).eval(), state),
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
                    # Emulate the float32 ULP differences between a GPU-built
                    # cache and CPU sin/cos during raw feature reconstruction.
                    side_effect=lambda records, args, device: (
                        {"route_lanes": torch.zeros(len(records), 1)},
                        torch.nextafter(
                            clean_target().repeat(len(records), 1, 1),
                            torch.full_like(
                                clean_target().repeat(len(records), 1, 1), float("inf")
                            ),
                        ),
                    ),
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
                patch("score_function.evaluation.long_refinement_plots.plot_long_refinement"),
            ):
                summary = diagnose_refinement(config, checkpoint, max_samples=1, steps=2)
                repeated = diagnose_refinement(
                    config, checkpoint, max_samples=1, steps=2, output=Path(directory) / "repeat"
                )
            self.assertEqual(summary["paths"], 2)
            self.assertEqual(
                [r["initial_sha256"] for r in summary["results"]],
                [r["initial_sha256"] for r in repeated["results"]],
            )
            self.assertAlmostEqual(summary["results"][0]["initial_metrics"]["ade_m"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
