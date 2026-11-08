"""Multiple K snapshots must follow the identical deterministic score path."""

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from score_function.model.score_branch import build_model
from score_function.navsim_dd.evaluate import score_pose_group
from score_function.navsim_dd.runtime import NavsimRefiner


class MultiKParityTests(unittest.TestCase):
    def setUp(self):
        # Use the actual NAVSIM architecture, including configured dropout and
        # both attention types; NavsimRefiner must put the whole model in eval.
        config = json.loads((Path(__file__).resolve().parents[1] /
                             "configs/navsim_dd_s10_g_ddaligned.json").read_text())
        with torch.random.fork_rng():
            torch.manual_seed(20260922)
            model = build_model(config, "cpu")
            self.scene = torch.randn(1, 64, 256)
            self.status = torch.randn(1, 256)
            self.raw = torch.randn(1, 8, 3)
            self.raw[0, :, 0] += torch.arange(1, 9) * 2.0
            self.raw[0, :, 2] = torch.linspace(-3.2, 3.2, 8)
        stats = {"mean": [10.0, -0.3, 0.0, 0.0], "std": [12.0, 2.5, 1.0, 1.0]}
        self.refiner = NavsimRefiner(model, stats, .1, .1, {"fixture": "parity"}, "cpu")

    def test_segmented_snapshots_equal_independent_runs_bit_for_bit(self):
        original = tuple(t.clone() for t in (self.raw, self.scene, self.status))
        initial_rng = torch.random.get_rng_state().clone()
        ks = [0, 5, 10, 20, 50, 100]
        counts = []
        hook = self.refiner.model.register_forward_hook(lambda *args: counts.append(1))
        try:
            snapshots = self.refiner.refine_many(self.raw, self.scene, self.status, ks)
        finally:
            hook.remove()
        self.assertEqual(len(counts), 100)  # A single path, not sum(ks) updates.
        self.assertIs(snapshots[0], self.raw)
        for k in ks[1:]:
            with self.subTest(k=k):
                independent = self.refiner.refine(self.raw, self.scene, self.status, k)
                self.assertTrue(torch.equal(snapshots[k], independent))
        self.assertTrue(torch.equal(initial_rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, (self.raw, self.scene, self.status))))
        self.assertTrue(all(not layer.training for layer in self.refiner.model.modules()))

    def test_sweep_scores_only_new_arms_and_keeps_reused_k0_exact(self):
        calls = []
        def scorer(poses):
            calls.append(poses.copy())
            return {"score": .2 + .1 * len(calls), "comfort": 1.0}

        ks = [0, 5, 10, 50, 100]
        counts = []
        hook = self.refiner.model.register_forward_hook(lambda *args: counts.append(1))
        try:
            trajectories, scores, deltas, _ = score_pose_group(
                self.raw, ks, self.refiner, self.scene, self.status, .1, scorer, "cpu",
                baseline_metrics={"score": .88, "comfort": 1.0},
            )
        finally:
            hook.remove()
        self.assertEqual(len(counts), 100)
        self.assertEqual(len(calls), 4)
        self.assertEqual(set(scores), set(ks))
        self.assertNotIn(20, scores)  # Previous K20 is joined from saved results later.
        np.testing.assert_array_equal(np.asarray(trajectories[0]), self.raw.numpy()[0])
        self.assertEqual(scores[0], {"score": .88, "comfort": 1.0})
        for k, poses in zip(ks[1:], calls):
            np.testing.assert_array_equal(np.asarray(trajectories[k]), poses)
            self.assertAlmostEqual(deltas[k]["score"], scores[k]["score"] - .88)


if __name__ == "__main__":
    unittest.main()
