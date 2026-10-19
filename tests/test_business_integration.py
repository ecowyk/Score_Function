"""Regression checks for the physical boundary used by business planners."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from examples.business_integration import refine_physical_ego, smoke_check


class Unavailable:
    def __getattribute__(self, name):
        raise AssertionError("Disabled refinement must not access external resources")


class ConstantScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.neighbor_conditions = None

    def forward(self, trajectory, scene, route, **kwargs):
        self.calls += 1
        self.neighbor_conditions = kwargs
        return trajectory.new_tensor([1.0, -2.0, 3.0, 4.0]).expand_as(trajectory)


class BusinessIntegration(unittest.TestCase):
    def setUp(self):
        self.trajectory = torch.tensor([[[2.0, -3.0, 1.0, 0.0]]]).expand(2, 80, 4).clone()
        self.scene = torch.zeros(2, 107, 192)
        self.route = torch.zeros(2, 192)
        self.stats = SimpleNamespace(
            mean=torch.tensor([[1.0, -1.0, 0.1, -0.2]]),
            std=torch.tensor([[2.0, 3.0, 0.5, 0.25]]),
        )

    def test_disabled_is_exact_and_needs_no_external_resources(self):
        self.trajectory[..., 2:] = 0.3  # Bypass must not project this heading.
        original = self.trajectory.clone()
        absent = Unavailable()
        for gamma, steps in ((0.0, 20), (0.1, 0)):
            with self.subTest(gamma=gamma, steps=steps):
                result, trace = refine_physical_ego(
                    self.trajectory, absent, absent, absent, absent,
                    sigma=0.1, gamma=gamma, steps=steps,
                )
                self.assertIs(result, self.trajectory)
                self.assertTrue(torch.equal(result, original))
                self.assertEqual(trace["completed_steps"], 0)
                self.assertTrue(trace["disabled"])

    def test_disabled_still_rejects_invalid_parameters_and_trajectory(self):
        cases = (
            {"sigma": 0}, {"sigma": -0.1}, {"sigma": float("nan")},
            {"sigma": float("inf")}, {"gamma": -1}, {"gamma": float("nan")},
            {"gamma": float("inf")}, {"steps": -1}, {"steps": True}, {"steps": 1.5},
        )
        for bad in cases:
            options = dict(sigma=0.1, gamma=0.0, steps=0)
            options.update(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                refine_physical_ego(self.trajectory, None, None, None, None, **options)
        for invalid in (torch.zeros(2, 80, 3), torch.full((2, 80, 4), float("nan"))):
            with self.assertRaises(ValueError):
                refine_physical_ego(invalid, None, None, None, None, sigma=0.1, gamma=0, steps=0)

    def test_exact_update_in_physical_units_and_inputs_unchanged(self):
        branch = ConstantScore().eval()
        original = self.trajectory.clone()
        result, trace = refine_physical_ego(
            self.trajectory, branch, self.scene, self.route, self.stats,
            sigma=0.5, gamma=0.8, steps=3, heading_projection=False,
        )
        delta = torch.tensor([1.0, -2.0, 3.0, 4.0]) * self.stats.std[0] * (0.8 * 0.5**2 * 3)
        torch.testing.assert_close(result, original + delta)
        self.assertTrue(torch.equal(self.trajectory, original))
        self.assertEqual(result.shape, original.shape)
        self.assertEqual(branch.calls, 3)
        self.assertEqual(trace["completed_steps"], 3)

    def test_projection_uses_physical_heading_and_neighbors_pass_through(self):
        branch = ConstantScore().eval()
        neighbor = torch.randn(2, 10, 80, 4)
        valid = torch.ones(2, 10, dtype=torch.bool)
        saved_neighbor = neighbor.clone()
        result, _ = refine_physical_ego(
            self.trajectory, branch, self.scene, self.route, self.stats,
            sigma=0.5, gamma=0.8, steps=1, neighbor_future=neighbor, neighbor_valid=valid,
        )
        expected = self.trajectory + torch.tensor([1.0, -2.0, 3.0, 4.0]) * self.stats.std[0] * 0.2
        expected[..., 2:] /= expected[..., 2:].norm(dim=-1, keepdim=True)
        torch.testing.assert_close(result, expected)
        self.assertIs(branch.neighbor_conditions["neighbor_future"], neighbor)
        self.assertIs(branch.neighbor_conditions["neighbor_valid"], valid)
        self.assertTrue(torch.equal(neighbor, saved_neighbor))

    def test_enabled_requires_eval_mode_and_conditions(self):
        with self.assertRaisesRegex(ValueError, "eval mode"):
            refine_physical_ego(
                self.trajectory, ConstantScore(), self.scene, self.route, self.stats,
                sigma=0.1, gamma=0.1, steps=1,
            )
        with self.assertRaisesRegex(ValueError, "conditions"):
            refine_physical_ego(
                self.trajectory, ConstantScore().eval(), None, self.route, self.stats,
                sigma=0.1, gamma=0.1, steps=1,
            )

    def test_full_width_g_cpu_smoke(self):
        report = smoke_check()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["shape"], [1, 80, 4])
        self.assertEqual(report["completed_steps"], 20)


if __name__ == "__main__":
    unittest.main()
