"""Small tensor-level contracts; these are not nuPlan research experiments."""

import inspect
import json
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from score_function.loss import fixed_sigma_dsm_loss
from score_function.model.module.temporal import TemporalResidualBlock
from score_function.model.refinement import refine_ego
from score_function.model.score_branch import ScoreFunctionBranch, build_model
from score_function.utils.normalizer import (
    denormalize_ego_future,
    heading_norm_deviation,
    normalize_ego_future,
    project_heading,
)


def normalizer():
    mean = torch.arange(44, dtype=torch.float32).reshape(11, 1, 4) / 10
    std = torch.arange(1, 45, dtype=torch.float32).reshape(11, 1, 4) / 3
    return SimpleNamespace(mean=mean, std=std)


class ConstantScore(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, trajectory, scene, route):
        self.calls += 1
        return torch.full_like(trajectory, self.value)


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(19)
        self.trajectory = torch.randn(2, 80, 4)
        self.scene = torch.randn(2, 7, 192)
        self.route = torch.randn(2, 192)

    def test_branch_shapes_no_timestep_and_configurable_width(self):
        self.assertEqual(
            list(inspect.signature(ScoreFunctionBranch.forward).parameters),
            [
                "self",
                "ego_traj_norm",
                "scene_context",
                "route_embedding",
                "neighbor_future",
                "neighbor_valid",
            ],
        )
        model = ScoreFunctionBranch(hidden_dim=24, num_heads=6)
        self.assertEqual(torch.count_nonzero(model.temporal_pos).item(), 0)
        score = model(self.trajectory, self.scene, self.route)
        self.assertEqual(score.shape, self.trajectory.shape)
        self.assertTrue(torch.isfinite(score).all())
        self.assertFalse(
            any(
                "t_embedder" in key or "timestep" in key or "diffusion" in key
                for key in model.state_dict()
            )
        )
        branch = build_model(
            {"model": {"architecture": "temporal_score_function", "hidden_dim": 24}}, "cpu"
        )
        self.assertIsInstance(branch, ScoreFunctionBranch)
        with self.assertRaises(ValueError):
            branch(self.trajectory[:, :79], self.scene, self.route)

    def test_temporal_blocks_are_noncausal_and_receptive_field_is_37(self):
        model = ScoreFunctionBranch(hidden_dim=12, num_heads=3, dropout=0).double().eval()
        self.assertEqual(model.temporal_receptive_field, 37)
        trajectory = self.trajectory[:1].double().requires_grad_()
        score = model(trajectory, self.scene[:1].double(), self.route[:1].double())
        gradient = torch.autograd.grad(score[0, 40].sum(), trajectory)[0].abs().sum(-1)[0]
        self.assertEqual(torch.count_nonzero(gradient[:22]).item(), 0)
        self.assertEqual(torch.count_nonzero(gradient[59:]).item(), 0)
        self.assertGreater(gradient[22].item(), 0)
        self.assertGreater(gradient[58].item(), 0)
        for dilation in (1, 2, 4):
            block = TemporalResidualBlock(12, dilation=dilation, dropout=0)
            self.assertEqual(block(torch.randn(2, 80, 12)).shape, (2, 80, 12))

    def test_only_trainable_branch_receives_gradients(self):
        encoder = nn.Linear(6, 192).eval().requires_grad_(False)
        before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
        with torch.no_grad():
            context = encoder(torch.randn(2, 7, 6))
            route = encoder(torch.randn(2, 6))
        model = ScoreFunctionBranch(hidden_dim=24, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        epsilon = torch.randn_like(self.trajectory)
        loss = fixed_sigma_dsm_loss(model(self.trajectory, context, route), epsilon, 0.05)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)
        head_before = model.score_head[-1].weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(head_before, model.score_head[-1].weight))
        for name, parameter in encoder.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))

    def test_dsm_has_correct_sign_scale_and_denoising_equivalence(self):
        epsilon = torch.randn_like(self.trajectory)
        sigma = 0.05
        self.assertLess(fixed_sigma_dsm_loss(-epsilon / sigma, epsilon, sigma).item(), 1e-13)
        score = torch.randn_like(epsilon)
        noisy = self.trajectory + sigma * epsilon
        denoised = noisy + sigma**2 * score
        equivalent = ((denoised - self.trajectory) / sigma).square().mean()
        torch.testing.assert_close(fixed_sigma_dsm_loss(score, epsilon, sigma), equivalent)
        with self.assertRaises(ValueError):
            fixed_sigma_dsm_loss(score, epsilon, 0)

    def test_synthetic_fixed_corruption_can_be_optimized(self):
        # This finite, fixed-batch optimizer smoke test checks implementation;
        # it makes no claim about held-out driving data or true score recovery.
        torch.manual_seed(41)
        model = ScoreFunctionBranch(hidden_dim=24, num_heads=6, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        clean = torch.zeros(2, 80, 4)
        clean[..., 0] = torch.linspace(0, 2, 80)
        clean[..., 2] = 1
        epsilon = torch.randn_like(clean)
        scene = torch.randn(2, 7, 192)
        route = torch.randn(2, 192)
        noisy = clean + 0.05 * epsilon
        initial = fixed_sigma_dsm_loss(model(noisy, scene, route), epsilon, 0.05).item()
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = fixed_sigma_dsm_loss(model(noisy, scene, route), epsilon, 0.05)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        final = fixed_sigma_dsm_loss(model(noisy, scene, route), epsilon, 0.05).item()
        self.assertLess(final, initial * 0.5)

    def test_ego_normalization_roundtrip_does_not_broadcast_agents(self):
        stats = normalizer()
        normalized = normalize_ego_future(self.trajectory, stats)
        self.assertEqual(normalized.shape, (2, 80, 4))
        torch.testing.assert_close(normalized, (self.trajectory - stats.mean[0]) / stats.std[0])
        torch.testing.assert_close(denormalize_ego_future(normalized, stats), self.trajectory)
        stats.std[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            normalize_ego_future(self.trajectory, stats)

    def test_heading_projection_preserves_xy_and_handles_zero_heading(self):
        stats = normalizer()
        physical = self.trajectory.clone()
        physical[..., 2:] = 0
        updated = normalize_ego_future(physical, stats)
        previous_physical = physical.clone()
        previous_physical[..., 3] = 2
        previous = normalize_ego_future(previous_physical, stats)
        projected = project_heading(updated, stats, previous)
        self.assertTrue(torch.equal(projected[..., :2], updated[..., :2]))
        heading = denormalize_ego_future(projected, stats)[..., 2:]
        torch.testing.assert_close(heading[..., 0], torch.zeros(2, 80), atol=1e-7, rtol=0)
        torch.testing.assert_close(heading[..., 1], torch.ones(2, 80))
        self.assertLess(heading_norm_deviation(projected, stats).max().item(), 1e-6)
        fallback = denormalize_ego_future(project_heading(updated, stats, updated), stats)
        torch.testing.assert_close(fallback[..., 2], torch.ones(2, 80))
        torch.testing.assert_close(fallback[..., 3], torch.zeros(2, 80), atol=1e-7, rtol=0)

    def test_disabled_refinement_is_bitwise_exact_and_skips_branch_and_projection(self):
        model = ConstantScore(float("nan"))
        for gamma, steps in ((0.0, 10), (0.5, 0)):
            result, trace = refine_ego(
                model,
                self.trajectory,
                self.scene,
                self.route,
                None,
                sigma=0.05,
                gamma=gamma,
                steps=steps,
            )
            self.assertIs(result, self.trajectory)
            self.assertTrue(torch.equal(result, self.trajectory))
            self.assertEqual(trace["completed_steps"], 0)
        self.assertEqual(model.calls, 0)

    def test_refinement_uses_exact_fixed_budget_and_records_finite_json(self):
        stats = normalizer()
        model = ConstantScore().eval()
        result, trace = refine_ego(
            model,
            self.trajectory,
            self.scene,
            self.route,
            stats,
            sigma=0.05,
            gamma=0.5,
            steps=3,
            heading_projection=False,
        )
        expected = self.trajectory
        for _ in range(3):
            expected = expected + 0.5 * 0.05**2
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(model.calls, 3)
        self.assertEqual(trace["completed_steps"], 3)
        self.assertEqual(len(trace["trajectories"]), 4)
        self.assertEqual(len(trace["scores"]), 3)
        self.assertEqual(len(trace["heading_norm_deviation_pre_projection"]), 3)
        json.dumps(trace, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "Nonfinite|nonfinite"):
            refine_ego(
                ConstantScore(float("nan")).eval(),
                self.trajectory,
                self.scene,
                self.route,
                stats,
                sigma=0.05,
                gamma=0.5,
                steps=1,
            )
        with self.assertRaisesRegex(ValueError, "eval mode"):
            refine_ego(
                ConstantScore(),
                self.trajectory,
                self.scene,
                self.route,
                stats,
                sigma=0.05,
                gamma=0.5,
                steps=1,
            )


if __name__ == "__main__":
    unittest.main()
