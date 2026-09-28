"""Numerical and training checks for conservative score parameterization.

These small synthetic checks verify differentiation and integration contracts;
they are not evidence of better score estimation or closed-loop driving.
"""

import json
import socket
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from score_function.loss import fixed_sigma_dsm_loss
from score_function.model.refinement import refine_ego
from score_function.model.score_branch import ScoreFunctionBranch
from score_function.utils.normalizer import heading_norm_deviation, normalize_ego_future


def small_branch(parameterization="energy", **options):
    settings = dict(
        future_len=8,
        hidden_dim=12,
        num_heads=3,
        context_dim=12,
        pre_dilations=(1,),
        post_dilations=(1,),
        dropout=0.0,
        parameterization=parameterization,
    )
    settings.update(options)
    return ScoreFunctionBranch(**settings)


def inputs(batch=2, future_len=8, *, neighbors=False, dtype=torch.float32):
    trajectory = torch.randn(batch, future_len, 4, dtype=dtype)
    scene = torch.randn(batch, 3, 12, dtype=dtype)
    route = torch.randn(batch, 12, dtype=dtype)
    kwargs = {}
    if neighbors:
        valid = torch.zeros(batch, 10, dtype=torch.bool)
        valid[0, :2] = True
        kwargs = {
            "neighbor_future": torch.randn(batch, 10, future_len, 4, dtype=dtype),
            "neighbor_valid": valid,
        }
    return trajectory, scene, route, kwargs


def energy_ddp_worker(rank, directory):
    """A second iteration detects unused-parameter/reducer regressions."""
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "store").resolve().as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(130)
        model = small_branch(temporal_attention=True, neighbor_future=True)
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        initial = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        # Different rank-local batches make synchronization observable.
        torch.manual_seed(140 + rank)
        clean, scene, route, kwargs = inputs(batch=1, neighbors=True)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            epsilon = torch.randn_like(clean)
            prediction = wrapped(clean + 0.05 * epsilon, scene, route, **kwargs)
            loss = fixed_sigma_dsm_loss(prediction, epsilon, 0.05)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f"Missing/nonfinite DSM gradient for {name}")
            optimizer.step()
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        replicas = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(replicas, flat)
        if torch.equal(flat, initial):
            raise AssertionError("DDP training did not update energy parameters")
        torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=1e-7)
        if rank == 0:
            (directory / "result.json").write_text(
                json.dumps({"steps": 2, "updated": True, "synchronized": True}),
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


class EnergyParameterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(103)

    def test_energy_score_matches_finite_differences_and_is_batch_independent(self):
        model = small_branch().double().eval()
        trajectory, scene, route, _ = inputs(dtype=torch.float64)
        energy = model.energy_value(trajectory, scene, route)
        score = model.predict_score(trajectory, scene, route)
        self.assertEqual(energy.shape, (2,))
        self.assertEqual(score.shape, trajectory.shape)
        torch.testing.assert_close(model(trajectory, scene, route), score)

        # Compare to values, rather than reproducing the implementation's
        # autograd expression: this detects a wrong sign or time reduction.
        step = 1e-5
        for time_index, coordinate in ((0, 0), (4, 2), (7, 3)):
            direction = torch.zeros_like(trajectory)
            direction[0, time_index, coordinate] = step
            with torch.no_grad():
                upper = model.energy_value(trajectory + direction, scene, route)
                lower = model.energy_value(trajectory - direction, scene, route)
            derivative = (upper - lower) / (2 * step)
            torch.testing.assert_close(
                -score[0, time_index, coordinate], derivative[0], rtol=2e-4, atol=2e-7
            )
            self.assertEqual(derivative[1].item(), 0.0)

        single = model.predict_score(trajectory[:1], scene[:1], route[:1])
        torch.testing.assert_close(single[0], score[0], rtol=1e-9, atol=1e-10)
        changed = trajectory.clone()
        changed[1] += 20
        changed_scene, changed_route = scene.clone(), route.clone()
        changed_scene[1] -= 10
        changed_route[1] += 10
        other = model.predict_score(changed, changed_scene, changed_route)
        torch.testing.assert_close(other[0], score[0], rtol=1e-9, atol=1e-10)
        self.assertFalse(trajectory.requires_grad)

    def test_frozen_energy_runs_under_no_grad_and_real_inference_tensors(self):
        for global_attention, neighbors in ((False, False), (True, False), (False, True)):
            with self.subTest(global_attention=global_attention, neighbors=neighbors):
                model = (
                    small_branch(temporal_attention=global_attention, neighbor_future=neighbors)
                    .eval()
                    .requires_grad_(False)
                )
                trajectory, scene, route, kwargs = inputs(neighbors=neighbors)
                expected = model.predict_score(trajectory, scene, route, **kwargs)
                with torch.no_grad():
                    actual = model(trajectory, scene, route, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertFalse(actual.requires_grad)
                with torch.inference_mode():
                    # Cloning inside the context deliberately creates inference
                    # tensors, as an upstream frozen planner may do.
                    inference_kwargs = {key: value.clone() for key, value in kwargs.items()}
                    actual = model(
                        trajectory.clone(), scene.clone(), route.clone(), **inference_kwargs
                    )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_two_dsm_updates_reach_parameters_in_all_attention_paths(self):
        for parameterization in ("score", "energy"):
            for global_attention, neighbors in ((False, False), (True, False), (False, True)):
                with self.subTest(
                    parameterization=parameterization,
                    global_attention=global_attention,
                    neighbors=neighbors,
                ):
                    model = small_branch(
                        parameterization,
                        temporal_attention=global_attention,
                        neighbor_future=neighbors,
                    ).train()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    clean, scene, route, kwargs = inputs(neighbors=neighbors)
                    before = {name: p.detach().clone() for name, p in model.named_parameters()}
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        epsilon = torch.randn_like(clean)
                        score = model(clean + 0.05 * epsilon, scene, route, **kwargs)
                        loss = fixed_sigma_dsm_loss(score, epsilon, 0.05)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        gradient_mass = 0.0
                        for name, parameter in model.named_parameters():
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                            gradient_mass += parameter.grad.abs().sum().item()
                        self.assertGreater(gradient_mass, 0)
                        optimizer.step()
                    self.assertTrue(
                        any(
                            not torch.equal(before[name], parameter)
                            for name, parameter in model.named_parameters()
                        )
                    )

    def test_fixed_corruption_dsm_improves_with_energy_training(self):
        # A small optimizer smoke test, not a generalization benchmark.
        torch.manual_seed(41)
        model = small_branch().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        clean, scene, route, _ = inputs()
        clean.zero_()
        clean[..., 0] = torch.linspace(0, 1, clean.shape[1])
        clean[..., 2] = 1
        epsilon = torch.randn_like(clean)
        noisy = clean + 0.05 * epsilon
        with torch.no_grad():
            initial = fixed_sigma_dsm_loss(model(noisy, scene, route), epsilon, 0.05).item()
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            loss = fixed_sigma_dsm_loss(model(noisy, scene, route), epsilon, 0.05)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            final = fixed_sigma_dsm_loss(model(noisy, scene, route), epsilon, 0.05).item()
        self.assertLess(final, initial * 0.9)

    def test_energy_refines_with_projection_and_preserves_disabled_identity(self):
        model = small_branch(future_len=80).eval().requires_grad_(False)
        physical, scene, route, _ = inputs(future_len=80)
        physical[..., 2] = 1
        physical[..., 3] = 0
        stats = SimpleNamespace(
            mean=torch.tensor([[[0.1, -0.2, 0.3, -0.1]]]),
            std=torch.tensor([[[2.0, 3.0, 0.7, 1.3]]]),
        )
        initial = normalize_ego_future(physical, stats)
        with torch.inference_mode():
            result, trace = refine_ego(
                model,
                initial,
                scene,
                route,
                stats,
                sigma=0.05,
                gamma=0.1,
                steps=2,
                heading_projection=True,
            )
        self.assertEqual(result.shape, initial.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(trace["completed_steps"], 2)
        self.assertLess(heading_norm_deviation(result, stats).max().item(), 2e-6)
        self.assertFalse(torch.equal(result[..., :2], initial[..., :2]))
        json.dumps(trace, allow_nan=False)

        # A disabled planner must not invoke the energy/gradient path or even
        # require a valid normalizer, irrespective of output parameterization.
        with patch.object(model, "forward", side_effect=AssertionError("branch called")):
            disabled, trace = refine_ego(
                model,
                initial,
                scene,
                route,
                None,
                sigma=0.05,
                gamma=0,
                steps=20,
                heading_projection=True,
            )
        self.assertIs(disabled, initial)
        self.assertTrue(torch.equal(disabled, initial))
        self.assertEqual(trace["completed_steps"], 0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_rank_energy_dsm_updates_synchronize(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
        except OSError as error:
            self.skipTest(f"Local sockets unavailable for Gloo: {error}")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                mp.spawn(energy_ddp_worker, args=(tmp,), nprocs=2, join=True)
            except mp.ProcessRaisedException as error:
                text = str(error).lower()
                restricted_socket = any(
                    marker in text
                    for marker in (
                        "operation not permitted",
                        "permission denied",
                        "address family not supported",
                    )
                )
                if "gloo" in text and "init_process_group" in text and restricted_socket:
                    self.skipTest("Execution environment forbids Gloo socket initialization")
                raise
            result = json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result, {"steps": 2, "updated": True, "synchronized": True})


if __name__ == "__main__":
    unittest.main()
