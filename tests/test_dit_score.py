"""DiT contracts and optional parity with the actual pinned DP source."""

import importlib.util
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from score_function.loss import fixed_sigma_dsm_loss
from score_function.model.dit_score_branch import DiTScoreFunctionBranch
from score_function.model.module.dit import DiTBlock, FinalLayer, Mlp
from score_function.model.refinement import refine_ego
from score_function.model.score_branch import ScoreFunctionBranch, build_model

OFFICIAL_ROOT = os.environ.get("SCORE_FUNCTION_OFFICIAL_ROOT")


class DiTScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(731)
        self.inputs = (torch.randn(2, 80, 4), torch.randn(2, 7, 192), torch.randn(2, 192))

    def model(self):
        return DiTScoreFunctionBranch(hidden_dim=24, num_heads=3, preproj_hidden_dim=48)

    def test_factory_interface_and_official_zero_initialization(self):
        model = build_model(
            {"model": {"architecture": "dit_score_function", "hidden_dim": 24}}, "cpu"
        )
        self.assertIsInstance(model, DiTScoreFunctionBranch)
        self.assertEqual(
            inspect.signature(model.forward).parameters.keys(),
            inspect.signature(ScoreFunctionBranch().forward).parameters.keys(),
        )
        self.assertEqual(model.temporal_receptive_field, 80)
        self.assertFalse(model.uses_neighbor_future)
        self.assertEqual(model.parameterization, "score")
        for module in (*model.blocks, model.final_layer):
            self.assertEqual(module.adaLN_modulation[-1].weight.count_nonzero(), 0)
            self.assertEqual(module.adaLN_modulation[-1].bias.count_nonzero(), 0)
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
        self.assertGreater(model.temporal_embedding.weight.count_nonzero(), 0)
        model.eval()
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                score = model(*self.inputs)
            self.assertEqual(score.shape, (2, 80, 4))
            self.assertEqual(score.count_nonzero(), 0)
            self.assertFalse(score.requires_grad)
        self.assertFalse(any("t_embedder" in name for name in model.state_dict()))

    def test_zero_output_initialization_opens_up_under_dsm_backward(self):
        model = self.model().train()
        frozen_encoder = nn.Linear(5, 192).eval().requires_grad_(False)
        with torch.no_grad():
            scene = frozen_encoder(torch.randn(2, 7, 5))
            route = frozen_encoder(torch.randn(2, 5))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        epsilon = torch.randn_like(self.inputs[0])
        noisy = self.inputs[0] + 0.05 * epsilon
        # An initially zero head and gated self-attention legitimately delay
        # upstream nonzero gradients. Check several actual optimizer updates.
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = fixed_sigma_dsm_loss(model(noisy, scene, route), epsilon, 0.05)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
        self.assertTrue(all(p.grad is None for p in frozen_encoder.parameters()))
        self.assertGreater(model.blocks[0].attn.in_proj_weight.grad.abs().sum(), 0)
        self.assertGreater(model.preproj.fc1.weight.grad.abs().sum(), 0)
        self.assertGreater(model.route_projection.weight.grad.abs().sum(), 0)
        self.assertGreater(model(*self.inputs).abs().sum(), 0)

    def test_full_temporal_dependency_and_both_conditioning_paths(self):
        model = self.model().double().eval()
        # Probe a trained-like state, not the deliberately zero initial score.
        with torch.no_grad():
            nn.init.normal_(model.final_layer.proj[-1].weight, std=0.02)
            for block in model.blocks:
                nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
        inputs = [x[:1].double().requires_grad_() for x in self.inputs]
        grads = torch.autograd.grad(model(*inputs)[0, 40].sum(), inputs)
        self.assertGreater(grads[0][0, 0].abs().sum(), 0)
        self.assertGreater(grads[0][0, -1].abs().sum(), 0)
        self.assertGreater(grads[1].abs().sum(), 0)
        self.assertGreater(grads[2].abs().sum(), 0)

    def test_refinement_uses_direct_score_without_extra_sigma_scaling(self):
        model = self.model().eval().requires_grad_(False)
        nn.init.normal_(model.final_layer.proj[-1].weight, std=0.02)
        stats = SimpleNamespace(mean=torch.zeros(1, 1, 4), std=torch.ones(1, 1, 4))
        actual, trace = refine_ego(
            model,
            *self.inputs,
            stats,
            sigma=0.05,
            gamma=0.25,
            steps=3,
            heading_projection=False,
        )
        expected = self.inputs[0]
        with torch.no_grad():
            for _ in range(3):
                expected = expected + 0.25 * 0.05**2 * model(expected, *self.inputs[1:])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(trace["completed_steps"], 3)
        self.assertGreater((actual - self.inputs[0]).abs().sum(), 0)

    def test_rejects_unsupported_parameterization_and_inputs(self):
        for settings in ({"parameterization": "energy"}, {"depth": 0}, {"mlp_ratio": 0}):
            with self.assertRaises(ValueError):
                DiTScoreFunctionBranch(**settings)
        model = self.model()
        for inputs in (
            (self.inputs[0][:, :79], *self.inputs[1:]),
            (self.inputs[0], self.inputs[1][:, :0], self.inputs[2]),
            (*self.inputs[:2], self.inputs[2][:, :5]),
        ):
            with self.assertRaises(ValueError):
                model(*inputs)
        with self.assertRaisesRegex(ValueError, "neighbor"):
            model(*self.inputs, neighbor_future=torch.zeros(2, 2, 80, 4))


@unittest.skipUnless(OFFICIAL_ROOT, "Set SCORE_FUNCTION_OFFICIAL_ROOT for DP DiT parity")
class OfficialDiTParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        path = Path(OFFICIAL_ROOT) / "diffusion_planner/model/module/dit.py"
        spec = importlib.util.spec_from_file_location("official_dit_parity", path)
        cls.official = importlib.util.module_from_spec(spec)
        # Use real timm, without importing the planner or replacing its layers.
        spec.loader.exec_module(cls.official)

    def assert_module_parity(self, local, official, inputs, **kwargs):
        official.load_state_dict(local.state_dict(), strict=True)
        left = [x.detach().clone().requires_grad_() for x in inputs]
        right = [x.detach().clone().requires_grad_() for x in inputs]
        torch.manual_seed(81)
        actual = local(*left, **kwargs)
        torch.manual_seed(81)
        expected = official(*right, **kwargs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        for first, second in zip(left, right):
            torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
        reference = dict(official.named_parameters())
        for name, parameter in local.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, reference[name].grad, atol=0, rtol=0)

    def test_block_order_mask_dropout_outputs_and_gradients_match_official(self):
        torch.manual_seed(99)
        inputs = (torch.randn(2, 9, 24), torch.randn(2, 7, 24), torch.randn(2, 24))
        mask = torch.zeros(2, 9, dtype=torch.bool)
        mask[0, -2:] = True
        for training in (False, True):
            with self.subTest(training=training):
                # Nonzero modulation at construction exercises all gate paths.
                local = DiTBlock(dim=24, heads=3, dropout=0.1).train(training)
                official = self.official.DiTBlock(dim=24, heads=3, dropout=0.1).train(training)
                self.assert_module_parity(local, official, inputs, attn_mask=mask)

    def test_final_layer_outputs_and_gradients_match_official(self):
        self.assert_module_parity(
            FinalLayer(24, 4),
            self.official.FinalLayer(24, 4),
            (torch.randn(2, 9, 24), torch.randn(2, 24)),
        )

    def test_preprojection_matches_real_timm(self):
        from timm.models.layers import Mlp as TimmMlp

        self.assert_module_parity(
            Mlp(4, 512, 192),
            TimmMlp(
                in_features=4, hidden_features=512, out_features=192, act_layer=nn.GELU, drop=0
            ),
            (torch.randn(2, 80, 4),),
        )


if __name__ == "__main__":
    unittest.main()
