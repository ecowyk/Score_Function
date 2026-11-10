"""Conditional-score behavior with full, variable-length VLM observations."""

import unittest

import torch

from score_function.navsim_chainflow.model import ChainFlowScore


def small_model_config(dropout=0.0):
    return {
        "architecture": "chainflow_scene_score",
        "parameterization": "score",
        "future_len": 8,
        "input_dim": 4,
        "context_dim": 256,
        "vlm_dim": 1536,
        "semantic_tokens": 2,
        "semantic_heads": 8,
        "hidden_dim": 12,
        "num_heads": 3,
        "dropout": dropout,
        "pre_dilations": [1],
        "post_dilations": [1],
        "temporal_attention": True,
        "neighbor_future": False,
    }


def inputs(batch=2, length=5):
    generator = torch.Generator().manual_seed(731)
    return {
        "noisy": torch.randn(batch, 8, 4, generator=generator),
        "scene": torch.randn(batch, 64, 256, generator=generator),
        "ego": torch.randn(batch, 256, generator=generator),
        "vlm": torch.randn(batch, length, 1536, generator=generator),
        "vlm_mask": torch.ones(batch, length, dtype=torch.bool),
    }


class ChainFlowModelTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(37)
        self.model = ChainFlowScore(small_model_config()).eval()

    def test_padding_has_no_influence_on_score(self):
        sample = inputs()
        reference = self.model(**sample)
        padded = dict(sample)
        padded["vlm"] = torch.cat((sample["vlm"], torch.full((2, 7, 1536), 100.0)), dim=1)
        padded["vlm_mask"] = torch.cat((sample["vlm_mask"], torch.zeros(2, 7, dtype=torch.bool)), dim=1)
        actual = self.model(**padded)
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
        self.assertEqual(tuple(actual.shape), (2, 8, 4))

    def test_every_condition_branch_changes_predictions_and_receives_gradients(self):
        sample = inputs()
        reference = self.model(**sample)
        for name in ("scene", "ego", "vlm"):
            altered = dict(sample)
            altered[name] = torch.flip(sample[name], (-1,)) * 1.7
            self.assertGreater((self.model(**altered) - reference).abs().max().item(), 1e-7, name)
        differentiable = dict(sample)
        for name in ("scene", "ego", "vlm"):
            differentiable[name] = sample[name].clone().requires_grad_()
        self.model(**differentiable).square().mean().backward()
        for name in ("scene", "ego", "vlm"):
            gradient = differentiable[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_invalid_or_empty_vlm_mask_is_rejected(self):
        sample = inputs()
        sample["vlm_mask"][0] = False
        with self.assertRaises(ValueError):
            self.model(**sample)
        sample = inputs()
        sample["vlm_mask"] = torch.ones(2, 4, dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.model(**sample)

    def test_small_batch_dsm_can_be_learned(self):
        # This detects disconnected conditioning/model outputs, not convergence
        # of the real dataset. The same fixed corruption is deliberately reused.
        sample = inputs(batch=2, length=4)
        epsilon = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(992))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002, weight_decay=0.0)
        initial = (0.1 * self.model(**sample) + epsilon).square().mean().item()
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = (0.1 * self.model(**sample) + epsilon).square().mean()
            loss.backward()
            optimizer.step()
        final = (0.1 * self.model(**sample) + epsilon).square().mean().item()
        self.assertLess(final, 0.65 * initial)

    def test_conditioning_is_deterministic_in_eval_mode(self):
        model = ChainFlowScore(small_model_config(dropout=0.2)).eval()
        sample = inputs()
        torch.testing.assert_close(model(**sample), model(**sample), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            ChainFlowScore({**small_model_config(), "parameterization": "energy"})


if __name__ == "__main__":
    unittest.main()
