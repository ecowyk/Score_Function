"""Eight-pose physical interface, deterministic K snapshots and checkpoint identity."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from score_function.model.score_branch import build_model
from score_function.navsim_dd.runtime import CHECKPOINT_FORMAT, NavsimRefiner, poses_to_channels


class KnownScore(nn.Module):
    def forward(self, trajectory, scene, status):
        # Constant in the two spatial dimensions; no angle component.
        result = torch.zeros_like(trajectory)
        result[..., 0] = 2
        result[..., 1] = -1
        return result


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.stats = {"mean": [3., 1., 0., 0.], "std": [10., 5., 1., 1.]}
        self.refiner = NavsimRefiner(KnownScore(), self.stats, 0.1, 0.1, {"id": "a"}, "cpu")
        self.raw = torch.zeros(2, 8, 3)
        self.raw[..., 2] = torch.tensor([3.15, -3.14])[:, None]
        self.scene = torch.zeros(2, 64, 256)
        self.status = torch.zeros(2, 256)

    def test_same_start_physical_units_and_heading(self):
        out = self.refiner.refine_many(self.raw, self.scene, self.status, [20, 0, 1, 5])
        self.assertIs(out[0], self.raw)
        for k in (1, 5, 20):
            self.assertTrue(torch.allclose(out[k][..., 0], torch.full((2, 8), .02*k), atol=1e-5))
            self.assertTrue(torch.allclose(out[k][..., 1], torch.full((2, 8), -.005*k), atol=1e-5))
            torch.testing.assert_close(poses_to_channels(out[k])[..., 2:], poses_to_channels(self.raw)[..., 2:])
        torch.testing.assert_close(out[20], self.refiner.refine(self.raw, self.scene, self.status, 20))
        self.assertEqual(self.raw[..., :2].abs().max().item(), 0.)

    def test_disabled_path_keeps_dtype_and_skips_conditioner(self):
        raw = self.raw.double()
        self.assertIs(self.refiner.refine(raw, None, None, 0), raw)
        self.assertIs(self.refiner.refine(raw, None, None, 20, gamma=0), raw)
        with self.assertRaises(ValueError):
            self.refiner.refine(raw, self.scene[:, :32], self.status, 1)
        with self.assertRaises(ValueError):
            self.refiner.require_conditioner({"id": "other"})

    def test_portable_ema_load_and_mismatched_format(self):
        settings = dict(architecture="temporal_score_function", future_len=8, input_dim=4,
                        context_dim=256, hidden_dim=24, num_heads=3, temporal_attention=True,
                        parameterization="score", neighbor_future=False)
        model = build_model({"model": settings}, torch.device("cpu")).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ema.pt"
            ckpt = {"format": CHECKPOINT_FORMAT, "model": settings, "state_dict": model.state_dict(),
                    "normalizer": self.stats, "sigma": .1, "gamma": .1, "conditioner_identity": {"id": "a"}}
            torch.save(ckpt, path)
            refiner = NavsimRefiner.from_checkpoint(path, "cpu")
            result = refiner.refine(self.raw, self.scene, self.status, 1)
            self.assertEqual(result.shape, (2, 8, 3))
            self.assertTrue(result.isfinite().all())
            ckpt["format"] = "nuplan"
            torch.save(ckpt, path)
            with self.assertRaises(ValueError):
                NavsimRefiner.from_checkpoint(path, "cpu")
