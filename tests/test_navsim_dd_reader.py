"""Native DD prefix parity, frozen semantics and noisy-trajectory conditioning."""

import copy
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from score_function.navsim_dd.reader import (
    ARCHITECTURE, DDReaderScore, FrozenDDReader, build_reader_score,
    export_reader_assets, verify_native_reader,
)


def native_position_embedding(pos_tensor, hidden_dim=256):
    """Independent original-DD positional-expression fixture."""
    half_hidden_dim = hidden_dim // 2
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    pos_x = pos_tensor[..., 0, None] * (2 * math.pi) / dim_t
    pos_y = pos_tensor[..., 1, None] * (2 * math.pi) / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


class NativeSpatial(nn.Module):
    """Original module contract: perform value projection inside each call."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.attention_weights = nn.Linear(256, 8)
        self.output_proj = nn.Linear(256, 256)
        self.dropout = nn.Dropout(.1)
        self.value_proj = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

    def forward(self, query, points, bev, shape):
        batch, candidates, count, _ = points.shape
        normalized = points.clone()
        normalized[..., 0] /= self.config.lidar_max_y
        normalized[..., 1] /= self.config.lidar_max_x
        normalized = normalized[..., [1, 0]]
        weights = self.attention_weights(query).view(batch, candidates, count).softmax(-1)
        values = self.value_proj(bev)
        sampled = F.grid_sample(values, normalized.view(batch, candidates, count, 2),
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        out = (weights[:, None] * sampled).sum(-1).permute(0, 2, 1).contiguous()
        return self.dropout(self.output_proj(out)) + query


class NativeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.c5_conv = nn.Conv2d(512, 16, 1)
        self.up_conv5 = nn.Conv2d(16, 16, 3, padding=1)
        self.up_conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsample2 = nn.Upsample(size=(8, 8), mode="bilinear", align_corners=False)
        self.relu = nn.ReLU()

    def top_down(self, latent):
        p5 = self.relu(self.c5_conv(latent))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        return self.relu(self.up_conv4(self.upsample2(p4)))


def make_native_agent():
    """Independently assembled native field layout with small spatial workload."""
    config = SimpleNamespace(
        tf_d_model=256, tf_num_head=8, tf_d_ffn=64, tf_dropout=0., tf_num_layers=1,
        num_bounding_boxes=3, bev_features_channels=16, lidar_resolution_height=32,
        lidar_resolution_width=32, bev_down_sample_factor=4, bev_upsample_factor=2,
        # Unequal extents catch swapped-axis/normalization errors.
        lidar_max_x=24., lidar_max_y=40.,
    )
    agent = nn.Module()
    model = agent._transfuser_model = nn.Module()
    model._config = config
    model._query_splits = [1, 3]
    model._backbone = NativeBackbone()
    model._bev_downscale = nn.Conv2d(512, 256, 1)
    model._keyval_embedding = nn.Embedding(65, 256)
    model._query_embedding = nn.Embedding(4, 256)
    model._tf_decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(256, 8, 64, 0., batch_first=True), 1,
    )
    model.bev_proj = nn.Sequential(nn.Linear(272, 256), nn.ReLU(), nn.LayerNorm(256))
    head = model._trajectory_head = nn.Module()
    head.plan_anchor_encoder = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                             nn.LayerNorm(256), nn.Linear(256, 256))
    head.diff_decoder = nn.Module()
    layer = nn.Module()
    layer.cross_bev_attention = NativeSpatial(config)
    layer.cross_agent_attention = nn.MultiheadAttention(256, 8, dropout=0., batch_first=True)
    layer.cross_ego_attention = nn.MultiheadAttention(256, 8, dropout=0., batch_first=True)
    layer.dropout = nn.Dropout(.1)
    layer.dropout1 = nn.Dropout(.1)
    layer.norm1, layer.norm2, layer.norm3 = (nn.LayerNorm(256) for _ in range(3))
    layer.ffn = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 256))
    head.diff_decoder.layers = nn.ModuleList([layer])
    return agent.eval().requires_grad_(False)


def make_assets():
    """Reusable CPU fixture for training/runtime integration tests."""
    return export_reader_assets(make_native_agent())


def model_config():
    return dict(architecture=ARCHITECTURE, parameterization="score", future_len=8,
                input_dim=4, context_dim=256, hidden_dim=24, num_heads=3,
                dropout=0., pre_dilations=[1], post_dilations=[1],
                temporal_attention=True, neighbor_future=False)


NORMALIZER = {"mean": [4., -1., 0., 0.], "std": [7., 2., 1., 1.]}


class ReaderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)
        self.agent = make_native_agent()
        self.assets = export_reader_assets(self.agent)
        self.latent = torch.randn(2, 512, 8, 8)
        self.status = torch.randn(2, 256)
        self.noisy = torch.randn(2, 8, 4)

    def test_native_scene_and_full_trajectory_prefix_parity_and_rng(self):
        before = torch.get_rng_state().clone()
        fake_modules = {"navsim.agents.diffusiondrive.modules.blocks":
                        SimpleNamespace(gen_sineembed_for_position=native_position_embedding)}
        xy = self.noisy[..., :2] * torch.tensor([7., 2.]) + torch.tensor([4., -1.])
        with patch.dict("sys.modules", fake_modules):
            errors = verify_native_reader(self.agent, self.latent, self.status, xy, self.assets)
            # The same cached FP16 data enters both paths for cache-contract parity.
            errors16 = verify_native_reader(self.agent, self.latent.half(), self.status.half(), xy, self.assets)
        self.assertEqual(set(errors), {"context_max_abs_error", "ego_query_max_abs_error",
                                      "agent_queries_max_abs_error", "reader_max_abs_error"})
        self.assertLess(max(errors.values()), 1e-6)
        self.assertLess(max(errors16.values()), 1e-6)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_reader_is_frozen_while_score_learns_and_sees_noisy_coordinates(self):
        model = build_reader_score(model_config(), NORMALIZER, self.assets, "cpu").train()
        self.assertTrue(model.score.training)
        self.assertFalse(any(module.training for module in model.reader.modules()))
        before = {k: v.clone() for k, v in model.reader.state_dict().items()}
        observed = []
        original = model.reader.read_trajectory

        def capture(physical_xy, scene):
            observed.append(physical_xy.clone())
            return original(physical_xy, scene)

        with patch.object(model.reader, "read_trajectory", side_effect=capture):
            result = model(self.noisy, self.latent.half(), self.status.half())
        self.assertEqual(tuple(result.shape), (2, 8, 4))
        expected = (self.noisy * torch.tensor(NORMALIZER["std"]) + torch.tensor(NORMALIZER["mean"]))[..., :2]
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
        loss = (.1 * result + torch.randn_like(result)).square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.reader.parameters()))
        self.assertTrue(all(parameter.grad is not None and parameter.grad.isfinite().all()
                            for parameter in model.score.parameters()))
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        optimizer.step()
        for key, value in before.items():
            torch.testing.assert_close(model.reader.state_dict()[key], value, rtol=0, atol=0)

    def test_scene_preparation_reused_but_read_location_updates(self):
        model = build_reader_score(model_config(), NORMALIZER, self.assets, "cpu").eval()
        scene = model.prepare_scene(self.latent, self.status)
        self.assertFalse(any(v.requires_grad for v in scene.values()))
        xy = self.noisy[..., :2]
        first = model.reader.read_trajectory(xy, scene)
        second = model.reader.read_trajectory(xy + 2., scene)
        self.assertGreater((first - second).abs().max().item(), 1e-4)
        direct = model(self.noisy, self.latent, self.status)
        cached = model.score_prepared(self.noisy, scene)
        torch.testing.assert_close(direct, cached, rtol=0, atol=0)
        shifted = model.score_prepared(self.noisy + .2, scene)
        self.assertGreater((cached - shifted).abs().max().item(), 1e-4)

    def test_portable_checkpoint_recreates_frozen_reader_without_dd_imports(self):
        model = build_reader_score(model_config(), NORMALIZER, self.assets, "cpu").eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "portable.pt"
            torch.save({"spec": model.reader_spec, "state": model.state_dict()}, path)
            saved = torch.load(path, weights_only=True)
            restored = DDReaderScore(model_config(), NORMALIZER, saved["spec"]).eval()
            restored.load_state_dict(saved["state"], strict=True)
        torch.testing.assert_close(model(self.noisy, self.latent, self.status),
                                   restored(self.noisy, self.latent, self.status), rtol=0, atol=0)

    def test_compatibility_rejects_wrong_architecture_assets_and_dimensions(self):
        corrupt = copy.deepcopy(self.assets)
        corrupt["dd_commit"] = "different"
        with self.assertRaises(ValueError):
            build_reader_score(model_config(), NORMALIZER, corrupt, "cpu")
        with self.assertRaises(ValueError):
            DDReaderScore({**model_config(), "parameterization": "energy"}, NORMALIZER, self.assets["spec"])
        reader = FrozenDDReader(self.assets["spec"])
        with self.assertRaises(ValueError):
            reader.prepare_scene(self.latent[:, :, :7], self.status)
        with self.assertRaises(ValueError):
            reader.prepare_scene(self.latent, self.status[:, :128])


if __name__ == "__main__":
    unittest.main()
