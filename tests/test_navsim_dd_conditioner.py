"""Exercise the native memory contract without NAVSIM data or CUDA."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from score_function.navsim_dd.conditioner import DDConditioner, load_diffusiondrive, verify_native_conditions


class _Backbone(nn.Module):
    def forward(self, image, lidar):
        return None, image + lidar, None


class _Decoder(nn.Module):
    def forward(self, query, memory):
        return query + memory.mean(dim=1, keepdim=True)


class _Agent(nn.Module):
    """Synthetic planner independently assembling its native decoder memory."""

    def __init__(self):
        super().__init__()
        model = self._transfuser_model = nn.Module()
        model._backbone = _Backbone()
        model._bev_downscale = nn.Identity()
        model._keyval_embedding = nn.Embedding(65, 256)
        model._status_encoding = nn.Linear(8, 256)
        model._tf_decoder = _Decoder()
        self.planner_calls = 0
        self.simulate_failure = False

    def forward(self, features):
        self.planner_calls += 1
        model = self._transfuser_model
        _, bev, _ = model._backbone(features["camera_feature"], features["lidar_feature"])
        status = model._status_encoding(features["status_feature"])
        scene = model._bev_downscale(bev).reshape(bev.shape[0], 256, 64).transpose(1, 2)
        memory = torch.cat([scene, status.unsqueeze(1)], 1) + model._keyval_embedding.weight
        # The real planner draws random anchor noise in its trajectory decoder.
        query = torch.randn_like(status).unsqueeze(1)
        output = model._tf_decoder(query, memory)
        if self.simulate_failure:
            raise RuntimeError("simulated agent failure")
        return output


class DDConditionerTest(unittest.TestCase):
    def setUp(self):
        self.agent = _Agent()
        self.features = {
            "camera_feature": torch.randn(2, 256, 8, 8),
            "lidar_feature": torch.randn(2, 256, 8, 8),
            "status_feature": torch.randn(2, 8),
        }

    def test_extracts_native_tokens_without_sampling_or_updating_encoder(self):
        conditioner = DDConditioner(self.agent)
        rng = torch.get_rng_state().clone()
        scene, global_context = conditioner.encode_features(self.features)
        self.assertEqual(self.agent.planner_calls, 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(scene.requires_grad)
        self.assertFalse(self.agent.training)
        self.assertFalse(any(p.requires_grad for p in self.agent.parameters()))
        self.assertEqual(tuple(scene.shape), (2, 64, 256))
        self.assertEqual(tuple(global_context.shape), (2, 256))
        # A particular non-origin cell tests row/column order and addition of
        # the learned positional embeddings. Status must exclude the position.
        raw = self.features["camera_feature"] + self.features["lidar_feature"]
        model = self.agent._transfuser_model
        torch.testing.assert_close(scene[:, 19], raw[:, :, 2, 3] + model._keyval_embedding.weight[19])
        torch.testing.assert_close(global_context, model._status_encoding(self.features["status_feature"]))
        report = verify_native_conditions(self.agent, self.features, rtol=0, atol=0)
        self.assertEqual(report["scene_max_abs_error"], 0.0)
        self.assertEqual(report["status_max_abs_error"], 0.0)
        self.assertEqual(self.agent.planner_calls, 1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_diagnostic_removes_hooks_and_restores_rng_on_native_error(self):
        self.agent.simulate_failure = True
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "simulated agent failure"):
            verify_native_conditions(self.agent, self.features)
        model = self.agent._transfuser_model
        self.assertFalse(model._tf_decoder._forward_pre_hooks)
        self.assertFalse(model._status_encoding._forward_hooks)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_loader_rejects_partial_checkpoint_and_loads_exact_native_weights(self):
        class NativeAgent(nn.Module):
            def __init__(self, config, lr, checkpoint_path):
                super().__init__()
                self.encoder = nn.Linear(2, 2)

        modules = {
            "navsim.agents.diffusiondrive.transfuser_agent": SimpleNamespace(TransfuserAgent=NativeAgent),
            "navsim.agents.diffusiondrive.transfuser_config": SimpleNamespace(TransfuserConfig=SimpleNamespace),
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", modules):
            checkpoint, anchor, backbone = (Path(tmp) / name for name in ("dd.ckpt", "anchor", "r34"))
            anchor.touch()
            backbone.touch()
            source = NativeAgent(None, None, None)
            state = {"agent." + key: value for key, value in source.state_dict().items()}
            torch.save({"state_dict": state}, checkpoint)
            agent = load_diffusiondrive(checkpoint, anchor, backbone, "cpu")
            self.assertFalse(agent.training)
            self.assertFalse(any(p.requires_grad for p in agent.parameters()))
            torch.testing.assert_close(agent.encoder.weight, source.encoder.weight)
            del state["agent.encoder.bias"]
            torch.save({"state_dict": state}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_diffusiondrive(checkpoint, anchor, backbone, "cpu")
