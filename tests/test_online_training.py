"""CPU contracts plus opt-in parity with the actual upstream augmentation code."""

import copy
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from score_function.data_process.online_training import (
    RAW_INPUT_KEYS,
    RAW_KEYS,
    OnlineTrainingProvider,
    RawTrainingDataset,
)
from score_function.utils.dataset import collate_cpu
from score_function.utils.normalizer import normalize_ego_future
from score_function.utils.train_utils import atomic_write, file_hash


def raw_sample():
    """Valid raw DP layouts with both populated and padded entities."""
    row = {
        "ego_current_state": torch.tensor([0.0, 0.0, 1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "ego_agent_future": torch.zeros(80, 3),
        "neighbor_agents_past": torch.zeros(32, 21, 11),
        "neighbor_agents_future": torch.zeros(10, 80, 3),
        "static_objects": torch.zeros(5, 10),
        "lanes": torch.zeros(70, 20, 12),
        "route_lanes": torch.zeros(25, 20, 12),
        "lanes_speed_limit": torch.ones(70, 1),
        "lanes_has_speed_limit": torch.ones(70, 1, dtype=torch.bool),
        "route_lanes_speed_limit": torch.ones(25, 1),
        "route_lanes_has_speed_limit": torch.ones(25, 1, dtype=torch.bool),
    }
    row["ego_agent_future"][:, 0] = torch.arange(1, 81) * 0.6
    row["neighbor_agents_past"][0, :, 0] = 10
    row["neighbor_agents_past"][0, :, 2] = 1
    row["neighbor_agents_future"][0, :, 0] = 10 + torch.arange(1, 81) * 0.5
    for key in ("lanes", "route_lanes"):
        row[key][0, :, 0] = torch.arange(20)
        row[key][0, :, 1] = 2
        row[key][0, :, 2] = 1
    row["static_objects"][0, :4] = torch.tensor([3.0, 4.0, 1.0, 0.0])
    return row


def batch_samples(size=3):
    rows = [
        {
            **{f"raw_{key}": value for key, value in raw_sample().items()},
            "record": {"token": str(i)},
        }
        for i in range(size)
    ]
    return collate_cpu(rows)


def config(root=".", neighbors=False):
    return {
        "paths": {"root": str(root), "manifest": "manifest.json"},
        "model": {"neighbor_future": neighbors},
        "training": {
            "data_augmentation": {
                "enabled": True,
                "probability": 0.5,
                "encoding_batch_size": 2,
            }
        },
    }


class StubAugmentation:
    def __init__(self, augment_prob, device):
        self.threshold = augment_prob

    def __call__(self, raw, ego, neighbors):
        shift = torch.rand(ego.shape[0], device=ego.device)
        for key in ("lanes", "route_lanes", "neighbor_agents_past"):
            raw[key][..., 0] += shift[:, None, None]
        ego[..., 0] += shift[:, None]
        raw["ego_current_state"][:, 4] += shift
        return raw, ego, neighbors


class StubPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.observed = []
        self.decoded = []

    def decoder(self, encoding, inputs):
        self.decoded.append(
            (encoding["encoding"].clone(), {key: value.clone() for key, value in inputs.items()})
        )
        result = torch.zeros(len(inputs["lanes"]), 11, 80, 4)
        result[:, 1:, :, 0] = inputs["neighbor_agents_past"][:, :10, -1, 0, None]
        result[..., 2] = 1
        return {"prediction": result}


def planner_args():
    return SimpleNamespace(
        agent_num=32,
        predicted_neighbor_num=10,
        state_normalizer=SimpleNamespace(mean=torch.zeros(11, 1, 4), std=torch.ones(11, 1, 4)),
        observation_normalizer=lambda raw: {
            key: value * 2 if value.is_floating_point() else value for key, value in raw.items()
        },
    )


def encode_stub(planner, inputs):
    planner.observed.append({key: value.clone() for key, value in inputs.items()})
    signal = inputs["lanes"].mean(dim=(1, 2, 3)) * planner.weight
    return (signal[:, None, None].expand(-1, 107, 192), signal[:, None].expand(-1, 192))


def augmentation_module(cls):
    module = ModuleType("diffusion_planner.utils.data_augmentation")
    module.StatePerturbation = cls
    return module


class OnlineTrainingTests(unittest.TestCase):
    def test_replayed_training_rng_reproduces_online_batch(self):
        planner, args = StubPlanner(), planner_args()
        batch = batch_samples()
        with (
            patch.dict(
                sys.modules,
                {
                    "diffusion_planner.utils.data_augmentation": augmentation_module(
                        StubAugmentation
                    )
                },
            ),
            patch(
                "score_function.data_process.online_training.load_frozen_planner",
                return_value=(planner, args),
            ),
            patch("score_function.data_process.online_training.encode_conditions", encode_stub),
        ):
            provider = OnlineTrainingProvider(config(), "cpu")
            rng = torch.get_rng_state()
            first = provider(batch)
            torch.set_rng_state(rng)
            replay = provider(batch)
        for key in ("target", "context", "route"):
            torch.testing.assert_close(first[key], replay[key], atol=0, rtol=0)
        self.assertEqual(planner.decoded, [])

    def test_fresh_paired_inputs_bounded_encoding_neighbor_refresh_and_no_planner_gradients(self):
        planner, args = StubPlanner(), planner_args()
        original = batch_samples()
        before = {key: value.clone() for key, value in original.items() if torch.is_tensor(value)}
        cfg = config(neighbors=True)
        cfg["training"]["data_augmentation"]["probability"] = 0.75
        with (
            patch.dict(
                sys.modules,
                {
                    "diffusion_planner.utils.data_augmentation": augmentation_module(
                        StubAugmentation
                    )
                },
            ),
            patch(
                "score_function.data_process.online_training.load_frozen_planner",
                return_value=(planner, args),
            ),
            patch("score_function.data_process.online_training.encode_conditions", encode_stub),
        ):
            provider = OnlineTrainingProvider(cfg, "cpu")
            self.assertEqual(provider.augmentation.threshold, 0.25)
            first, second = provider(original), provider(original)
        self.assertFalse(torch.equal(first["target"], second["target"]))
        self.assertEqual([len(row["lanes"]) for row in planner.observed], [2, 1, 2, 1])
        observed = {
            key: torch.cat([row[key] for row in planner.observed[:2]]) for key in RAW_INPUT_KEYS
        }
        # The same per-item shift reaches the ego target and the map condition,
        # with observation normalization applied strictly after augmentation.
        torch.testing.assert_close(
            first["target"][:, 0, 0] - 0.6, observed["lanes"][:, 0, 0, 0] / 2
        )
        self.assertEqual(len(planner.decoded), 4)
        for (context, decoded), encoded in zip(planner.decoded, planner.observed):
            for key in RAW_INPUT_KEYS:
                torch.testing.assert_close(decoded[key], encoded[key], rtol=0, atol=0)
            torch.testing.assert_close(context[:, 0, 0], encoded["lanes"].mean((1, 2, 3)))
        self.assertFalse(torch.equal(first["neighbor_future"], second["neighbor_future"]))
        self.assertFalse(planner.training)
        self.assertFalse(planner.weight.requires_grad)
        branch = torch.nn.Linear(4, 1)
        branch(first["target"]).square().mean().backward()
        self.assertIsNotNone(branch.weight.grad)
        self.assertIsNone(planner.weight.grad)
        self.assertTrue(
            all(not value.requires_grad for value in first.values() if torch.is_tensor(value))
        )
        for key, value in before.items():
            torch.testing.assert_close(original[key], value, rtol=0, atol=0)

    def test_raw_hash_detects_modified_bytes_and_manifest_fallback_checks_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature.npz"
            np.savez(path, **{key: value.numpy() for key, value in raw_sample().items()})
            record = {
                "token": "frame",
                "recording": "recording",
                "split": "train",
                "start_time_us": 123,
                "feature": path.name,
                "feature_sha256": file_hash(path),
            }
            cfg = config(tmp)
            atomic_write(Path(tmp) / "manifest.json", [record])
            cached = SimpleNamespace(
                records=[
                    {key: record[key] for key in ("token", "recording", "split", "start_time_us")}
                ],
                metadata={},
                sha256="cache_hash",
                close=lambda: None,
            )
            dataset = RawTrainingDataset(cached, cfg)
            row = dataset[0]
            self.assertEqual(set(row), {f"raw_{key}" for key in RAW_KEYS} | {"record"})
            np.savez(path, **{key: value.numpy() + 1 for key, value in raw_sample().items()})
            with self.assertRaisesRegex(ValueError, "Changed raw feature"):
                dataset[0]
            mismatch = copy.deepcopy(record)
            mismatch["split"] = "val"
            atomic_write(Path(tmp) / "manifest.json", [mismatch])
            with self.assertRaisesRegex(ValueError, "frame/split identity"):
                RawTrainingDataset(cached, cfg)


@unittest.skipUnless(
    os.environ.get("SCORE_FUNCTION_OFFICIAL_ROOT"),
    "Set SCORE_FUNCTION_OFFICIAL_ROOT for actual augmentation parity",
)
class OfficialAugmentationParityTests(unittest.TestCase):
    def test_exact_pairing_matches_upstream_training_order(self):
        source = Path(os.environ["SCORE_FUNCTION_OFFICIAL_ROOT"])
        # Execute the actual official augmentation source. Only its scalar
        # vehicle-parameter dependency is stubbed, not any augmentation math.
        vehicle = ModuleType("nuplan.common.actor_state.vehicle_parameters")
        vehicle.get_pacifica_parameters = lambda: SimpleNamespace(wheel_base=3.089)
        spec = importlib.util.spec_from_file_location(
            "official_augmentation_parity",
            source / "diffusion_planner/utils/data_augmentation.py",
        )
        official = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {vehicle.__name__: vehicle}):
            spec.loader.exec_module(official)
        planner, args = StubPlanner(), planner_args()
        batch = batch_samples(2)
        cfg = config()
        cfg["training"]["data_augmentation"]["probability"] = 1.0
        with (
            patch.dict(sys.modules, {"diffusion_planner.utils.data_augmentation": official}),
            patch(
                "score_function.data_process.online_training.load_frozen_planner",
                return_value=(planner, args),
            ),
            patch("score_function.data_process.online_training.encode_conditions", encode_stub),
        ):
            provider = OnlineTrainingProvider(cfg, "cpu")
            torch.manual_seed(912)
            output = provider(batch)
            torch.manual_seed(912)
            raw = {key: batch[f"raw_{key}"].clone() for key in RAW_INPUT_KEYS}
            raw, future, _ = official.StatePerturbation(augment_prob=0.0, device="cpu")(
                raw,
                batch["raw_ego_agent_future"].clone(),
                batch["raw_neighbor_agents_future"].clone(),
            )
        packed = torch.cat((future[..., :2], future[..., 2:3].cos(), future[..., 2:3].sin()), -1)
        torch.testing.assert_close(
            output["target"], normalize_ego_future(packed, args.state_normalizer), atol=0, rtol=0
        )
        expected = args.observation_normalizer(raw)
        for key in RAW_INPUT_KEYS:
            torch.testing.assert_close(planner.observed[0][key], expected[key], atol=0, rtol=0)
        self.assertFalse(torch.equal(future, batch["raw_ego_agent_future"]))
        self.assertFalse(torch.equal(raw["lanes"], batch["raw_lanes"]))
        self.assertTrue(
            torch.equal(
                raw["neighbor_agents_past"][:, 1:],
                torch.zeros_like(raw["neighbor_agents_past"][:, 1:]),
            )
        )


if __name__ == "__main__":
    unittest.main()
