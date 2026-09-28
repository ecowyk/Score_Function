"""Configuration/checkpoint compatibility and real energy training-loop checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import test_training_pipeline as pipeline
import torch

from score_function.model.score_branch import build_model
from score_function.train import train
from score_function.train_epoch import validate_epoch
from score_function.utils.checkpoint import load_selected
from score_function.utils.config import load_config
from score_function.utils.dataset import ShardedDataset
from score_function.utils.train_utils import atomic_write, load_tensor


class ParameterizationPipelineTests(unittest.TestCase):
    def test_legacy_config_override_and_invalid_parameterization(self):
        source = Path(__file__).resolve().parents[1] / "configs/score_function.json"
        with tempfile.TemporaryDirectory() as tmp:
            legacy = json.loads(source.read_text())
            legacy["model"].pop("parameterization")
            path = Path(tmp) / "legacy.json"
            atomic_write(path, legacy)
            self.assertEqual(load_config(path)["model"]["parameterization"], "score")
            selected = load_config(path, overrides=['model.parameterization="energy"'])
            self.assertEqual(selected["model"]["parameterization"], "energy")
            with self.assertRaisesRegex(ValueError, "parameterization"):
                load_config(path, overrides=['model.parameterization="unknown"'])

    def test_legacy_selected_checkpoint_loads_with_identical_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = pipeline.fixture(tmp)
            legacy = copy.deepcopy(config)
            legacy["model"].pop("parameterization")
            torch.manual_seed(123)
            model = build_model(legacy, "cpu").eval().requires_grad_(False)
            checkpoint = Path(tmp) / "legacy.pt"
            atomic_write(
                checkpoint,
                {
                    "schema_version": 1,
                    "method": config["method"],
                    "weight_kind": "ema",
                    "sigma_score": config["training"]["sigma"],
                    "config": legacy,
                    "score_branch": model.state_dict(),
                },
                tensor=True,
            )
            loaded, _ = load_selected(config, checkpoint, "cpu")
            inputs = (torch.randn(2, 80, 4), torch.randn(2, 7, 192), torch.randn(2, 192))
            with torch.no_grad():
                torch.testing.assert_close(loaded(*inputs), model(*inputs), rtol=0, atol=0)
            config["model"]["parameterization"] = "energy"
            with self.assertRaisesRegex(ValueError, "parameterization"):
                load_selected(config, checkpoint, "cpu")

    def test_energy_training_validation_checkpoint_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = pipeline.fixture(tmp)
            config["model"]["parameterization"] = "energy"
            train(config, device_override="cpu")
            full = load_tensor(Path(config["output"]) / "score/last.pt")
            self.assertEqual(full["config"]["model"]["parameterization"], "energy")
            checkpoint = Path(config["output"]) / "score/best.pt"
            model, selected = load_selected(config, checkpoint, "cpu")
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            dataset = ShardedDataset(config["cache"], "val")
            try:
                loss = validate_epoch(
                    {"selected": model}, dataset, config["training"], torch.device("cpu")
                )
            finally:
                dataset.close()
            self.assertEqual(loss["selected"], selected["val_dsm"])
            wrong = copy.deepcopy(config)
            wrong["model"]["parameterization"] = "score"
            with self.assertRaisesRegex(ValueError, "parameterization"):
                load_selected(wrong, checkpoint, "cpu")

            config["output"] = str(Path(tmp) / "resumed")
            train(config, device_override="cpu", stop_after_updates=1)
            train(config, resume=True, device_override="cpu")
            resumed = load_tensor(Path(config["output"]) / "score/last.pt")
            pipeline.PipelineTests._assert_state_equal(self, full, resumed)
            changed = copy.deepcopy(config)
            changed["model"]["parameterization"] = "score"
            with self.assertRaisesRegex(ValueError, "Resume configuration"):
                train(changed, resume=True, device_override="cpu")


if __name__ == "__main__":
    unittest.main()
