"""Training budget floors, actual accounting, and official LR parity on CPU."""

import copy
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from test_training_pipeline import fixture

from score_function.train import train
from score_function.utils.config import load_config
from score_function.utils.train_utils import load_tensor, read_json
from score_function.utils.training_budget import (
    early_stop_allowed,
    learning_rate_at_step,
    training_budget,
)


class TrainingBudgetTests(unittest.TestCase):
    def test_default_budget_covers_actual_smaller_training_split(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/score_function.json")
        cfg = config["training"]
        budget = training_budget(cfg, 950000)
        self.assertEqual(budget["updates_per_epoch"], 463)
        self.assertEqual(budget["planned_epochs"], 527)
        self.assertEqual(budget["planned_optimizer_updates"], 244001)
        self.assertEqual(budget["planned_sample_presentations"], 244001 * 2048)
        self.assertFalse(early_stop_allowed(cfg, 527, 244001))

    def test_legacy_budget_and_independent_minimum_gates(self):
        cfg = {"max_epochs": 30, "minimum_epochs": 5, "batch_size": 4}
        self.assertEqual(training_budget(cfg, 10)["planned_epochs"], 30)
        self.assertFalse(early_stop_allowed(cfg, 4, 999))
        self.assertTrue(early_stop_allowed(cfg, 5, 10))
        cfg["minimum_updates"] = 20
        self.assertFalse(early_stop_allowed(cfg, 20, 19))
        self.assertTrue(early_stop_allowed(cfg, 5, 20))

    def test_official_epoch_warmup_matches_pytorch_scheduler(self):
        cfg = {
            "lr_schedule": "constant_after_warmup",
            "warmup_epochs": 5,
            "learning_rate": 5e-4,
            "warmup_learning_rate": 5e-5,
        }
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=cfg["learning_rate"])
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=4),
                torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 1.0),
            ],
            milestones=[5],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for epoch in range(8):
                for offset in range(3):
                    self.assertAlmostEqual(
                        learning_rate_at_step(cfg, epoch * 3 + offset, 3, 1e-7),
                        optimizer.param_groups[0]["lr"],
                        places=14,
                    )
                optimizer.step()
                scheduler.step()

    def test_update_floor_prevents_plateau_stop_and_survives_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            cfg = config["training"]
            cfg.update(max_epochs=3, minimum_epochs=1, minimum_updates=7)
            cfg["plateau"].update(lr_patience=1, stop_patience=2)
            # Constant validation produces a stop request at epoch 3 / step 6.
            # The update floor requires another complete epoch, reaching step 8.
            with patch(
                "score_function.train.validate_epoch", return_value={"raw": 1.0, "ema": 1.0}
            ):
                train(config, device_override="cpu")
                full = load_tensor(Path(config["output"]) / "score/last.pt")
                self.assertEqual((full["epoch"], full["step"]), (4, 8))
                self.assertEqual(full["sample_presentations"], 32)
                self.assertEqual(full["training_budget"]["planned_epochs"], 4)
                status = read_json(Path(config["output"]) / "score/status.json")
                self.assertEqual(status["sample_presentations"], 32)
                config["output"] = str(Path(tmp) / "resumed")
                train(config, device_override="cpu", stop_after_updates=3)
                train(config, resume=True, device_override="cpu")
                resumed = load_tensor(Path(config["output"]) / "score/last.pt")
            self.assertEqual(resumed["sample_presentations"], full["sample_presentations"])
            for key, value in full["score_branch"].items():
                torch.testing.assert_close(value, resumed["score_branch"][key], rtol=0, atol=0)

    def test_legacy_config_without_new_fields_still_loads(self):
        path = Path(__file__).resolve().parents[1] / "configs/score_function.json"
        legacy = copy.deepcopy(read_json(path))
        for key in ("minimum_updates", "lr_schedule", "early_stopping", "data_augmentation"):
            legacy["training"].pop(key)
        with tempfile.TemporaryDirectory() as tmp:
            import json

            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(legacy))
            cfg = load_config(path)["training"]
        self.assertNotIn("minimum_updates", cfg)
        self.assertEqual(training_budget(cfg, 4096)["planned_epochs"], 500)

    def test_fresh_run_refusal_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            output = Path(config["output"]) / "score"
            output.mkdir(parents=True)
            status = output / "status.json"
            status.write_text('{"state": "complete", "step": 123}\n')
            previous = status.read_bytes()
            with self.assertRaises(FileExistsError):
                train(config, device_override="cpu")
            self.assertEqual(status.read_bytes(), previous)
            self.assertFalse((output / "failure_rank0.json").exists())


if __name__ == "__main__":
    unittest.main()
