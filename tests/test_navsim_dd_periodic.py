"""Periodic portable weights preserve training and recover after export failure."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_navsim_dd_aligned import aligned_config
from test_navsim_dd_training import _tiny_cache
from score_function.navsim_dd import train as trainer
from score_function.navsim_dd.cache import CONDITIONER_KEYS
from score_function.navsim_dd.runtime import NavsimRefiner
from score_function.utils.train_utils import load_tensor, read_json


class NavsimPeriodicCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.identity = {key: key + "-fixture" for key in CONDITIONER_KEYS}
        _tiny_cache(self.cache, self.identity)
        self.device = torch.device("cpu")
        self.cfg = aligned_config(epochs=5, warmup_epochs=1)

    def assert_weights_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0, msg=key)

    def assert_portable(self, path, epoch, weights, kind="raw"):
        saved = load_tensor(path)
        self.assertEqual(saved["epoch"], epoch)
        self.assertEqual(saved["weight_kind"], kind)
        self.assert_weights_equal(weights, saved["state_dict"])
        refiner = NavsimRefiner.from_checkpoint(path, self.device)
        refiner.require_conditioner(self.identity)
        result = refiner.refine_many(
            torch.zeros(1, 8, 3), torch.zeros(1, 64, 256), torch.zeros(1, 256), [1],
        )
        self.assertTrue(torch.isfinite(result[1]).all())

    def test_periodic_options_require_positive_integer_interval_and_start(self):
        trainer._check_config(self.cfg)
        trainer._check_config({**self.cfg, "checkpoint_every_epochs": 2})
        for settings in (
            {"checkpoint_start_epoch": 3},
            *({"checkpoint_every_epochs": value} for value in (0, -1, True, 1.5, None)),
            *({"checkpoint_every_epochs": 2, "checkpoint_start_epoch": value}
              for value in (0, -1, True, 1.5, None)),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                trainer._check_config({**self.cfg, **settings})

    def test_snapshots_match_their_epochs_and_do_not_change_training(self):
        baseline, output = self.root / "baseline", self.root / "periodic"
        trainer.train(self.cache, baseline, self.cfg, self.device)
        cfg = {**self.cfg, "checkpoint_every_epochs": 2, "checkpoint_start_epoch": 3}
        committed = {}
        original_write = trainer.atomic_write

        def record_committed(path, value, **kwargs):
            if Path(path).name == "last.pt":
                committed[len(value["history"])] = {
                    key: tensor.detach().clone() for key, tensor in value["raw"].items()
                }
            return original_write(path, value, **kwargs)

        with patch.object(trainer, "atomic_write", side_effect=record_committed):
            trainer.train(self.cache, output, cfg, self.device)
        self.assertEqual(set(committed), {1, 2, 3, 4, 5})  # Recovery still commits every epoch.
        self.assertFalse((baseline / "checkpoints").exists())
        self.assertEqual(sorted(path.name for path in (output / "checkpoints").iterdir()),
                         ["epoch_0003_raw.pt", "epoch_0005_raw.pt"])
        for epoch in (3, 5):
            self.assert_portable(output / "checkpoints" / f"epoch_{epoch:04d}_raw.pt", epoch, committed[epoch])
        self.assertEqual(read_json(baseline / "history.json"), read_json(output / "history.json"))
        self.assert_weights_equal(load_tensor(baseline / "last.pt")["raw"], load_tensor(output / "last.pt")["raw"])
        self.assert_portable(output / "final_raw.pt", 5, committed[5])

    def test_resume_repairs_snapshot_after_committed_checkpoint_export_failure(self):
        output = self.root / "interrupted"
        cfg = {**self.cfg, "checkpoint_every_epochs": 2, "checkpoint_start_epoch": 3}
        snapshot = output / "checkpoints/epoch_0003_raw.pt"
        original_write = trainer.atomic_write

        def fail_snapshot(path, value, **kwargs):
            if Path(path) == snapshot:
                raise OSError("injected periodic export failure")
            return original_write(path, value, **kwargs)

        with patch.object(trainer, "atomic_write", side_effect=fail_snapshot):
            with self.assertRaisesRegex(RuntimeError, "injected periodic export failure"):
                trainer.train(self.cache, output, cfg, self.device)
        committed = load_tensor(output / "last.pt")
        self.assertEqual(len(committed["history"]), 3)
        self.assertFalse(snapshot.exists())
        self.assertFalse((output / "final_raw.pt").exists())
        original_epoch = trainer.train_epoch

        def require_repair_before_training(*args, **kwargs):
            self.assert_portable(snapshot, 3, committed["raw"])
            return original_epoch(*args, **kwargs)

        # Loading a model for inspection consumes RNG; inspect only in a forked
        # CPU RNG scope so this assertion cannot perturb the resumed training.
        def checked_epoch(*args, **kwargs):
            with torch.random.fork_rng(devices=[]):
                generator = require_repair_before_training(*args, **kwargs)
            return generator

        with patch.object(trainer, "train_epoch", side_effect=checked_epoch):
            trainer.train(self.cache, output, cfg, self.device, resume=True)
        reference = self.root / "uninterrupted"
        trainer.train(self.cache, reference, cfg, self.device)
        self.assertEqual(read_json(reference / "history.json"), read_json(output / "history.json"))
        self.assert_weights_equal(load_tensor(reference / "last.pt")["raw"], load_tensor(output / "last.pt")["raw"])
        self.assert_weights_equal(load_tensor(reference / "checkpoints/epoch_0003_raw.pt")["state_dict"],
                                  load_tensor(snapshot)["state_dict"])

    def test_default_start_and_ema_snapshot_use_the_selected_weight_kind(self):
        output = self.root / "ema"
        cfg = {**self.cfg, "epochs": 2, "ema_decay": 0.9, "checkpoint_every_epochs": 2}
        trainer.train(self.cache, output, cfg, self.device)
        self.assertEqual([path.name for path in (output / "checkpoints").iterdir()], ["epoch_0002_ema.pt"])
        last = load_tensor(output / "last.pt")
        self.assert_portable(output / "checkpoints/epoch_0002_ema.pt", 2, last["ema"], kind="ema")
        self.assertTrue(any(not torch.equal(last["raw"][key], last["ema"][key]) for key in last["raw"]))

    def test_start_after_completed_budget_does_not_create_future_snapshot(self):
        output = self.root / "later"
        cfg = {**self.cfg, "epochs": 2, "checkpoint_every_epochs": 2, "checkpoint_start_epoch": 3}
        trainer.train(self.cache, output, cfg, self.device)
        self.assertFalse((output / "checkpoints").exists())
        self.assertEqual(load_tensor(output / "final_raw.pt")["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
