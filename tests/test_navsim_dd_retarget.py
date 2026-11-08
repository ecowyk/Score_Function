"""Retarget an interrupted continuation without replaying committed epochs."""

import fcntl
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import test_navsim_dd_continuation as fixtures
from test_navsim_dd_aligned import InterruptedTraining, aligned_config, interrupt_at_epoch
from test_navsim_dd_training import _tiny_cache
from score_function.navsim_dd import continuation, train as trainer
from score_function.navsim_dd.cache import CONDITIONER_KEYS
from score_function.utils.train_utils import atomic_write, file_hash, load_tensor, read_json


class NavsimRetargetTests(unittest.TestCase):
    assert_state_equal = fixtures.NavsimContinuationTests.assert_state_equal

    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.previous, self.cache = (self.root / name for name in ("base", "previous", "cache"))
        self.cfg = aligned_config(epochs=4, warmup_epochs=1)
        self.device = torch.device("cpu")
        self.options = dict(target_epochs=8, checkpoint_every_epochs=2, checkpoint_start_epoch=6)
        _tiny_cache(self.cache, {key: key + "-fixture" for key in CONDITIONER_KEYS})
        trainer.train(self.cache, self.source, self.cfg, self.device)

    def prepare_previous(self, train_one=False):
        plan = continuation.prepare_continuation(self.source, self.previous, self.cache, additional_epochs=2)
        if train_one:
            with patch.object(trainer, "train_epoch", side_effect=interrupt_at_epoch(2)):
                with self.assertRaises(InterruptedTraining):
                    trainer.train(self.cache, self.previous, plan["config"], self.device, resume=True)
        return plan

    def test_latest_durable_state_matches_continuous_training_and_preserves_sources(self):
        self.prepare_previous(train_one=True)
        previous_last = load_tensor(self.previous / "last.pt")
        self.assertEqual(len(previous_last["history"]), 5)
        # Simulate termination after last.pt was committed but before history.json.
        atomic_write(self.previous / "history.json", read_json(self.source / "history.json"))
        snapshots = {path: fixtures.file_snapshot(path) for path in (self.source, self.previous)}
        target = self.root / "retargeted"
        plan = continuation.prepare_retarget_continuation(
            self.source, self.previous, target, self.cache, **self.options)
        provenance = plan["provenance"]
        self.assertEqual(provenance["source_output"], str(self.source))
        self.assertEqual(provenance["source_epochs"], 4)
        self.assertEqual(provenance["additional_epochs"], 4)
        self.assertEqual(provenance["target_epochs"], 8)
        self.assertEqual(provenance["restore_output"], str(self.previous))
        self.assertEqual(provenance["restored_epoch"], 5)
        self.assertEqual(provenance["restore_sha256"]["last.pt"], file_hash(self.previous / "last.pt"))
        self.assertEqual(plan["config"]["lr_schedule_epochs"], 4)
        for key in ("raw", "ema", "optimizer", "scaler", "rng_by_rank", "history", "best_weights", "step"):
            self.assert_state_equal(previous_last[key], plan["last"][key], key)
        self.assertEqual(len(read_json(target / "history.json")), 5)
        full = self.root / "continuous"
        trainer.train(self.cache, full, plan["config"], self.device)
        trainer.train(self.cache, target, plan["config"], self.device, resume=True)
        expected, actual = load_tensor(full / "last.pt"), load_tensor(target / "last.pt")
        for key in ("raw", "ema", "optimizer", "scaler", "rng_by_rank", "history", "best_weights", "step"):
            self.assert_state_equal(expected[key], actual[key], key)
        self.assertEqual(read_json(full / "history.json"), read_json(target / "history.json"))
        for name in ("final_raw.pt", "best_raw.pt", "final_ema.pt", "best_ema.pt"):
            self.assert_state_equal(load_tensor(full / name), load_tensor(target / name), name)
        for path, before in snapshots.items():
            self.assertEqual(fixtures.file_snapshot(path), before)

    def test_prepared_previous_without_trainer_lock_remains_unchanged(self):
        previous_plan = self.prepare_previous()
        self.assertFalse((self.previous / ".train.lock").exists())
        before = fixtures.file_snapshot(self.previous)
        target = self.root / "retargeted"
        plan = continuation.prepare_retarget_continuation(
            self.source, self.previous, target, self.cache, **self.options)
        self.assertEqual(plan["provenance"]["restored_epoch"], 4)
        for key in ("raw", "optimizer", "rng_by_rank", "scaler", "best_weights", "history", "step"):
            self.assert_state_equal(previous_plan["last"][key], plan["last"][key], key)
        self.assertEqual(fixtures.file_snapshot(self.previous), before)
        self.assertFalse((self.previous / ".train.lock").exists())

    def test_no_previous_starts_at_original_and_existing_target_is_rejected(self):
        before = fixtures.file_snapshot(self.source)
        target = self.root / "retargeted"
        plan = continuation.prepare_retarget_continuation(
            self.source, None, target, self.cache, **self.options)
        self.assertEqual(plan["provenance"]["restore_output"], str(self.source))
        self.assertEqual(plan["provenance"]["restored_epoch"], 4)
        self.assertEqual(read_json(target / "config.json"), plan["config"])
        self.assertEqual(plan["config"]["checkpoint_every_epochs"], 2)
        self.assertEqual(plan["config"]["checkpoint_start_epoch"], 6)
        target_before = fixtures.file_snapshot(target)
        with self.assertRaises(FileExistsError):
            continuation.prepare_retarget_continuation(
                self.source, None, target, self.cache, **self.options)
        self.assertEqual(fixtures.file_snapshot(target), target_before)
        self.assertEqual(fixtures.file_snapshot(self.source), before)

    def test_previous_history_signature_provenance_and_rng_must_match(self):
        self.prepare_previous(train_one=True)
        for case in ("history_prefix", "signature", "provenance", "rng", "best_shape", "optimizer", "missing_lock"):
            with self.subTest(case=case):
                previous = self.root / case
                shutil.copytree(self.previous, previous)
                last = load_tensor(previous / "last.pt")
                if case == "history_prefix":
                    last["history"][0]["train_dsm"] += 1
                    atomic_write(previous / "history.json", last["history"])
                elif case == "signature":
                    last["signature"] = "wrong-experiment"
                elif case == "rng":
                    last["rng_by_rank"] = []
                elif case == "best_shape":
                    key = next(iter(last["best_weights"]))
                    last["best_weights"][key] = torch.zeros(1)
                elif case == "optimizer":
                    last["optimizer"]["state"] = {}
                elif case == "provenance":
                    provenance = read_json(previous / "continuation.json")
                    provenance["source_signature"] = "different-base-run"
                    atomic_write(previous / "continuation.json", provenance)
                else:
                    (previous / ".train.lock").unlink()
                atomic_write(previous / "last.pt", last, tensor=True)
                before = fixtures.file_snapshot(previous)
                target = self.root / (case + "-target")
                with self.assertRaises(ValueError):
                    continuation.prepare_retarget_continuation(
                        self.source, previous, target, self.cache, **self.options)
                self.assertFalse(target.exists())
                self.assertEqual(fixtures.file_snapshot(previous), before)

    def test_existing_previous_trainer_lock_is_enforced(self):
        self.prepare_previous(train_one=True)
        with (self.previous / ".train.lock").open("r+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "active"):
                continuation.plan_retarget_continuation(
                    self.source, self.previous, self.cache, **self.options)


if __name__ == "__main__":
    unittest.main()
