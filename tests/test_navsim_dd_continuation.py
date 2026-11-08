"""Completed-run forks retain the original LR horizon and exact resume state."""

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_navsim_dd_aligned import InterruptedTraining, aligned_config, interrupt_at_epoch
from test_navsim_dd_training import _tiny_cache
from score_function.navsim_dd import continuation, train as trainer
from score_function.navsim_dd.cache import CONDITIONER_KEYS
from score_function.utils.train_utils import atomic_write, load_tensor, read_json
from score_function.utils.training_budget import learning_rate_at_step


def file_snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


class NavsimContinuationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache, self.source = self.root / "cache", self.root / "source"
        self.identity = {key: key + "-fixture" for key in CONDITIONER_KEYS}
        self.cfg = aligned_config(epochs=4, warmup_epochs=1)
        self.device = torch.device("cpu")

    def create_source(self):
        _tiny_cache(self.cache, self.identity)
        trainer.train(self.cache, self.source, self.cfg, self.device)

    def assert_state_equal(self, expected, actual, location="state"):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0, msg=location)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual, err_msg=location)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys(), location)
            for key in expected:
                self.assert_state_equal(expected[key], actual[key], f"{location}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(expected), type(actual), location)
            self.assertEqual(len(expected), len(actual), location)
            for index, value in enumerate(expected):
                self.assert_state_equal(value, actual[index], f"{location}[{index}]")
        else:
            self.assertEqual(expected, actual, location)

    def test_extended_budget_preserves_all_original_rates_then_stays_at_floor(self):
        original = aligned_config(epochs=100)
        continued = {**original, "epochs": 250, "lr_schedule_epochs": 100}
        for epoch in range(100):
            for offset in (0, 2):
                step = epoch * 3 + offset
                self.assertEqual(learning_rate_at_step(original, step, 3, 0),
                                 learning_rate_at_step(continued, step, 3, 0))
        for epoch in range(100, 250):
            for offset in (0, 2):
                self.assertEqual(learning_rate_at_step(continued, epoch * 3 + offset, 3, 0), 1e-6)
        # Merely changing epochs would restart the tail at a substantially higher LR.
        naive = {**original, "epochs": 250}
        self.assertGreater(learning_rate_at_step(naive, 100 * 3, 3, 0), 100 * 1e-6)

    def test_fork_adds_150_and_copies_full_state_without_changing_source(self):
        self.create_source()
        old = load_tensor(self.source / "last.pt")
        # Persistence-only fixture: this does not claim to exercise CUDA AMP numerics.
        old["scaler"] = {"scale": 32768.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                         "growth_interval": 2000, "_growth_tracker": 9}
        atomic_write(self.source / "last.pt", old, tensor=True)
        before = file_snapshot(self.source)
        target = self.root / "continued"
        result = continuation.prepare_continuation(self.source, target, self.cache, additional_epochs=150)
        self.assertEqual(result["config"]["epochs"], 154)
        self.assertEqual(result["config"]["lr_schedule_epochs"], 4)
        for key, value in self.cfg.items():
            if key != "epochs":
                self.assertEqual(result["config"][key], value, key)
        self.assertEqual(read_json(target / "config.json"), result["config"])
        self.assertEqual(read_json(target / "history.json"), old["history"])
        copied = load_tensor(target / "last.pt")
        self.assertNotEqual(copied["signature"], old["signature"])
        self.assertEqual(copied["signature"], read_json(target / "run.json")["signature"])
        for key in ("raw", "ema", "optimizer", "scaler", "rng_by_rank", "history", "best_weights", "step"):
            self.assert_state_equal(old[key], copied[key], key)
        self.assertTrue((target / "continuation.json").is_file())
        self.assertFalse((target / "final_raw.pt").exists())
        self.assertEqual(file_snapshot(self.source), before)

    def test_actual_continuation_and_interrupted_resume_match_continuous_training(self):
        self.create_source()
        before = file_snapshot(self.source)
        full = self.root / "continuous"
        continued = self.root / "continued"
        interrupted = self.root / "interrupted"
        cfg = {**self.cfg, "epochs": 6, "lr_schedule_epochs": 4}
        trainer.train(self.cache, full, cfg, self.device)
        expected_history = read_json(full / "history.json")
        self.assertEqual(expected_history[:4], read_json(self.source / "history.json"))
        for output in (continued, interrupted):
            plan = continuation.prepare_continuation(self.source, output, self.cache, additional_epochs=2)
            self.assertEqual(plan["config"], cfg)
            if output == interrupted:
                with patch.object(trainer, "train_epoch", side_effect=interrupt_at_epoch(2)):
                    with self.assertRaises(InterruptedTraining):
                        trainer.train(self.cache, output, cfg, self.device, resume=True)
                saved = load_tensor(output / "last.pt")
                self.assertEqual(len(saved["history"]), 5)
                self.assertEqual(saved["step"], 10)
                self.assertFalse((output / "final_raw.pt").exists())
            trainer.train(self.cache, output, cfg, self.device, resume=True)
            self.assertEqual(expected_history, read_json(output / "history.json"))
            self.assertEqual([row["lr"] for row in expected_history[4:]], [1e-6, 1e-6])
            expected, actual = load_tensor(full / "last.pt"), load_tensor(output / "last.pt")
            for key in ("raw", "ema", "optimizer", "scaler", "rng_by_rank", "history", "best_weights", "step"):
                self.assert_state_equal(expected[key], actual[key], key)
            for name in ("final_raw.pt", "best_raw.pt", "final_ema.pt", "best_ema.pt"):
                self.assert_state_equal(load_tensor(full / name), load_tensor(output / name), name)
        self.assertEqual(file_snapshot(self.source), before)

    def test_rejects_incomplete_source_and_inconsistent_final_export(self):
        self.create_source()
        cases = ("missing_final", "short_history", "wrong_final_epoch", "wrong_final_weights")
        for case in cases:
            with self.subTest(case=case):
                source = self.root / case
                shutil.copytree(self.source, source)
                if case == "missing_final":
                    (source / "final_raw.pt").unlink()
                elif case == "short_history":
                    atomic_write(source / "history.json", read_json(source / "history.json")[:-1])
                else:
                    saved = load_tensor(source / "final_raw.pt")
                    if case == "wrong_final_epoch":
                        saved["epoch"] -= 1
                    else:
                        key = next(iter(saved["state_dict"]))
                        saved["state_dict"][key] = saved["state_dict"][key] + 1
                    atomic_write(source / "final_raw.pt", saved, tensor=True)
                before = file_snapshot(source)
                output = self.root / (case + "-target")
                with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                    continuation.prepare_continuation(source, output, self.cache, additional_epochs=2)
                self.assertFalse(output.exists())
                self.assertEqual(file_snapshot(source), before)

    def test_rejects_different_cache_and_existing_target_without_mutations(self):
        self.create_source()
        other = self.root / "other-cache"
        _tiny_cache(other, {**self.identity, "checkpoint_sha256": "different-frozen-backbone"})
        source_before = file_snapshot(self.source)
        output = self.root / "new-output"
        with self.assertRaises((ValueError, RuntimeError)):
            continuation.prepare_continuation(self.source, output, other, additional_epochs=2)
        self.assertFalse(output.exists())
        output.mkdir()
        (output / "user-file.txt").write_text("keep this existing output\n")
        target_before = file_snapshot(output)
        with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
            continuation.prepare_continuation(self.source, output, self.cache, additional_epochs=2)
        self.assertEqual(file_snapshot(output), target_before)
        self.assertEqual(file_snapshot(self.source), source_before)


if __name__ == "__main__":
    unittest.main()
