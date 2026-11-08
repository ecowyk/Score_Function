"""DD learning-rate parity, retained batches, raw exports and precision recovery."""

import copy
import math
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from test_navsim_dd_training import _tiny_cache

from score_function.navsim_dd import train as trainer
from score_function.navsim_dd.cache import CONDITIONER_KEYS
from score_function.navsim_dd.runtime import NavsimRefiner
from score_function.train_epoch import train_epoch
from score_function.utils.train_utils import load_tensor, read_json
from score_function.utils.training_budget import learning_rate_at_step


class InterruptedTraining(RuntimeError):
    """Simulate a stopped process after its previous epoch was committed."""


class ScalarScore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, noisy, context, route):
        return self.weight * noisy


class RecordingScaler:
    """Exercise AMP control flow on CPU without pretending to test FP16 numerics."""

    def __init__(self, *, skip_step=False):
        self.skip_step = skip_step
        self.scale_value = 128.0
        self.events = []

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        self.events.append("scale")
        return loss

    def unscale_(self, optimizer):
        self.events.append("unscale")

    def step(self, optimizer):
        self.events.append("step")
        if not self.skip_step:
            optimizer.step()

    def update(self):
        self.events.append("update")
        if self.skip_step:
            self.scale_value /= 2


def aligned_config(**overrides):
    cfg = read_json(trainer.DEFAULT_CONFIG)
    cfg["model"].update(hidden_dim=12, num_heads=3)
    cfg.update(
        epochs=4, microbatch_size=2, num_workers=0, validation_repeats=1,
        learning_rate=6e-4, min_learning_rate=1e-6, warmup_epochs=3,
        lr_schedule="dd_warmup_cosine", gradient_clip=0.0,
        ema_decay=0.0, precision="32-true", drop_last=False,
    )
    cfg.update(overrides)
    return cfg


def interrupt_at_epoch(epoch_to_interrupt):
    calls = 0
    original = trainer.train_epoch

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == epoch_to_interrupt:
            raise InterruptedTraining("fixture interruption")
        return original(*args, **kwargs)

    return run


class NavsimAlignedTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.identity = {key: key + "-fixture" for key in CONDITIONER_KEYS}
        self.device = torch.device("cpu")

    def assert_same_weights(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for key, weight in expected.items():
            torch.testing.assert_close(weight, actual[key], rtol=0, atol=0, msg=key)

    def test_dd_schedule_matches_official_epoch_formula_through_step_100(self):
        cfg = aligned_config(epochs=100)
        peak, minimum = 6e-4, 1e-6
        for epoch in range(101):
            expected = (
                peak * (epoch + 1) / 3 if epoch < 3 else
                minimum + (peak - minimum) * (1 + math.cos(math.pi * (epoch - 3) / 97)) / 2
            )
            for offset in range(3):
                with self.subTest(epoch=epoch, offset=offset):
                    self.assertAlmostEqual(
                        learning_rate_at_step(cfg, epoch * 3 + offset, 3, 9e-2),
                        expected, places=15,
                    )
        self.assertGreater(learning_rate_at_step(cfg, 99, 1, 0), minimum)
        self.assertEqual(learning_rate_at_step(cfg, 100, 1, 0), minimum)
        for epoch, expected in enumerate((2e-4, 4e-4, 6e-4)):
            self.assertAlmostEqual(learning_rate_at_step(cfg, epoch, 1, 0), expected, places=15)

    def test_zero_clip_updates_raw_weights_without_calling_gradient_clipping(self):
        cfg = aligned_config(warmup_epochs=0, lr_schedule="constant_after_warmup")
        model = ScalarScore()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        batch = {
            "target": torch.zeros(2, 8, 4),
            "context": torch.zeros(2, 64, 256),
            "route": torch.zeros(2, 256),
        }
        with patch("torch.nn.utils.clip_grad_norm_") as clip:
            result = list(train_epoch(
                [batch], model, optimizer, cfg, None, self.device,
                start_step=0, start_update=0, updates_per_epoch=1,
                accumulation=1, next_lr=cfg["learning_rate"],
            ))
        clip.assert_not_called()
        self.assertNotEqual(model.weight.item(), 0.0)
        self.assertEqual(result[0]["step"], 1)
        self.assertNotIn("amp_scale", result[0])

    def test_raw_export_and_cosine_resume_match_uninterrupted_training(self):
        cfg = aligned_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, full, resumed = root / "cache", root / "full", root / "resume"
            _tiny_cache(cache, self.identity)
            trainer.train(cache, full, cfg, self.device)
            with patch.object(trainer, "train_epoch", side_effect=interrupt_at_epoch(3)):
                with self.assertRaises(InterruptedTraining):
                    trainer.train(cache, resumed, cfg, self.device)
            self.assertEqual(len(load_tensor(resumed / "last.pt")["history"]), 2)
            trainer.train(cache, resumed, cfg, self.device, resume=True)
            self.assertEqual(read_json(full / "history.json"), read_json(resumed / "history.json"))
            final = load_tensor(resumed / "final_raw.pt")
            last = load_tensor(resumed / "last.pt")
            self.assertEqual(final["weight_kind"], "raw")
            self.assert_same_weights(last["raw"], final["state_dict"])
            for name in ("final_raw.pt", "best_raw.pt", "final_ema.pt", "best_ema.pt"):
                expected, actual = load_tensor(full / name), load_tensor(resumed / name)
                self.assertEqual(actual["weight_kind"], "raw")
                self.assertEqual(expected["epoch"], actual["epoch"])
                self.assert_same_weights(expected["state_dict"], actual["state_dict"])
            refiner = NavsimRefiner.from_checkpoint(resumed / "final_raw.pt", self.device)
            refiner.require_conditioner(self.identity)
            refined = refiner.refine_many(
                torch.zeros(1, 8, 3), torch.zeros(1, 64, 256), torch.zeros(1, 256), [1],
            )
            self.assertTrue(torch.isfinite(refined[1]).all())
            with self.assertRaisesRegex(RuntimeError, "Resume rejected"):
                trainer.train(cache, resumed, {**cfg, "epochs": 5}, self.device, resume=True)

    def test_last_partial_batch_is_retained_and_loss_is_sample_weighted(self):
        cfg = aligned_config(
            epochs=1, warmup_epochs=0, microbatch_size=3,
            lr_schedule="constant_after_warmup",
        )
        batch_sizes, seen_tokens, losses = [], [], []
        original = trainer.train_epoch

        def observe(loader, *args, **kwargs):
            batches = list(loader)
            batch_sizes.extend(len(batch["tokens"]) for batch in batches)
            seen_tokens.extend(token for batch in batches for token in batch["tokens"])
            for result in original(batches, *args, **kwargs):
                losses.append(result["train_dsm"])
                yield result

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tiny_cache(root / "cache", self.identity)
            with patch.object(trainer, "train_epoch", side_effect=observe):
                trainer.train(root / "cache", root / "run", cfg, self.device)
            run = read_json(root / "run" / "run.json")
            row = read_json(root / "run" / "history.json")[0]
            self.assertEqual(run["updates_per_epoch"], 2)
            self.assertEqual(row["step"], 2)
            self.assertEqual(batch_sizes, [3, 1])
            self.assertEqual(sorted(seen_tokens), [f"train-{i}" for i in range(4)])
            self.assertAlmostEqual(row["train_dsm"], (3 * losses[0] + losses[1]) / 4, places=13)

    def test_implicit_legacy_precision_matches_explicit_fp32(self):
        cfg = read_json(trainer.DEFAULT_CONFIG)
        cfg["model"].update(hidden_dim=12, num_heads=3)
        cfg.update(epochs=1, microbatch_size=2, num_workers=0, validation_repeats=1)
        self.assertNotIn("precision", cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tiny_cache(root / "cache", self.identity)
            trainer.train(root / "cache", root / "implicit", cfg, self.device)
            trainer.train(root / "cache", root / "explicit", {**cfg, "precision": "32-true"}, self.device)
            self.assertEqual(
                read_json(root / "implicit" / "history.json"),
                read_json(root / "explicit" / "history.json"),
            )
            self.assert_same_weights(
                load_tensor(root / "implicit" / "last.pt")["raw"],
                load_tensor(root / "explicit" / "last.pt")["raw"],
            )

    def test_mixed_precision_rejects_cpu_before_reading_cache(self):
        cfg = aligned_config(precision="16-mixed")
        with patch.object(trainer, "load_cache_metadata") as load_cache:
            with self.assertRaisesRegex(ValueError, "CUDA|cuda"):
                trainer.train("nonexistent-cache", "unused-output", cfg, self.device)
        load_cache.assert_not_called()

    def test_amp_unscales_before_clipping_and_skips_ema_after_overflow(self):
        for clipping in (0.0, 5.0):
            for skip_step in (False, True):
                with self.subTest(clipping=clipping, skip_step=skip_step):
                    cfg = aligned_config(
                        warmup_epochs=0, lr_schedule="constant_after_warmup",
                        gradient_clip=clipping, ema_decay=0.9,
                    )
                    model = ScalarScore()
                    ema = copy.deepcopy(model).requires_grad_(False)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
                    scaler = RecordingScaler(skip_step=skip_step)
                    batch = {
                        "target": torch.zeros(2, 8, 4),
                        "context": torch.zeros(2, 64, 256),
                        "route": torch.zeros(2, 256),
                    }
                    original_clip = torch.nn.utils.clip_grad_norm_

                    def clip(*args, **kwargs):
                        scaler.events.append("clip")
                        return original_clip(*args, **kwargs)

                    with patch("torch.autocast", return_value=nullcontext()), patch(
                        "torch.nn.utils.clip_grad_norm_", side_effect=clip,
                    ):
                        result = list(train_epoch(
                            [batch], model, optimizer, cfg, ema, self.device,
                            start_step=0, start_update=0, updates_per_epoch=1,
                            accumulation=1, next_lr=cfg["learning_rate"], scaler=scaler,
                        ))[0]
                    expected_events = ["scale"]
                    if clipping:
                        expected_events += ["unscale", "clip"]
                    self.assertEqual(scaler.events, expected_events + ["step", "update"])
                    self.assertEqual(result["optimizer_step_skipped"], skip_step)
                    self.assertEqual(result["amp_scale"], 64.0 if skip_step else 128.0)
                    if skip_step:
                        self.assertEqual(model.weight.item(), 0.0)
                        self.assertEqual(ema.weight.item(), 0.0)
                    else:
                        self.assertNotEqual(model.weight.item(), 0.0)
                        self.assertNotEqual(ema.weight.item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for real FP16/scaler recovery")
    def test_cuda_scaler_state_is_saved_and_restored(self):
        cfg = aligned_config()
        cfg["precision"] = "16-mixed"
        device = torch.device("cuda", 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tiny_cache(root / "cache", self.identity)
            with patch.object(trainer, "train_epoch", side_effect=interrupt_at_epoch(2)):
                with self.assertRaises(InterruptedTraining):
                    trainer.train(root / "cache", root / "run", cfg, device)
            state = load_tensor(root / "run" / "last.pt")["scaler"]
            self.assertTrue(state)
            self.assertGreater(state["scale"], 0)
            original = torch.cuda.amp.GradScaler.load_state_dict
            with patch.object(
                torch.cuda.amp.GradScaler, "load_state_dict", autospec=True, side_effect=original,
            ) as restore:
                trainer.train(root / "cache", root / "run", cfg, device, resume=True)
            restore.assert_called_once()
            self.assertEqual(restore.call_args.args[1], state)
            self.assertTrue(load_tensor(root / "run" / "last.pt")["scaler"])
            self.assertEqual(len(read_json(root / "run" / "history.json")), cfg["epochs"])


if __name__ == "__main__":
    unittest.main()
