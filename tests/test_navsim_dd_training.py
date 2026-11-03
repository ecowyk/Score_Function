"""Exercise cached training, epoch-boundary resume and the physical-pose inference adapter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException

from score_function.navsim_dd import train as trainer
from score_function.navsim_dd.cache import CONDITIONER_KEYS, _publish_cache
from score_function.navsim_dd.runtime import NavsimRefiner
from score_function.utils.train_utils import load_tensor, read_json


def _tiny_cache(directory, identity):
    def encode(token):
        # The tensor layout matches a frozen DD extraction. No sensor files are required.
        rng = np.random.default_rng(int(token.split("-")[-1]))
        return rng.normal(size=(8, 3)), rng.normal(size=(64, 256)), rng.normal(size=256)

    chunks = [
        (split, split + "-log", [f"{split}-{i}" for i in range(count)], encode)
        for split, count in (("train", 4), ("val", 3))
    ]
    _publish_cache(directory, {**identity, "data_root": "fixture"}, chunks)

def _train_rank(rank, folder):
    root = Path(folder)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(root / "gloo").resolve().as_uri(),
                            rank=rank, world_size=2)
    try:
        cfg = read_json(trainer.DEFAULT_CONFIG)
        cfg["model"].update(hidden_dim=12, num_heads=3)
        cfg.update(microbatch_size=2, num_workers=0, validation_repeats=1, epochs=1)
        trainer.train(root / "cache", root / "run", cfg, torch.device("cpu"))
    finally:
        dist.destroy_process_group()


class NavsimTrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_and_exports_physical_refiner(self):
        cfg = read_json(trainer.DEFAULT_CONFIG)
        cfg["model"].update(hidden_dim=12, num_heads=3)
        cfg.update(microbatch_size=2, num_workers=0, validation_repeats=1, epochs=2)
        identity = {key: key + "-fixture" for key in CONDITIONER_KEYS}
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, uninterrupted, resumed = root / "cache", root / "full", root / "resume"
            _tiny_cache(cache, identity)
            device = torch.device("cpu")
            trainer.train(cache, uninterrupted, cfg, device)
            trainer.train(cache, resumed, {**cfg, "epochs": 1}, device)
            trainer.train(cache, resumed, cfg, device, resume=True)

            self.assertEqual(read_json(uninterrupted / "history.json"), read_json(resumed / "history.json"))
            for name in ("final_ema.pt", "best_ema.pt"):
                expected = load_tensor(uninterrupted / name)
                actual = load_tensor(resumed / name)
                self.assertEqual(expected["epoch"], actual["epoch"])
                for key, weight in expected["state_dict"].items():
                    self.assertTrue(torch.equal(weight, actual["state_dict"][key]), key)
            refiner = NavsimRefiner.from_checkpoint(resumed / "final_ema.pt", device)
            refiner.require_conditioner(identity)
            raw = torch.zeros(1, 8, 3)
            refined = refiner.refine_many(raw, torch.zeros(1, 64, 256), torch.zeros(1, 256), [0, 2])
            self.assertIs(refined[0], raw)
            self.assertTrue(torch.isfinite(refined[2]).all())

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                trainer.train(cache, resumed, cfg, device)
            other_cache = root / "other"
            _tiny_cache(other_cache, {**identity, "checkpoint_sha256": "other weights"})
            with self.assertRaisesRegex(RuntimeError, "Resume rejected"):
                trainer.train(other_cache, resumed, cfg, device, resume=True)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_two_rank_training_publishes_one_complete_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tiny_cache(root / "cache", {key: key + "-fixture" for key in CONDITIONER_KEYS})
            try:
                mp.spawn(_train_rank, args=(tmp,), nprocs=2, join=True)
            except ProcessRaisedException as exc:
                if "gloo/transport" in str(exc) and "Operation not permitted" in str(exc):
                    self.skipTest("Local execution environment forbids Gloo sockets")
                raise
            last = load_tensor(root / "run" / "last.pt")
            self.assertEqual(len(last["rng_by_rank"]), 2)
            self.assertEqual(last["step"], 1)
            self.assertEqual(read_json(root / "run" / "run.json")["global_batch"], 4)
            self.assertEqual(load_tensor(root / "run" / "final_ema.pt")["epoch"], 1)
