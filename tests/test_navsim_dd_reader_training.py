"""Full-reader training updates, frozen weights, recovery and portable inference."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from score_function.navsim_dd import train as trainer
from score_function.navsim_dd.reader import ASSET_FORMAT, DD_COMMIT, FrozenDDReader
from score_function.navsim_dd.reader_runtime import CHECKPOINT_FORMAT, DDReaderRefiner
from score_function.utils.train_utils import atomic_write, file_hash, load_tensor, read_json


class TinyLatentDataset(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + (10 if self.split == "val" else 0))
        return {
            "context": torch.randn(512, 8, 8, generator=generator).half().float(),
            "route": torch.randn(256, generator=generator).half().float(),
            "target": torch.randn(8, 4, generator=generator),
            "record": {"token": f"{self.split}-{index}", "split": self.split},
        }


def fixture_assets():
    # The neural path is real, with a smaller spatial grid and scene decoder
    # solely to make the training/recovery integration check inexpensive on CPU.
    spec = {
        "latent_channels": 512, "latent_shape": [8, 8], "d_model": 256,
        "bev_channels": 8, "bev_shape": [8, 8], "bev_upsample_factor": 1,
        "num_agents": 2, "scene_num_layers": 1, "num_heads": 8,
        "dim_feedforward": 128, "scene_dropout": 0.1,
        "lidar_max_x": 32., "lidar_max_y": 32., "num_poses": 8,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        state = FrozenDDReader(spec).state_dict()
    return {"format": ASSET_FORMAT, "dd_commit": DD_COMMIT, "spec": spec, "state_dict": state}


class FullReaderTrainingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.assets = fixture_assets()
        atomic_write(self.cache / "reader_assets.pt", self.assets, tensor=True)
        self.identity = {
            "condition": "dd-backbone512-status-reader-v1", "dd_commit": DD_COMMIT,
            "checkpoint_sha256": "checkpoint", "anchor_sha256": "anchor", "backbone_sha256": "backbone",
            "reader_assets_sha256": file_hash(self.cache / "reader_assets.pt"),
        }
        self.metadata = {
            "normalizer": {"mean": [3., -1., 0., 0.], "std": [10., 2., 1., 1.]},
            "conditioner_identity": self.identity,
            "reader_assets": {"path": "reader_assets.pt", "sha256": self.identity["reader_assets_sha256"],
                              "spec": self.assets["spec"]},
        }
        self.cfg = read_json(Path(__file__).resolve().parents[1] / "configs/navsim_dd_s10_g_reader.json")
        self.cfg["model"].update(hidden_dim=12, num_heads=3)
        self.cfg.update(epochs=2, microbatch_size=2, num_workers=0, validation_repeats=1,
                        precision="32-true", lr_schedule="constant_after_warmup", warmup_epochs=0,
                        checkpoint_every_epochs=1, checkpoint_start_epoch=1)
        self.device = torch.device("cpu")
        self.data = (self.metadata, TinyLatentDataset("train"), TinyLatentDataset("val"))

    def test_optimizer_updates_score_keeps_reader_and_restores_exact_resume(self):
        trainer._seed(self.cfg["seed"])
        initial = trainer.build_training_model(self.cfg, self.metadata, self.cache, self.device)
        initial_weights = {k: v.clone() for k, v in initial.state_dict().items()}
        trainable_count = sum(1 for parameter in initial.parameters() if parameter.requires_grad)
        del initial
        full, resumed = self.root / "full", self.root / "resumed"
        original_epoch = trainer.train_epoch
        calls = 0

        def interrupt_second_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("test interruption after durable epoch")
            return original_epoch(*args, **kwargs)

        with patch.object(trainer, "load_training_data", return_value=self.data):
            trainer.train(self.cache, full, self.cfg, self.device)
            with patch.object(trainer, "train_epoch", side_effect=interrupt_second_epoch):
                with self.assertRaisesRegex(RuntimeError, "test interruption"):
                    trainer.train(self.cache, resumed, self.cfg, self.device)
            self.assertEqual(len(load_tensor(resumed / "last.pt")["history"]), 1)
            trainer.train(self.cache, resumed, self.cfg, self.device, resume=True)
        self.assertEqual(read_json(full / "history.json"), read_json(resumed / "history.json"))
        final = load_tensor(resumed / "final_raw.pt")
        self.assertEqual(final["format"], CHECKPOINT_FORMAT)
        self.assertEqual(final["reader_spec"], self.assets["spec"])
        self.assertEqual(final["epoch"], 2)
        expected_state = load_tensor(full / "final_raw.pt")["state_dict"]
        for key, tensor in final["state_dict"].items():
            torch.testing.assert_close(tensor, expected_state[key],
                                       atol=0, rtol=0, msg=key)
            if key.startswith("reader."):
                torch.testing.assert_close(tensor, initial_weights[key], atol=0, rtol=0, msg=key)
        self.assertTrue(any(not torch.equal(tensor, initial_weights[key])
                            for key, tensor in final["state_dict"].items() if key.startswith("score.")))
        last = load_tensor(resumed / "last.pt")
        self.assertEqual(last["format"], trainer.READER_TRAIN_FORMAT)
        self.assertEqual(len(last["optimizer"]["param_groups"][0]["params"]), trainable_count)
        self.assertEqual(sorted(p.name for p in (resumed / "checkpoints").iterdir()),
                         ["epoch_0001_raw.pt", "epoch_0002_raw.pt"])

        refiner = DDReaderRefiner.from_checkpoint(resumed / "final_raw.pt", "cpu")
        refiner.require_conditioner(self.identity)
        refiner.require_native_conditioner({k: v for k, v in self.identity.items() if k != "reader_assets_sha256"})
        with self.assertRaisesRegex(ValueError, "assets differ"):
            refiner.require_native_conditioner({**self.identity, "checkpoint_sha256": "changed"})
        raw = torch.zeros(1, 8, 3)
        row = self.data[1][0]
        latent, status = row["context"][None], row["route"][None]
        with patch.object(refiner.model, "prepare_scene", wraps=refiner.model.prepare_scene) as prepare:
            with patch.object(refiner.model.reader, "read_trajectory", wraps=refiner.model.reader.read_trajectory) as read:
                outputs = refiner.refine_many(raw, latent, status, [0, 1, 3])
        self.assertIs(outputs[0], raw)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(read.call_count, 3)
        self.assertFalse(torch.equal(read.call_args_list[0].args[0], read.call_args_list[-1].args[0]))
        torch.testing.assert_close(outputs[3], refiner.refine(raw, latent, status, 3), atol=0, rtol=0)
        disabled = raw.double()
        self.assertIs(refiner.refine(disabled, None, None, 0), disabled)
        self.assertIs(refiner.refine(disabled, None, None, 3, gamma=0), disabled)
        self.assertTrue(outputs[3].isfinite().all())

    def test_asset_change_and_incompatible_checkpoint_are_rejected(self):
        assets = copy.deepcopy(self.assets)
        assets["state_dict"]["bev_downscale.weight"].add_(1.)
        atomic_write(self.cache / "reader_assets.pt", assets, tensor=True)
        with self.assertRaisesRegex(ValueError, "assets changed"):
            trainer.build_training_model(self.cfg, self.metadata, self.cache, self.device)
        path = self.root / "wrong.pt"
        torch.save({"format": "navsim-dd-score-v1"}, path)
        with self.assertRaisesRegex(ValueError, "full DD reader"):
            DDReaderRefiner.from_checkpoint(path, "cpu")

    def test_real_cache_dispatch_trains_and_rejects_legacy_cache(self):
        from test_navsim_dd_reader_cache import encode_reader, source
        from score_function.navsim_dd import reader_cache

        original = self.root / "source"
        original.mkdir()
        source_metadata, tasks = source(original)
        identity = reader_cache.make_identity(original, source_metadata,
                                               file_hash(self.cache / "reader_assets.pt"))
        atomic_write(self.cache / "build.json", identity)
        for task in tasks:
            reader_cache.write_chunk(self.cache, original, identity, task, encode_reader)
        reader_cache.publish_cache(self.cache, original, identity, tasks,
                                   source_metadata, self.assets["spec"])
        metadata, training, validation = trainer.load_training_data(self.cache, self.cfg)
        self.assertEqual((len(training), len(validation)), (2, 1))
        self.assertEqual(metadata["normalizer"], source_metadata["normalizer"])
        with self.assertRaisesRegex(ValueError, "schema"):
            trainer.load_training_data(original, self.cfg)
        trainer.train(self.cache, self.root / "real-cache-run", {**self.cfg, "epochs": 1}, self.device)
        self.assertEqual(load_tensor(self.root / "real-cache-run/final_raw.pt")["epoch"], 1)

    def test_production_budget_is_complete_500_epoch_cosine(self):
        cfg = read_json(Path(__file__).resolve().parents[1] / "configs/navsim_dd_s10_g_reader.json")
        trainer._check_config(cfg)
        self.assertEqual(cfg["epochs"], 500)
        self.assertEqual(cfg["checkpoint_every_epochs"], 50)
        self.assertEqual(cfg["checkpoint_start_epoch"], 50)
        self.assertNotIn("lr_schedule_epochs", cfg)
        self.assertEqual(cfg["ema_decay"], 0.0)
        self.assertEqual(cfg["model"]["architecture"], "dd_scene_reader_score")


if __name__ == "__main__":
    unittest.main()
