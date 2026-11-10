"""Training recovery, variable-length conditioning and portable refinement."""

import copy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from score_function.navsim_chainflow import train as trainer
from score_function.navsim_chainflow.runtime import ChainFlowRefiner
from score_function.utils.train_utils import load_tensor, read_json
from test_chainflow_model import small_model_config


class TinyConditionDataset(Dataset):
    def __init__(self, split):
        self.records = [
            {"token": f"{split}-{i}", "log_name": f"{split}-log", "split": split}
            for i in range(4)
        ]
        self.split = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + (110 if self.split == "val" else 60))
        target = torch.zeros(8, 4)
        target[:, 0] = torch.linspace(0.1, 0.8, 8) + index * 0.05
        target[:, 1] = torch.linspace(0.0, 0.03, 8) * index
        target[:, 2] = 1.0
        return {
            "scene": torch.randn(64, 256, generator=generator),
            "ego": torch.randn(256, generator=generator),
            "vlm": torch.randn(3 + index, 1536, generator=generator),
            "target": target,
            "record": dict(self.records[index]),
        }

    def close(self):
        pass


def assert_nested_equal(case, actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(actual, dict):
        case.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            assert_nested_equal(case, actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        case.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            assert_nested_equal(case, a, b)
    else:
        case.assertEqual(actual, expected)


def train_gloo_rank(rank, root, cfg, metadata):
    """Spawn entrypoint: exercise real DDP hooks and collective reductions."""
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "gloo", init_method=(root / "gloo-rendezvous").as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=90),
    )
    try:
        training, validation = TinyConditionDataset("train"), TinyConditionDataset("val")
        validation.records = validation.records[:3]
        models = []
        build = trainer.build_training_model

        def record_model(*args, **kwargs):
            model = build(*args, **kwargs)
            models.append(model)
            return model

        run = root / "distributed"
        with patch.object(trainer, "load_training_data", return_value=(metadata, training, validation)), \
                patch.object(trainer, "build_training_model", side_effect=record_model):
            trainer.train(root / "cache", run, cfg, torch.device("cpu"))
            state_before = {key: value.detach().clone() for key, value in models[-1].state_dict().items()}
            trainer.train(root / "cache", run, cfg, torch.device("cpu"), resume=True)
        state_after = models[-1].state_dict()
        for key, value in state_before.items():
            torch.testing.assert_close(state_after[key], value, atol=0, rtol=0)
            assert torch.isfinite(value).all(), f"Rank {rank} has nonfinite model state: {key}"
        torch.save(state_before, root / f"rank_{rank}_weights.pt")
    finally:
        dist.destroy_process_group()


class ChainFlowTrainingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.identity = {
            "condition": "chainflow-scene-vlm-v1",
            "checkpoint_sha256": "a" * 64,
            "backbone_sha256": "b" * 64,
        }
        self.metadata = {
            "normalizer": {"mean": [3.0, -1.0, 0.0, 0.0], "std": [10.0, 2.0, 1.0, 1.0]},
            "conditioner_identity": self.identity,
            "splits": {name: {"num_records": 4} for name in ("train", "val")},
        }
        self.cfg = read_json(Path(__file__).resolve().parents[1] / "configs/navsim_chainflow_s10g.json")
        self.cfg["model"] = small_model_config(dropout=0.1)
        self.cfg.update(
            epochs=3, microbatch_size=2, global_batch_size=2, num_workers=0,
            validation_repeats=1, diagnostic_train_samples=3,
            precision="32-true", warmup_epochs=0,
            checkpoint_every_epochs=1, checkpoint_start_epoch=1, ema_decay=0.8,
        )
        self.data = (self.metadata, TinyConditionDataset("train"), TinyConditionDataset("val"))
        self.device = torch.device("cpu")

    def test_variable_length_collation_retains_every_token(self):
        rows = [self.data[1][0], self.data[1][3]]
        batch = trainer.collate_rows(rows)
        self.assertEqual(tuple(batch["vlm"].shape), (2, 6, 1536))
        self.assertEqual(batch["vlm_mask"].dtype, torch.bool)
        self.assertEqual(batch["vlm_mask"].sum(dim=1).tolist(), [3, 6])
        self.assertEqual(batch["tokens"], ["train-0", "train-3"])
        torch.testing.assert_close(batch["vlm"][0, :3], rows[0]["vlm"], atol=0, rtol=0)
        torch.testing.assert_close(batch["vlm"][1], rows[1]["vlm"], atol=0, rtol=0)

    def test_resume_restores_weights_optimizer_rng_and_all_loss_metrics(self):
        uninterrupted, resumed = self.root / "full", self.root / "resumed"
        original = trainer.run_training_epoch
        calls = 0

        def fail_second_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("deliberate interruption")
            return original(*args, **kwargs)

        with patch.object(trainer, "load_training_data", return_value=self.data):
            trainer.train(self.cache, uninterrupted, self.cfg, self.device)
            with patch.object(trainer, "run_training_epoch", side_effect=fail_second_epoch):
                with self.assertRaisesRegex(RuntimeError, "deliberate interruption"):
                    trainer.train(self.cache, resumed, self.cfg, self.device)
            self.assertEqual(len(load_tensor(resumed / "last.pt")["history"]), 1)
            trainer.train(self.cache, resumed, self.cfg, self.device, resume=True)

        self.assertEqual(read_json(uninterrupted / "history.json"), read_json(resumed / "history.json"))
        full_state, resumed_state = load_tensor(uninterrupted / "last.pt"), load_tensor(resumed / "last.pt")
        for key in ("raw", "ema", "optimizer"):
            assert_nested_equal(self, resumed_state[key], full_state[key])
        self.assertEqual(resumed_state["step"], 6)
        history = read_json(resumed / "history.json")
        for row in history:
            for suffix in ("raw", "ema"):
                self.assertAlmostEqual(row[f"val_dsm_{suffix}"],
                                       (row[f"val_xy_{suffix}"] + row[f"val_heading_{suffix}"]) / 2,
                                       places=10)
                self.assertTrue(torch.isfinite(torch.tensor(row[f"diagnostic_train_dsm_{suffix}"])))
        for suffix in ("raw", "ema"):
            exported = load_tensor(resumed / f"final_{suffix}.pt")
            self.assertEqual(exported["format"], "navsim-chainflow-score-v1")
            self.assertEqual(exported["weight_kind"], suffix)
            self.assertEqual(exported["epoch"], 3)
            self.assertEqual(exported["conditioner_identity"], self.identity)
            self.assertEqual(len(list(resumed.rglob(f"epoch_*_{suffix}.pt"))), 3)
            self.assertTrue((resumed / f"best_{suffix}.pt").is_file())

    def test_changed_experiment_configuration_cannot_resume(self):
        run = self.root / "run"
        with patch.object(trainer, "load_training_data", return_value=self.data):
            trainer.train(self.cache, run, self.cfg, self.device)
            cfg = copy.deepcopy(self.cfg)
            cfg["sigma"] = 0.2
            with self.assertRaises((ValueError, RuntimeError)):
                trainer.train(self.cache, run, cfg, self.device, resume=True)
        self.assertEqual(read_json(run / "history.json")[-1]["epoch"], 3)

    def test_microbatches_preserve_the_requested_global_update_budget(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["global_batch_size"] = 4
        run = self.root / "accumulated"
        with patch.object(trainer, "load_training_data", return_value=self.data):
            trainer.train(self.cache, run, cfg, self.device)
        # Four examples, microbatch two: two forwards form one optimizer step.
        self.assertEqual(load_tensor(run / "last.pt")["step"], 3)
        self.assertEqual(read_json(run / "run.json")["global_batch"], 4)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_rank_ddp_accumulation_uneven_validation_and_completed_resume(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(global_batch_size=4, microbatch_size=1)
        metadata = copy.deepcopy(self.metadata)
        metadata["splits"]["val"]["num_records"] = 3
        try:
            mp.spawn(train_gloo_rank, args=(str(self.root), cfg, metadata), nprocs=2, join=True)
        except mp.ProcessRaisedException as exc:
            # Some execution hosts prohibit even loopback TCP sockets. Skip
            # only that backend-init denial, never a trainer/DDP-hook failure.
            detail = str(exc)
            if ("dist.init_process_group(" in detail
                    and "gloo/transport/tcp/device.cc:" in detail
                    and "Operation not permitted" in detail):
                self.skipTest("Host denies loopback sockets required by the Gloo backend")
            raise

        rank_zero = load_tensor(self.root / "rank_0_weights.pt")
        rank_one = load_tensor(self.root / "rank_1_weights.pt")
        assert_nested_equal(self, rank_zero, rank_one)
        run = self.root / "distributed"
        checkpoint = load_tensor(run / "last.pt")
        assert_nested_equal(self, checkpoint["raw"], rank_zero)
        self.assertEqual(len(checkpoint["rng_by_rank"]), 2)
        self.assertEqual(checkpoint["step"], 3)
        self.assertEqual(len(checkpoint["history"]), 3)
        for row in checkpoint["history"]:
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in row.values()))
        metadata = read_json(run / "run.json")
        self.assertEqual(metadata["gradient_accumulation"], 2)
        self.assertEqual(metadata["val_samples"], 3)
        self.assertEqual(len(metadata["diagnostic_train_indices"]), 3)
        self.assertTrue((run / "TRAINING_COMPLETE.json").is_file())

    def test_portable_final_ema_refines_one_shared_path_and_preserves_k0(self):
        run = self.root / "run"
        with patch.object(trainer, "load_training_data", return_value=self.data):
            trainer.train(self.cache, run, self.cfg, self.device)
        refiner = ChainFlowRefiner.from_checkpoint(run / "final_ema.pt", "cpu")
        refiner.require_conditioner(self.identity)
        with self.assertRaises(ValueError):
            refiner.require_conditioner({**self.identity, "checkpoint_sha256": "changed"})
        raw = torch.zeros(1, 8, 3)
        raw[..., 0] = torch.linspace(1.0, 8.0, 8)
        raw[..., 2] = 3.8  # K0 must preserve yaw outside the principal branch too.
        row = self.data[1][0]
        condition = {name: row[name][None] for name in ("scene", "ego", "vlm")}
        condition["vlm_mask"] = torch.ones(1, row["vlm"].shape[0], dtype=torch.bool)
        results = refiner.refine_many(raw, condition, [0, 1, 3])
        self.assertIs(results[0], raw)
        self.assertTrue(torch.isfinite(results[3]).all())
        torch.testing.assert_close(results[3], refiner.refine_many(raw, condition, [3])[3], atol=0, rtol=0)
        disabled = raw.double()
        self.assertIs(refiner.refine_many(disabled, None, [0])[0], disabled)
        self.assertIs(refiner.refine_many(disabled, None, [0, 3], gamma=0)[3], disabled)


if __name__ == "__main__":
    unittest.main()
