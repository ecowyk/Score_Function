"""Real training/checkpoint wiring on temporary synthetic shards, not a toy study."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import test_training_pipeline as pipeline
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from score_function.model.dit_score_branch import DiTScoreFunctionBranch
from score_function.train import train
from score_function.train_epoch import validate_epoch
from score_function.utils.checkpoint import load_selected
from score_function.utils.config import load_config
from score_function.utils.dataset import ShardedDataset
from score_function.utils.train_utils import load_tensor

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_dit_config", ROOT / "scripts/prepare_dit_config.py"
)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def fixture(root):
    config, rows = pipeline.fixture(root)
    config["model"] = json.loads((ROOT / "configs/score_function_dit.json").read_text())["model"]
    config["model"].update(hidden_dim=24, num_heads=3, depth=2, preproj_hidden_dim=48)
    return config, rows


class DiTPipelineTests(unittest.TestCase):
    def test_config_derivation_preserves_current_experiment_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            original = json.loads((ROOT / "configs/score_function.json").read_text())
            original["paths"].update(root=tmp, run_dir="outputs/current")
            original["training"].update(sigma=0.1, microbatch_size=128, learning_rate=0.0002)
            original["model"].update(hidden_dim=96, num_heads=3, dropout=0.2)
            source.write_text(json.dumps(original))
            before = source.read_bytes()
            target = Path(tmp) / "dit.json"
            HELPER.prepare(source, target, "outputs/new_dit")
            raw = json.loads(target.read_text())
            config = load_config(target)
            self.assertEqual(source.read_bytes(), before)
            for key in original.keys() - {"model", "paths"}:
                self.assertEqual(raw[key], original[key], key)
            for key in original["paths"].keys() - {"run_dir"}:
                self.assertEqual(raw["paths"][key], original["paths"][key], key)
            self.assertEqual(config["model"]["architecture"], "dit_score_function")
            self.assertEqual(config["model"]["depth"], 3)
            for key in ("hidden_dim", "num_heads", "dropout"):
                self.assertEqual(config["model"][key], original["model"][key])
            self.assertFalse(Path(config["output"]).exists())
            with self.assertRaises(FileExistsError):
                HELPER.prepare(source, target, "outputs/new_dit")
            with self.assertRaisesRegex(ValueError, "new run directory"):
                HELPER.prepare(source, Path(tmp) / "bad.json", "outputs/current")
            Path(config["output"]).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                HELPER.prepare(source, Path(tmp) / "bad.json", "outputs/new_dit")

    def test_config_rejects_unsupported_or_malformed_dit_options(self):
        source = ROOT / "configs/score_function_dit.json"
        for override in (
            'model.parameterization="energy"',
            "model.depth=0",
            "model.depth=true",
            "model.preproj_hidden_dim=0",
            "model.mlp_ratio=0",
            "model.mlp_ratio=true",
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(source, overrides=[override])
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads(source.read_text())
            raw["model"]["neighbor_future"] = True
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "neighbor_future"):
                load_config(path)

    def test_training_selection_validation_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            train(config, device_override="cpu")
            full = load_tensor(Path(config["output"]) / "score/last.pt")
            checkpoint = Path(config["output"]) / "score/best.pt"
            model, selected = load_selected(config, checkpoint, "cpu")
            self.assertIsInstance(model, DiTScoreFunctionBranch)
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
            wrong["model"] = json.loads((ROOT / "configs/score_function.json").read_text())["model"]
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_selected(wrong, checkpoint, "cpu")
            wrong = copy.deepcopy(config)
            wrong["training"]["sigma"] *= 2
            with self.assertRaisesRegex(ValueError, "sigma"):
                load_selected(wrong, checkpoint, "cpu")
            config["output"] = str(Path(tmp) / "resumed")
            train(config, device_override="cpu", stop_after_updates=1)
            train(config, resume=True, device_override="cpu")
            resumed = load_tensor(Path(config["output"]) / "score/last.pt")
            pipeline.PipelineTests._assert_state_equal(self, full, resumed)

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo unavailable")
    def test_two_rank_ddp_exact_resume_with_zero_initialized_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            mp.spawn(
                pipeline.ddp_worker,
                args=(config, str(Path(tmp) / "full_store"), False, None),
                nprocs=2,
                join=True,
            )
            full = load_tensor(Path(config["output"]) / "score/last.pt")
            config["output"] = str(Path(tmp) / "resumed")
            for resume, stop, name in ((False, 1, "partial"), (True, None, "resume")):
                mp.spawn(
                    pipeline.ddp_worker,
                    args=(config, str(Path(tmp) / name), resume, stop),
                    nprocs=2,
                    join=True,
                )
            resumed = load_tensor(Path(config["output"]) / "score/last.pt")
            pipeline.PipelineTests._assert_state_equal(self, full, resumed)


if __name__ == "__main__":
    unittest.main()
