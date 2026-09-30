"""Migration and CLI guards for existing experiment artifacts."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from score_function.cli import main
from score_function.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_aligned_config", ROOT / "scripts/prepare_aligned_config.py"
)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class AlignmentEntrypointTests(unittest.TestCase):
    def test_existing_variant_data_and_sigma_survive_budget_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((ROOT / "configs/score_function_energy.json").read_text())
            config["paths"].update(root=str(root), run_dir="outputs/old_energy")
            config["model"].update(neighbor_future=True, post_dilations=[2, 4])
            config["paths"]["neighbor_cache"] = "cache/neighbors/index.json"
            config["data"]["neighbor_batch_size"] = 16
            config["training"].update(sigma=0.1, max_epochs=30, minimum_updates=0)
            source = root / "old.json"
            source.write_text(json.dumps(config))
            before = source.read_bytes()
            destination = root / "new.json"
            HELPER.prepare(source, destination, "outputs/new", microbatch=256, gpus=8)
            aligned = load_config(destination)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(aligned["model"], config["model"])
            self.assertEqual(aligned["data"], config["data"])
            self.assertEqual(aligned["training"]["sigma"], 0.1)
            self.assertEqual(aligned["training"]["minimum_updates"], 244000)
            self.assertFalse(aligned["training"]["early_stopping"])
            self.assertTrue(aligned["training"]["data_augmentation"]["enabled"])
            self.assertEqual(aligned["paths"]["neighbor_cache"], config["paths"]["neighbor_cache"])
            self.assertFalse((root / "outputs/new").exists())
            with self.assertRaises(FileExistsError):
                HELPER.prepare(source, destination, "outputs/new")
            with self.assertRaises(ValueError):
                HELPER.prepare(source, root / "bad.json", "outputs/old_energy")
            (root / "existing").mkdir()
            with self.assertRaises(FileExistsError):
                HELPER.prepare(source, root / "bad.json", "existing")
            with self.assertRaises(ValueError):
                HELPER.prepare(source, root / "bad.json", "outputs/new", microbatch=1024, gpus=8)

    def test_long_diagnostic_cli_dispatch_and_cpu_runtime(self):
        config = ROOT / "configs/score_function.json"
        with (
            patch("score_function.cli.configure_runtime") as runtime,
            patch("score_function.evaluation.long_refinement.diagnose_refinement") as diagnose,
        ):
            main(
                [
                    "visualize-refinement",
                    "--config",
                    str(config),
                    "--checkpoint",
                    "old/best.pt",
                    "--device",
                    "cpu",
                    "--steps",
                    "5000",
                    "--max-samples",
                    "2",
                    "--initializations",
                    "expert_noise",
                    "--snapshot-steps",
                    "0",
                    "20",
                    "5000",
                ]
            )
            self.assertFalse(runtime.call_args.kwargs["require_cuda"])
            self.assertEqual(diagnose.call_args.kwargs["steps"], 5000)
            self.assertEqual(diagnose.call_args.kwargs["initializations"], ["expert_noise"])
            self.assertEqual(diagnose.call_args.kwargs["snapshot_steps"], [0, 20, 5000])
        with (
            self.assertRaises(SystemExit),
            patch("score_function.cli.configure_runtime") as runtime,
        ):
            main(["train", "--config", str(config), "--steps", "5000"])
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
