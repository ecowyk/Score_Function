"""Opt-in production pipeline wiring, using synthetic data and random official weights.

This is an engineering test, not evidence about learned driving scores or nuPlan.
No resources are downloaded and every artifact is scoped to a temporary folder.
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from test_official_boundary import feature_record, official_arguments

from score_function.data_process.feature_cache import build_cache
from score_function.evaluation.diagnostics import evaluate
from score_function.evaluation.planner_evaluation import evaluate_planner
from score_function.tools.check_small_batch import run_smoke
from score_function.train import train
from score_function.utils.config import load_config
from score_function.utils.train_utils import atomic_write, file_hash, read_json

OFFICIAL_ROOT = os.environ.get("SCORE_FUNCTION_OFFICIAL_ROOT")


@unittest.skipUnless(
    OFFICIAL_ROOT, "Set SCORE_FUNCTION_OFFICIAL_ROOT for actual official-code pipeline"
)
class EndToEndTests(unittest.TestCase):
    def test_cache_smoke_training_and_both_offline_evaluators(self):
        source = Path(OFFICIAL_ROOT).resolve()
        self.assertTrue((source / "diffusion_planner/model/diffusion_planner.py").is_file())
        sys.path.insert(0, str(source))
        from diffusion_planner.model.diffusion_planner import Diffusion_Planner
        from diffusion_planner.utils.config import Config

        torch.set_num_threads(1)
        torch.manual_seed(232)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args_path = root / "args.json"
            checkpoint_path = root / "synthetic_model.pth"
            atomic_write(args_path, official_arguments())
            original = Diffusion_Planner(Config(str(args_path), None)).eval()
            atomic_write(checkpoint_path, {"ema_state_dict": original.state_dict()}, tensor=True)
            del original
            config = load_config(
                Path(__file__).resolve().parents[1] / "configs/score_function.json", root
            )
            config["paths"].update(
                planner_dir=str(source),
                planner_args=str(args_path),
                planner_checkpoint=str(checkpoint_path),
                manifest=str(root / "manifest.json"),
                train_log_allowlist=str(root / "allowed.json"),
                cache=str(root / "cache/index.json"),
                run_dir=str(root / "run"),
            )
            config["cache"] = config["paths"]["cache"]
            config["output"] = config["paths"]["run_dir"]
            config["runtime"].update(device="cpu", cpu_threads=1)
            config["model"].update(hidden_dim=24, num_heads=3, dropout=0.0)
            config["data"].update(encoding_batch_size=2, shard_size=3)
            config["smoke"].update(frames=4, updates=100, learning_rate=1e-3)
            config["training"].update(
                max_epochs=1,
                minimum_epochs=1,
                warmup_epochs=0,
                batch_size=4,
                microbatch_size=1,
                num_workers=0,
                validation_repeats=1,
                checkpoint_every_updates=1,
                log_every_updates=1,
            )
            config["evaluation"].update(noise_repeats=1, visualize_samples=1, batch_size=2)
            config["refinement"].update(steps=2)
            records = []
            for index in range(6):
                frame_dir = root / f"frame_{index}"
                frame_dir.mkdir()
                record, _ = feature_record(frame_dir)
                record.update(
                    token=f"synthetic_{index}",
                    recording=f"recording_{index}",
                    log=f"fake_train_db_{index}",
                    start_time_us=index * 100_000,
                    split="train" if index < 4 else "val",
                    feature_sha256=file_hash(record["feature"]),
                )
                records.append(record)
            atomic_write(config["paths"]["manifest"], records)
            atomic_write(config["paths"]["train_log_allowlist"], [row["log"] for row in records])

            build_cache(config)
            cache_index = Path(config["cache"])
            index_hash = file_hash(cache_index)
            shard_files = {
                str(path): (file_hash(path), path.stat().st_mtime_ns)
                for path in cache_index.parent.glob("shard_*/*.npy")
            }
            self.assertEqual(read_json(cache_index.parent / "status.json")["state"], "complete")
            self.assertEqual(
                read_json(cache_index)["metadata"]["split_counts"], {"train": 4, "val": 2}
            )
            build_cache(config)
            self.assertEqual(file_hash(cache_index), index_hash)
            self.assertEqual(
                shard_files,
                {
                    str(path): (file_hash(path), path.stat().st_mtime_ns)
                    for path in cache_index.parent.glob("shard_*/*.npy")
                },
            )

            smoke = run_smoke(config, device_override="cpu")
            self.assertEqual(smoke["state"], "passed")
            self.assertLess(smoke["final_dsm"], smoke["initial_dsm"])
            self.assertTrue(math.isfinite(smoke["final_dsm"]))
            train(config, device_override="cpu")
            run = Path(config["output"])
            self.assertEqual(read_json(run / "score/status.json")["state"], "complete")
            selected = run / "score/best.pt"
            corruption = evaluate(config, selected, split="val")
            candidates = evaluate_planner(config, selected, split="val", max_samples=1)
            self.assertEqual(corruption["samples"], 2)
            self.assertEqual(candidates["samples"], 1)
            for report in (corruption, candidates):
                self.assertEqual(report["numerical_failures"], 0)
                self._assert_finite_numbers(report)
            for stage in ("offline_val", "planner_offline_val"):
                folder = run / stage
                self.assertEqual(read_json(folder / "status.json")["state"], "complete")
                self.assertTrue((folder / "report.md").is_file())
                self.assertTrue((folder / "per_sample.csv").is_file())
                self.assertGreater((folder / "trajectory_000.png").stat().st_size, 1024)

            # Exercise the optional branch against the actual official decoder,
            # then run the same train/evaluate interfaces with the sidecar cache.
            from score_function.data_process.neighbor_cache import build_neighbor_cache

            config["model"]["neighbor_future"] = True
            config["data"]["neighbor_batch_size"] = 2
            config["paths"]["neighbor_cache"] = str(root / "neighbors/index.json")
            config["output"] = str(root / "neighbor_run")
            build_neighbor_cache(config)
            neighbor_hash = file_hash(config["paths"]["neighbor_cache"])
            build_neighbor_cache(config)
            self.assertEqual(neighbor_hash, file_hash(config["paths"]["neighbor_cache"]))
            self.assertEqual(index_hash, file_hash(cache_index))
            train(config, device_override="cpu")
            selected = Path(config["output"]) / "score/best.pt"
            self.assertEqual(evaluate(config, selected, split="val")["numerical_failures"], 0)
            self.assertEqual(
                evaluate_planner(config, selected, split="val", max_samples=1)[
                    "numerical_failures"
                ],
                0,
            )

    def _assert_finite_numbers(self, value):
        if isinstance(value, dict):
            for child in value.values():
                self._assert_finite_numbers(child)
        elif isinstance(value, list):
            for child in value:
                self._assert_finite_numbers(child)
        elif isinstance(value, (int, float)):
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
