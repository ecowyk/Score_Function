"""Matrix isolation, real temporal coverage, and optional neighbor conditioning."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_model_contracts import normalizer
from test_training_pipeline import fixture

from score_function.model.score_branch import ScoreFunctionBranch
from score_function.tools.experiment_suite import Suite, make_plan, parser
from score_function.train import train
from score_function.utils.dataset import ShardedDataset, collate_cpu
from score_function.utils.neighbor import prediction_neighbors
from score_function.utils.train_utils import atomic_write, file_hash, load_tensor


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matrix_isolated_outputs_shared_inputs_and_matched_wide_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args(["--root", tmp])
            plan = make_plan(args)
            entries = {row["id"]: row for row in plan["experiments"]}
            self.assertEqual(
                len({row["config"]["paths"]["run_dir"] for row in entries.values()}), 8
            )
            self.assertEqual(len({row["config"]["paths"]["cache"] for row in entries.values()}), 1)
            for row in entries.values():
                config = row["config"]
                self.assertIsNone(config["data"]["timestamp_spacing_s"])
                self.assertIsNone(config["data"]["max_scenarios_per_db"])
                self.assertEqual(config["training"]["batch_size"], 2048)
            local = ScoreFunctionBranch(hidden_dim=24, num_heads=3)
            wide = ScoreFunctionBranch(hidden_dim=24, num_heads=3, post_dilations=(4, 12))
            self.assertEqual(
                sum(p.numel() for p in local.parameters()),
                sum(p.numel() for p in wide.parameters()),
            )
            self.assertEqual(wide.temporal_receptive_field, 77)
            for identifier in ("S05-W", "S10-W", "S05-N"):
                self.assertEqual(entries[identifier]["config"]["model"]["post_dilations"], [4, 12])

    def test_global_attention_connects_first_and_last_ego_points(self):
        torch.manual_seed(12)
        x = torch.randn(1, 80, 4, requires_grad=True)
        scene, route = torch.randn(1, 107, 192), torch.randn(1, 192)
        local = ScoreFunctionBranch(hidden_dim=24, num_heads=3, dropout=0).eval()
        full = ScoreFunctionBranch(
            hidden_dim=24, num_heads=3, dropout=0, temporal_attention=True
        ).eval()
        grad_local = torch.autograd.grad(local(x, scene, route)[0, 0].sum(), x)[0]
        grad_full = torch.autograd.grad(full(x, scene, route)[0, 0].sum(), x)[0]
        self.assertEqual(grad_local[0, -1].abs().sum().item(), 0)
        self.assertGreater(grad_full[0, -1].abs().sum().item(), 0)

    def test_neighbor_padding_is_invariant_and_all_missing_is_finite(self):
        model = ScoreFunctionBranch(
            hidden_dim=24, num_heads=3, dropout=0, neighbor_future=True
        ).eval()
        x, scene, route = torch.randn(2, 80, 4), torch.randn(2, 107, 192), torch.randn(2, 192)
        future = torch.randn(2, 10, 80, 4)
        valid = torch.zeros(2, 10, dtype=torch.bool)
        valid[0, 0] = True
        a = model(x, scene, route, future, valid)
        corrupt = future.clone()
        corrupt[~valid] = float("nan")
        torch.testing.assert_close(a, model(x, scene, route, corrupt, valid), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(a).all())
        with self.assertRaisesRegex(ValueError, "requires predicted"):
            model(x, scene, route)
        a.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_neighbor_normalization_uses_neighbor_statistics(self):
        joint = torch.randn(2, 11, 80, 4)
        past = torch.ones(2, 32, 21, 11)
        past[:, 3] = 0
        stats = normalizer()
        condition = prediction_neighbors(joint, {"neighbor_agents_past": past}, stats)
        expected = (joint[:, 1:] - stats.mean[1:]) / stats.std[1:]
        expected[:, 3] = 0
        torch.testing.assert_close(condition["neighbor_future"], expected)
        self.assertFalse(condition["neighbor_valid"][:, 3].any())

    def test_neighbor_cache_alignment_and_exact_training_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, rows = fixture(tmp)
            config["model"]["neighbor_future"] = True
            root = Path(tmp)
            index = root / "neighbors/index.json"
            config["paths"]["neighbor_cache"] = str(index)
            source = json.loads(Path(config["cache"]).read_text())
            items = []
            for number, item in enumerate(source["shards"]):
                directory = index.parent / item["path"]
                directory.mkdir(parents=True)
                count = len(rows[number * 4 : number * 4 + 4])
                hashes = {}
                for key, value in (
                    ("neighbor_future", np.ones((count, 10, 80, 4), dtype=np.float32)),
                    ("neighbor_valid", np.ones((count, 10), dtype=bool)),
                ):
                    path = directory / f"{key}.npy"
                    np.save(path, value)
                    hashes[path.name] = file_hash(path)
                atomic_write(
                    directory / "complete.json",
                    {
                        "identity": "neighbors",
                        "count": count,
                        "hashes": hashes,
                        "source_marker_sha256": item["marker_sha256"],
                    },
                )
                items.append(
                    {"path": item["path"], "marker_sha256": file_hash(directory / "complete.json")}
                )
            atomic_write(
                index,
                {
                    "identity": "neighbors",
                    "shards": items,
                    "source_cache_sha256": file_hash(config["cache"]),
                },
            )
            dataset = ShardedDataset(config["cache"], "train", neighbor_index=index)
            batch = collate_cpu([dataset[0], dataset[1]])
            self.assertEqual(batch["neighbor_future"].shape, (2, 10, 80, 4))
            dataset.close()
            train(config, device_override="cpu", stop_after_updates=1)
            train(config, resume=True, device_override="cpu")
            resumed = load_tensor(Path(config["output"]) / "score/last.pt")
            config["output"] = str(root / "uninterrupted")
            train(config, device_override="cpu")
            full = load_tensor(Path(config["output"]) / "score/last.pt")
            for key in full["score_branch"]:
                torch.testing.assert_close(
                    full["score_branch"][key], resumed["score_branch"][key], rtol=0, atol=0
                )
            invalid = json.loads(index.read_text())
            invalid["source_cache_sha256"] = "wrong"
            atomic_write(index, invalid)
            with self.assertRaisesRegex(ValueError, "does not match"):
                ShardedDataset(config["cache"], neighbor_index=index)

    def test_suite_stage_graph_starts_eight_trainers_and_no_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(parser().parse_args(["--root", tmp]))
            suite = Suite(plan, resume=False)
            commands = []

            def stage(entry, command, **kwargs):
                commands.append((entry["id"], command, kwargs))
                if command == "train":
                    directory = Path(entry["config"]["paths"]["run_dir"]) / "score"
                    atomic_write(directory / "status.json", {"state": "complete"})
                    atomic_write(directory / "selection.json", {"weight_kind": "ema", "step": 1})
                    (directory / "best.pt").write_bytes(b"fixture")

            with patch.object(suite, "stage", side_effect=stage):
                self.assertEqual(suite.run(), 0)
            self.assertEqual(sum(command == "train" for _, command, _ in commands), 8)
            self.assertEqual(sum(command == "prepare" for _, command, _ in commands), 1)
            self.assertEqual(sum(command == "cache" for _, command, _ in commands), 1)
            self.assertEqual(sum(command == "cache-neighbors" for _, command, _ in commands), 1)
            self.assertFalse(any("evaluate" in command for _, command, _ in commands))

    def test_failed_trainer_is_reported_without_hiding_successful_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(parser().parse_args(["--root", tmp]))
            suite = Suite(plan, resume=False)

            def stage(entry, command, **kwargs):
                if command != "train":
                    return
                if entry["id"] == "S05-G":
                    raise RuntimeError("injected worker failure")
                directory = Path(entry["config"]["paths"]["run_dir"]) / "score"
                atomic_write(directory / "status.json", {"state": "complete"})
                atomic_write(directory / "selection.json", {"weight_kind": "ema", "step": 1})
                (directory / "best.pt").write_bytes(b"fixture")

            with patch.object(suite, "stage", side_effect=stage):
                self.assertEqual(suite.run(), 1)
            summary = json.loads((Path(plan["output"]) / "summary.json").read_text())
            self.assertEqual(summary["experiments"]["S05-G"]["state"], "failed")
            self.assertEqual(
                sum(item["state"] == "complete" for item in summary["experiments"].values()), 7
            )


if __name__ == "__main__":
    unittest.main()
