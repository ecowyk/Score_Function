"""Synthetic storage and exact-resume tests; these are not driving experiments."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from score_function.model.score_branch import build_model
from score_function.train import train
from score_function.train_epoch import validate_epoch
from score_function.utils.config import METHOD, load_config
from score_function.utils.dataset import (
    EpochBatchSampler,
    ShardedDataset,
    build_data_loader,
    validate_shard,
    write_shard,
)
from score_function.utils.planner_utils import planner_identity
from score_function.utils.train_utils import atomic_write, file_hash, load_tensor


def fixture(root):
    """Real branch + small shards; fake provenance is never loaded as a planner."""
    root = Path(root)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/score_function.json", root)
    config["model"].update(hidden_dim=24, num_heads=3)
    config["training"].update(
        max_epochs=2,
        minimum_epochs=1,
        batch_size=4,
        microbatch_size=1,
        warmup_epochs=0,
        validation_repeats=2,
        num_workers=0,
        checkpoint_every_updates=1,
        log_every_updates=1,
    )
    config["runtime"].update(device="cpu", cpu_threads=1)
    config["cache"] = str(root / "cache" / "index.json")
    config["output"] = str(root / "run")
    official = root / "official" / "diffusion_planner"
    official.mkdir(parents=True)
    (official / "__init__.py").write_text("# synthetic provenance fixture", encoding="utf-8")
    (root / "args.json").write_text("{}", encoding="utf-8")
    (root / "model.pth").write_bytes(b"Synthetic identity only: training needs no planner load.")
    config["paths"].update(
        planner_dir=str(official.parent),
        planner_args=str(root / "args.json"),
        planner_checkpoint=str(root / "model.pth"),
    )
    generator = torch.Generator().manual_seed(92)
    rows = []
    for index in range(11):
        rows.append(
            {
                "record": {
                    "token": f"t{index}",
                    "recording": f"drive{index}",
                    "start_time_us": index,
                    "split": "train" if index < 8 else "val",
                },
                "target": torch.randn(80, 4, generator=generator),
                "context": torch.randn(107, 192, generator=generator),
                "route": torch.randn(192, generator=generator),
            }
        )
    shards = []
    for index, start in enumerate(range(0, len(rows), 4)):
        directory = root / "cache" / f"shard_{index:06d}"
        write_shard(directory, rows[start : start + 4], "fixture")
        shards.append(
            {"path": directory.name, "marker_sha256": file_hash(directory / "complete.json")}
        )
    atomic_write(
        config["cache"],
        {
            "schema_version": 1,
            "method": METHOD,
            "metadata": {
                "identity": "fixture",
                "samples": len(rows),
                "planner": planner_identity(config),
                "ego_normalizer": {"mean": [0.0] * 4, "std": [1.0] * 4},
            },
            "shards": shards,
        },
    )
    return config, rows


def ddp_worker(rank, config, store, resume, stop):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(store).resolve().as_uri(), rank=rank, world_size=2
    )
    train(config, resume=resume, device_override="cpu", stop_after_updates=stop)


class PipelineTests(unittest.TestCase):
    def test_initial_branch_is_retained_when_ema_validation_worsens(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            # Better raw weights must not silently enter an EMA-only selection policy.
            with patch(
                "score_function.train.validate_epoch",
                side_effect=[
                    {"raw": 1.0, "ema": 1.0},
                    {"raw": 0.2, "ema": 3.0},
                    {"raw": 0.1, "ema": 5.0},
                ],
            ):
                train(config, device_override="cpu")
            selected = load_tensor(Path(config["output"]) / "score/best.pt")
            self.assertEqual((selected["step"], selected["weight_kind"]), (0, "initial"))
            self.assertNotIn("model", selected)
            self.assertNotIn("head", selected)
            torch.manual_seed(config["training"]["seed"])
            initial = build_model(config, "cpu").state_dict()
            self.assertEqual(set(initial), set(selected["score_branch"]))
            for name, value in selected["score_branch"].items():
                torch.testing.assert_close(value, initial[name], atol=0, rtol=0)

    def test_shards_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, rows = fixture(tmp)
            dataset = ShardedDataset(config["cache"], "train", max_open_shards=1)
            try:
                self.assertEqual(len(dataset), 8)
                for index in (7, 0, 5):
                    for key in ("target", "context", "route"):
                        torch.testing.assert_close(
                            dataset[index][key], rows[index][key], rtol=0, atol=0
                        )
            finally:
                dataset.close()
            directory = Path(config["cache"]).parent / "shard_000000"
            with (directory / "target.npy").open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaises(ValueError):
                validate_shard(directory, "fixture")

    def test_other_project_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            index = json.loads(Path(config["cache"]).read_text(encoding="utf-8"))
            index["method"] = "frozen_planner_score_head_v1"
            atomic_write(config["cache"], index)
            with self.assertRaises(ValueError):
                ShardedDataset(config["cache"], "train")

    def test_disjoint_global_sampler_and_resume_cursor(self):
        kwargs = dict(size=19, global_batch=8, microbatch=2, world_size=2, seed=19, epoch=2)
        left = list(EpochBatchSampler(rank=0, **kwargs))
        right = list(EpochBatchSampler(rank=1, **kwargs))
        self.assertFalse(set(sum(left, [])) & set(sum(right, [])))
        self.assertEqual(len(set(sum(left + right, []))), 16)
        resumed = list(EpochBatchSampler(rank=0, start_update=1, **kwargs))
        self.assertEqual(resumed, left[2:])

    def test_spawn_loader_reads_memory_mapped_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, rows = fixture(tmp)
            config["training"]["num_workers"] = 2
            dataset = ShardedDataset(config["cache"], "train")
            try:
                seen = []
                for batch in build_data_loader(dataset, config["training"]):
                    seen.extend(batch["tokens"])
                    index = int(batch["tokens"][0][1:])
                    torch.testing.assert_close(batch["target"][0], rows[index]["target"])
                self.assertEqual(seen, [f"t{index}" for index in range(8)])
            finally:
                dataset.close()

    def test_official_allowlist_excludes_unlisted_logs(self):
        from score_function.data_process.data_processor import allowed_databases

        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            folder = Path(tmp) / "db"
            folder.mkdir()
            for name in ("official_train", "official_val"):
                (folder / f"{name}.db").touch()
            allowlist = Path(tmp) / "allow.json"
            atomic_write(allowlist, ["official_train"])
            config["paths"].update(database_dir=str(folder), train_log_allowlist=str(allowlist))
            selected, excluded = allowed_databases(config)
            self.assertEqual([path.stem for path in selected], ["official_train"])
            self.assertEqual(excluded, 1)

    def test_validation_reduction_matches_selected_checkpoint(self):
        from score_function.utils.checkpoint import load_selected

        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            train(config, device_override="cpu")
            state = load_tensor(Path(config["output"]) / "score/best.pt")
            model = build_model(config, "cpu")
            model.load_state_dict(state["score_branch"])
            dataset = ShardedDataset(config["cache"], "val")
            try:
                loss = validate_epoch(
                    {"selected": model}, dataset, config["training"], torch.device("cpu")
                )
            finally:
                dataset.close()
            self.assertEqual(loss["selected"], state["val_dsm"])
            checkpoint = Path(config["output"]) / "score/best.pt"
            deployed, loaded = load_selected(config, checkpoint, "cpu")
            self.assertFalse(deployed.training)
            self.assertFalse(any(parameter.requires_grad for parameter in deployed.parameters()))
            self.assertEqual(loaded["sigma_score"], config["training"]["sigma"])
            changed = copy.deepcopy(config)
            changed["training"]["sigma"] *= 2
            with self.assertRaisesRegex(ValueError, "sigma"):
                load_selected(changed, checkpoint, "cpu")
            changed = copy.deepcopy(config)
            changed["model"]["hidden_dim"] = 48
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_selected(changed, checkpoint, "cpu")

    def _assert_state_equal(self, full, resumed):
        for name in ("step", "epoch", "cursor", "plateau", "best"):
            self.assertEqual(full[name], resumed[name])
        for kind in ("score_branch", "ema_branch"):
            self.assertEqual(set(full[kind]), set(resumed[kind]))
            for key in full[kind]:
                torch.testing.assert_close(full[kind][key], resumed[kind][key], rtol=0, atol=0)
        self.assertEqual(full["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
        for index, parameters in full["optimizer"]["state"].items():
            for key, value in parameters.items():
                other = resumed["optimizer"]["state"][index][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    self.assertEqual(value, other)
        self.assertEqual(len(full["history"]), len(resumed["history"]))
        for first, second in zip(full["history"], resumed["history"]):
            self.assertEqual(
                {key: value for key, value in first.items() if key != "elapsed_s"},
                {key: value for key, value in second.items() if key != "elapsed_s"},
            )

    def test_single_process_exact_resume_and_config_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            train(config, device_override="cpu")
            full = load_tensor(Path(config["output"]) / "score/last.pt")
            for stop in (1, 2):
                # 2 is the last update before the first epoch's validation.
                config["output"] = str(Path(tmp) / f"resume_{stop}")
                train(config, device_override="cpu", stop_after_updates=stop)
                train(config, resume=True, device_override="cpu")
                resumed = load_tensor(Path(config["output"]) / "score/last.pt")
                self._assert_state_equal(full, resumed)
                choice = json.loads(
                    (Path(config["output"]) / "score/selection.json").read_text(encoding="utf-8")
                )
                initial = json.loads(
                    (Path(config["output"]) / "score/initial_validation.json").read_text(
                        encoding="utf-8"
                    )
                )["raw"]
                expected = min(initial, min(row["val_dsm_ema"] for row in resumed["history"]))
                self.assertEqual(choice["val_dsm"], expected)
            for section, key, value in (
                ("training", "sigma", 0.2),
                ("model", "dropout", 0.2),
            ):
                changed = copy.deepcopy(config)
                changed[section][key] = value
                with self.assertRaisesRegex(ValueError, "Resume configuration"):
                    train(changed, resume=True, device_override="cpu")

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo unavailable")
    def test_two_rank_exact_resume_including_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = fixture(tmp)
            mp.spawn(
                ddp_worker,
                args=(config, str(Path(tmp) / "store_full"), False, None),
                nprocs=2,
                join=True,
            )
            full = load_tensor(Path(config["output"]) / "score/last.pt")
            for stop in (1, 2):
                config["output"] = str(Path(tmp) / f"resumed_{stop}")
                mp.spawn(
                    ddp_worker,
                    args=(config, str(Path(tmp) / f"store_partial_{stop}"), False, stop),
                    nprocs=2,
                    join=True,
                )
                mp.spawn(
                    ddp_worker,
                    args=(config, str(Path(tmp) / f"store_resume_{stop}"), True, None),
                    nprocs=2,
                    join=True,
                )
                resumed = load_tensor(Path(config["output"]) / "score/last.pt")
                self._assert_state_equal(full, resumed)
                self.assertEqual(len(resumed["ranks"]), 2)


if __name__ == "__main__":
    unittest.main()
