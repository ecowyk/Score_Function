"""Global cap/filter wiring, frozen selection, and reuse without sampling bias."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from score_function.data_process.data_processor import allowed_databases, reusable_records
from score_function.data_process.scenario_selection import (
    freeze_selection,
    log_name,
    official_scenarios,
    restore_scenario,
)
from score_function.tools.experiment_suite import make_plan, parser
from score_function.utils.train_utils import atomic_write, file_hash


class GlobalPreprocessingTests(unittest.TestCase):
    def config(self, root):
        return make_plan(parser().parse_args(["--root", str(root)]))["experiments"][0]["config"]

    def test_dotted_log_names_with_and_without_db_extension(self):
        name = "2021.06.14.18.42.45_veh-12_03445_03902"
        self.assertEqual(log_name(name), name)
        self.assertEqual(log_name("/somewhere/" + name + ".db"), name)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / (name + ".db")).touch()
            (root / "excluded.db").touch()
            cfg = self.config(root)
            cfg["paths"].update(
                database_dir=str(root), train_log_allowlist=str(root / "allow.json")
            )
            for entry in (name, name + ".db"):
                atomic_write(root / "allow.json", [entry])
                files, excluded = allowed_databases(cfg)
                self.assertEqual([file.stem for file in files], [name])
                self.assertEqual(excluded, 1)

    def test_official_builder_receives_all_logs_and_one_global_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            cfg["data"]["total_scenarios"] = 7
            plan = [{"log": f"log{i}", "db": f"/db/log{i}.db"} for i in range(3)]
            builder, worker, filters = MagicMock(), MagicMock(), MagicMock()
            builder.return_value.get_scenarios.side_effect = lambda *args: random.sample(
                range(30), 7
            )
            modules = {
                "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder": SimpleNamespace(
                    NuPlanScenarioBuilder=builder
                ),
                "nuplan.planning.scenario_builder.scenario_filter": SimpleNamespace(
                    ScenarioFilter=filters
                ),
                "nuplan.planning.utils.multithreading.worker_parallel": SimpleNamespace(
                    SingleMachineParallelExecutor=worker
                ),
            }
            before = random.getstate()
            with patch.dict("sys.modules", modules):
                selected = official_scenarios(cfg, plan)
                again = official_scenarios(cfg, plan)
            self.assertEqual(selected, again)
            self.assertEqual(before, random.getstate())
            self.assertEqual(builder.call_args.args[3], [row["db"] for row in plan])
            self.assertEqual(filters.call_args.kwargs["limit_total_scenarios"], 7)
            self.assertEqual(filters.call_args.kwargs["log_names"], ["log0", "log1", "log2"])
            self.assertTrue(filters.call_args.kwargs["shuffle"])
            self.assertTrue(filters.call_args.kwargs["expand_scenarios"])
            self.assertFalse(filters.call_args.kwargs["remove_invalid_goals"])
            self.assertIsNone(filters.call_args.kwargs["timestamp_threshold_s"])
            self.assertEqual(builder.return_value.get_scenarios.call_count, 2)
            self.assertEqual(worker.return_value._executor.shutdown.call_count, 2)

    def test_selection_is_frozen_and_resume_never_reselects(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            cfg["data"]["total_scenarios"] = 3
            plan = [
                {"log": "a", "db": "/a.db", "split": "train", "recording": "a"},
                {"log": "b", "db": "/b.db", "split": "val", "recording": "b"},
            ]
            candidates = [
                SimpleNamespace(
                    log_name=name,
                    token=str(i),
                    _initial_lidar_timestamp=i,
                    _map_name="map",
                    scenario_type="turn",
                    _scenario_extraction_info=None,
                )
                for i, name in enumerate(("a", "b", "a"))
            ]
            loader = MagicMock(return_value=candidates)
            first = freeze_selection(cfg, plan, "protocol", loader=loader)
            self.assertEqual(first["selected_scenarios"], 3)
            self.assertEqual(sum(row["count"] for row in first["logs"]), 3)
            self.assertEqual([row["split"] for row in first["logs"]], ["train", "val"])
            self.assertEqual(freeze_selection(cfg, plan, "protocol", loader=loader), first)
            loader.assert_called_once()
            with self.assertRaisesRegex(ValueError, "selection changed"):
                freeze_selection(cfg, plan, "changed", loader=loader)
            path = Path(cfg["paths"]["data_output"]) / "selection" / first["logs"][0]["path"]
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modified"):
                freeze_selection(cfg, plan, "protocol", loader=loader)

    def test_over_cap_is_rejected_before_feature_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            cfg["data"]["total_scenarios"] = 1
            with self.assertRaisesRegex(ValueError, "exceeded"):
                freeze_selection(cfg, [], "id", loader=lambda *args: [None, None])

    def test_restore_preserves_official_constructor_fields(self):
        cfg = self.config("/workspace")
        constructor, extraction = MagicMock(), MagicMock()
        modules = {
            "nuplan.common.actor_state.vehicle_parameters": SimpleNamespace(
                get_pacifica_parameters=lambda: "vehicle"
            ),
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario": SimpleNamespace(
                NuPlanScenario=constructor
            ),
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils": SimpleNamespace(
                ScenarioExtractionInfo=extraction
            ),
        }
        descriptor = {
            "token": "t1",
            "start_time_us": 123,
            "map": "boston",
            "scenario_type": "turn",
            "extraction": {"scenario_duration": 20},
        }
        with patch.dict("sys.modules", modules):
            restore_scenario(cfg, {"db": "/db/a.db"}, descriptor)
        self.assertEqual(constructor.call_args.kwargs["initial_lidar_timestamp"], 123)
        self.assertEqual(constructor.call_args.kwargs["log_file_load_path"], "/db/a.db")
        extraction.assert_called_once_with(scenario_duration=20)

    def test_reuse_only_selected_frames_and_recompute_corrupt_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root)
            cfg["paths"]["reuse_features_from"] = str(root / "old")
            row = {"log": "a", "db": str(root / "a.db"), "recording": "a", "split": "val"}
            records, descriptors = [], []
            for i in range(3):
                path = root / f"{i}.npz"
                np.savez(path, ego_agent_future=np.zeros((80, 3)))
                descriptor = {"token": str(i), "start_time_us": i, "map": "map"}
                descriptors.append(descriptor)
                records.append(
                    {
                        **row,
                        **descriptor,
                        "split": "train",
                        "feature": str(path),
                        "feature_sha256": file_hash(path),
                    }
                )
            atomic_write(root / "old/manifests/a.json", {"records": records, "complete": False})
            reused = reusable_records(cfg, row, [descriptors[1]])
            self.assertEqual(list(reused), ["1"])
            self.assertEqual(reused["1"]["split"], "val")
            (root / "1.npz").write_bytes(b"interrupted old write")
            self.assertEqual(reusable_records(cfg, row, [descriptors[1]]), {})
            wrong = {**descriptors[0], "start_time_us": -1}
            with self.assertRaisesRegex(ValueError, "does not match"):
                reusable_records(cfg, row, [wrong])

    def test_suite_new_output_global_default_and_override(self):
        cfg = self.config("/workspace")
        self.assertEqual(cfg["data"]["total_scenarios"], 1_000_000)
        self.assertFalse(cfg["data"]["remove_invalid_goals"])
        self.assertIn("score_matrix_1m", cfg["paths"]["data_output"])
        plan = make_plan(parser().parse_args(["--root", "/workspace", "--total-scenarios", "123"]))
        self.assertEqual(plan["experiments"][0]["config"]["data"]["total_scenarios"], 123)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            make_plan(parser().parse_args(["--root", "/workspace", "--total-scenarios", "0"]))


if __name__ == "__main__":
    unittest.main()
