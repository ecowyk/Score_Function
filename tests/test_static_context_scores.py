"""Exact per-scenario metric binding without nuPlan, checkpoints, or a GPU."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

SPEC = importlib.util.spec_from_file_location(
    "static_context_scores", Path(__file__).parents[1] / "scripts/static_context_scores.py"
)
SCORES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORES)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class StaticContextScoresTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.suite = Path(self.temp.name)
        model = self.suite / "models/S05-G"
        self.config_path = model / "config.json"
        self.checkpoint_path = model / "final_ema.pt"
        write_json(
            self.config_path,
            {
                "reference_seed": 123,
                "refinement": {"heading_projection": True},
            },
        )
        self.checkpoint_path.write_bytes(b"fixture weights, never torch-loaded")
        write_json(
            self.suite / "plan.json",
            {
                "gamma": 0.1,
                "heading_projection": True,
                "reference_seed": 123,
                "tokens": ["first_token", "second_token"],
                "files": {
                    str(path): SCORES._digest(path)
                    for path in (self.config_path, self.checkpoint_path)
                },
            },
        )
        self.runs = {}
        for k, raw in ((0, 0.222), (20, 0.81234)):
            job_id = "DP_K0" if k == 0 else "S05-G_K20"
            run = self.suite / "closed_loop_nonreactive_agents" / job_id
            self.runs[k] = run
            (run / "aggregator_metric").mkdir(parents=True)
            (run / "code/hydra").mkdir(parents=True)
            planner = {
                "score_config": str(self.config_path),
                "score_checkpoint": str(self.checkpoint_path),
                "steps": k,
                "gamma": 0.1,
                "heading_projection": None,
            }
            (run / "code/hydra/config.yaml").write_text(
                yaml.safe_dump({"planner": {"score_function_planner": planner}})
            )
            write_json(
                self.suite / "jobs" / f"{job_id}.json",
                {
                    "id": job_id,
                    "model": "S05-G",
                    "K": k,
                    "state": "reused" if k else "complete",
                    "result_dir": str(run),
                },
            )
            rows = [
                {
                    "scenario": "first_token",
                    "scenario_type": "turn",
                    "log_name": "record_A",
                    "score": raw,
                },
                {
                    "scenario": "second_token",
                    "scenario_type": "straight",
                    "log_name": "record_B",
                    "score": 0.5,
                },
                # Summary rows deliberately disagree with scenario scores.
                {"scenario": "final_score", "scenario_type": None, "log_name": None, "score": 0.99},
                {
                    "scenario": "first_token",
                    "scenario_type": "turn",
                    "log_name": None,
                    "score": 0.01,
                },
            ]
            for row in rows:
                row["no_ego_at_fault_collisions"] = 1.0
            pd.DataFrame(rows).to_parquet(run / "aggregator_metric/scores.parquet")
        self.payload = {
            "case_id": "W01",  # A chosen diagnostic group is not a metric filter.
            "token": "first_token",
            "scenario_type": "turn",
            "metadata": {
                "suite": str(self.suite),
                "plan_sha256": SCORES._digest(self.suite / "plan.json"),
                "config_sha256": SCORES._digest(self.config_path),
                "checkpoint_sha256": SCORES._digest(self.checkpoint_path),
                "source_log": str(self.runs[0] / "simulation_log/first_token.msgpack.xz"),
                "reference_seed": 123,
            },
        }

    def update_table(self, k, transform):
        path = self.runs[k] / "aggregator_metric/scores.parquet"
        transform(pd.read_parquet(path)).to_parquet(path)

    def test_exact_scenario_score_reused_job_and_one_read_per_job(self):
        provider = SCORES.ScenarioScores(self.suite)
        original = pd.read_parquet
        with mock.patch.object(pd, "read_parquet", wraps=original) as reader:
            result = provider.for_payload(self.payload, (20,))
            second = copy.deepcopy(self.payload)
            second.update(token="second_token", scenario_type="straight")
            provider.for_payload(second, (20,))
            self.assertEqual(reader.call_count, 2)
        self.assertEqual(result["scope"], "scenario_closed_loop")
        self.assertEqual(result["scale"], 100)
        self.assertAlmostEqual(result["baseline"]["score"], 22.2)
        self.assertAlmostEqual(result["by_k"]["20"]["score"], 81.234)
        self.assertEqual(result["by_k"]["20"]["metrics"]["score"], 0.81234)
        self.assertEqual(result["by_k"]["20"]["log_name"], "record_A")
        self.assertTrue(result["by_k"]["20"]["metric_file"].endswith("scores.parquet"))

    def test_summary_rows_never_replace_missing_scenario(self):
        self.update_table(20, lambda table: table.loc[table.log_name.ne("record_A")])
        with self.assertRaisesRegex(ValueError, "found 0.*summary rows excluded"):
            SCORES.ScenarioScores(self.suite).for_payload(self.payload, (20,))

    def test_duplicate_scenario_is_rejected(self):
        self.update_table(20, lambda table: pd.concat((table, table.iloc[[0]]), ignore_index=True))
        with self.assertRaisesRegex(ValueError, "found 2"):
            SCORES.ScenarioScores(self.suite).for_payload(self.payload, (20,))

    def test_wrong_recording_is_rejected(self):
        def change(table):
            table.loc[0, "log_name"] = "other_recording"
            return table

        self.update_table(20, change)
        with self.assertRaisesRegex(ValueError, "Recording mismatch"):
            SCORES.ScenarioScores(self.suite).for_payload(self.payload, (20,))

    def test_capture_identity_hashes_and_source_run_are_bound(self):
        bad = {
            "suite": str(self.suite / "different"),
            "plan_sha256": "wrong",
            "config_sha256": "wrong",
            "checkpoint_sha256": "wrong",
            "reference_seed": 456,
            "source_log": str(self.runs[20] / "simulation_log/first_token.msgpack.xz"),
        }
        provider = SCORES.ScenarioScores(self.suite)
        for key, value in bad.items():
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload["metadata"][key] = value
                with self.assertRaises(ValueError):
                    provider.for_payload(payload, (20,))

    def test_score_outside_unit_interval_or_nonfinite_is_rejected(self):
        path = self.runs[20] / "aggregator_metric/scores.parquet"
        original = pd.read_parquet(path)
        for value in (float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(value=value):
                table = original.copy()
                table.loc[0, "score"] = value
                table.to_parquet(path)
                with self.assertRaises(ValueError):
                    SCORES.ScenarioScores(self.suite).for_payload(self.payload, (20,))

    def test_missing_or_unfinished_job_has_no_fallback(self):
        with self.assertRaisesRegex(ValueError, "Missing completed"):
            SCORES.ScenarioScores(self.suite).for_payload(self.payload, (50,))
        path = self.suite / "jobs/S05-G_K20.json"
        job = json.loads(path.read_text())
        job["state"] = "running"
        write_json(path, job)
        with self.assertRaisesRegex(ValueError, "not complete/reused"):
            SCORES.ScenarioScores(self.suite).for_payload(self.payload, (20,))

    def test_saved_hydra_k_and_weights_are_verified(self):
        path = self.runs[20] / "code/hydra/config.yaml"
        original = yaml.safe_load(path.read_text())
        alien = self.suite / "alien.pt"
        alien.write_bytes(b"different weights")
        for key, value in (("steps", 20.5), ("gamma", 0.2), ("score_checkpoint", str(alien))):
            with self.subTest(key=key):
                saved = copy.deepcopy(original)
                saved["planner"]["score_function_planner"][key] = value
                path.write_text(yaml.safe_dump(saved))
                with self.assertRaises(ValueError):
                    SCORES.ScenarioScores(self.suite).for_payload(self.payload, (20,))

    def test_only_positive_integer_g_k_values_are_allowed(self):
        provider = SCORES.ScenarioScores(self.suite)
        for ks in ((), (0,), (-1,), (20.5,), (True,), ("20",)):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                provider.for_payload(self.payload, ks)


if __name__ == "__main__":
    unittest.main()
