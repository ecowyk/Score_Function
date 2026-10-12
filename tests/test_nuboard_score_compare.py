"""Viewer setup reads official paths; no Bokeh server, GPU, or nuPlan needed."""

import importlib.util
import io
import json
import lzma
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import msgpack

SCRIPT = Path(__file__).parents[1] / "scripts/nuboard_score_compare.py"
SPEC = importlib.util.spec_from_file_location("nuboard_score_compare", SCRIPT)
VIEWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEWER)


class NuBoardScoreCompareTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = self.root / "suite"
        (self.suite / "jobs").mkdir(parents=True)
        self.runs = {}
        for job_id, k in VIEWER.JOBS:
            parent = self.root / "reused_old" if k == 20 else self.suite
            run = parent / "closed_loop_nonreactive_agents" / f"{job_id}_original_run"
            (run / "simulation_log").mkdir(parents=True)
            (run / "metrics").mkdir()
            (run / "aggregator_metric").mkdir()
            header = {
                "simulation_main_path": str(run),
                "simulation_folder": "simulation_log",
                "metric_main_path": str(run),
                "metric_folder": "metrics",
                "aggregator_metric_folder": "aggregator_metric",
            }
            (run / "nuboard_123.nuboard").write_bytes(pickle.dumps(header))
            job = {
                "id": job_id,
                "model": "S05-G",
                "K": k,
                "state": "reused" if k == 20 else "complete",
                "result_dir": str(run),
            }
            (self.suite / "jobs" / f"{job_id}.json").write_text(json.dumps(job))
            self.runs[k] = run

    def test_four_jobs_and_reused_run_keep_exact_result_paths(self):
        rows = VIEWER.discover_runs(self.suite)
        self.assertEqual([row["K"] for row in rows], [0, 20, 50, 200])
        self.assertEqual(rows[1]["state"], "reused")
        self.assertEqual(Path(rows[1]["result_dir"]), self.runs[20])
        self.assertTrue(all(Path(row["simulation_dir"]).name == "simulation_log" for row in rows))

    def test_list_cli_works_without_loading_nuplan_or_models(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--suite", str(self.suite), "--list"],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("DP_K0 [complete]", result.stdout)
        self.assertIn("S05-G_K20 [reused]", result.stdout)
        self.assertIn(str(self.runs[200]), result.stdout)

    def test_incomplete_and_wrong_model_jobs_fail(self):
        path = self.suite / "jobs/S05-G_K50.json"
        original = json.loads(path.read_text())
        for key, value in (("state", "running"), ("model", "S05-L"), ("K", 20)):
            with self.subTest(key=key):
                path.write_text(json.dumps({**original, key: value}))
                with self.assertRaises(ValueError):
                    VIEWER.discover_runs(self.suite)

    def test_missing_saved_logs_and_ambiguous_nuboard_fail(self):
        run = self.runs[200]
        (run / "simulation_log").rmdir()
        with self.assertRaisesRegex(ValueError, "saved simulation folder is unavailable"):
            VIEWER.discover_runs(self.suite)
        (run / "simulation_log").mkdir()
        (run / "nuboard_456.nuboard").write_bytes((run / "nuboard_123.nuboard").read_bytes())
        with self.assertRaisesRegex(ValueError, "expected one .nuboard"):
            VIEWER.discover_runs(self.suite)

    def test_notebook_environment_is_set_and_distributed_state_is_removed(self):
        env, paths = VIEWER.runtime_environment(
            self.root,
            self.suite,
            {
                "RANK": "3",
                "CUDA_VISIBLE_DEVICES": "4,5,6,7",
                "NUPLAN_HYDRA_CONFIG_PATH": "/old",
                "TORCHELASTIC_RUN_ID": "old",
                "PYTHONPATH": "/keep",
                "HOME": "/home/unchanged",
            },
        )
        self.assertNotIn("RANK", env)
        self.assertNotIn("NUPLAN_HYDRA_CONFIG_PATH", env)
        self.assertNotIn("TORCHELASTIC_RUN_ID", env)
        self.assertEqual(env["NUPLAN_SIMULATION_ALLOW_ANY_BUILDER"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["HOME"], "/home/unchanged")
        self.assertEqual(paths[0], self.suite / "code/lg")
        self.assertTrue(env["PYTHONPATH"].endswith("/keep"))

    def test_log_decoder_preserves_history_and_discards_unused_planner(self):
        serialized = pickle.dumps(
            SimpleNamespace(
                planner={"unused_model": True},
                scenario="saved_scenario",
                simulation_history=[1, 2, 3],
            )
        )
        for suffix, data in (
            (".pkl.xz", serialized),
            (".msgpack.xz", msgpack.packb(serialized)),
        ):
            with self.subTest(suffix=suffix):
                path = self.root / ("log" + suffix)
                original = lzma.compress(data)
                path.write_bytes(original)
                log = VIEWER.load_log_cpu(path)
                self.assertIsNone(log.planner)
                self.assertEqual(log.scenario, "saved_scenario")
                self.assertEqual(log.simulation_history, [1, 2, 3])
                self.assertEqual(path.read_bytes(), original)
        loader = VIEWER.CPUUnpickler(io.BytesIO(serialized))
        self.assertIs(
            loader.find_class("torch.storage", "_load_from_bytes"), VIEWER.cpu_torch_bytes
        )


if __name__ == "__main__":
    unittest.main()
