"""Regression: read saved nuPlan results before Hydra has ever initialized."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_urban_driver_score.py"
spec = importlib.util.spec_from_file_location("prepare_ud_score", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SavedConfigTests(unittest.TestCase):
    def test_smoke_reservations_work_with_real_sequential_worker(self):
        from nuplan.planning.utils.multithreading.worker_pool import Task
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential

        full = {"output_dir": "/result", "number_of_cpus_allocated_per_simulation": 2,
                "number_of_gpus_allocated_per_simulation": 1.0,
                "worker": {"_target_": "RayDistributed"},
                "scenario_filter": {"limit_total_scenarios": None},
                "planner": {"audit_dir": "/result/refinement_audit"}}
        smoke = prepare.make_smoke_config(full, Path("/result"))
        task = Task(fn=lambda value: value + 1,
                    num_cpus=smoke["number_of_cpus_allocated_per_simulation"],
                    num_gpus=smoke["number_of_gpus_allocated_per_simulation"])
        self.assertEqual(Sequential().map(task, [1]), [2])
        self.assertEqual(full["number_of_cpus_allocated_per_simulation"], 2)
        self.assertEqual(full["worker"]["_target_"], "RayDistributed")
        self.assertIsNone(full["scenario_filter"]["limit_total_scenarios"])
        self.assertEqual(smoke["planner"]["audit_dir"], "/result/smoke/refinement_audit")

    def test_read_result_registers_nested_now_resolver_in_fresh_process(self):
        OmegaConf.clear_resolver("now")
        self.assertFalse(OmegaConf.has_resolver("now"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "code/hydra").mkdir(parents=True)
            (root / "aggregator_metric").mkdir()
            OmegaConf.save(OmegaConf.create({
                "date_format": "%Y.%m.%d.%H.%M.%S",
                "output_dir": str(root),
                "runner_report_file": "runner_report.parquet",
                "model": {"future_poses": 16},
                "planner": {"ml_planner": {"model_config": "${model}"}},
                "metric_aggregator": {"file_name": "metrics_${now:${date_format}}",
                                      "save_path": "${output_dir}/aggregator_metric"},
            }), root / "code/hydra/config.yaml")
            original = (root / "code/hydra/config.yaml").read_bytes()
            pd.DataFrame([{"log_name": "log", "scenario_name": "abc", "succeeded": True}]).to_parquet(root / "runner_report.parquet")
            pd.DataFrame([
                {"log_name": "log", "scenario": "abc", "score": .43, "num_scenarios": 1},
                {"log_name": None, "scenario": "final_score", "score": .43, "num_scenarios": 1},
            ]).to_parquet(root / "aggregator_metric/result.parquet")
            cfg, _, keys, score = prepare.read_result(root, 1)
            self.assertTrue(cfg["metric_aggregator"]["file_name"].startswith("metrics_"))
            self.assertNotIn("${", cfg["metric_aggregator"]["file_name"])
            self.assertEqual(cfg["planner"]["ml_planner"]["model_config"], {"future_poses": 16})
            self.assertEqual(keys, [("log", "abc")])
            self.assertEqual(score, 43.)
            self.assertEqual((root / "code/hydra/config.yaml").read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
