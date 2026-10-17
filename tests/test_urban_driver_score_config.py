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
