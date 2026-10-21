"""Exercise nuPlan's real metric callbacks against relocated queue outputs."""

import copy
import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import nuplan
import pandas as pd
from omegaconf import OmegaConf

from nuplan.planning.metrics.metric_engine import JSON_FILE_EXTENSION
from nuplan.planning.script.builders.metric_aggregator_builder import build_metrics_aggregators
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_native_score_queue.py"
spec = importlib.util.spec_from_file_location("native_score_queue_metrics", SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)

CHALLENGE = "closed_loop_nonreactive_agents"


def official_config(output):
    """Use the installed devkit's actual nonreactive aggregator definition."""
    config_path = (Path(nuplan.__file__).resolve().parent / "planning/script/config/simulation"
                   / "metric_aggregator/closed_loop_nonreactive_agents_weighted_average.yaml")
    aggregators = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    # Only the output filename is fixed; retain the official weights and challenge.
    next(iter(aggregators.values()))["file_name"] = "official_metrics.parquet"
    metrics = str(output / "metrics")
    return {
        "output_dir": str(output), "metric_dir": "metrics", "scenario_builder": {},
        "runner_report_file": "runner_report.parquet", "metric_aggregator": aggregators,
        "aggregator_save_path": str(output / "aggregator_metric"),
        "main_callback": {
            "metric_file_callback": {
                "metric_file_output_path": metrics, "scenario_metric_paths": [metrics],
                "delete_scenario_metric_files": True,
            },
            "metric_aggregator_callback": {"metric_save_path": metrics},
        },
    }


def integrate_and_aggregate(cfg):
    """Write two synthetic scenario metric records, then use both official callbacks."""
    output = Path(cfg["output_dir"])
    metrics = output / cfg["metric_dir"]
    metrics.mkdir(parents=True, exist_ok=True)
    keys = [("log_0", "0000000000000000"), ("log_1", "0000000000000001")]
    for (log, token), score in zip(keys, (.6, .8)):
        record = {
            "log_name": log, "scenario_name": token, "scenario_type": "synthetic_type",
            "planner_name": "synthetic_planner",
            "metric_statistics_name": "ego_progress_along_expert_route", "metric_score": [score],
        }
        (metrics / (token + JSON_FILE_EXTENSION)).write_bytes(pickle.dumps([record]))
    MetricFileCallback(**cfg["main_callback"]["metric_file_callback"]).on_run_simulation_end()
    aggregators = build_metrics_aggregators(OmegaConf.create(cfg))
    MetricAggregatorCallback(
        metric_aggregators=aggregators, **cfg["main_callback"]["metric_aggregator_callback"]
    ).on_run_simulation_end()
    (output / "code/hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), output / "code/hydra/config.yaml")
    pd.DataFrame([
        {"log_name": log, "scenario_name": token, "succeeded": True} for log, token in keys
    ]).to_parquet(output / "runner_report.parquet")
    return keys, aggregators


class NativeQueueOfficialMetricsTests(unittest.TestCase):
    def test_official_callbacks_reproduce_missing_aggregate_and_accept_fixed_paths(self):
        with tempfile.TemporaryDirectory(prefix="native_queue_metrics_") as temporary:
            root = Path(temporary)
            original = official_config(root / "baseline_smoke/pdm_open")
            untouched = copy.deepcopy(original)
            with self.assertLogs(
                "nuplan.planning.simulation.main_callback.metric_aggregator_callback", level="WARNING"
            ) as logs:
                integrate_and_aggregate(original)
            self.assertTrue(any("No metric files found" in line for line in logs.output))
            self.assertEqual(len(list((Path(original["output_dir"]) / "metrics").glob("*.parquet"))), 1)
            self.assertEqual(list(Path(original["aggregator_save_path"]).glob("*.parquet")), [])
            with self.assertRaisesRegex(ValueError, "official aggregate"):
                queue.read_result(original["output_dir"], 2)

            fixed = queue.execution_config(original, root / "queue/pdm_open/raw", 8)
            self.assertEqual(fixed["metric_aggregator"], original["metric_aggregator"])
            keys, aggregators = integrate_and_aggregate(fixed)
            self.assertEqual(aggregators[0].challenge, CHALLENGE)
            _, detail, actual_keys, score = queue.read_result(fixed["output_dir"], 2)
            self.assertEqual(actual_keys, keys)
            self.assertEqual(len(detail), 2)
            self.assertAlmostEqual(score, 70.)
            self.assertEqual(original, untouched)

            # K20 and its preflight relocate already-qualified metric paths again.
            previous = fixed
            for stage in ("S10G_K20", "K20_smoke"):
                current = queue.execution_config(previous, root / "queue/pdm_open" / stage, 8)
                self.assertEqual(current["metric_dir"], fixed["metric_dir"])
                _, aggregators = integrate_and_aggregate(current)
                self.assertEqual(aggregators[0].challenge, CHALLENGE)
                self.assertAlmostEqual(queue.read_result(current["output_dir"], 2)[3], 70.)
                previous = current


if __name__ == "__main__":
    unittest.main(verbosity=2)
