"""Guard PLUTO paired-run scheduling and result integrity without GPU/data."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load_module("pluto_queue", ROOT / "scripts/evaluate_pluto_score_queue.py")
fixture = load_module("native_queue_fixtures", ROOT / "tests/test_native_score_queue.py").fixture


class PlutoQueueTests(unittest.TestCase):
    def prepare(self, root):
        exp = root / "dataset/nuplan/exp"
        baseline = (exp / "exp/simulation/closed_loop_nonreactive_agents/urbandriver"
                    / "test14-hard/raw_8gpu_reference")
        cfg, _, _, keys = fixture(baseline)
        cfg["scenario_builder"]["data_root"] = str(root / "dataset/nuplan/nuplan-v1.1/test")
        OmegaConf.save(OmegaConf.create(cfg), baseline / "code/hydra/config.yaml")
        code = root / "pluto"
        (code / "config/planner").mkdir(parents=True)
        model = {
            "_target_": queue.MODEL_TARGET, "future_steps": 80, "num_modes": 12,
            "feature_builder": {
                "_target_": "src.feature_builders.pluto_feature_builder.PlutoFeatureBuilder",
                "build_reference_line": True, "future_horizon": 8, "sample_interval": .1,
            },
        }
        OmegaConf.save(OmegaConf.create({"pluto_planner": {"planner": model}}),
                       code / "config/planner/pluto_planner.yaml")
        models = root / "models"
        models.mkdir()
        for name in ("pluto.ckpt", "score.pt", "score.json"):
            (models / name).write_bytes(b"synthetic weight identity")
        environment = {
            "UD_ROOT": str(root), "NUPLAN_EXP_ROOT": str(exp),
            "NUPLAN_DEVKIT_ROOT": str(root / "nuplan-devkit"),
            "PLUTO_CODE": str(code), "PLUTO_CHECKPOINT": str(models / "pluto.ckpt"),
            "BASELINE_RESULT": str(baseline),
            "SCORE_CONFIG": str(models / "score.json"),
            "SCORE_CHECKPOINT": str(models / "score.pt"),
            "FROZEN_SCORE_CODE": str(root / "frozen_score"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
        return exp, cfg, model, keys, environment

    def complete(self, cfg, out, entry, environment):
        count = len(cfg["scenario_filter"]["scenario_tokens"])
        fixture(out, count)
        OmegaConf.save(OmegaConf.create(cfg), out / "code/hydra/config.yaml")
        planner = cfg["planner"]["pluto_score"]
        if planner["steps"] == 20:
            audits = Path(planner["audit_dir"])
            audits.mkdir()
            audit = {
                "completed_steps": 20, "K": 20, "gamma": .1, "sigma_score": .1,
                "epoch": 529, "weight_kind": "ema", "finite": True,
                "native_pose_count": 80, "score_pose_count": 80,
            }
            for index in range(count):
                (audits / f"{index}.json").write_text(json.dumps(audit))

    def run_main(self, out, environment, simulation):
        with mock.patch.dict(os.environ, environment), \
                mock.patch.object(sys, "argv", ["queue", "--output", str(out)]), \
                mock.patch.object(queue, "verify_pluto_source", return_value=queue.PLUTO_COMMIT), \
                mock.patch.object(queue.shared, "score_preflight"), \
                mock.patch.object(queue, "native_preflight", return_value={"strict": True}), \
                mock.patch.object(queue.shared, "run_simulation", side_effect=simulation), \
                contextlib.redirect_stdout(io.StringIO()):
            queue.main()

    def test_native_and_refined_configs_preserve_the_official_model_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, base, model, keys, env = self.prepare(root)
            untouched = copy.deepcopy(base)
            for steps in (0, 20):
                directory = exp / f"stage_{steps}"
                cfg = queue.make_config(base, model, env["PLUTO_CHECKPOINT"], directory, keys,
                                        steps, env["SCORE_CONFIG"], env["SCORE_CHECKPOINT"])
                planner = cfg["planner"]["pluto_score"]
                self.assertEqual(planner["model_config"], model)
                self.assertEqual(planner["_target_"], queue.PLANNER_TARGET)
                self.assertFalse(planner["_recursive_"])
                self.assertEqual(planner["steps"], steps)
                self.assertEqual(planner["native_device"], "cuda")
                self.assertEqual(planner["device"], "cuda")
                self.assertEqual(planner["score_checkpoint"], env["SCORE_CHECKPOINT"] if steps else "unused")
                for name in ("seed", "observation", "ego_controller", "simulation_metric",
                             "simulation_time_controller", "simulation_history_buffer_duration"):
                    self.assertEqual(cfg[name], base[name])
                self.assertEqual(cfg["worker"]["threads_per_node"], 16)
                self.assertEqual(cfg["number_of_gpus_allocated_per_simulation"], 1)
                self.assertEqual(cfg["number_of_cpus_allocated_per_simulation"], 2)
                self.assertEqual(cfg["scenario_filter"]["scenario_tokens"], sorted(k[1] for k in keys))
                self.assertEqual(cfg["metric_dir"], "metrics/closed_loop_nonreactive_agents")
                self.assertEqual(cfg["main_callback"]["aggregate"]["metric_save_path"],
                                 str(directory / cfg["metric_dir"]))
            self.assertEqual(base, untouched)

    def test_smokes_precede_full_native_then_K20_and_completed_runs_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out, calls = exp / "pluto_queue", []
            def simulation(cfg, directory, entry, environment):
                calls.append(directory.name)
                if "smoke" in directory.name:
                    self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
                    self.assertTrue(cfg["worker"]["_target_"].endswith(".Sequential"))
                else:
                    self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
                    self.assertTrue(cfg["worker"]["_target_"].endswith(".RayDistributed"))
                self.complete(cfg, directory, entry, environment)
            self.run_main(out, env, simulation)
            self.assertEqual(calls, ["K0_smoke", "K20_smoke", "raw", "S10G_K20"])
            summary = json.loads((out / "summary.json").read_text())[queue.VARIANT]
            self.assertEqual(summary["scenarios"], 272)
            self.assertEqual(summary["raw_100"], 50.)
            self.assertEqual(summary["K20_100"], 50.)
            self.assertEqual(summary["tied"], 272)
            self.assertEqual(len(pd.read_csv(out / queue.VARIANT / "paired_scores.csv")), 272)
            self.assertEqual((exp / "pluto_score_queue_latest.txt").read_text().strip(), str(out))
            calls.clear()
            self.run_main(out, env, simulation)
            self.assertEqual(calls, [])

    def test_failed_native_smoke_stops_before_score_or_full_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out, calls = exp / "pluto_queue", []
            def simulation(cfg, directory, entry, environment):
                calls.append(directory.name)
                self.complete(cfg, directory, entry, environment)
                report = pd.read_parquet(directory / "runner_report.parquet")
                report.loc[0, "succeeded"] = False
                report.to_parquet(directory / "runner_report.parquet")
            with self.assertRaisesRegex(ValueError, "successful scenarios"):
                self.run_main(out, env, simulation)
            self.assertEqual(calls, ["K0_smoke"])
            self.assertFalse((out / "summary.json").exists())

    def test_invalid_score_audit_stops_before_expensive_full_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out, calls = exp / "pluto_queue", []
            def simulation(cfg, directory, entry, environment):
                calls.append(directory.name)
                self.complete(cfg, directory, entry, environment)
                if directory.name == "K20_smoke":
                    path = directory / "refinement_audit/0.json"
                    audit = json.loads(path.read_text())
                    audit["native_pose_count"] = 16
                    path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "native80"):
                self.run_main(out, env, simulation)
            self.assertEqual(calls, ["K0_smoke", "K20_smoke"])

    def test_missing_aggregate_is_not_accepted_even_for_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out = exp / "pluto_queue"
            def simulation(cfg, directory, entry, environment):
                self.complete(cfg, directory, entry, environment)
                (directory / "aggregator_metric/result.parquet").unlink()
            with self.assertRaisesRegex(ValueError, "official aggregate"):
                self.run_main(out, env, simulation)

    def test_incomplete_existing_stage_is_preserved_without_relaunch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out = exp / "pluto_queue"
            directory = out / queue.VARIANT / "K0_smoke"
            directory.mkdir(parents=True)
            (directory / "console.log").write_text("original failure\n")
            simulation = mock.Mock()
            with self.assertRaises(FileNotFoundError):
                self.run_main(out, env, simulation)
            simulation.assert_not_called()
            self.assertEqual((directory / "console.log").read_text(), "original failure\n")

    def test_checkpoint_change_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out = exp / "pluto_queue"
            self.run_main(out, env, self.complete)
            Path(env["PLUTO_CHECKPOINT"]).write_bytes(b"changed checkpoint")
            simulation = mock.Mock()
            with self.assertRaisesRegex(ValueError, "Resume protocol changed"):
                self.run_main(out, env, simulation)
            simulation.assert_not_called()

    def test_source_pin_and_clean_tracked_files_are_required(self):
        for responses, message in ((["wrong\n"], "pinned"),
                                   ([queue.PLUTO_COMMIT + "\n", " M src/model.py\n"], "local changes")):
            with self.subTest(message=message), mock.patch.object(
                    queue.subprocess, "check_output", side_effect=responses):
                with self.assertRaisesRegex(ValueError, message):
                    queue.verify_pluto_source(Path("/unused"))

    def test_official_model_config_keeps_reference_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, model, _, env = self.prepare(root)
            self.assertEqual(queue.load_official_model_config(env["PLUTO_CODE"]), model)
            model["feature_builder"]["build_reference_line"] = False
            OmegaConf.save(OmegaConf.create({"pluto_planner": {"planner": model}}),
                           Path(env["PLUTO_CODE"]) / "config/planner/pluto_planner.yaml")
            with self.assertRaisesRegex(ValueError, "reference lines"):
                queue.load_official_model_config(env["PLUTO_CODE"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
