"""Guard Flow Planner paired-run scheduling and result integrity without GPU/data."""

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


queue = load_module("flow_queue", ROOT / "scripts/evaluate_flow_score_queue.py")
fixture = load_module("native_queue_fixtures", ROOT / "tests/test_native_score_queue.py").fixture


class FlowQueueTests(unittest.TestCase):
    def prepare(self, root):
        exp = root / "dataset/nuplan/exp"
        baseline = (exp / "exp/simulation/closed_loop_nonreactive_agents/urbandriver"
                    / "test14-hard/raw_8gpu_reference")
        cfg, _, _, keys = fixture(baseline)
        cfg["scenario_builder"]["data_root"] = str(root / "dataset/nuplan/nuplan-v1.1/test")
        OmegaConf.save(OmegaConf.create(cfg), baseline / "code/hydra/config.yaml")
        code = root / "Flow-Planner"
        code.mkdir()
        model = {
            "_target_": queue.MODEL_TARGET, "future_len": 80, "state_dim": 4,
            "model_type": "x_start", "cfg_weight": 1.0,
            "flow_ode": {"sample_steps": 4, "sample_method": "midpoint",
                         "sample_temperature": 1.0, "cfg_weight": "${..cfg_weight}"},
        }
        models = root / "models"
        models.mkdir()
        config_path = models / "model_config.yaml"
        # Published configuration contains unresolved training-only references;
        # reading inference metadata must not require a training environment.
        OmegaConf.save(OmegaConf.create({"model": model,
                       "save_dir": "${oc.env:UNSET_FLOW_TRAINING_PATH}"}), config_path)
        for name in ("flow.pth", "score.pt", "score.json"):
            (models / name).write_bytes(b"synthetic weight identity")
        environment = {
            "UD_ROOT": str(root), "NUPLAN_EXP_ROOT": str(exp),
            "NUPLAN_DEVKIT_ROOT": str(root / "nuplan-devkit"),
            "FLOW_CODE": str(code), "FLOW_CHECKPOINT": str(models / "flow.pth"),
            "FLOW_CONFIG": str(config_path),
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
        planner = cfg["planner"]["flow_score"]
        if planner["steps"] == 20:
            audits = Path(planner["audit_dir"])
            audits.mkdir()
            audit = {
                "completed_steps": 20, "K": 20, "gamma": .1, "sigma_score": .1,
                "epoch": 529, "weight_kind": "ema", "finite": True,
                "native_pose_count": 80, "score_pose_count": 80,
                "evaluation_seed": planner["evaluation_seed"],
                "seed_policy": "once_per_scenario_before_native_initialization",
            }
            for index in range(count):
                (audits / f"{index}.json").write_text(json.dumps(audit))

    def run_main(self, out, environment, simulation, native_metadata=None):
        if native_metadata is None:
            native_metadata = {"strict": True}
        with mock.patch.dict(os.environ, environment), \
                mock.patch.object(sys, "argv", ["queue", "--output", str(out)]), \
                mock.patch.object(queue, "verify_flow_source", return_value=queue.FLOW_COMMIT), \
                mock.patch.object(queue.shared, "score_preflight"), \
                mock.patch.object(queue, "native_preflight", return_value=native_metadata), \
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
                cfg = queue.make_config(base, env["FLOW_CONFIG"], env["FLOW_CHECKPOINT"], directory, keys,
                                        steps, env["SCORE_CONFIG"], env["SCORE_CHECKPOINT"])
                planner = cfg["planner"]["flow_score"]
                self.assertEqual(planner["config_path"], env["FLOW_CONFIG"])
                self.assertNotIn("model", cfg)
                self.assertNotIn("sample_steps", planner)
                self.assertEqual(planner["_target_"], queue.PLANNER_TARGET)
                self.assertFalse(planner["_recursive_"])
                self.assertEqual(planner["steps"], steps)
                self.assertEqual(planner["evaluation_seed"], base["seed"])
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
            out, calls = exp / "flow_queue", []
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
            self.assertEqual((exp / "flow_score_queue_latest.txt").read_text().strip(), str(out))
            protocol = json.loads((out / "protocol.json").read_text())
            self.assertEqual(protocol["native_rng_policy"], "seed_each_scenario_once_from_reference")
            self.assertEqual(protocol["seed"], OmegaConf.load(env["BASELINE_RESULT"] + "/code/hydra/config.yaml").seed)
            calls.clear()
            self.run_main(out, env, simulation)
            self.assertEqual(calls, [])

    def test_failed_native_smoke_stops_before_score_or_full_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out, calls = exp / "flow_queue", []
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
            out, calls = exp / "flow_queue", []
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
            out = exp / "flow_queue"
            def simulation(cfg, directory, entry, environment):
                self.complete(cfg, directory, entry, environment)
                (directory / "aggregator_metric/result.parquet").unlink()
            with self.assertRaisesRegex(ValueError, "official aggregate"):
                self.run_main(out, env, simulation)

    def test_wrong_scenario_seed_audit_stops_before_full_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out, calls = exp / "flow_queue", []
            def simulation(cfg, directory, entry, environment):
                calls.append(directory.name)
                self.complete(cfg, directory, entry, environment)
                if directory.name == "K20_smoke":
                    path = directory / "refinement_audit/0.json"
                    audit = json.loads(path.read_text())
                    audit["evaluation_seed"] += 1
                    path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "evaluation seed"):
                self.run_main(out, env, simulation)
            self.assertEqual(calls, ["K0_smoke", "K20_smoke"])

    def test_incomplete_existing_stage_is_preserved_without_relaunch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out = exp / "flow_queue"
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
            out = exp / "flow_queue"
            self.run_main(out, env, self.complete)
            Path(env["FLOW_CHECKPOINT"]).write_bytes(b"changed checkpoint")
            simulation = mock.Mock()
            with self.assertRaisesRegex(ValueError, "Resume protocol changed"):
                self.run_main(out, env, simulation)
            simulation.assert_not_called()

    def test_source_pin_and_clean_tracked_files_are_required(self):
        for responses, message in ((["wrong\n"], "pinned"),
                                   ([queue.FLOW_COMMIT + "\n", " M src/model.py\n"], "local changes")):
            with self.subTest(message=message), mock.patch.object(
                    queue.subprocess, "check_output", side_effect=responses):
                with self.assertRaisesRegex(ValueError, message):
                    queue.verify_flow_source(Path("/unused"))

    def test_published_sampling_is_preserved_without_resolving_training_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, model, _, env = self.prepare(root)
            sampling = queue.load_flow_sampling(env["FLOW_CONFIG"])
            self.assertEqual(sampling["sample_steps"], 4)
            self.assertEqual(sampling["sample_method"], "midpoint")
            self.assertEqual(sampling["cfg_weight"], 1.0)
            model["flow_ode"]["sample_steps"] = 20
            OmegaConf.save(OmegaConf.create({"model": model}), env["FLOW_CONFIG"])
            with self.assertRaisesRegex(ValueError, "published Flow sampling"):
                queue.load_flow_sampling(env["FLOW_CONFIG"])

    def test_config_change_is_rejected_on_resume_even_with_same_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out = exp / "flow_queue"
            self.run_main(out, env, self.complete)
            path = Path(env["FLOW_CONFIG"])
            path.write_text(path.read_text() + "\n# Changed model config identity\n")
            simulation = mock.Mock()
            with self.assertRaisesRegex(ValueError, "Resume protocol changed"):
                self.run_main(out, env, simulation)
            simulation.assert_not_called()

    def test_effective_native_config_metadata_is_recorded_and_required_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, _, env = self.prepare(root)
            out = exp / "flow_queue"
            metadata = {"strict": True, "effective_model_config_sha256": "identity1",
                        "dataset_defaults_source": "flow_planner/config/data/dataset/nuplan.yaml",
                        "normalizer_defaults": {"future_len": 80}}
            self.run_main(out, env, self.complete, metadata)
            protocol = json.loads((out / "protocol.json").read_text())
            self.assertEqual(protocol["native_metadata"], metadata)
            changed = dict(metadata, effective_model_config_sha256="identity2")
            simulation = mock.Mock()
            with self.assertRaisesRegex(ValueError, "Resume protocol changed"):
                self.run_main(out, env, simulation, changed)
            simulation.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
