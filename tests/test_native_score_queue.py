"""Guard paired evaluation integrity without loading models or nuPlan data."""

import copy
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from omegaconf import OmegaConf


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_native_score_queue.py"
spec = importlib.util.spec_from_file_location("native_score_queue", SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def fixture(directory, count=272):
    """An official-shaped aggregate plus independent scenario runner report."""
    directory = Path(directory)
    (directory / "code/hydra").mkdir(parents=True)
    (directory / "aggregator_metric").mkdir()
    keys = sorted((f"log_{i % 4}", f"{i:016x}") for i in range(count))
    cfg = {
        "output_dir": str(directory),
        "runner_report_file": "runner_report.parquet",
        "metric_dir": "metrics",
        "aggregator_save_path": str(directory / "aggregator_metric"),
        "date_format": "%Y.%m.%d.%H.%M.%S",
        "seed": 123,
        "scenario_builder": {"_target_": "NuPlanScenarioBuilder", "data_root": "/data/test"},
        "scenario_filter": {
            "scenario_tokens": [k[1] for k in keys], "log_names": sorted({k[0] for k in keys}),
            "remove_invalid_goals": True, "limit_total_scenarios": None,
            "num_scenarios_per_type": None, "shuffle": False,
        },
        "observation": {"_target_": "nuplan.TracksObservation"},
        "ego_controller": {"_target_": "LQR", "gain": 2},
        "simulation_time_controller": {"_target_": "StepSimulationTimeController"},
        "simulation_history_buffer_duration": 2.0,
        "simulation_metric": {"metrics": {"ttc": {"horizon": 3.0}}},
        "metric_aggregator": {"weighted": {"save_path": str(directory / "aggregator_metric"),
                                           "challenge_name": "closed_loop_nonreactive_agents"}},
        "metric_file_name": "metrics_${now:${date_format}}",
        "model": {"_target_": "GC_PGP", "horizon": 8},
        "planner": {"ml_planner": {
            "_target_": "MLPlanner", "model_config": "${model}",
            "checkpoint_path": "/models/gc_pgp.ckpt",
        }},
        "worker": {"_target_": "worker", "max_workers": 4},
        "main_callback": {
            "metric": {"metric_file_output_path": str(directory / "metrics"),
                       "scenario_metric_paths": [str(directory / "metrics")]},
            "aggregate": {"metric_save_path": str(directory / "metrics")},
        },
    }
    OmegaConf.save(OmegaConf.create(cfg), directory / "code/hydra/config.yaml")
    report = pd.DataFrame([
        {"log_name": log, "scenario_name": token, "succeeded": True}
        for log, token in keys
    ])
    report.to_parquet(directory / "runner_report.parquet")
    metric = pd.DataFrame([
        {"log_name": log, "scenario": token, "score": .5, "num_scenarios": 1}
        for log, token in keys
    ] + [{"log_name": None, "scenario": "final_score", "score": .5, "num_scenarios": count}])
    metric.to_parquet(directory / "aggregator_metric/result.parquet")
    return cfg, report, metric, keys


class NativeQueueIntegrityTests(unittest.TestCase):
    def test_smoke_completion_needs_no_aggregate_but_benchmark_still_requires_one(self):
        for count in (0, 2):
            with self.subTest(aggregates=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, _, metric, expected = fixture(root, 2)
                (root / "aggregator_metric/result.parquet").unlink()
                for index in range(count):
                    metric.to_parquet(root / f"aggregator_metric/result_{index}.parquet")
                cfg, keys = queue.read_completed_run(root, 2)
                self.assertEqual(keys, expected)
                self.assertIn("planner", cfg)
                with self.assertRaisesRegex(ValueError, f"found {count}"):
                    queue.read_result(root, 2)

    def test_smoke_rejects_failed_missing_duplicate_and_null_scenario_identities(self):
        for mode in ("failed", "missing", "duplicate", "null"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, report, _, _ = fixture(root, 2)
                if mode == "failed":
                    report.loc[0, "succeeded"] = False
                elif mode == "missing":
                    report = report.iloc[:1]
                elif mode == "duplicate":
                    report.loc[1, ["log_name", "scenario_name"]] = report.loc[
                        0, ["log_name", "scenario_name"]].values
                else:
                    report.loc[0, "scenario_name"] = None
                report.to_parquet(root / "runner_report.parquet")
                with self.assertRaises(ValueError):
                    queue.read_completed_run(root, 2)

    def test_complete_272_and_fresh_hydra_resolver_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, expected = fixture(root)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            OmegaConf.clear_resolver("now")
            cfg, detail, keys, score = queue.read_result(root)
            self.assertEqual(keys, expected)
            self.assertEqual(len(detail), 272)
            self.assertEqual(score, 50.)
            self.assertEqual(cfg["planner"]["ml_planner"]["model_config"], cfg["model"])
            self.assertNotIn("${", cfg["metric_file_name"])
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_failed_scenario_is_not_accepted_as_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, report, _, _ = fixture(root)
            report.loc[0, "succeeded"] = False
            report.to_parquet(root / "runner_report.parquet")
            with self.assertRaises((ValueError, RuntimeError)):
                queue.read_result(root)

    def test_count_and_duplicate_identity_are_not_silently_dropped(self):
        for mode in ("missing", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, report, _, _ = fixture(root)
                if mode == "missing":
                    report = report.iloc[:-1]
                else:
                    report.loc[1, ["log_name", "scenario_name"]] = report.loc[
                        0, ["log_name", "scenario_name"]
                    ].values
                report.to_parquet(root / "runner_report.parquet")
                with self.assertRaises((ValueError, RuntimeError)):
                    queue.read_result(root)

    def test_aggregate_must_cover_the_same_scenario_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, metric, _ = fixture(root)
            metric.loc[0, "scenario"] = "not_in_runner_report"
            metric.to_parquet(root / "aggregator_metric/result.parquet")
            with self.assertRaises((ValueError, RuntimeError)):
                queue.read_result(root)

    def test_nonfinite_score_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, metric, _ = fixture(root)
            metric.loc[0, "score"] = float("nan")
            metric.to_parquet(root / "aggregator_metric/result.parquet")
            with self.assertRaises((ValueError, RuntimeError)):
                queue.read_result(root)

    def test_every_nested_output_is_relocated_without_mutating_source(self):
        original = {
            "output_dir": "/old/run",
            "callback": [{"metric_file_output_path": "/old/run/metrics"}],
            "aggregator": {"save_path": "/old/run/aggregator_metric"},
            "checkpoint": "/old/run_other/model.ckpt",
            "map_root": "/data/maps",
        }
        untouched = copy.deepcopy(original)
        moved = queue.relocate(original, "/old/run", "/new/run")
        self.assertEqual(moved["callback"][0]["metric_file_output_path"], "/new/run/metrics")
        self.assertEqual(moved["aggregator"]["save_path"], "/new/run/aggregator_metric")
        self.assertEqual(moved["checkpoint"], "/old/run_other/model.ckpt")
        self.assertEqual(moved["map_root"], "/data/maps")
        self.assertEqual(original, untouched)

    def test_raw_and_refined_configs_keep_evaluation_and_native_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, keys = fixture(root / "baseline")
            base = queue.load_saved_config(root / "baseline")
            untouched = copy.deepcopy(base)
            smoke = copy.deepcopy(base)
            smoke["model"]["horizon"] = 8
            raw_dir, refined_dir = root / "new_raw", root / "new_K20"
            raw = queue.make_native_config(base, smoke, raw_dir, keys, 8)
            refined = queue.make_refined_config(
                raw, refined_dir, "/score/config.json", "/score/final_ema.pt", 8,
                planner_name="gc_pgp",
            )
            for field in ("seed", "ego_controller", "observation", "simulation_metric",
                          "simulation_time_controller", "simulation_history_buffer_duration"):
                self.assertEqual(raw[field], base[field])
                self.assertEqual(refined[field], base[field])
            raw_wrapper = raw["planner"]["native_score"]
            refined_wrapper = refined["planner"]["native_score"]
            self.assertEqual(refined_wrapper["native_planner_config"], raw_wrapper["native_planner_config"])
            self.assertEqual(raw_wrapper["steps"], 0)
            self.assertEqual(refined["planner"]["native_score"]["steps"], 20)
            self.assertEqual(refined_wrapper["device"], "cuda")
            self.assertEqual(refined_wrapper["native_device"], "cuda")
            self.assertTrue(raw["worker"]["_target_"].endswith(".RayDistributed"))
            self.assertEqual(raw["worker"]["threads_per_node"], 16)
            self.assertEqual(refined["number_of_gpus_allocated_per_simulation"], 1)
            self.assertEqual(refined["number_of_cpus_allocated_per_simulation"], 2)
            self.assertEqual(raw["scenario_filter"]["scenario_tokens"], sorted(k[1] for k in keys))
            self.assertTrue(raw["scenario_filter"]["remove_invalid_goals"])
            self.assertEqual(refined["main_callback"]["metric"]["metric_file_output_path"],
                             str(refined_dir / "metrics/closed_loop_nonreactive_agents"))
            for config, directory in ((raw, raw_dir), (refined, refined_dir)):
                metric_path = str(directory / "metrics/closed_loop_nonreactive_agents")
                self.assertEqual(config["metric_dir"], "metrics/closed_loop_nonreactive_agents")
                self.assertEqual(config["main_callback"]["metric"]["scenario_metric_paths"], [metric_path])
                self.assertEqual(config["main_callback"]["aggregate"]["metric_save_path"], metric_path)
                self.assertEqual(config["metric_aggregator"]["weighted"]["challenge_name"],
                                 base["metric_aggregator"]["weighted"]["challenge_name"])
                self.assertEqual(config["aggregator_save_path"], str(directory / "aggregator_metric"))
            self.assertEqual(base, untouched)
            with self.assertRaisesRegex(ValueError, "never another refiner"):
                queue.make_refined_config(
                    refined, root / "double_refined", "/score/config.json", "/score/final_ema.pt", 8)

    def test_reuse_allows_scheduling_changes_but_rejects_semantic_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, keys = fixture(root)
            cfg = queue.load_saved_config(root)
            expected = queue.relocate(cfg, str(root), str(root / "another_output"))
            expected["worker"]["max_workers"] = 1
            expected["scenario_builder"]["max_workers"] = 1
            self.assertEqual(queue.validate_reuse(root, expected, keys)[2], keys)
            edits = (
                ("seed", lambda x: x.update(seed=999)),
                ("controller", lambda x: x["ego_controller"].update(gain=99)),
                ("filter", lambda x: x["scenario_filter"].update(remove_invalid_goals=False)),
                ("model", lambda x: x["planner"]["ml_planner"]["model_config"].update(horizon=7)),
                ("metric", lambda x: x["metric_aggregator"]["weighted"].update(ttc_weight=9)),
            )
            for label, change in edits:
                with self.subTest(change=label):
                    changed = copy.deepcopy(expected)
                    change(changed)
                    with self.assertRaises((ValueError, RuntimeError)):
                        queue.validate_reuse(root, changed, keys)
            wrong_keys = keys[:-1] + [("another_log", "another_token")]
            with self.assertRaises((ValueError, RuntimeError)):
                queue.validate_reuse(root, expected, wrong_keys)

    def test_existing_unverified_output_is_preserved_and_never_launched_over(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "console.log"
            marker.write_text("existing failed experiment\n")
            with mock.patch.object(queue.subprocess, "Popen") as launch:
                with self.assertRaises(FileExistsError):
                    queue.run_simulation({}, root, Path("/unused/entry.py"), {})
                launch.assert_not_called()
            self.assertEqual(marker.read_text(), "existing failed experiment\n")

    def test_old_queue_identity_requires_matching_inherited_suite_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            pid = 987654321
            (suite / "queue.pid").write_text(str(pid))
            proc_files = {
                f"/proc/{pid}/environ": f"BASELINE_FULL_ROOT={suite}\0".encode(),
                f"/proc/{pid}/cmdline": b"python\0-u\0-\0",
            }
            def read_proc(path):
                if str(path) not in proc_files:
                    raise FileNotFoundError(path)
                return proc_files[str(path)]
            with mock.patch.object(Path, "read_bytes", read_proc):
                self.assertTrue(queue.old_queue_active(suite))
                unrelated = suite / "other_suite"
                unrelated.mkdir()
                (unrelated / "queue.pid").write_text(str(pid))
                self.assertFalse(queue.old_queue_active(unrelated))
                proc_files.clear()
                self.assertFalse(queue.old_queue_active(suite))


class NativeQueueSequenceTests(unittest.TestCase):
    """Run the actual queue orchestration against synthetic completed results."""

    def prepare(self, root):
        exp = root / "dataset/nuplan/exp"
        baseline = exp / "exp/simulation/closed_loop_nonreactive_agents/urbandriver/test14-hard/raw_8gpu_test"
        base, _, _, keys = fixture(baseline)
        base["scenario_builder"]["data_root"] = str(root / "dataset/nuplan/nuplan-v1.1/test")
        OmegaConf.save(OmegaConf.create(base), baseline / "code/hydra/config.yaml")
        models = root / "models"
        models.mkdir()
        for name in ("pdm_open", "gc_pgp"):
            checkpoint = models / f"{name}.ckpt"
            checkpoint.write_bytes(b"synthetic checkpoint identity only")
            smoke = exp / "baseline_smoke" / f"{name}_test"
            cfg, _, _, _ = fixture(smoke, 2)
            if name == "pdm_open":
                cfg["planner"] = {"pdm_open_planner": {
                    "_target_": "PDMOpenPlanner", "model": {"horizon": 8},
                    "checkpoint_path": str(checkpoint),
                }}
            else:
                cfg["planner"]["ml_planner"]["checkpoint_path"] = str(checkpoint)
            OmegaConf.save(OmegaConf.create(cfg), smoke / "code/hydra/config.yaml")
            # Real custom smoke paths can finish successfully without an
            # aggregate: nuPlan filters metric paths by challenge name.
            (smoke / "aggregator_metric/result.parquet").unlink()
        (models / "score.json").write_text("{}")
        (models / "score.pt").write_bytes(b"synthetic EMA identity only")
        environment = {
            "UD_ROOT": str(root), "NUPLAN_EXP_ROOT": str(exp),
            "NUPLAN_DEVKIT_ROOT": str(root / "nuplan-devkit"),
            "BASELINE_RESULT": str(baseline),
            "SCORE_CONFIG": str(models / "score.json"),
            "SCORE_CHECKPOINT": str(models / "score.pt"),
            "FROZEN_SCORE_CODE": str(root / "frozen_score"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
        return exp, baseline, keys, environment

    def complete_simulation(self, cfg, out, entry, environment):
        count = 1 if out.name == "K20_smoke" else 272
        fixture(out, count)
        OmegaConf.save(OmegaConf.create(cfg), out / "code/hydra/config.yaml")
        if cfg["planner"].get("native_score", {}).get("steps") == 20:
            audits = Path(cfg["planner"]["native_score"]["audit_dir"])
            audits.mkdir()
            audit = {"completed_steps": 20, "K": 20, "gamma": .1,
                     "sigma_score": .1, "epoch": 529, "weight_kind": "ema", "finite": True}
            for index in range(count):
                (audits / f"{index}.json").write_text(json.dumps(audit))

    def run_main(self, output, environment, simulation, raw_suite=None):
        argv = [str(SCRIPT), "--output", str(output)]
        if raw_suite is not None:
            argv += ["--raw-suite", str(raw_suite)]
        with mock.patch.dict(os.environ, environment), mock.patch.object(sys, "argv", argv), \
                mock.patch.object(queue, "score_preflight"), \
                mock.patch.object(queue, "run_simulation", side_effect=simulation), \
                contextlib.redirect_stdout(io.StringIO()):
            queue.main()

    def test_each_native_is_immediately_followed_by_its_K20(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, environment = self.prepare(root)
            output, calls = exp / "paired_queue", []
            def simulation(cfg, out, entry, env):
                calls.append(str(out.relative_to(output)))
                wrapper = cfg["planner"]["native_score"]
                self.assertEqual(wrapper["steps"], 0 if out.name == "raw" else 20)
                self.assertEqual(wrapper["device"], "cuda")
                self.assertEqual(wrapper["native_device"],
                                 "cpu" if out.parent.name == "pdm_open" else "cuda")
                self.assertEqual(cfg["number_of_gpus_allocated_per_simulation"], 1)
                if out.name == "K20_smoke":
                    self.assertTrue(cfg["worker"]["_target_"].endswith(".Sequential"))
                    self.assertEqual(cfg["number_of_cpus_allocated_per_simulation"], 1)
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
                else:
                    self.assertTrue(cfg["worker"]["_target_"].endswith(".RayDistributed"))
                    self.assertEqual(cfg["worker"]["threads_per_node"], 16)
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
                self.complete_simulation(cfg, out, entry, env)
            self.run_main(output, environment, simulation)
            self.assertEqual(calls, [
                "pdm_open/raw", "pdm_open/K20_smoke", "pdm_open/S10G_K20",
                "gc_pgp/raw", "gc_pgp/K20_smoke", "gc_pgp/S10G_K20",
            ])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(set(summary), {"pdm_open", "gc_pgp"})
            self.assertTrue(all(result["scenarios"] == 272 for result in summary.values()))

    def test_two_completed_external_raw_runs_are_reused_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, baseline, keys, environment = self.prepare(root)
            raw_suite, output, calls = exp / "baseline_full_existing", exp / "paired_queue", []
            base = queue.load_saved_config(baseline)
            for name in ("pdm_open", "gc_pgp"):
                smoke = queue.load_saved_config(exp / "baseline_smoke" / f"{name}_test")
                raw_dir = raw_suite / name
                cfg = queue.make_native_config(base, smoke, raw_dir, keys, 8)
                self.complete_simulation(cfg, raw_dir, None, {})
            before = {p.relative_to(raw_suite): p.read_bytes()
                      for p in raw_suite.rglob("*") if p.is_file()}
            def simulation(cfg, out, entry, env):
                calls.append(str(out.relative_to(output)))
                self.complete_simulation(cfg, out, entry, env)
            self.run_main(output, environment, simulation, raw_suite)
            self.assertEqual(calls, [
                "pdm_open/K20_smoke", "pdm_open/S10G_K20",
                "gc_pgp/K20_smoke", "gc_pgp/S10G_K20",
            ])
            after = {p.relative_to(raw_suite): p.read_bytes()
                     for p in raw_suite.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_legacy_CPU_PDM_is_reused_but_CPU_GCPGP_is_rerun_on_GPU(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, baseline, keys, environment = self.prepare(root)
            raw_suite, output, calls = exp / "baseline_full_cpu", exp / "paired_queue", []
            base = queue.load_saved_config(baseline)
            for name in ("pdm_open", "gc_pgp"):
                smoke = queue.load_saved_config(exp / "baseline_smoke" / f"{name}_test")
                raw_dir = raw_suite / name
                cfg = queue.make_native_config(base, smoke, raw_dir, keys, 8)
                cfg["planner"] = cfg["planner"]["native_score"]["native_planner_config"]
                cfg["number_of_gpus_allocated_per_simulation"] = 0
                cfg["worker"] = {"_target_": "SingleMachineParallelExecutor", "max_workers": 4}
                self.complete_simulation(cfg, raw_dir, None, {})
            before = {p.relative_to(raw_suite): p.read_bytes()
                      for p in raw_suite.rglob("*") if p.is_file()}
            def simulation(cfg, out, entry, env):
                calls.append(str(out.relative_to(output)))
                if out.name == "raw":
                    self.assertEqual(cfg["planner"]["native_score"]["native_device"], "cuda")
                self.complete_simulation(cfg, out, entry, env)
            self.run_main(output, environment, simulation, raw_suite)
            self.assertEqual(calls, [
                "pdm_open/K20_smoke", "pdm_open/S10G_K20", "gc_pgp/raw",
                "gc_pgp/K20_smoke", "gc_pgp/S10G_K20",
            ])
            after = {p.relative_to(raw_suite): p.read_bytes()
                     for p in raw_suite.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["pdm_open"]["raw_dir"], str(raw_suite / "pdm_open"))
            self.assertEqual(summary["gc_pgp"]["raw_dir"], str(output / "gc_pgp/raw"))

    def test_native_failure_stops_before_any_K20_or_next_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, environment = self.prepare(root)
            output, calls = exp / "paired_queue", []
            def fail(cfg, out, entry, env):
                calls.append(str(out.relative_to(output)))
                out.mkdir(parents=True)
                (out / "console.log").write_text("native simulation failure\n")
                raise RuntimeError("native simulation failure")
            with self.assertRaisesRegex(RuntimeError, "native simulation failure"):
                self.run_main(output, environment, fail)
            self.assertEqual(calls, ["pdm_open/raw"])
            self.assertFalse((output / "summary.json").exists())
            self.assertEqual((output / "pdm_open/raw/console.log").read_text(),
                             "native simulation failure\n")

    def test_failed_external_raw_is_not_silently_replaced_or_refined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, _, _, environment = self.prepare(root)
            raw_suite, output = exp / "baseline_full_failed", exp / "paired_queue"
            raw_dir = raw_suite / "pdm_open"
            _, report, _, _ = fixture(raw_dir)
            report.loc[0, "succeeded"] = False
            report.to_parquet(raw_dir / "runner_report.parquet")
            before = {p.relative_to(raw_suite): p.read_bytes()
                      for p in raw_suite.rglob("*") if p.is_file()}
            simulation = mock.Mock()
            with self.assertRaisesRegex(ValueError, "successful scenarios"):
                self.run_main(output, environment, simulation, raw_suite)
            simulation.assert_not_called()
            after = {p.relative_to(raw_suite): p.read_bytes()
                     for p in raw_suite.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
