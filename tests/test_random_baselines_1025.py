"""Dataset-free checks of the twelve-job random NR queue and paired identity guards."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_random_baselines_1025 as queue

spec = importlib.util.spec_from_file_location("native_tests", ROOT / "tests/test_native_score_queue.py")
native_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_tests)


def source(name):
    if name == "urban_driver":
        planner = {"ml_planner": {"_target_": "nuplan.MLPlanner",
            "model_config": {"_target_": "urban_driver.UrbanDriver"}, "checkpoint_path": "/models/ud.ckpt"}}
    elif name.startswith("plantf_"):
        checkpoint = "last.ckpt" if name.endswith("24") else "epoch=17-val_minFDE=0.000.ckpt"
        planner = {"imitation_planner": {"_target_": "src.planners.imitation_planner.ImitationPlanner",
            "planner_ckpt": "/models/" + checkpoint, "planner": {"_target_": "src.PlanTF", "future_steps": 80}}}
    elif name == "pluto_wo_refine":
        planner = {"pluto_score": {"_target_": "pluto_score.planner.PlutoScorePlanner", "steps": 0,
            "model_config": {"_target_": "src.Pluto", "feature_builder": {"build_reference_line": True}},
            "checkpoint_path": "/models/pluto.ckpt", "native_device": "cuda"}}
    elif name == "pdm_open":
        planner = {"pdm_open_planner": {"_target_": "pdm.PDMOpen", "checkpoint_path": "/models/pdm.ckpt"}}
    else:
        planner = {"ml_planner": {"_target_": "nuplan.MLPlanner", "model_config": {"_target_": "gc_pgp.PGP"},
            "checkpoint_path": "/models/pgp.ckpt"}}
    return {"planner": planner, "model": {"_target_": name}}


class RandomQueueTests(unittest.TestCase):
    def test_sorted_plan_roundtrip_restores_callback_order_without_protocol_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, _, _, _ = native_tests.fixture(root / "reference", 3)
            cfg = queue.shared().load_saved_config(root / "reference")
            cfg["main_callback"] = {
                "a_start": {"_target_": "callbacks.Start", "value": 1},
                "metric_file_callback": {"_target_": "callbacks.MetricFileCallback", "path": "/metrics"},
                "metric_other": {"_target_": "callbacks.Other", "value": 2},
                "metric_aggregator_callback": {"_target_": "callbacks.MetricAggregatorCallback", "path": "/metrics"},
                "metric_summary_callback": {"_target_": "callbacks.MetricSummaryCallback", "path": "/summary"},
                "z_end": {"_target_": "callbacks.End", "value": 3},
            }
            path = root / "plan.json"
            queue.write(path, {"configs": {"urban_driver": {"0": cfg}}})
            loaded = queue.read(path)["configs"]["urban_driver"]["0"]
            before = list(loaded["main_callback"])
            self.assertLess(before.index("metric_aggregator_callback"), before.index("metric_file_callback"))
            fixed = queue.order_metric_callbacks(loaded)
            after = list(fixed["main_callback"])
            self.assertLess(after.index("metric_file_callback"), after.index("metric_aggregator_callback"))
            self.assertLess(after.index("metric_aggregator_callback"), after.index("metric_summary_callback"))
            for key in ("a_start", "metric_other", "z_end"):
                self.assertEqual(before.index(key), after.index(key))
            self.assertEqual(fixed, loaded)
            self.assertEqual(list(loaded["main_callback"]), before)
            self.assertEqual(queue.shared().evaluation_signature(fixed), queue.shared().evaluation_signature(loaded))
            simulation_yaml = root / "simulation.yaml"
            OmegaConf.save(OmegaConf.create(fixed), simulation_yaml)
            self.assertEqual(list(OmegaConf.load(simulation_yaml).main_callback), after)
            fixed["main_callback"]["metric_file_callback"]["path"] = "/different"
            self.assertEqual(loaded["main_callback"]["metric_file_callback"]["path"], "/metrics")

    def test_metric_callback_aliases_use_targets_and_standard_keys_also_work(self):
        for callbacks in (
            {"first": {"_target_": "nuplan.callbacks.MetricSummaryCallback"},
             "keep": {"_target_": "nuplan.callbacks.Other"},
             "second": {"_target_": "nuplan.callbacks.MetricAggregatorCallback"},
             "third": {"_target_": "nuplan.callbacks.MetricFileCallback"}},
            {"metric_summary_callback": {}, "keep": {},
             "metric_aggregator_callback": {}, "metric_file_callback": {}},
        ):
            with self.subTest(callbacks=callbacks):
                cfg = {"main_callback": callbacks}
                fixed = queue.order_metric_callbacks(cfg)
                keys = list(callbacks)
                self.assertEqual(list(fixed["main_callback"]), [keys[3], "keep", keys[2], keys[0]])
                self.assertEqual(fixed, cfg)
                self.assertEqual(queue.order_metric_callbacks(fixed), fixed)

    def test_stage_generates_aggregate_after_sorted_plan_for_smoke_and_full(self):
        for smoke in (True, False):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base, _, _, cohort = native_tests.fixture(root / "reference", 3)
                cfg = queue.make_config("urban_driver", base, source("urban_driver"),
                                        root / "urban_driver/native", cohort, 0,
                                        "/score/config.json", "/score/final_ema.pt", 20260920)
                cfg["main_callback"] = {
                    "metric_file_callback": {"_target_": "callbacks.MetricFileCallback"},
                    "metric_aggregator_callback": {"_target_": "callbacks.MetricAggregatorCallback"},
                    "metric_summary_callback": {"_target_": "callbacks.MetricSummaryCallback"},
                }
                path = root / "plan.json"
                queue.write(path, dict(scenario_keys=cohort, configs={"urban_driver": {"0": cfg}},
                    sources={"urban_driver": {"runtime": {"entry": "/simulate.py"}}}))
                plan = queue.read(path)
                events = []

                def callbacks_simulation(config, directory, entry, environment):
                    # Reproduce the official forward iteration dependency with
                    # real parquet files and the real final result validator.
                    directory = Path(directory)
                    (directory / "code/hydra").mkdir(parents=True)
                    OmegaConf.save(OmegaConf.create(config), directory / "code/hydra/config.yaml")
                    selected = [(log, token) for log, token in cohort
                                if token in config["scenario_filter"]["scenario_tokens"]]
                    pd.DataFrame([dict(log_name=log, scenario_name=token, succeeded=True)
                                  for log, token in selected]).to_parquet(directory / "runner_report.parquet")
                    frame = pd.DataFrame([dict(log_name=log, scenario=token, score=.5, num_scenarios=1)
                                          for log, token in selected] + [dict(log_name=None,
                                          scenario="final_score", score=.5, num_scenarios=len(selected))])
                    metric = directory / "metric_file.parquet"
                    aggregate = Path(config["aggregator_save_path"]) / "result.parquet"
                    for callback in config["main_callback"].values():
                        name = callback["_target_"].rsplit(".", 1)[-1]
                        events.append(name)
                        if name == "MetricFileCallback":
                            frame.to_parquet(metric)
                        elif name == "MetricAggregatorCallback" and metric.exists():
                            aggregate.parent.mkdir(parents=True, exist_ok=True)
                            pd.read_parquet(metric).to_parquet(aggregate)

                bad = queue.shared().execution_config(plan["configs"]["urban_driver"]["0"], root / "bad", 1)
                callbacks_simulation(bad, root / "bad", Path("/simulate.py"), {})
                with self.assertRaisesRegex(ValueError, "found 0"):
                    queue.shared().read_result(root / "bad", len(cohort))
                events.clear()
                before = copy.deepcopy(plan)
                with mock.patch.object(queue, "verify_hashes"), \
                        mock.patch.object(queue, "environment", return_value={}), \
                        mock.patch.object(queue.shared(), "run_simulation", side_effect=callbacks_simulation):
                    result = queue.stage(plan, root, "urban_driver", 0, smoke, False)
                self.assertEqual(events, ["MetricFileCallback", "MetricAggregatorCallback", "MetricSummaryCallback"])
                self.assertEqual(len(result[2]), 1 if smoke else len(cohort))
                self.assertEqual(result[3], 50.)
                self.assertEqual(plan, before)

    def test_prepare_resolves_saved_hydra_paths_without_shell_exports(self):
        # Exercise the failing read_result call, not just the path helper.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "completed_dp"
            cfg, _, _, cohort = native_tests.fixture(baseline, 3)
            cfg["scenario_builder"]["data_root"] = "${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/test"
            cfg["maps_root"] = "${oc.env:NUPLAN_MAPS_ROOT}"
            cfg["exp_root"] = "${oc.env:NUPLAN_EXP_ROOT}"
            cfg["devkit_root"] = "${oc.env:NUPLAN_DEVKIT_ROOT}"
            cfg["dp_root"] = "${oc.env:DP_DIR}"
            OmegaConf.save(OmegaConf.create(cfg), baseline / "code/hydra/config.yaml")
            matrix = root / "matrix/plan.json"
            queue.write(matrix, {"scenario_keys": {queue.SPLIT: cohort, "test14-hard": cohort}})
            queue.write(matrix.parent / "jobs/random_NR_seed20260920_K0.json",
                        dict(state="complete", K=0, seed=20260920, setting="random_NR",
                             scenario_keys=cohort, result_dir=str(baseline)))
            args = SimpleNamespace(root=root, output=root / "new_queue", matrix=matrix,
                                   seed=20260920, frozen_score_code=None, score_config=None,
                                   score_checkpoint=None)
            read_result = queue.shared().read_result

            class ReferenceChecked(Exception):
                pass

            def check_reference(*args):
                resolved, _, actual, _ = read_result(*args)
                self.assertEqual(resolved["scenario_builder"]["data_root"],
                                 str(root / "dataset/nuplan/nuplan-v1.1/test"))
                self.assertEqual(resolved["maps_root"], str(root / "dataset/nuplan/maps"))
                self.assertEqual(resolved["exp_root"], str(root / "dataset/nuplan/exp"))
                self.assertEqual(resolved["devkit_root"], str(root / "nuplan-devkit"))
                self.assertEqual(resolved["dp_root"], str(root / "Diffusion-Planner"))
                self.assertEqual(actual, sorted(cohort))
                raise ReferenceChecked()

            for inherited in ({}, {key: "/stale" for key in queue.nuplan_paths(root)}):
                with self.subTest(inherited=inherited), mock.patch.dict("os.environ", inherited, clear=True), \
                        mock.patch.object(queue.shared(), "read_result", side_effect=check_reference), \
                        self.assertRaises(ReferenceChecked):
                    queue.prepare(args)
            self.assertFalse(args.output.exists())

    def test_six_variants_twelve_jobs_no_dp(self):
        self.assertEqual(len(queue.VARIANTS), 6)
        self.assertEqual(set(queue.VARIANTS), {"urban_driver", "plantf_epoch17", "plantf_epoch24",
            "pdm_open", "gc_pgp", "pluto_wo_refine"})

    def test_cohort_count_is_derived_and_duplicate_keys_rejected(self):
        rows = [["log", f"{i:016x}"] for i in range(17)]
        self.assertEqual(len(queue.keys(rows)), 17)
        for bad in ([], rows + [rows[0]], [["log", "invalid-token"]], [[None, "0" * 16]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                queue.keys(bad)

    def test_all_configs_preserve_protocol_exact_cohort_and_eight_gpu_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _, _, cohort = native_tests.fixture(root / "dp", 17)
            original = copy.deepcopy(base)
            for name in queue.VARIANTS:
                native = source(name)
                unchanged = copy.deepcopy(native)
                for steps in (0, 20):
                    out = root / name / str(steps)
                    cfg = queue.make_config(name, base, native, out, cohort, steps,
                                            "/score/config.json", "/score/final_ema.pt", 20260920)
                    wrapper = next(iter(cfg["planner"].values()))
                    self.assertEqual(wrapper["steps"], steps)
                    self.assertEqual(wrapper["score_checkpoint"], "/score/final_ema.pt" if steps else "unused")
                    self.assertEqual(cfg["seed"], 20260920)
                    self.assertEqual(cfg["number_of_gpus_allocated_per_simulation"], 1.)
                    self.assertTrue(cfg["worker"]["_target_"].endswith(".RayDistributed"))
                    self.assertEqual(cfg["worker"]["threads_per_node"], 16)
                    self.assertEqual(cfg["scenario_builder"]["max_workers"], 8)
                    self.assertEqual(cfg["scenario_filter"]["scenario_tokens"], sorted({t for _, t in cohort}))
                    self.assertIsNone(cfg["scenario_filter"]["num_scenarios_per_type"])
                    self.assertIsNone(cfg["scenario_filter"]["limit_total_scenarios"])
                    for key in ("observation", "ego_controller", "simulation_metric", "simulation_time_controller",
                                "simulation_history_buffer_duration"):
                        self.assertEqual(cfg[key], base[key])
                    self.assertIn(queue.CHALLENGE, str(out / cfg["metric_dir"]))
                    self.assertEqual(cfg["output_dir"], str(out))
                    if name == "pdm_open":
                        self.assertEqual(wrapper["native_device"], "cpu")
                    if name == "pluto_wo_refine":
                        self.assertTrue(wrapper["model_config"]["feature_builder"]["build_reference_line"])
                self.assertEqual(native, unchanged)
            self.assertEqual(base, original)

    def test_only_native_references_and_exact_plantf_checkpoint_names_allowed(self):
        good = source("gc_pgp")
        wrapped = {"planner": {"native_score": {"steps": 0, "native_planner_config": good["planner"]}}}
        self.assertEqual(queue.native_planner("gc_pgp", wrapped), good["planner"])
        wrapped["planner"]["native_score"]["steps"] = 20
        with self.assertRaisesRegex(ValueError, "K0"):
            queue.native_planner("gc_pgp", wrapped)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            queue.native_planner("plantf_epoch24", source("plantf_epoch17"))
        bad = source("pluto_wo_refine")
        bad["planner"]["pluto_score"]["steps"] = 20
        with self.assertRaises(ValueError):
            queue.native_planner("pluto_wo_refine", bad)

    def test_native_checkpoint_discovery_excludes_score_checkpoint(self):
        original = source("pluto_wo_refine")["planner"]
        original["pluto_score"]["score_checkpoint"] = "/score/final_ema.pt"
        self.assertEqual(queue.checkpoint_paths(original), [Path("/models/pluto.ckpt")])

    def test_existing_native_checkpoint_provenance_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "run/full"
            source_dir.mkdir(parents=True)
            checkpoint = root / "last.ckpt"
            checkpoint.write_bytes(b"original")
            protocol = source_dir.parent / "protocol.json"
            protocol.write_text(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": queue.digest(checkpoint)}))
            self.assertEqual(queue.source_checkpoint_provenance(source_dir, checkpoint), [protocol])
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since"):
                queue.source_checkpoint_provenance(source_dir, checkpoint)

    def test_src_packages_isolated_and_all_eight_devices_visible(self):
        plan = {"root": "/server", "frozen_score_code": "/frozen"}
        with mock.patch.dict("os.environ", {"RANK": "7", "RAY_ADDRESS": "auto", "WORLD_SIZE": "8"}):
            for name in queue.VARIANTS:
                env = queue.environment(plan, name)
                paths = env["PYTHONPATH"].split(":")
                self.assertEqual(paths[0], "/frozen")
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], queue.GPU_IDS)
                self.assertNotIn("RANK", env)
                self.assertNotIn("RAY_ADDRESS", env)
                if name.startswith("plantf_"):
                    self.assertIn("/server/planTF", paths)
                    self.assertNotIn("/server/pluto", paths)
                elif name == "pluto_wo_refine":
                    self.assertIn("/server/pluto", paths)
                    self.assertNotIn("/server/planTF", paths)

    def test_missing_or_wrong_random_scenario_is_not_a_completed_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, report, _, cohort = native_tests.fixture(root, 17)
            cfg = queue.shared().load_saved_config(root)
            result = queue.validate_stage(root, cfg, cohort, "gc_pgp", 0)
            self.assertEqual(len(result[1]), 17)
            report.loc[0, "scenario_name"] = "f" * 16
            report.to_parquet(root / "runner_report.parquet")
            with self.assertRaises(ValueError):
                queue.validate_stage(root, cfg, cohort, "gc_pgp", 0)

    def test_k20_audits_require_all_scenarios_and_correct_pose_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, _, _, cohort = native_tests.fixture(root, 3)
            cfg = queue.shared().load_saved_config(root)
            folder = root / "refinement_audit"
            folder.mkdir()
            audit = {"completed_steps": 20, "K": 20, "gamma": .1, "sigma_score": .1,
                     "epoch": 529, "weight_kind": "ema", "finite": True,
                     "native_pose_count": 80, "score_pose_count": 80}
            for i in range(3):
                (folder / f"{i}.json").write_text(json.dumps(audit))
            queue.validate_stage(root, cfg, cohort, "plantf_epoch24", 20)
            with self.assertRaisesRegex(ValueError, "sampling"):
                queue.validate_stage(root, cfg, cohort, "urban_driver", 20)
            (folder / "0.json").unlink()
            with self.assertRaisesRegex(ValueError, "audits"):
                queue.validate_stage(root, cfg, cohort, "plantf_epoch24", 20)

    def test_retry_preserves_failed_attempt_and_runs_again_without_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            target = out / "urban_driver/native"
            target.mkdir(parents=True)
            (target / "failure.log").write_text("original evidence")
            cfg = {"output_dir": str(target)}
            plan = {"scenario_keys": [["log", "0" * 16]], "configs": {"urban_driver": {"0": cfg}},
                    "sources": {"urban_driver": {"runtime": {"entry": "/runner.py"}}}}
            result = ({}, None, [], 51.)
            with mock.patch.object(queue, "verify_hashes"), mock.patch.object(queue, "environment", return_value={}), \
                    mock.patch.object(queue, "validate_stage", side_effect=[ValueError("incomplete"), result]), \
                    mock.patch.object(queue.shared(), "run_simulation") as launch:
                self.assertEqual(queue.stage(plan, out, "urban_driver", 0, False, True), result)
                self.assertEqual(launch.call_count, 1)
            archives = list(target.parent.glob("native.failed.*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / "failure.log").read_text(), "original evidence")


if __name__ == "__main__":
    unittest.main()
