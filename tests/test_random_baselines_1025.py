"""Dataset-free checks of the twelve-job random NR queue and paired identity guards."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
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
