"""Paired native K0 reuse, protocol binding, recovery, and launcher contracts."""
import copy
import fcntl
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import socket
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from score_function.navsim_chainflow import evaluate as ev
from score_function.utils.train_utils import atomic_write

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chainflow_eval_launcher", ROOT / "scripts/eval_navsim_chainflow.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@dataclass
class FakeMetrics:
    no_at_fault_collisions: float = 1.
    drivable_area_compliance: float = 1.
    ego_progress: float = .5
    time_to_collision_within_bound: float = 1.
    comfort: float = 1.
    driving_direction_compliance: float = 1.
    score: float = .9


class FakeRefiner:
    def require_conditioner(self, identity):
        if identity != {"native": "chainflow"}:
            raise ValueError("Wrong frozen conditioner")

    @classmethod
    def from_checkpoint(cls, path, device):
        return cls()

    def refine_many(self, raw, condition, ks, gamma=None):
        out = {0: raw}
        for k in ks:
            if k:
                out[k] = raw.clone()
                out[k][..., 0] += k * .01
        return out


class ChainFlowEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.devkit = self.root / "upstream"
        self.devkit.mkdir()
        for name in (
            "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml",
            "navsim/evaluate/pdm_score.py",
            "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
            "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
        ):
            path = self.devkit / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.metric = self.root / "metrics"
        (self.metric / "metadata").mkdir(parents=True)
        tokens = ["a", "b", "c", "d"]
        names = []
        for token in tokens:
            path = self.metric / token / "metric_cache.pkl"
            path.parent.mkdir()
            path.write_text(token)
            names.append(str(path))
        (self.metric / "metadata/worker0.csv").write_text("file_name\n" + "\n".join(names) + "\n")
        self.native_calls = 0
        self.condition_change = False

        def condition(scene):
            value = float(ord(scene.token)) / 100.
            if self.condition_change:
                value += .05
            return {"scene": torch.full((1, 64, 256), value),
                    "ego": torch.zeros(1, 256), "vlm": torch.ones(1, 3, 1536),
                    "vlm_mask": torch.ones(1, 3, dtype=torch.bool)}

        def native_predict(agent, scene, device):
            self.native_calls += 1
            poses = torch.zeros(1, 8, 3)
            poses[..., 0] = float(ord(scene.token)) / 100.
            return poses, condition(scene)

        class FakeLoader:
            tokens = ["a", "b", "c", "d"]
            def get_tokens_list_per_log(self):
                return {"log": self.tokens}
            def get_scene_from_token(self, token):
                return SimpleNamespace(token=token)

        class FakeMetricCache:
            def __init__(self, path):
                self.metric_cache_paths = self._load_metric_cache_paths(path)
            def get_from_token(self, token):
                return token

        horizon = object()
        simulator = SimpleNamespace(proposal_sampling=horizon)
        scorer = SimpleNamespace(proposal_sampling=horizon)
        def module(name, **attrs):
            result = ModuleType(name)
            for k, v in attrs.items():
                setattr(result, k, v)
            return result
        self.modules = {
            "hydra.utils": module("hydra.utils", instantiate=lambda value: value),
            "omegaconf": module("omegaconf", OmegaConf=SimpleNamespace(load=lambda p: SimpleNamespace(simulator=simulator, scorer=scorer))),
            "navsim.common.dataloader": module("navsim.common.dataloader", MetricCacheLoader=FakeMetricCache),
            "navsim.common.dataclasses": module("navsim.common.dataclasses", Trajectory=lambda poses: SimpleNamespace(poses=poses)),
            "navsim.evaluate.pdm_score": module("navsim.evaluate.pdm_score", pdm_score=lambda metric, trajectory, *args: FakeMetrics(score=.8+float(trajectory.poses[0, 0])*.01)),
            "score_function.navsim_chainflow.native": module(
                "score_function.navsim_chainflow.native",
                activate_devkit=lambda root: root,
                load_agent=lambda *args: SimpleNamespace(_score_bridge_identity={"native": "chainflow"}),
                make_scene_loader=lambda *args: FakeLoader(), native_predict=native_predict,
                encode_scene=lambda agent, scene, device: condition(scene)),
            "score_function.navsim_chainflow.runtime": module("score_function.navsim_chainflow.runtime", ChainFlowRefiner=FakeRefiner),
        }
        self.score_path = self.root / "final_ema.pt"
        torch.save({"epoch": 500, "weight_kind": "ema"}, self.score_path)

    def args(self, mode="baseline", shards=2, index=0):
        return SimpleNamespace(mode=mode, devkit_root=self.devkit,
            checkpoint=self.root / "native.ckpt", backbone=self.root / "dino.safetensors",
            vlm_cache=self.root / "vlm", data_root=self.root / "data", metric_cache=self.metric,
            output=self.root / mode, score_checkpoint=self.score_path, baseline=self.root / "baseline",
            device="cpu", seed=20260922, ks=[5, 10, 20], gamma=.1, num_shards=shards,
            shard_index=index, max_scenes=4)

    def baseline(self):
        with patch.dict("sys.modules", self.modules):
            for index in range(2):
                ev.run(self.args(index=index))
        return ev.merge_shards(self.root / "baseline")

    def test_full_native_and_refine_preserve_k0_without_resampling(self):
        original = self.baseline()
        self.assertEqual(self.native_calls, 4)
        # Refinement may use a different shard count without changing token RNG/K0.
        with patch.dict("sys.modules", self.modules):
            ev.run(self.args("refine", shards=1))
        report = ev.merge_shards(self.root / "refine")
        self.assertEqual(self.native_calls, 4)
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["valid_count"], 4)
        self.assertEqual(report["by_k"]["0"], original["by_k"]["0"])
        self.assertEqual(report["by_k"]["20"]["improved"], 4)
        _, rows, _ = ev.collect_completed(self.root / "refine")
        _, source, _ = ev.collect_completed(self.root / "baseline")
        source = {row["token"]: row for row in source}
        for row in rows:
            self.assertEqual(row["trajectories"]["0"], source[row["token"]]["trajectories"]["0"])
            self.assertEqual(row["metrics"]["0"], source[row["token"]]["metrics"]["0"])

    def test_conditions_or_metric_cache_change_fails_before_refinement(self):
        self.baseline()
        self.condition_change = True
        with patch.dict("sys.modules", self.modules):
            with self.assertRaisesRegex(RuntimeError, "Evaluation failed") as raised:
                ev.run(self.args("refine", shards=1))
        self.assertIn("conditions differ", str(raised.exception.__cause__))
        self.assertFalse((self.root / "refine/summary.json").exists())
        self.assertEqual(self.native_calls, 4)

    def external_fixture(self):
        """Real adapter is independently tested; exercise its evaluator contract."""
        self.baseline()
        protocol, native_rows, _ = ev.collect_completed(self.root / "baseline")
        identity = dict(chainflow_commit="pinned", checkpoint_sha256="a"*64, backbone_sha256="b"*64)
        rows = {}
        for row in native_rows:
            source = copy.deepcopy(row)
            for key in ("condition_sha256", "metric_cache_sha256", "native_ms", "conditioner_ms", "refine_ms"):
                source.pop(key)
            source.update(native_origin="external-handoff", trajectory_dtype="float64", trajectory_sha256="c"*64)
            source["trajectories"]["0"][0][0] += 0.123456789012345
            source["metrics"]["0"]["score"] = .8 + source["trajectories"]["0"][0][0] * .01
            rows[source["token"]] = source
        imported = dict(schema="chainflow-external-baseline-v1", rows=rows,
                        identity=identity, manifest={"metric_cache": str(self.metric)},
                        summary={"count": 4, "valid_count": 4, "failed_count": 0},
                        provenance={"source": "external/reproduction_ready.json", "files_sha256": {"index": "d"*64}})
        self.modules["score_function.navsim_chainflow.native"].load_agent = lambda *a: SimpleNamespace(_score_bridge_identity=identity)
        return imported

    def test_external_float64_k0_is_replayed_then_reused_without_native_forward(self):
        imported = self.external_fixture()
        from score_function.navsim_chainflow import handoff
        with patch.dict("sys.modules", self.modules), patch.object(handoff, "is_handoff", return_value=True), \
                patch.object(handoff, "load_handoff", return_value=imported), \
                patch.object(FakeRefiner, "require_conditioner"):
            ev.run(self.args("refine", shards=1))
        self.assertEqual(self.native_calls, 4)
        _, rows, _ = ev.collect_completed(self.root / "refine")
        for row in rows:
            source = imported["rows"][row["token"]]
            self.assertEqual(row["trajectories"]["0"], source["trajectories"]["0"])
            self.assertEqual(row["metrics"]["0"], source["metrics"]["0"])
            self.assertEqual(row["k0_metric_audit"]["max_abs_error"], 0.)
            self.assertFalse(row["native_forward_executed"])
            ev.validate_row(row, row["token"], "log", [0, 5, 10, 20], baseline=source)
        # A missing/different original condition hash is never invented.
        self.assertNotIn("condition_sha256", imported["rows"]["a"])

    def test_external_incompatible_metrics_fail_before_any_refinement(self):
        imported = self.external_fixture()
        imported["rows"]["a"]["metrics"]["0"]["time_to_collision_within_bound"] = .5
        from score_function.navsim_chainflow import handoff
        with patch.dict("sys.modules", self.modules), patch.object(handoff, "is_handoff", return_value=True), \
                patch.object(handoff, "load_handoff", return_value=imported), \
                patch.object(FakeRefiner, "require_conditioner"), patch.object(FakeRefiner, "refine_many") as refine:
            with self.assertRaisesRegex(RuntimeError, "Evaluation failed") as error:
                ev.run(self.args("refine", shards=1))
            self.assertIn("metric replay differs", str(error.exception.__cause__))
            refine.assert_not_called()
        self.assertEqual(self.native_calls, 4)

    def test_external_cpu_preflight_checks_saved_trajectories_without_encoder(self):
        imported = self.external_fixture()
        from score_function.navsim_chainflow import handoff
        with patch.dict("sys.modules", self.modules), patch.object(handoff, "is_handoff", return_value=True), \
                patch.object(handoff, "load_handoff", return_value=imported):
            report = ev.audit_external_baseline(self.root / "baseline", self.devkit, self.metric,
                                                imported["identity"], max_scenes=3)
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["total_count"], 4)
        self.assertTrue(report["passed"])
        self.assertTrue(report["full_replay_required_during_evaluation"])
        self.assertEqual(self.native_calls, 4)

    def test_metric_audit_requires_all_finite_matching_components(self):
        original = vars(FakeMetrics())
        self.assertEqual(ev.verify_k0_metrics(original, original), 0.)
        for altered in ({**original, "comfort": float("nan")},
                        {**original, "ego_progress": .50001}, {"score": .9}):
            with self.assertRaises(ValueError):
                ev.verify_k0_metrics(altered, original)

    def test_external_native_handoff_validates_assets_coverage_and_summary(self):
        self.baseline()
        report = ev.validate_native_baseline(
            self.root / "baseline", expected_identity={"native": "chainflow"},
            expected_scoring=ev.scoring_identity(self.devkit), expected_count=4)
        self.assertEqual(report["count"], 4)
        with self.assertRaisesRegex(ValueError, "complete canonical"):
            ev.validate_native_baseline(self.root / "baseline")
        with self.assertRaisesRegex(ValueError, "conditioner identity"):
            ev.validate_native_baseline(self.root / "baseline", expected_identity={"native": "DD"}, expected_count=4)
        summary_path = self.root / "baseline/summary.json"
        invalid = json.loads(summary_path.read_text())
        invalid["by_k"]["0"]["mean"]["score"] += .001
        atomic_write(summary_path, invalid)
        with self.assertRaisesRegex(ValueError, "summary does not match"):
            ev.validate_native_baseline(self.root / "baseline", expected_count=4)

    def test_metric_cache_mutation_is_rejected_before_score_refinement(self):
        self.baseline()
        (self.metric / "a/metric_cache.pkl").write_text("different simulation state")
        with patch.dict("sys.modules", self.modules):
            with self.assertRaisesRegex(RuntimeError, "Evaluation failed") as raised:
                ev.run(self.args("refine", shards=1))
        self.assertIn("metric cache changed", str(raised.exception.__cause__))
        self.assertEqual(self.native_calls, 4)
        self.assertFalse((self.root / "refine/summary.json").exists())

    def test_changed_completed_results_and_partial_coverage_rejected(self):
        self.baseline()
        path = self.root / "baseline/part-0000-of-0002/scenes.jsonl"
        path.write_text(path.read_text().replace('"score": 0.', '"score": 0.0', 1))
        with self.assertRaisesRegex(ValueError, "records changed"):
            ev.collect_completed(self.root / "baseline")

    def test_incomplete_shard_resumes_only_complete_valid_lines(self):
        self.baseline()
        directory = self.root / "baseline/part-0000-of-0002"
        manifest = json.loads((directory / "run.json").read_text())
        source = directory / "scenes.jsonl"
        row = json.loads(source.read_text().splitlines()[0])
        source.unlink()
        partial = directory / "scenes.jsonl.tmp"
        partial.write_text(json.dumps(row) + '\n{"token":"c"')
        resumed = ev.resume_prefix(directory, manifest, {"a": "log", "c": "log"})
        self.assertEqual(len(resumed), 1)
        self.assertEqual(partial.read_text(), json.dumps(row) + "\n")
        row["metrics"]["0"]["score"] = float("nan")
        partial.write_text(json.dumps(row) + "\n")
        before = partial.read_bytes()
        with self.assertRaises(ValueError):
            ev.resume_prefix(directory, manifest, {"a": "log", "c": "log"})
        self.assertEqual(partial.read_bytes(), before)

    def test_baseline_protocol_rejects_dd_and_changed_native_seed(self):
        self.baseline()
        protocol, rows, _ = ev.collect_completed(self.root / "baseline")
        expected = copy.deepcopy(protocol)
        expected["native_seed"] = 20260922
        with self.assertRaisesRegex(ValueError, "native_seed"):
            ev.load_baseline(self.root / "baseline", expected, ["a", "b", "c", "d"])
        expected = copy.deepcopy(protocol)
        expected["identity"] = {"native": "diffusiondrive"}
        with self.assertRaisesRegex(ValueError, "identity"):
            ev.load_baseline(self.root / "baseline", expected, ["a", "b", "c", "d"])

    def test_refiner_k0_guard_and_launcher_workers(self):
        class BrokenRefiner(FakeRefiner):
            def refine_many(self, raw, condition, ks, gamma=None):
                return {k: raw + 1 for k in ks}
        with self.assertRaisesRegex(ValueError, "changed K0"):
            ev.score_arms(torch.zeros(1, 8, 3), {}, [0, 5], BrokenRefiner(), lambda p: {}, "cpu")
        args = launcher.parse_args(["run", "--root", str(self.root), "--mode", "refine", "--workers", "2"])
        cmd = launcher.worker_command(args, 1)
        self.assertEqual(cmd[cmd.index("--num-shards")+1], "2")
        self.assertEqual(cmd[cmd.index("--device")+1], "cuda:1")
        self.assertIn("--baseline", cmd)
        self.assertNotIn("--native-only", cmd)
        env = launcher.child_environment(args)
        self.assertEqual(env["CHAINFLOW_ENABLE_RAY"], "0")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertNotIn("DiffusionDrive", env["PYTHONPATH"])

    def test_completed_shards_cannot_bypass_changed_launch_without_launcher_state(self):
        original = self.baseline()["protocol"]
        request = {"mode": original["mode"], "workers": original["num_shards"],
                   "seed": original["seed"], "gamma": .1, "ks": original["ks"],
                   "max_scenes": original["max_scenes"], "data_root": original["data_root"],
                   "metric_cache": original["metric_cache"], "native_identity": original["identity"],
                   "scoring_identity": original["scoring_identity"]}
        self.assertEqual(launcher.completed_indices(self.root / "baseline", request), {0, 1})
        request["native_identity"] = {"native": "different checkpoint"}
        with self.assertRaisesRegex(ValueError, "current launch: identity"):
            launcher.completed_indices(self.root / "baseline", request)

    def test_standalone_launcher_holds_idle_lease_and_restores_environment(self):
        from score_function.navsim_chainflow import resources
        args = launcher.parse_args(["run", "--root", str(self.root), "--workers", "1"])
        args.output.mkdir(parents=True)
        gpu = resources.GPU(3, "GPU-test-3", 81920, 0, 0)
        self.assertEqual(args.gpu_policy, "wait-idle")
        with patch.dict(os.environ, {}, clear=True), patch.object(resources, "query_gpus", return_value=[gpu]):
            with self.assertRaisesRegex(RuntimeError, "body stops"):
                with launcher.evaluation_gpu_lease(args):
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-test-3")
                    self.assertEqual(os.environ["CF_CUDA_MEMORY_GB"], "12.0")
                    with (args.gpu_lock_dir / "GPU-test-3.lock").open("a+") as other:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise RuntimeError("body stops")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
            with (args.gpu_lock_dir / "GPU-test-3.lock").open("a+") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_parent_lease_requires_live_owner_metadata_and_actual_lock(self):
        args = launcher.parse_args(["run", "--root", str(self.root), "--workers", "1", "--gpu-lease-held"])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "live parent's"):
                launcher.validate_inherited_lease(args)
        args.gpu_lock_dir.mkdir(parents=True)
        path = args.gpu_lock_dir / "GPU-test-4.lock"
        record = {"pid": os.getppid(), "gpu_uuid": "GPU-test-4", "host": socket.gethostname(),
                  "budget_gb_per_worker": 12., "selected_uuids": ["GPU-test-4"]}
        path.write_text(json.dumps(record))
        env = {"CF_GPU_LEASE_OWNER_PID": str(os.getppid()), "CF_GPU_LEASE_LOCK_DIR": str(args.gpu_lock_dir),
               "CUDA_VISIBLE_DEVICES": "GPU-test-4", "CF_CUDA_MEMORY_GB": "12"}
        with patch.dict(os.environ, env, clear=True):
            with path.open("a+") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with launcher.evaluation_gpu_lease(args):
                    pass
            with self.assertRaisesRegex(ValueError, "no longer held"):
                launcher.validate_inherited_lease(args)


if __name__ == "__main__":
    unittest.main()
