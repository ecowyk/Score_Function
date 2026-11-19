"""Cross-planner launcher source binding, report and resume contracts (CPU-only)."""
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts import eval_chainflow_score_on_dd as queue
from score_function.navsim_chainflow import cross_dd


class CrossDDLauncherTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.args = queue.parse_args(["run", "--root", str(self.root), "--workers", "2"])
        exp = self.root / "dataset/navsim/exp"
        self.exp = exp
        prior = queue.pipeline.paths_for(self.root, "b64-e40")
        prior["navtest_vlm_cache"] = self.root / "original-test-vlm-release"
        prior["vlm_cache"] = self.root / "different-train-cache"
        prior["python"] = Path(sys.executable)
        self.plan = {"schema": "chainflow-score-pipeline-v1", "recipe": "b64-e40", "primary_k": 20,
                     "final_weights": "ema", "config": {"global_batch_size": 64, "epochs": 40},
                     "source_sha256": {p: queue.pipeline.sha256(queue.REPO / p) for p in queue.SCORE_SOURCES},
                     "paths": {k: str(p) for k, p in prior.items()}}
        queue.pipeline.atomic_json(self.args.plan, self.plan)
        queue.load_plan(self.args)
        self.args.training.mkdir(parents=True)
        self.args.score_checkpoint.write_bytes(b"original-score")
        self.checksum = queue.pipeline.sha256(self.args.score_checkpoint)
        queue.pipeline.atomic_json(self.args.training / "TRAINING_COMPLETE.json",
                                   {"signature": "score-run-a", "epochs": 40, "final_ema_sha256": self.checksum})
        queue.pipeline.atomic_json(self.args.training / "run.json",
                                   {"signature": "score-run-a", "config": self.plan["config"],
                                    "protocol": {"world_size": 8}, "global_batch": 64,
                                    "planned_optimizer_updates": 53160})
        self.dd = self.exp / "original-dd"
        self.dd.mkdir()
        self.exp.joinpath("dd_score_e200_eval_latest.txt").write_text(str(self.dd))

    def test_explicit_source_override_and_first_known_pointer(self):
        self.assertEqual(queue.dd_input(self.args), self.exp / "dd_score_e200_eval_latest.txt")
        self.args.dd_source = self.dd
        self.assertEqual(queue.dd_input(self.args), self.dd)
        self.assertEqual(self.args.score_checkpoint, self.args.training / "final_ema.pt")
        self.assertEqual(self.args.navtest_vlm_cache.name, "original-test-vlm-release")
        self.assertTrue(self.args.output.is_relative_to(self.exp))
        self.assertNotEqual(self.args.output, self.args.evaluation)

    def test_requires_original_final_checkpoint(self):
        self.assertEqual(queue.score_sha(self.args, self.plan), self.checksum)
        self.args.score_checkpoint.write_bytes(b"other-checkpoint")
        with self.assertRaisesRegex(ValueError, "differs"):
            queue.score_sha(self.args, self.plan)

    def test_request_binds_dd_and_reference_to_exact_score_and_conditions(self):
        from score_function.navsim_chainflow import cross_reference, dd_baseline
        cf = {"mode": "refine", "split": "navtest", "count": 12146, "max_scenes": None,
              "ks": queue.KS, "seed": 20260922, "native_seed": 27, "score_epoch": 40,
              "score_weight_kind": "ema", "gamma": .1, "score_checkpoint_sha256": self.checksum,
              "identity": {"features": 1}, "scoring_identity": {"weights": 1},
              "data_root": str(self.args.data_root), "metric_cache": str(self.args.metric_cache),
              "token_digest": "same-membership"}
        dd = {"mean_k0_score": .880337, "protocol": {k: cf[k] for k in ("token_digest", "data_root", "metric_cache")}}
        ref = {"summary_sha256": "checked-reference"}
        scoring = types.ModuleType("score_function.navsim_chainflow.evaluate")
        scoring.scoring_identity = mock.Mock(return_value=cf["scoring_identity"])
        native = types.ModuleType("score_function.navsim_chainflow.native")
        native.conditioner_identity = mock.Mock(return_value=cf["identity"])
        with mock.patch.dict(sys.modules, {scoring.__name__: scoring, native.__name__: native}), \
                mock.patch.object(cross_reference, "load_reference", return_value=({}, {"protocol": cf}, ref)), \
                mock.patch.object(dd_baseline, "load_dd_baseline", return_value=({}, dd)):
            req = queue.build_request(self.args, self.plan)
            self.assertEqual(req["dd_source"], str(self.dd))
            self.assertEqual(req["reference_provenance"], ref)
            self.assertEqual(req["vlm_cache"], str(self.args.navtest_vlm_cache))
            self.assertEqual(req["ks"], [0, 5, 10, 15, 20, 30])
            cf["score_checkpoint_sha256"] = "old_ema"
            with self.assertRaisesRegex(ValueError, "score_checkpoint_sha256"):
                queue.build_request(self.args, self.plan)
            self.plan["source_sha256"]["score_function/utils/normalizer.py"] = "changed"
            with self.assertRaisesRegex(ValueError, "inference source"):
                queue.build_request(self.args, self.plan)

    def test_resume_only_runs_missing_shards_and_preserves_report_if_relaunch_mismatches(self):
        req = {"workers": 2, "dd_source": str(self.dd)}
        queue.pipeline.atomic_json(self.args.output / "request.json", req)
        queue.pipeline.atomic_json(queue.shard_path(self.args.output, 1, 2) / "complete.json", {"count": 6073})
        for_k = {str(k): {"mean": {"score": .88}, "mean_delta": {"score": 0.},
                          "improved": 0, "worsened": 0, "tied": 12146} for k in queue.KS}

        def merge(output, request):
            self.assertEqual(request, req)
            s = {"count": 12146, "valid_count": 12146, "by_k": for_k, "chainflow_reference_by_k": for_k}
            queue.pipeline.atomic_json(output / "summary.json", s)
            return s

        with mock.patch.object(queue, "build_request", return_value=req) as check, \
                mock.patch.object(queue, "acquire_gpus", return_value=nullcontext(["gpu0"])) as lease, \
                mock.patch.object(queue, "gpu_environment", return_value={"CUDA_VISIBLE_DEVICES": "gpu0"}), \
                mock.patch.object(queue, "launch_workers") as workers, mock.patch.object(cross_dd, "merge", side_effect=merge), \
                mock.patch.dict(queue.os.environ), mock.patch.object(sys, "path", sys.path.copy()), redirect_stdout(io.StringIO()):
            queue.run(self.args)
            self.assertEqual(lease.call_args.kwargs["policy"], "wait-idle")
            self.assertEqual(lease.call_args.kwargs["workers"], 1)
            self.assertEqual(workers.call_args.args[1], [0])
            queue.pipeline.atomic_json(queue.shard_path(self.args.output, 0, 2) / "complete.json", {"count": 6073})
            queue.run(self.args)
            self.assertEqual(lease.call_count, 1)
            check.return_value = dict(req, dd="new")
            with self.assertRaisesRegex(ValueError, "different request"):
                queue.run(self.args)
            self.assertEqual(queue.pipeline.read_json(self.args.output / "launcher.json")["status"], "complete")
            queue.print_report(self.args.output)

    def test_no_result_reported_with_changed_summary(self):
        queue.pipeline.atomic_json(self.args.output / "launcher.json", {"status": "complete", "summary_sha256": "old"})
        queue.pipeline.atomic_json(self.args.output / "summary.json", {"invalid": True})
        with self.assertRaisesRegex(ValueError, "verified"):
            queue.print_report(self.args.output)


if __name__ == "__main__":
    unittest.main()
