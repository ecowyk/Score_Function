"""CPU checks for cross-planner resume, merge provenance and exact scene pairing."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from score_function.navsim_chainflow import cross_dd as cross
from score_function.navsim_chainflow.handoff import METRICS


def records():
    token = "0000000000000001"
    poses = [[1., 0., 0.]] * 8
    metrics = {m: .8 for m in METRICS}
    source = dict(log="same-log", native_seed=123, top1_index=6, trajectory_sha256="c"*64,
                  trajectories={"0": poses}, metrics={"0": metrics})
    reference = dict(log="same-log", metric_cache_sha256="a"*64, condition_sha256="b"*64)
    row = dict(token=token, log="same-log", valid=True, native_origin="diffusiondrive", native_seed=123,
               top1_index=6, native_ms=0., native_forward_executed=False, conditioner_ms=1., refine_ms=2.,
               metric_cache_sha256="a"*64, condition_sha256="b"*64,
               k0_metric_audit=dict(metrics=metrics.copy(), max_abs_error=0., tolerance=1e-9, trajectory_sha256="c"*64),
               metrics={}, trajectories={}, delta={})
    for k in cross.KS:
        row["metrics"][str(k)] = {m: .75 if k and m == "score" else .8 for m in METRICS}
        row["trajectories"][str(k)] = copy.deepcopy(poses)
        row["delta"][str(k)] = {m: row["metrics"][str(k)][m] - v for m, v in metrics.items()}
    return token, row, source, reference


class CrossDiffusionDriveTests(unittest.TestCase):
    def test_checkpoints_and_different_planner_seeds_are_not_interchangeable(self):
        token, row, source, reference = records()
        self.assertEqual(cross.validate_row(row, token, source, reference)["metrics"]["20"]["score"], .75)
        for key in ("metric_cache_sha256", "condition_sha256", "native_origin", "native_seed", "top1_index"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                cross.validate_row(dict(row, **{key:"other"}), token, source, reference)
        wrong = copy.deepcopy(row)
        wrong["trajectories"]["0"][2][0] += 1e-5
        with self.assertRaisesRegex(ValueError, "DD K0"):
            cross.validate_row(wrong, token, source, reference)
        wrong = copy.deepcopy(row)
        wrong["k0_metric_audit"]["metrics"]["ego_progress"] += .01
        with self.assertRaisesRegex(ValueError, "replay"):
            cross.validate_row(wrong, token, source, reference)
        wrong = copy.deepcopy(row)
        wrong["delta"]["20"]["score"] = .05
        with self.assertRaisesRegex(ValueError, "paired delta"):
            cross.validate_row(wrong, token, source, reference)

    def test_resume_only_drops_an_unfinished_last_line(self):
        token, row, source, reference = records()
        manifest = dict(shard_tokens=[token], request_sha256="a")
        with tempfile.TemporaryDirectory() as temp:
            part = Path(temp)
            payload = (json.dumps(row)+"\n").encode()
            final = part/"scenes.jsonl"
            partial = part/"scenes.jsonl.tmp"
            final.write_bytes(payload+b"{")
            args = part, manifest, {token: source}, {token: reference}
            self.assertEqual(len(cross.read_records(*args, completed=False)), 1)
            self.assertEqual(partial.read_bytes(), payload)
            partial.replace(final)
            marker = dict(manifest=manifest, count=1, scenes_sha256=hashlib.sha256(payload).hexdigest())
            (part/"complete.json").write_text(json.dumps(marker))
            self.assertEqual(len(cross.read_records(*args, completed=True)), 1)
            final.write_bytes(payload+b" ")
            with self.assertRaisesRegex(ValueError, "completion"):
                cross.read_records(*args, completed=True)

    def test_math_policy_matches_reference_without_requiring_identical_device_names(self):
        reference = dict(device_type="cuda", matmul_allow_tf32=False, cudnn_allow_tf32=True,
                         matmul_precision="highest", cudnn_deterministic=False, cudnn_benchmark=False,
                         deterministic_algorithms=False, cublas_workspace_config=None, device_name="A100")
        cross.require_matching_math(dict(reference, device_name="A800"), reference)
        with self.assertRaisesRegex(ValueError, "matmul_allow_tf32"):
            cross.require_matching_math(dict(reference, matmul_allow_tf32=True), reference)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for tensor inference")
    def test_real_tensor_path_replays_then_refines_dd_origin_with_cf_condition(self):
        import torch
        from score_function.navsim_chainflow.evaluate import condition_digest

        token, _, source, reference = records()
        condition = {"scene": torch.ones(1, 64, 256), "ego": torch.zeros(1, 256),
                     "vlm": torch.ones(1, 2, 1536)}
        reference["condition_sha256"] = condition_digest(condition)
        calls = []
        class FakeScore:
            def refine_many(self, raw, ctx, ks, gamma):
                calls.append("refine")
                self.raw = raw.tolist()[0]
                self.condition = ctx
                return {k: raw + .01*k for k in ks}

        refiner = FakeScore()
        def score(trajectory):
            calls.append("score")
            return {**dict.fromkeys(METRICS, .8), "score": float(trajectory[0, 0]) * .1 + .7}
        def encode():
            calls.append("encode")
            return condition
        kwargs = dict(metric_hash=reference["metric_cache_sha256"], refiner=refiner, device="cpu",
                      gamma=.1, score=score, get_condition=encode)
        result = cross.evaluate_pair(token, source, reference, **kwargs)
        self.assertEqual(calls, ["score", "encode", "refine", *(["score"]*5)])
        self.assertEqual(refiner.raw, source["trajectories"]["0"])
        self.assertEqual(result["metrics"]["0"], source["metrics"]["0"])
        self.assertIs(refiner.condition, condition)
        self.assertGreater(result["metrics"]["20"]["score"], result["metrics"]["0"]["score"])

        calls.clear()
        source["metrics"]["0"]["score"] -= .1
        with self.assertRaisesRegex(ValueError, "metric replay"):
            cross.evaluate_pair(token, source, reference, **kwargs)
        self.assertEqual(calls, ["score"])

    def test_sources_bind_same_scene_score_weights_and_frozen_context_protocol(self):
        token, _, dd, reference = records()
        dd_provenance = dict(protocol={"data_root":"/data", "metric_cache":"/metrics"})
        cf_provenance = dict(summary_sha256="c"*64)
        protocol = dict(score_epoch=40, score_weight_kind="ema", gamma=.1, ks=cross.KS,
                        score_checkpoint_sha256="d"*64, identity={"conditioner":"CF"}, scoring_identity={"yaml":"pinned"})
        summary = dict(protocol=protocol)
        request = dict(ks=cross.KS, seed=20260922, gamma=.1, dd_source="/dd", dd_baseline=dd_provenance,
                       data_root="/data", metric_cache="/metrics", reference_evaluation="/cf",
                       reference_provenance=cf_provenance, reference_summary_sha256=cf_provenance["summary_sha256"],
                       native_identity=protocol["identity"], scoring_identity=protocol["scoring_identity"],
                       score_checkpoint_sha256=protocol["score_checkpoint_sha256"])
        with patch("score_function.navsim_chainflow.dd_baseline.load_dd_baseline", return_value=({token: dd}, dd_provenance)), \
             patch("score_function.navsim_chainflow.cross_reference.load_reference", return_value=({token: reference}, summary, cf_provenance)):
            self.assertEqual(cross.sources(request, [token])[0][token], dd)
            with self.assertRaisesRegex(ValueError, "score_checkpoint_sha256"):
                cross.sources(dict(request, score_checkpoint_sha256="other"))
            with self.assertRaisesRegex(ValueError, "membership"):
                cross.sources(request, ["other"])
            reference["log"] = "mismatched-log"
            with self.assertRaisesRegex(ValueError, "Log mismatch"):
                cross.sources(request)
