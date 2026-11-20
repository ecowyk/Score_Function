import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from score_function.navsim_chainflow import dd_checkpoint_matrix as matrix
from score_function.navsim_chainflow.assets import atomic_json, sha256
from score_function.navsim_chainflow.handoff import METRICS


def metrics(score):
    return {key: score for key in METRICS}


def scene(token, base, score):
    normal = metrics(base)
    poses = [[1., 0., 0.]] * 8
    dd = dict(log="log", trajectory_sha256="start-"+token, metrics={"0": normal},
              trajectories={"0": poses})
    cf = dict(log="log", condition_sha256="condition-"+token, metric_cache_sha256="metric-"+token)
    row = dict(token=token, log="log", dd_k0_sha256=dd["trajectory_sha256"], k0=normal, k0_replay=normal,
               condition_sha256=cf["condition_sha256"], metric_cache_sha256=cf["metric_cache_sha256"],
               k20={"lr/best": dict(metrics=metrics(score), trajectory=poses)})
    return dd, cf, row


def runtime():
    return dict(device_type="cuda", matmul_allow_tf32=False, cudnn_allow_tf32=False,
                float32_matmul_precision="highest", cudnn_deterministic=False, cudnn_benchmark=False,
                deterministic_algorithms=False, cublas_workspace_config=None)


class MatrixTests(unittest.TestCase):
    def test_matching_policy_uses_real_production_runtime_key(self):
        baseline = runtime()
        alternate = runtime()
        alternate["matmul_precision"] = alternate.pop("float32_matmul_precision")
        matrix.matching_math(baseline, alternate)
        with self.assertRaises(ValueError):
            matrix.matching_math(baseline, dict(alternate, matmul_precision="medium"))

    def test_resume_preserves_prefix_and_rejects_mixed_arm_or_changed_base(self):
        src, ref, row = scene("a", .7, .8)
        d, r, names = {"a": src}, {"a": ref}, ["lr/best"]
        manifest = {"shard_tokens": ["a"]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows = directory / "scenes.jsonl.tmp"
            good = json.dumps(row).encode() + b"\n"
            rows.write_bytes(good + b'{"token"')
            self.assertEqual(matrix.read_records(directory, manifest, d, r, names), 1)
            self.assertEqual(rows.read_bytes(), good)
            for wrong in (dict(row, k20={}), dict(row, dd_k0_sha256="different"),
                          dict(row, condition_sha256="different"), dict(row, k0_replay=metrics(.9))):
                rows.write_text(json.dumps(wrong) + "\n")
                with self.assertRaises(ValueError):
                    matrix.read_records(directory, manifest, d, r, names)

    def test_merge_pairs_all_candidates_using_shared_k0_and_rejects_corruption(self):
        env = runtime()
        tokens = ["a", "b"]
        data = [scene("a", .7, .8), scene("b", .9, .6)]
        dd = {t: item[0] for t, item in zip(tokens, data)}
        cf = {t: item[1] for t, item in zip(tokens, data)}
        protocol = {"protocol": {"execution_identity": env}}
        req = dict(base={}, workers=2, checkpoints={"lr/best": {}, "reference/final": {}},
                   reuse_reference_final=dict(by_k20=dict(mean=metrics(.8), mean_delta=metrics(0.),
                       improved=0, worsened=0, tied=2), output="/historic", summary_sha256="historical"))
        token_digest = hashlib.sha256(b"a\nb").hexdigest()
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(matrix.cross_dd, "sources", return_value=(dd, cf, protocol)))
            stack.enter_context(patch.object(matrix, "TOTAL", 2))
            stack.enter_context(patch.object(matrix, "NAVTEST_TOKEN_DIGEST", token_digest))
            output = Path(temp)
            atomic_json(output / "request.json", req)
            for i, token in enumerate(tokens):
                part = matrix.cross_dd.shard_directory(output, i, 2)
                manifest = dict(schema=matrix.SCHEMA, request_sha256=matrix.cross_dd.request_digest(req),
                                shard_index=i, shard_tokens=[token], runtime=env)
                atomic_json(part / "run.json", manifest)
                rows = part / "scenes.jsonl"
                rows.write_text(json.dumps(data[i][2]) + "\n")
                atomic_json(part / "complete.json", dict(manifest=manifest, count=1, scenes_sha256=sha256(rows)))
            result = matrix.merge(output, req)
            self.assertAlmostEqual(result["k0"], .8)
            stats = result["by_checkpoint"]["lr/best"]
            self.assertAlmostEqual(stats["mean"]["score"], .7)
            self.assertAlmostEqual(stats["mean_delta"]["score"], -.1)
            self.assertEqual([stats[k] for k in ("improved", "worsened", "tied")], [1, 1, 0])
            historical = result["by_checkpoint"]["reference/final"]
            self.assertEqual(historical["per_scene_directory"], "/historic")
            rows.write_bytes(rows.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "published matrix shard"):
                matrix.merge(output, req)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_shared_conditions_one_k0_replay_and_only_one_k20_per_distinct_checkpoint(self):
        import torch
        from score_function.navsim_chainflow import evaluate as cf

        dd, ref, _ = scene("a", .7, .8)
        cond = {key: torch.ones(1, 1) for key in ("scene", "ego", "vlm")}
        ref["condition_sha256"] = cf.condition_digest(cond)
        calls = {"condition": 0, "score": 0}
        class Refiner:
            def __init__(self, shift):
                self.shift = shift

            def refine_many(self, raw, condition, ks, gamma):
                self.args = ks, gamma
                return {0: raw, 20: raw + self.shift}
        x, y = Refiner(.1), Refiner(.2)

        def condition():
            calls["condition"] += 1
            return cond

        def score(poses):
            calls["score"] += 1
            return metrics(.7 if calls["score"] == 1 else .8)

        refiners = {"lr/best": x, "lr/final": y, "copy": x}
        row = matrix.evaluate_scene("a", dd, ref, ref["metric_cache_sha256"],
                                    score, condition, refiners, "cpu")
        self.assertEqual(calls, dict(condition=1, score=3))
        self.assertEqual(row["k20"]["lr/best"], row["k20"]["copy"])
        self.assertEqual(x.args, ([0, 20], .1))
        self.assertEqual(y.args, x.args)


if __name__ == "__main__":
    unittest.main()
