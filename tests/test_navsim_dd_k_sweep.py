"""Paired K sweep comparisons must reuse the same checkpoint and exact K0."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from score_function.navsim_dd import k_sweep as sweep


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(root, ks, *, shards=2):
    tokens = [f"scene-{i}" for i in range(4)]
    run = dict(split="navtest", seed=20260922, gamma=0.1, identity={"dd": "same"},
               score_checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
               token_digest=sweep._token_digest(tokens), count=len(tokens),
               data_root="/data/navsim", metric_cache="/data/metrics", max_scenes=None,
               ks=ks, num_shards=shards)
    records = []
    for i, token in enumerate(tokens):
        values = {str(k): {"score": 0.5 + (k * (1 if i % 2 else -1) + i) / 1000,
                           "drivable_area_compliance": 1.0} for k in ks}
        records.append(dict(token=token, log=f"log-{i}", seed=sweep._token_seed(run["seed"], token), top1_index=i,
                            metrics=values, delta={k: {m: v[m] - values["0"][m] for m in v} for k, v in values.items()},
                            trajectories={str(k): [[float(j), k / 100, 0.0] for j in range(8)] for k in ks}))
    for i in range(shards):
        part = root / f"part-{i:04}-of-{shards:04}"
        rows = records[i::shards]
        manifest = dict(run=run, shard_index=i, shard_tokens=[r["token"] for r in rows])
        write_json(part / "run.json", manifest)
        write_json(part / "complete.json", manifest)
        (part / "scenes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {"count": len(tokens), "by_k": {str(k): {"mean": {
        m: sum(r["metrics"][str(k)][m] for r in records) / len(records) for m in values["0"]}}
        for k in ks}}
    write_json(root / "summary.json", summary)
    write_json(root / "launcher.json", {"epochs": 500, "status": "complete"})
    return records, run


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class KSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference, self.output = self.root / "reference", self.root / "new"
        fixture(self.reference, [0, 20])
        fixture(self.output, [0, 5, 10, 50, 100])
        self.checkpoint = self.root / "final_raw.pt"
        self.checkpoint.write_bytes(b"checkpoint")

    def merge(self):
        return sweep.merge_comparison(self.output, self.reference, expected_count=4)

    def mutate_run(self, directory, key, value):
        for path in directory.glob("part-*/run.json"):
            manifest = json.loads(path.read_text())
            manifest["run"][key] = value
            write_json(path, manifest)
            write_json(path.with_name("complete.json"), manifest)

    def mutate_row(self, key, value):
        path = self.output / "part-0000-of-0002/scenes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0][key] = value
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_merges_all_k_values_and_keeps_evaluation_files_untouched(self):
        old = tree_bytes(self.reference)
        new = tree_bytes(self.output)
        result = self.merge()
        self.assertEqual(result["ks"], [0, 5, 10, 20, 50, 100])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["by_k"]["0"]["origin"], "reused")
        self.assertEqual(result["by_k"]["20"]["origin"], "reused")
        self.assertEqual(result["by_k"]["5"]["origin"], "computed")
        self.assertEqual(result["by_k"]["20"]["improved"], 2)
        self.assertEqual(result["by_k"]["20"]["worsened"], 2)
        self.assertEqual(result["by_k"]["0"]["tied"], 4)
        self.assertAlmostEqual(result["by_k"]["100"]["mean_delta"]["score"], 0.0)
        self.assertEqual(tree_bytes(self.reference), old)
        self.assertEqual({k: v for k, v in tree_bytes(self.output).items() if not k.startswith("comparison.")}, new)
        self.assertIn("PDMS_100", (self.output / "comparison.csv").read_text())
        self.assertEqual(json.loads((self.output / "comparison.json").read_text()), result)

    def test_print_table_labels_reused_and_computed_on_hundred_point_scale(self):
        self.merge()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            sweep.print_comparison(self.output)
        text = stream.getvalue()
        self.assertIn("500 epoch RAW", text)
        self.assertIn("4 paired scenes", text)
        self.assertIn("50.1500", text)
        self.assertIn("reused", text)
        self.assertIn("computed", text)

    def test_reference_checkpoint_bound_to_bytes_not_filename(self):
        result = sweep.validate_reference(self.reference, self.checkpoint, expected_count=4)
        self.assertIn("launcher.json", result["provenance"]["files_sha256"])
        self.checkpoint.write_bytes(b"other checkpoint")
        with self.assertRaisesRegex(ValueError, "different score checkpoint"):
            sweep.validate_reference(self.reference, self.checkpoint, expected_count=4)

    def test_wrong_source_epoch_and_incomplete_launcher_rejected(self):
        for change in ({"epochs": 100}, {"status": "running"}):
            write_json(self.reference / "launcher.json", {"epochs": 500, "status": "complete", **change})
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "500-epoch"):
                self.merge()
        self.assertFalse((self.output / "comparison.json").exists())

    def test_every_scientific_protocol_field_must_match(self):
        for key, value in (("gamma", 0.2), ("identity", {"dd": "different"}),
                           ("score_checkpoint_sha256", "different"), ("data_root", "/other"),
                           ("metric_cache", "/other"), ("max_scenes", 4)):
            with self.subTest(key=key):
                original = tree_bytes(self.output)
                self.mutate_run(self.output, key, value)
                with self.assertRaisesRegex(ValueError, key):
                    self.merge()
                for path, contents in original.items():
                    (self.output / path).write_bytes(contents)

    def test_k0_exact_coordinates_log_and_native_mode_must_match(self):
        path = self.output / "part-0000-of-0002/scenes.jsonl"
        original = path.read_bytes()
        first = json.loads(path.read_text().splitlines()[0])
        changed_poses = copy.deepcopy(first["trajectories"])
        changed_poses["0"][0][0] += 1e-8
        for key, value, expected in (("log", "different-log", "log mismatch"),
                                      ("top1_index", 19, "top1_index mismatch"),
                                      ("trajectories", changed_poses, "K0 trajectory mismatch")):
            with self.subTest(key=key):
                self.mutate_row(key, value)
                with self.assertRaisesRegex(ValueError, expected):
                    self.merge()
                path.write_bytes(original)

    def test_scene_seed_mismatch_rejected(self):
        self.mutate_row("seed", 1)
        with self.assertRaisesRegex(ValueError, "seed"):
            self.merge()

    def test_k0_metrics_must_match_even_when_new_summary_is_self_consistent(self):
        path = self.output / "part-0000-of-0002/scenes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["metrics"]["0"]["score"] += 0.001
        for k in rows[0]["delta"]:
            rows[0]["delta"][k]["score"] = rows[0]["metrics"][k]["score"] - rows[0]["metrics"]["0"]["score"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = json.loads((self.output / "summary.json").read_text())
        summary["by_k"]["0"]["mean"]["score"] += 0.001 / 4
        write_json(self.output / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "K0 metric mismatch"):
            self.merge()

    def test_reference_protocol_requires_same_fixed_gamma_and_source_commit(self):
        before = tree_bytes(self.reference)
        for key, value, expected in (("gamma", 0.2, "gamma"),
                                    ("identity", {"dd_commit": "other"}, "source commit")):
            with self.subTest(key=key):
                self.mutate_run(self.reference, key, value)
                with self.assertRaisesRegex(ValueError, expected):
                    sweep.validate_reference(self.reference, self.checkpoint, expected_count=4)
                for path, contents in before.items():
                    (self.reference / path).write_bytes(contents)

    def test_duplicate_tokens_and_nonfinite_submetrics_rejected(self):
        path = self.output / "part-0000-of-0002/scenes.jsonl"
        before = path.read_bytes()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(row) + "\n" for row in [rows[0], rows[0]]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.merge()
        path.write_bytes(before)
        rows[0]["metrics"]["5"]["drivable_area_compliance"] = float("nan")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "metric set"):
            self.merge()

    def test_incomplete_shard_and_wrong_summary_rejected(self):
        part = self.output / "part-0001-of-0002/complete.json"
        content = part.read_bytes()
        part.unlink()
        with self.assertRaises(FileNotFoundError):
            self.merge()
        part.write_bytes(content)
        summary = json.loads((self.output / "summary.json").read_text())
        summary["by_k"]["5"]["mean"]["score"] += 0.01
        write_json(self.output / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "Summary differs"):
            self.merge()

    def test_k20_cannot_be_silently_replaced_by_newly_computed_arm(self):
        other = self.root / "wrong-new"
        fixture(other, [0, 5, 20])
        with self.assertRaisesRegex(ValueError, "excluding reused K20"):
            sweep.merge_comparison(other, self.reference, expected_count=4)

    def test_independent_shard_partition_is_allowed(self):
        other = self.root / "one-shard"
        fixture(other, [0, 1, 3], shards=1)
        report = sweep.merge_comparison(other, self.reference, expected_count=4)
        self.assertEqual(report["ks"], [0, 1, 3, 20])

    def test_reference_cannot_be_overwritten(self):
        before = tree_bytes(self.reference)
        with self.assertRaisesRegex(ValueError, "independent"):
            sweep.merge_comparison(self.reference, self.reference, expected_count=4)
        self.assertEqual(tree_bytes(self.reference), before)


if __name__ == "__main__":
    unittest.main()
