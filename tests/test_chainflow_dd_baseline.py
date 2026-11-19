"""DiffusionDrive source import never substitutes ChainFlow K0 or accepts partial data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from score_function.navsim_chainflow import dd_baseline as dd


class DDBaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "DD"
        self.tokens = [f"{i:016x}" for i in range(4)]
        self.digest = hashlib.sha256("\n".join(self.tokens).encode()).hexdigest()
        self.protocol = {
            "split": "navtest", "count": 4, "ks": [0, 20], "seed": 20260922, "num_shards": 2,
            "token_digest": self.digest, "max_scenes": None, "metric_cache": "/cache", "data_root": "/data",
            "identity": {"dd_commit": dd.DD_COMMIT, "condition": "dd-keyval64-status-v1",
                         "checkpoint_sha256": dd.DD_CHECKPOINT_SHA256, "anchor_sha256": "e" * 64,
                         "backbone_sha256": dd.DD_BACKBONE_SHA256},
        }
        self.rows = []
        for i, token in enumerate(self.tokens):
            poses = [[float(t) + 0.125, -0.5 * i, 0.25] for t in range(8)]
            scores = dict.fromkeys(dd.METRICS, i * 0.25)
            self.rows.append({"token": token, "log": f"log{i}", "top1_index": i, "seed": dd._token_seed(dd.DD_SEED, token),
                              "trajectories": {"0": poses, "20": [[0., 0., 0.]] * 8},
                              "metrics": {"0": scores}, "delta": {"0": dict.fromkeys(dd.METRICS, 0.)}})
        self.summary = {"count": 4, "score_scale": "0..1", "by_k": {"0": {
            "mean": dict.fromkeys(dd.METRICS, .375)}}}
        self.publish()

    def publish(self):
        for i in range(2):
            path = self.root / f"part-{i:04d}-of-0002"
            path.mkdir(parents=True, exist_ok=True)
            manifest = {"run": self.protocol, "shard_index": i, "shard_tokens": [r["token"] for r in self.rows[i::2]]}
            for name in ("run.json", "complete.json"):
                (path / name).write_text(json.dumps(manifest))
            (path / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows[i::2]))
        (self.root / "summary.json").write_text(json.dumps(self.summary))

    def load(self, path=None, **kw):
        return dd.load_dd_baseline(path or self.root, expected_count=4, expected_token_digest=self.digest, **kw)

    def test_exact_k0_source_identity_per_shard_and_pointer(self):
        pointer = self.root.parent / "dd_score_e200_eval_latest.txt"
        pointer.write_text(str(self.root) + "\n")
        subset, provenance = self.load(pointer, needed=self.tokens[1::2])
        all_rows, other_provenance = self.load()
        self.assertEqual(provenance, other_provenance)
        self.assertEqual(self.load(needed=[]), ({}, provenance))
        self.assertEqual(set(subset), set(self.tokens[1::2]))
        self.assertEqual(provenance["mean_k0_score"], 0.375)
        self.assertEqual(provenance["source_seed"], 20260922)
        self.assertEqual(provenance["protocol"], self.protocol)
        for row in self.rows:
            saved = all_rows[row["token"]]
            self.assertEqual(saved["trajectories"], {"0": row["trajectories"]["0"]})
            self.assertEqual(saved["native_seed"], row["seed"])
            self.assertEqual(saved["metrics"], {"0": row["metrics"]["0"]})
            self.assertEqual(saved["log"], row["log"])
            self.assertEqual(saved["top1_index"], row["top1_index"])
            self.assertEqual(saved["trajectory_dtype"], "float32")
        files = [p for p in self.root.rglob("*") if p.is_file()]
        self.assertEqual(provenance["files_sha256"], {str(p.relative_to(self.root)):
                         hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        self.rows[0]["trajectories"]["0"][0][0] += 1
        self.publish()
        new, updated = self.load()
        self.assertNotEqual(all_rows[self.tokens[0]]["trajectory_sha256"], new[self.tokens[0]]["trajectory_sha256"])
        self.assertNotEqual(updated["files_sha256"], provenance["files_sha256"])

    def test_rejects_mixed_or_incomplete_run(self):
        manifest = self.root / "part-0000-of-0002/complete.json"
        original = json.loads(manifest.read_text())
        manifest.write_text(json.dumps({**original, "shard_tokens": []}))
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            self.load()
        self.publish()
        (self.root / "part-0001-of-0002/scenes.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "order/coverage"):
            self.load()

    def test_rejects_changed_metrics_seed_and_validity_outside_worker_selection(self):
        for key, value in (("seed", 27), ("valid", False), ("log", ""), ("top1_index", 20)):
            with self.subTest(key=key):
                original = self.rows[1]
                self.rows[1] = {**original, key: value}
                self.publish()
                with self.assertRaisesRegex(ValueError, "Invalid DD token"):
                    self.load(needed=[self.tokens[0]])
                self.rows[1] = original
        self.rows[1]["metrics"]["0"]["comfort"] = .11
        self.publish()
        with self.assertRaisesRegex(ValueError, "K0 summary"):
            self.load()

    def test_rejects_wrong_native_assets_and_membership(self):
        original = copy.deepcopy(self.protocol)
        for field in ("checkpoint_sha256", "dd_commit", "backbone_sha256"):
            with self.subTest(field=field):
                self.protocol["identity"][field] = "0" * 64
                self.publish()
                with self.assertRaisesRegex(ValueError, "pinned"):
                    self.load()
                self.protocol = copy.deepcopy(original)
        self.protocol["token_digest"] = "0" * 64
        self.publish()
        with self.assertRaisesRegex(ValueError, "token digest"):
            self.load()

    def test_rejects_duplicated_token_and_precision_loss(self):
        self.rows[2] = copy.deepcopy(self.rows[0])
        self.publish()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()
        self.rows[0]["trajectories"]["0"][0][0] = 1.2345678912345
        self.publish()
        with self.assertRaisesRegex(ValueError, "float32"):
            self.load()
