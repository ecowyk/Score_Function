import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from score_function.navsim_chainflow.cross_reference import load_reference
from score_function.navsim_chainflow.handoff import METRICS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj) + "\n")


class ChainFlowStartReferenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.tokens = [f"{i:016x}" for i in range(4)]
        self.ks = [0, 5, 10, 15, 20, 30]
        self.protocol = dict(schema="navsim-chainflow-evaluation-v1", mode="refine", split="navtest",
                             seed=20260922, native_seed=27, count=4, num_shards=2,
                             token_digest=sha("\n".join(self.tokens).encode()), ks=self.ks,
                             gamma=.1, score_epoch=40, score_weight_kind="ema",
                             score_checkpoint_sha256="e"*64, max_scenes=None)
        self.by_k = {}
        self.rows = []
        for i, token in enumerate(self.tokens):
            metrics = {str(k): {**dict.fromkeys(METRICS, 1.), "score": .9 - i*.1 - k*.001}
                       for k in self.ks}
            self.rows.append(dict(token=token, metrics=metrics, valid=True, log=f"log-{i//2}",
                                  condition_sha256=sha(token.encode()),
                                  metric_cache_sha256=sha(("metric-" + token).encode())))
        for k in self.ks:
            mean = .75 - k*.001
            self.by_k[str(k)] = {"mean": {**dict.fromkeys(METRICS, 1.), "score": mean},
                                 "mean_delta": {**dict.fromkeys(METRICS, 0.), "score": -k*.001}}
        self.publish()

    def publish(self):
        write(self.root / "summary.json", dict(count=4, valid_count=4, failed_count=0,
                                               mode="refine", protocol=self.protocol, by_k=self.by_k))
        for i in range(2):
            directory = self.root / f"part-{i:04d}-of-0002"
            directory.mkdir(exist_ok=True)
            rows = self.rows[i::2]
            manifest = {"run": self.protocol, "shard_index": i, "shard_tokens": [r["token"] for r in rows]}
            write(directory / "run.json", manifest)
            data = "".join(json.dumps(r) + "\n" for r in rows).encode()
            (directory / "scenes.jsonl").write_bytes(data)
            write(directory / "complete.json", {"manifest": manifest, "count": len(rows),
                                                "scenes_sha256": sha(data)})

    def load(self, needed=None):
        return load_reference(self.root, expected_count=4, needed=needed)

    def test_subset_matches_all_and_pins_source_artifacts(self):
        rows, summary, provenance = self.load()
        subset, _, subset_provenance = self.load(self.tokens[:1])
        self.assertEqual(list(subset), self.tokens[:1])
        self.assertEqual(subset[self.tokens[0]], rows[self.tokens[0]])
        self.assertEqual(subset[self.tokens[0]]["metrics"], self.rows[0]["metrics"])
        self.assertEqual(self.load([]), ({}, summary, provenance))
        self.assertEqual(subset_provenance, provenance)
        self.assertEqual(provenance["files_sha256"], {
            str(p.relative_to(self.root)): sha(p.read_bytes()) for p in self.root.rglob("*") if p.is_file()
        })

    def test_tampering_outside_worker_selection_is_rejected(self):
        path = self.root / "part-0001-of-0002/scenes.jsonl"
        path.write_text(path.read_text().replace("log-0", "log-z"))
        with self.assertRaisesRegex(ValueError, "completion marker"):
            self.load([self.tokens[0]])

    def test_mixed_protocol_is_rejected(self):
        path = self.root / "part-0001-of-0002/run.json"
        manifest = json.loads(path.read_text())
        manifest["run"]["score_checkpoint_sha256"] = "f"*64
        write(path, manifest)
        with self.assertRaisesRegex(ValueError, "Mixed reference shards"):
            self.load()

    def test_duplicate_or_missing_tokens_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "token coverage"):
            self.load(["missing"])
        self.rows[1] = self.rows[0]
        self.publish()
        with self.assertRaisesRegex(ValueError, "token coverage"):
            self.load()

    def test_incompatible_experiment_and_stale_summary_are_rejected(self):
        for field, incompatible in (("score_epoch", 50), ("score_weight_kind", "raw"), ("seed", 27)):
            with self.subTest(field=field):
                original = self.protocol[field]
                self.protocol[field] = incompatible
                self.publish()
                with self.assertRaisesRegex(ValueError, "epoch-40"):
                    self.load()
                self.protocol[field] = original
        self.by_k["20"]["mean"]["score"] += .01
        self.publish()
        with self.assertRaisesRegex(ValueError, "summary"):
            self.load()
