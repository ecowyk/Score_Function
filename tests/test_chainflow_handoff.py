"""Independent CSV/float64 NPZ handoff checks; no Torch, CUDA, or network."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from score_function.navsim_chainflow import handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.trajectories = self.run / "trajectories"
        self.trajectories.mkdir()
        self.tokens = ["ff94ab2c81a25745", "8486dcafaee75d76", "f512480c969a5eab"]
        self.digest = hashlib.sha256("\n".join(sorted(self.tokens)).encode()).hexdigest()
        self.index, self.values, self.records = {}, {}, []
        for i, token in enumerate(self.tokens):
            # Deliberately distinguish the native value from a float32 roundtrip.
            value = np.arange(24, dtype=np.float64).reshape(8, 3) / 7 + 0.123456789012345
            value = value + i
            self.values[token] = value
            path = self.trajectories / f"{token}.npz"
            np.savez(path, trajectory=value)
            self.index[token] = {"npz": str(path), "shape": [8, 3], "sha256": handoff._sha256(path)}
            metrics = {name: 0.8 + i * 0.05 for name in handoff.METRICS}
            self.records.append({"token": token, "log_name": f"2021.06.28.18.03.27_veh-14_{i}",
                                 "valid": "True", **metrics})
        checkpoint, backbone = self.root / "checkpoint", self.root / "backbone"
        checkpoint.write_bytes(b"fixture pinned checkpoint")
        backbone.write_bytes(b"fixture pinned backbone")
        self.patch_checkpoint = mock.patch.object(handoff, "CHECKPOINT_SHA256", handoff._sha256(checkpoint))
        self.patch_backbone = mock.patch.object(handoff, "BACKBONE_SHA256", handoff._sha256(backbone))
        self.patch_checkpoint.start()
        self.patch_backbone.start()
        self.addCleanup(self.patch_checkpoint.stop)
        self.addCleanup(self.patch_backbone.stop)
        self.addCleanup(self.temp.cleanup)
        config = self.run / "code/hydra/config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("diffusion_eval_noise_seed: 27\n")
        vlm, cache = self.root / "vlm", self.root / "metric_cache"
        vlm.mkdir()
        cache.mkdir()
        self.manifest = {
            "schema": handoff.SCHEMA, "state": "complete", "upstream_repo": handoff.UPSTREAM_REPO,
            "upstream_commit": handoff.UPSTREAM_COMMIT, "training_split": "navtrain",
            "evaluation_split": "navtest", "count": 3, "valid_count": 3, "failed_count": 0,
            "score_scale": "0..1", "mean_pdms": 0.85, "mean_pdms_x100": 85.0,
            "subscores": {name: 0.85 for name in handoff.METRICS},
            "checkpoint": str(checkpoint), "checkpoint_sha256": handoff.CHECKPOINT_SHA256,
            "checkpoint_hf_revision": handoff.CHECKPOINT_REVISION,
            "backbone": str(backbone), "backbone_sha256": handoff.BACKBONE_SHA256,
            "vlm_cache": str(vlm), "vlm_cache_archive_sha256": handoff.VLM_ARCHIVE_SHA256,
            "resolved_config": str(config), "resolved_config_sha256": handoff._sha256(config),
            "native_noise_seed": 27, "native_sampling": "ddim_12step_64proposal",
            "metric_cache": str(cache), "metrics": str(self.run / "metrics_per_token.csv"),
            "summary_csv": str(self.run / "original.csv"), "trajectories": str(self.trajectories),
            "trajectory_shape": [8, 3], "trajectory_columns": ["x", "y", "heading"],
            "trajectory_frame": "ego_local", "trajectory_interval_s": 0.5,
            "trajectory_horizon_s": 4.0, "token_index": str(self.trajectories / "token_index.json"),
        }
        self.write_csv("metrics", self.records)
        self.write_csv("summary_csv", self.records + [{"token": "average", "log_name": "",
            "valid": "True", **{name: 0.85 for name in handoff.METRICS}}])
        self.write_index()
        self.write_manifest()

    def write_manifest(self):
        (self.run / "reproduction_ready.json").write_text(json.dumps(self.manifest))

    def write_index(self):
        path = Path(self.manifest["token_index"])
        path.write_text(json.dumps(self.index))
        self.manifest["token_index_sha256"] = handoff._sha256(path)

    def write_csv(self, name, records):
        path = Path(self.manifest[name])
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["token", "log_name", "valid", *handoff.METRICS])
            writer.writeheader()
            writer.writerows(records)
        self.manifest[name + "_sha256"] = handoff._sha256(path)

    def load(self, **kwargs):
        return handoff.load_handoff(self.run, expected_count=3, expected_token_digest=self.digest, **kwargs)

    def replace_npz(self, token, value):
        path = Path(self.index[token]["npz"])
        np.savez(path, trajectory=value)
        self.index[token]["sha256"] = handoff._sha256(path)
        self.write_index()
        self.write_manifest()

    def test_real_format_preserves_float64_k0_and_does_not_modify_inputs(self):
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        loaded = self.load()
        self.assertTrue(handoff.is_handoff(self.run))
        self.assertTrue(handoff.is_handoff(self.run / "reproduction_ready.json"))
        self.assertEqual(loaded["schema"], handoff.ADAPTER_SCHEMA)
        self.assertEqual(loaded["summary"]["count"], 3)
        self.assertAlmostEqual(loaded["summary"]["by_k"]["0"]["mean"]["score"], 0.85)
        for token, expected in self.values.items():
            row = loaded["rows"][token]
            self.assertEqual(row["native_origin"], "external-handoff")
            self.assertEqual(row["trajectory_dtype"], "float64")
            self.assertEqual(row["trajectory_sha256"], self.index[token]["sha256"])
            actual = np.asarray(row["trajectories"]["0"], dtype=np.float64)
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(np.array_equal(actual, expected.astype(np.float32).astype(np.float64)))
            for absent in ("condition_sha256", "metric_cache_sha256", "native_ms", "conditioner_ms", "refine_ms"):
                self.assertNotIn(absent, row)
        self.assertTrue(loaded["provenance"]["payloads_verified"])
        self.assertIsNone(loaded["provenance"]["native_condition_hashes"])
        self.assertIsNone(loaded["provenance"]["native_metric_cache_hashes"])
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_checksum_corruption_is_rejected(self):
        for field in ("metrics", "summary_csv", "resolved_config", "token_index", "checkpoint", "backbone"):
            with self.subTest(field=field):
                path = Path(self.manifest[field])
                content = path.read_bytes()
                path.write_bytes(content + b"corrupt")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    self.load()
                path.write_bytes(content)
        path = Path(self.index[self.tokens[0]]["npz"])
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch: trajectory"):
            self.load()

    def test_wrong_official_identity_and_seed_rejected(self):
        for key, value in (("native_noise_seed", 28), ("native_sampling", "ddim_12step_128proposal"),
                           ("checkpoint_hf_revision", "a" * 40), ("upstream_commit", "a" * 40),
                           ("checkpoint_sha256", "a" * 64), ("trajectory_frame", "global")):
            with self.subTest(key=key):
                previous = self.manifest[key]
                self.manifest[key] = value
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    self.load()
                self.manifest[key] = previous
        self.write_manifest()

    def test_canonical_membership_required(self):
        with self.assertRaisesRegex(ValueError, "canonical navtest"):
            handoff.load_handoff(self.run, expected_count=3)

    def test_duplicate_csv_token_rejected(self):
        self.write_csv("metrics", self.records + [self.records[0]])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate native token"):
            self.load()

    def test_duplicate_index_json_keys_rejected(self):
        path = Path(self.manifest["token_index"])
        token = self.tokens[0]
        entry = json.dumps(self.index[token])
        path.write_text('{"' + token + '":' + entry + ',"' + token + '":' + entry + '}')
        self.manifest["token_index_sha256"] = handoff._sha256(path)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.load()

    def test_invalid_scene_metric_and_counts_rejected(self):
        for key, value in (("valid", "False"), ("score", "nan"), ("comfort", 1.1)):
            with self.subTest(key=key):
                record = {**self.records[0], key: value}
                self.write_csv("metrics", [record] + self.records[1:])
                self.write_manifest()
                with self.assertRaises(ValueError):
                    self.load()
        self.write_csv("metrics", self.records)
        self.manifest["failed_count"] = False
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.load()

    def test_invalid_npz_dtype_shape_and_nonfinite_rejected(self):
        token = self.tokens[0]
        for values in (np.ones((8, 3), dtype=np.int64), np.ones((8, 4)), np.full((8, 3), np.inf)):
            with self.subTest(dtype=values.dtype, shape=values.shape):
                self.replace_npz(token, values)
                with self.assertRaisesRegex(ValueError, "shape/dtype/values"):
                    self.load()

    def test_filename_token_swap_is_rejected(self):
        self.index[self.tokens[0]]["npz"] = self.index[self.tokens[1]]["npz"]
        self.write_index()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "token/directory"):
            self.load()

    def test_original_csv_disagreement_rejected(self):
        self.write_csv("summary_csv", [{**self.records[0], "score": 0.1}] + self.records[1:])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "original summary CSV"):
            self.load()

    def test_declared_mean_disagreement_rejected(self):
        self.manifest["mean_pdms"] = 0.84
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "mean_pdms"):
            self.load()

    def test_display_only_load_is_explicitly_unaudited(self):
        Path(self.manifest["checkpoint"]).write_bytes(b"changed checkpoint")
        loaded = self.load(verify_payloads=False)
        self.assertFalse(loaded["provenance"]["payloads_verified"])
        self.assertNotIn(self.manifest["checkpoint"], loaded["provenance"]["files_sha256"])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch: checkpoint"):
            self.load()


if __name__ == "__main__":
    unittest.main()
