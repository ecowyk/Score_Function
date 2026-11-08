"""Compact reader-cache provenance, recovery and trainer-interface tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from score_function.navsim_dd import cache as old_cache
from score_function.navsim_dd import reader_cache as reader
from score_function.utils.dataset import collate_cpu
from score_function.utils.train_utils import atomic_write, file_hash, read_json


IDENTITY = {key: f"test-{key}" for key in old_cache.CONDITIONER_KEYS}


def encode_old(token):
    offset = {"a": 0., "b": 4., "c": 1000.}[token]
    poses = np.column_stack((np.arange(8) + offset, np.ones(8), np.full(8, np.pi / 2)))
    return poses, torch.ones(64, 256), torch.full((256,), 2.5)


def source(root):
    batches = [("train", "log1", ["a", "b"], encode_old), ("val", "log2", ["c"], encode_old)]
    old_cache._publish_cache(root, IDENTITY, batches, {"a", "b", "c"})
    return reader.source_tasks(root)


def initialize(tmp):
    original, output = Path(tmp) / "old", Path(tmp) / "reader"
    original.mkdir()
    output.mkdir()
    metadata, tasks = source(original)
    (output / reader.ASSETS_FILE).write_bytes(b"test-frozen-assets")
    identity = reader.make_identity(original, metadata, file_hash(output / reader.ASSETS_FILE))
    atomic_write(output / "build.json", identity)
    return original, output, metadata, tasks, identity


def encode_reader(record, target):
    # No clean trajectory is stored in the feature tensor; this synthetic
    # encoder just establishes the callback's exact source-target contract.
    assert target.shape == (8, 4)
    return torch.full((512, 8, 8), 1.25), torch.full((256,), 2.5)


def complete(original, output, metadata, tasks, identity, encode=encode_reader):
    for task in tasks:
        reader.write_chunk(output, original, identity, task, encode)
    return reader.publish_cache(output, original, identity, tasks, metadata, {"test": True})


class ReaderCacheTest(unittest.TestCase):
    def test_exact_source_membership_targets_and_normalization(self):
        with TemporaryDirectory() as tmp:
            original, output, metadata, tasks, identity = initialize(tmp)
            source_sha = file_hash(original / "manifest.json")
            result = complete(original, output, metadata, tasks, identity)
            self.assertEqual(result["normalizer"], metadata["normalizer"])
            self.assertEqual(file_hash(original / "manifest.json"), source_sha)
            self.assertNotEqual(result["schema"], metadata["schema"])
            with self.assertRaisesRegex(ValueError, "schema"):
                old_cache.load_cache_metadata(output)
            training = reader.DDReaderDataset(output, "train")
            validation = reader.ReaderScoreDataset(output, "val")
            self.assertEqual([r["token"] for r in training.records], ["a", "b"])
            self.assertEqual([r["token"] for r in validation.records], ["c"])
            batch = collate_cpu([training[0], training[1]])
            self.assertEqual(batch["context"].shape, (2, 512, 8, 8))
            self.assertEqual(batch["route"].shape, (2, 256))
            self.assertEqual(batch["target"].shape, (2, 8, 4))
            self.assertEqual(batch["context"].dtype, torch.float32)
            old_training = old_cache.NavsimScoreDataset(original, "train")
            torch.testing.assert_close(training[0]["target"], old_training[0]["target"], rtol=0, atol=0)
            self.assertGreater(float(validation[0]["target"][0, 0]), 50.)
            for task in tasks:
                self.assertEqual((output / task["path"] / "target.npy").read_bytes(),
                                 (original / task["path"] / "target.npy").read_bytes())
                self.assertEqual(np.load(output / task["path"] / "context.npy").dtype, np.float16)
            ready = read_json(output / "TRAINING_READY.json")
            self.assertEqual(ready["records"], {"train": 2, "val": 1})
            self.assertEqual(ready["manifest_sha256"], file_hash(output / "manifest.json"))

    def test_interruption_resumes_only_completed_matching_shards(self):
        with TemporaryDirectory() as tmp:
            original, output, metadata, tasks, identity = initialize(tmp)
            def interrupted(record, target):
                if record["token"] == "c":
                    raise RuntimeError("interrupted native encoding")
                return encode_reader(record, target)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                complete(original, output, metadata, tasks, identity, interrupted)
            self.assertTrue((output / "train/000000/complete.json").is_file())
            self.assertFalse((output / "val/000000").exists())
            self.assertFalse((output / "manifest.json").exists())
            calls = []
            def resumed(record, target):
                calls.append(record["token"])
                return encode_reader(record, target)
            complete(original, output, metadata, tasks, identity, resumed)
            self.assertEqual(calls, ["c"])
            self.assertEqual(reader.load_reader_cache_metadata(output)["schema"], reader.SCHEMA)

    def test_corrupt_completed_latent_is_rejected_not_silently_reused(self):
        with TemporaryDirectory() as tmp:
            original, output, metadata, tasks, identity = initialize(tmp)
            complete(original, output, metadata, tasks, identity)
            path = output / tasks[0]["path"] / "context.npy"
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                reader.write_chunk(output, original, identity, tasks[0], encode_reader)

    def test_source_target_change_is_rejected_before_encoding(self):
        with TemporaryDirectory() as tmp:
            original, output, metadata, tasks, identity = initialize(tmp)
            path = original / tasks[0]["path"] / "target.npy"
            values = np.load(path)
            values[0, 0, 0] += 1
            np.save(path, values, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "Source targets changed"):
                reader.write_chunk(output, original, identity, tasks[0], encode_reader)
            with self.assertRaisesRegex(ValueError, "Source cache changed"):
                reader.source_tasks(original)

    def test_record_order_change_and_incomplete_coverage_rejected(self):
        with TemporaryDirectory() as tmp:
            original, output, metadata, tasks, identity = initialize(tmp)
            reader.write_chunk(output, original, identity, tasks[0], encode_reader)
            changed = {**tasks[0], "records": list(reversed(tasks[0]["records"]))}
            with self.assertRaisesRegex(ValueError, "provenance"):
                reader.write_chunk(output, original, identity, changed, encode_reader)
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                reader.publish_cache(output, original, identity, tasks[:1], metadata, {})
            self.assertFalse((output / "manifest.json").exists())

    def test_asset_normalizer_identity_changes_rejected(self):
        with TemporaryDirectory() as tmp:
            original, output, metadata, tasks, identity = initialize(tmp)
            complete(original, output, metadata, tasks, identity)
            changed = read_json(output / "manifest.json")
            changed["normalizer"]["mean"][0] += 1
            atomic_write(output / "manifest.json", changed)
            with self.assertRaisesRegex(ValueError, "provenance"):
                reader.load_cache_metadata(output)
            complete(original, output, metadata, tasks, identity)
            (output / reader.ASSETS_FILE).write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "assets"):
                reader.load_cache_metadata(output)


if __name__ == "__main__":
    unittest.main()
