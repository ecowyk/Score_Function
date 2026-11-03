"""Portable cache tests; native ingestion is covered by the real devkit smoke command."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from score_function.navsim_dd.cache import (
    CONDITIONER_KEYS, NavsimScoreDataset, _publish_cache, load_cache_metadata,
)
from score_function.utils.dataset import collate_cpu


IDENTITY = {key: key + "-test" for key in CONDITIONER_KEYS}


def encoded(token):
    offset = {"a": 0., "b": 4., "c": 1000.}[token]
    poses = np.column_stack((np.arange(8) + offset, np.full(8, 1.), np.full(8, np.pi / 2)))
    return poses, torch.full((64, 256), 1.25), torch.full((256,), 2.5)


def chunks(fn=encoded):
    yield "train", "log1", ["a", "b"], fn
    yield "val", "log2", ["c"], fn


class NavsimDDCacheTest(unittest.TestCase):
    def test_train_statistics_and_existing_trainer_interface(self):
        with TemporaryDirectory() as tmp:
            _publish_cache(tmp, IDENTITY, chunks(), {"a", "b", "c"})
            metadata = load_cache_metadata(tmp)
            self.assertEqual(metadata["conditioner_identity"], IDENTITY)
            self.assertEqual(metadata["normalizer"]["mean"], [5.5, 1., 0., 0.])
            self.assertEqual(metadata["normalizer"]["std"][1:], [1., 1., 1.])
            train = NavsimScoreDataset(tmp, "train")
            batch = collate_cpu([train[0], train[1]])
            self.assertEqual(batch["tokens"], ["a", "b"])
            self.assertEqual(batch["target"].shape, (2, 8, 4))
            self.assertEqual(batch["context"].shape, (2, 64, 256))
            self.assertEqual(batch["route"].shape, (2, 256))
            self.assertEqual(batch["context"].dtype, torch.float32)
            self.assertAlmostEqual(float(batch["target"][..., 0].mean()), 0., places=6)
            torch.testing.assert_close(batch["target"][..., 3], torch.ones(2, 8))
            val = NavsimScoreDataset(tmp, "val")
            self.assertGreater(val[0]["target"][0, 0], 50.)
            shard = np.load(Path(tmp) / "train/000000/context.npy", mmap_mode="r")
            self.assertEqual(shard.dtype, np.float16)

    def test_resume_valid_shards_only_and_detect_tampering(self):
        with TemporaryDirectory() as tmp:
            def fail_at_validation(token):
                if token == "c":
                    raise RuntimeError("interrupted")
                return encoded(token)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                _publish_cache(tmp, IDENTITY, chunks(fail_at_validation))
            with self.assertRaises(FileNotFoundError):
                load_cache_metadata(tmp)
            calls = []
            def resumed(token):
                calls.append(token)
                return encoded(token)
            _publish_cache(tmp, IDENTITY, chunks(resumed))
            self.assertEqual(calls, ["c"])
            with self.assertRaisesRegex(ValueError, "identity"):
                _publish_cache(tmp, {**IDENTITY, "checkpoint_sha256": "different"}, chunks())
            path = Path(tmp) / "train/000000/context.npy"
            with path.open("r+b") as f:
                f.seek(-1, 2)
                byte = f.read(1)
                f.seek(-1, 2)
                f.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                _publish_cache(tmp, IDENTITY, chunks())

    def test_log_leakage_and_full_split_coverage_rejected(self):
        with TemporaryDirectory() as tmp:
            mixed = [("train", "same", ["a"], encoded), ("val", "same", ["b"], encoded)]
            with self.assertRaisesRegex(ValueError, "both splits"):
                _publish_cache(tmp, IDENTITY, mixed)
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                _publish_cache(tmp, IDENTITY, chunks(), {"a", "b", "c", "absent"})
            self.assertFalse((Path(tmp) / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
