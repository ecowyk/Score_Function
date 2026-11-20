"""Integration between the checkpoint matrix launcher, source reuse, resume and tabular report."""
from contextlib import nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import eval_chainflow_dd_checkpoints as queue


class MatrixLauncherTests(unittest.TestCase):
    def test_only_reuse_same_scored_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            base = {"workers": 1, "score_checkpoint_sha256": "ref", "dd_source": "pinned"}
            arm = dict(mean={"score": .89}, mean_delta={"score": .01}, improved=1, worsened=0, tied=0)
            summary = dict(protocol=base, count=1, valid_count=1, failed_count=0, by_k={"20": arm})
            write = queue.pipeline.atomic_json
            write(path / "request.json", base)
            write(path / "summary.json", summary)
            write(path / "launcher.json", dict(status="complete", summary_sha256=queue.pipeline.sha256(path / "summary.json")))
            shard = queue.legacy.shard_path(path, 0, 1)
            shard.mkdir()
            (shard / "scenes.jsonl").write_text("one record\n")
            write(shard / "complete.json", {"scenes_sha256": queue.pipeline.sha256(shard / "scenes.jsonl")})
            with patch.object(queue.legacy, "TOTAL", 1), redirect_stdout(io.StringIO()):
                self.assertEqual(queue.prior_result(path, {**base, "workers": 2})["by_k20"], arm)
                self.assertIsNone(queue.prior_result(path, {**base, "score_checkpoint_sha256": "other"}))
                (shard / "scenes.jsonl").write_text("changed\n")
                self.assertIsNone(queue.prior_result(path, base))

    def test_report_uses_actual_epoch_and_common_k0(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            key = "lr/final"
            meta = dict(selection="last_available", epoch=39, val_dsm=.081)
            arm = dict(mean={"score": .89}, mean_delta={"score": .01}, improved=1, worsened=0, tied=0)
            queue.pipeline.atomic_json(path / "request.json", {"checkpoints": {key: meta}})
            queue.pipeline.atomic_json(path / "summary.json", {"valid_count": 1, "k0": .88, "by_checkpoint": {key: arm}})
            marker = {"status": "complete", "summary_sha256": queue.pipeline.sha256(path / "summary.json")}
            queue.pipeline.atomic_json(path / "launcher.json", marker)
            text = io.StringIO()
            with redirect_stdout(text):
                queue.report(path)
            for label in ("DD K0 88.0000", "89.0000", "+1.0000", "last_available", "39"):
                self.assertIn(label, text.getvalue())

    def test_resume_keeps_frozen_checkpoints_and_does_not_repeat_workers(self):
        from score_function.navsim_chainflow import dd_checkpoint_matrix, sweep_checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = queue.parse_args(["run", "--root", tmp, "--workers", "1"])
            args.devkit_root = args.gpu_lock_dir = root
            model = root / "model.pt"
            model.write_bytes(b"fixed weights")
            digest = queue.pipeline.sha256(model)
            meta = dict(path=str(model), sha256=digest, selection="final", epoch=40, val_dsm=.08)
            records = {"reference/final": meta}
            base = dict(score_checkpoint_sha256=digest, dd_source=str(root), workers=1)
            arm = dict(mean={"score": .89}, mean_delta={"score": .01}, improved=1, worsened=0, tied=0)
            def merge(output, request):
                summary = dict(valid_count=1, k0=.88, by_checkpoint={"reference/final": arm})
                queue.pipeline.atomic_json(output / "summary.json", summary)
                return summary
            with patch.object(queue.legacy, "load_plan", return_value={}), \
                    patch.object(queue.legacy, "TOTAL", 1), patch.object(queue.pipeline, "environment", return_value={}), \
                    patch.object(queue.legacy, "build_request", return_value=base), \
                    patch.object(sweep_checkpoints, "prepare_checkpoints", return_value=records) as select, \
                    patch.object(queue.legacy, "acquire_gpus", return_value=nullcontext(["GPU0"])), \
                    patch.object(queue.legacy, "gpu_environment", return_value={}), \
                    patch.object(queue, "run_workers") as workers, patch.object(dd_checkpoint_matrix, "merge", merge), \
                    patch.dict(os.environ), patch.object(sys, "path", sys.path.copy()), redirect_stdout(io.StringIO()):
                queue.run(args)
                queue.run(args)
                self.assertEqual(select.call_count, 1)
                self.assertEqual(workers.call_count, 1)


if __name__ == "__main__":
    unittest.main()
