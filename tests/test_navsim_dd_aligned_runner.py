"""CPU checks for progress, checkpoint selection, and evaluation supervision."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "navsim_dd_aligned_runner", REPOSITORY / "scripts/eval_navsim_dd_aligned.py"
)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def checkpoint_metadata():
    config = json.loads((REPOSITORY / "configs/navsim_dd_s10_g_ddaligned.json").read_text())
    return {
        "format": "navsim-dd-score-v1", "epoch": 100, "weight_kind": "raw",
        "sigma": 0.1, "gamma": 0.1, "training_precision": "16-mixed",
        "model": config["model"],
    }


class AlignedEvaluationRunnerTests(unittest.TestCase):
    def test_checkpoint_guard_rejects_earlier_epoch_and_ema_export(self):
        valid = checkpoint_metadata()
        runner.validate_checkpoint(valid)
        for change in ({"epoch": 21}, {"weight_kind": "ema"}, {"weight_kind": None}):
            with self.subTest(change=change):
                with self.assertRaises((ValueError, RuntimeError)):
                    runner.validate_checkpoint({**valid, **change})

    def test_progress_counts_one_file_and_only_complete_newline_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(runner.progress_snapshot(output), [0] * 8)
            part = output / "part-0000-of-0008"
            part.mkdir()
            records = ''.join(json.dumps({"token": token}) + "\n" for token in ("a", "b"))
            (part / "scenes.jsonl.tmp").write_text(records + '{"token": "in-progress', encoding="utf-8")
            # A stale final file must not be added to the active temporary file.
            (part / "scenes.jsonl").write_text(records, encoding="utf-8")
            self.assertEqual(runner.progress_snapshot(output), [2] + [0] * 7)
            (part / "scenes.jsonl.tmp").unlink()
            self.assertEqual(runner.progress_snapshot(output), [2] + [0] * 7)
            second = output / "part-0001-of-0008"
            second.mkdir()
            (second / "scenes.jsonl").write_text(json.dumps({"token": "c"}) + "\n", encoding="utf-8")
            self.assertEqual(runner.progress_snapshot(output), [2, 1] + [0] * 6)

    def test_summary_displays_scores_and_delta_on_the_same_hundred_point_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_json(output / "summary.json", {
                "count": 193, "score_scale": "0..1",
                "by_k": {
                    "0": {"mean": {"score": 0.880337}, "mean_delta": {"score": 0.0},
                          "improved": 0, "worsened": 0, "tied": 193},
                    "20": {"mean": {"score": 0.900337}, "mean_delta": {"score": 0.02},
                           "improved": 20, "worsened": 10, "tied": 163},
                },
            })
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                runner.print_summary(output)
            shown = stream.getvalue()
            self.assertIn("193", shown)
            self.assertIn("88.0337", shown)
            self.assertIn("90.0337", shown)
            self.assertRegex(shown, r"\+2\.0+")

    def test_status_reads_running_progress_without_starting_any_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp = root / "dataset/navsim/exp"
            training, output = exp / "dd_score_ddaligned_v1", exp / "evaluation"
            write_json(training / "history.json", [{"epoch": 100}])
            (training / "final_raw.pt").touch()
            write_json(output / "launcher.json", {"status": "running"})
            (exp / runner.LATEST).write_text(str(output) + "\n", encoding="utf-8")
            part = output / "part-0000-of-0008"
            part.mkdir()
            (part / "scenes.jsonl.tmp").write_text('{"token":"done"}\n{"token":', encoding="utf-8")
            stream = io.StringIO()
            with patch.object(runner.subprocess, "Popen") as spawn, contextlib.redirect_stdout(stream):
                runner.status(root)
            spawn.assert_not_called()
            shown = stream.getvalue()
            self.assertIn("100/100", shown)
            self.assertIn("running", shown)
            self.assertIn("Scenes 1/12146", shown)
            self.assertNotIn("new K20", shown)

    def test_worker_command_reuses_existing_k0_and_requests_only_new_k20(self):
        root = Path("/fixture/project")
        output = root / "output"
        checkpoint = root / "final_raw.pt"
        baseline = root / "completed-e200"
        command = runner.worker_command(root, output, checkpoint, baseline, 3)
        self.assertIn("score_function.navsim_dd.evaluate", command)
        self.assertEqual(command[command.index("--reuse-k0-from") + 1], str(baseline))
        self.assertEqual(command[command.index("--score-checkpoint") + 1], str(checkpoint))
        index = command.index("--ks") + 1
        steps = []
        while index < len(command) and not command[index].startswith("--"):
            steps.append(command[index])
            index += 1
        self.assertEqual(steps, ["20"])
        self.assertEqual(command[command.index("--shard-index") + 1], "3")
        self.assertEqual(command[command.index("--num-shards") + 1], "8")
        self.assertNotIn("--native-only", command)
        self.assertIn("--verify-native", command)

    def test_failed_worker_cleans_up_owned_children_without_merge_or_external_kill(self):
        # Real CPU children exercise process ownership and cleanup without CUDA.
        actual_popen, actual_sleep = subprocess.Popen, time.sleep
        outsider = actual_popen([sys.executable, "-c", "import time; time.sleep(60)"])
        owned = []

        def worker_command(_root, _output, _checkpoint, _baseline, shard):
            code = (
                "import sys,time; time.sleep(0.2); sys.exit(7)" if shard == 0 else
                "import time; time.sleep(60)"
            )
            return [sys.executable, "-c", code]

        def spawn(command, *args, **kwargs):
            self.assertNotIn("--merge", command, "A failed shard must prevent merging")
            child = actual_popen(command, *args, **kwargs)
            owned.append(child)
            return child

        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkpoint, baseline = root / "final_raw.pt", root / "baseline"
                baseline.mkdir()
                with (
                    patch.object(runner, "preflight", return_value=(checkpoint, baseline)),
                    patch.object(runner, "worker_command", side_effect=worker_command),
                    patch.object(runner.subprocess, "Popen", side_effect=spawn),
                    patch.object(runner.time, "sleep", side_effect=lambda duration: actual_sleep(min(duration, 0.01))),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                        runner.run(root)
                self.assertEqual(len(owned), 8)
                self.assertTrue(all(child.poll() is not None for child in owned))
                self.assertIsNone(outsider.poll(), "Unrelated processes must remain running")
                self.assertFalse(list(root.rglob("summary.json")))
                latest = root / "dataset/navsim/exp" / runner.LATEST
                output = Path(latest.read_text().strip())
                state = json.loads((output / "launcher.json").read_text())
                self.assertEqual(state["status"], "failed")
                self.assertIn("exit 7", state["error"])
        finally:
            for child in [*owned, outsider]:
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
