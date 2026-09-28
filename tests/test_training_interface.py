"""Launch routing and reporting must preserve the scientific training result."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import test_training_pipeline as pipeline

from score_function.train import train
from score_function.utils.progress import Progress
from score_function.utils.train_utils import load_tensor


class TrainingInterfaceTests(unittest.TestCase):
    def test_bars_and_plain_logs_produce_identical_training(self):
        for parameterization in ("score", "energy"):
            with (
                self.subTest(parameterization=parameterization),
                tempfile.TemporaryDirectory() as tmp,
            ):
                config, _ = pipeline.fixture(tmp)
                config["model"]["parameterization"] = parameterization
                checkpoints = []
                for mode in ("off", "on"):
                    config["output"] = str(Path(tmp) / mode)
                    output, terminal = io.StringIO(), io.StringIO()
                    with patch.dict(os.environ, {"SCORE_FUNCTION_PROGRESS": mode}):
                        with redirect_stdout(output), redirect_stderr(terminal):
                            train(config, device_override="cpu")
                    directory = Path(config["output"]) / "score"
                    checkpoints.append(load_tensor(directory / "last.pt"))
                    self.assertIn("Training complete", output.getvalue())
                    log = (directory / "console.log").read_text()
                    self.assertIn("Initial validation", log)
                    self.assertIn("Training complete", log)
                    self.assertNotIn("\r", log)
                    self.assertEqual(
                        json.loads((directory / "status.json").read_text())["state"], "complete"
                    )
                    if mode == "on":
                        self.assertIn("100%", terminal.getvalue())
                    else:
                        self.assertEqual(terminal.getvalue(), "")
                pipeline.PipelineTests._assert_state_equal(self, *checkpoints)

    def test_only_rank_zero_displays_progress(self):
        output, terminal = io.StringIO(), io.StringIO()
        with patch("score_function.utils.progress.ddp.rank", return_value=1):
            with redirect_stdout(output), redirect_stderr(terminal):
                with Progress("worker", total=1) as display:
                    display(1)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(terminal.getvalue(), "")

    def test_launcher_preserves_arguments_gpu_mask_and_failure_exit_code(self):
        launcher = Path(__file__).resolve().parents[1] / "scripts/train.sh"
        with tempfile.TemporaryDirectory(prefix="training launcher ") as tmp:
            capture = Path(tmp) / "arguments.json"
            interpreter = Path(tmp) / "fake python"
            interpreter.write_text(
                f"#!{sys.executable}\n"
                "import json, os, pathlib, sys\n"
                "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({"
                "'args': sys.argv[1:], 'gpus': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
                "sys.exit(17)\n"
            )
            interpreter.chmod(0o755)
            arguments = [
                "--config",
                "config with spaces.json",
                "--root",
                tmp,
                "--set",
                'paths.run_dir="outputs/a b"',
                "--resume",
                "--progress",
                "on",
            ]
            for mask in ("7", "1,3"):
                with self.subTest(mask=mask):
                    result = subprocess.run(
                        [
                            "bash",
                            str(launcher),
                            "--python",
                            str(interpreter),
                            "--gpus",
                            mask,
                            *arguments,
                        ],
                        env={**os.environ, "CAPTURE": str(capture)},
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 17, result.stderr)
                    captured = json.loads(capture.read_text())
                    self.assertEqual(captured["gpus"], mask)
                    self.assertEqual(captured["args"][-len(arguments) :], arguments)
                    self.assertEqual("torch.distributed.run" in captured["args"], mask == "1,3")
                    if mask == "1,3":
                        self.assertIn("--nproc_per_node=2", captured["args"])


if __name__ == "__main__":
    unittest.main()
