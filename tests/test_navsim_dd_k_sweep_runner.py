"""Protect checkpoint/source selection and single-variable K-sweep commands."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_navsim_dd_k_sweep as runner


class KSweepRunnerTests(unittest.TestCase):
    def test_worker_changes_only_ks_and_reuses_reference(self):
        root, output, checkpoint, reference = map(Path, ("/r", "/out", "/e500/final_raw.pt", "/ref500"))
        command = runner.worker_command(root, output, checkpoint, reference, 3, [100, 5, 10, 50, 5])
        self.assertEqual(command[command.index("--ks") + 1:command.index("--gamma")], ["5", "10", "50", "100"])
        for flag, value in (("--score-checkpoint", str(checkpoint)), ("--reuse-k0-from", str(reference)),
                            ("--gamma", "0.1"), ("--seed", "20260922"), ("--device", "cuda:3")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--verify-native", command)
        for ks in ([], [0], [20], [-1], [True], [1.5]):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                runner.checked_ks(ks)

    def test_invalid_reference_cannot_publish_pointer_or_spawn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exp = root / "dataset/navsim/exp"
            exp.mkdir(parents=True)
            original_pointer = exp / runner.base.latest_pointer(500)
            original_pointer.write_text("/original500\n")
            with (patch.object(runner.base, "preflight", return_value=(Path("/final_raw.pt"), Path("/original500"))) as check,
                  patch.object(runner, "validate_reference", side_effect=ValueError("checkpoint mismatch")),
                  patch.object(runner.subprocess, "Popen") as spawn):
                with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                    runner.run(root)
                check.assert_called_once_with(root, epochs=500, baseline_pointer=runner.base.latest_pointer(500))
                spawn.assert_not_called()
            self.assertEqual(original_pointer.read_text(), "/original500\n")
            self.assertFalse((exp / runner.LATEST).exists())
            self.assertFalse(list(exp.glob("dd_score_ddaligned_e500_k_sweep_*")))

    def test_status_uses_sweep_pointer_and_never_launches_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exp = root / "dataset/navsim/exp"
            output = exp / "sweep"
            output.mkdir(parents=True)
            (exp / runner.LATEST).write_text(str(output) + "\n")
            (output / "launcher.json").write_text(json.dumps({"status": "running", "computed_ks": [5, 10, 50, 100]}))
            part = output / "part-0000-of-0008"
            part.mkdir()
            (part / "scenes.jsonl.tmp").write_text('{}\n{}\n{"partial":')
            stream = io.StringIO()
            with patch.object(runner.subprocess, "Popen") as spawn, contextlib.redirect_stdout(stream):
                runner.status(root)
            spawn.assert_not_called()
            self.assertIn("Scenes 2/12146", stream.getvalue())
            self.assertIn("computed K=[5, 10, 50, 100]", stream.getvalue())
            self.assertIn("reused K0/K20", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
