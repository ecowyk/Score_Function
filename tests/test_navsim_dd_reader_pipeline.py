"""The full experiment must evaluate only after a successful durable training stage."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "reader_pipeline", Path(__file__).resolve().parents[1] / "scripts/run_navsim_dd_reader.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class FinishedProcess:
    pid = 123456

    def __init__(self, returncode=0):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class ReaderPipelineTests(unittest.TestCase):
    def test_full_protocol_and_epoch_resume_commands(self):
        with TemporaryDirectory() as temporary:
            p = pipeline.paths(Path(temporary) / "root with spaces")
            stages = dict(pipeline.commands(p))
            self.assertEqual(list(stages), ["cache", "preflight", "train", "evaluate"])
            self.assertIn("--nproc_per_node=8", stages["train"])
            self.assertNotIn("--resume", stages["train"])
            self.assertIn(str(p["source"]), stages["cache"])
            self.assertIn(str(p["evaluation"]), stages["evaluate"])
            p["train"].mkdir(parents=True)
            (p["train"] / "run.json").write_text("{}")
            self.assertIn("--resume", dict(pipeline.commands(p))["train"])
            cfg = pipeline.read_json(pipeline.CONFIG)
            self.assertEqual(cfg["epochs"], 500)
            self.assertEqual(cfg.get("lr_schedule_epochs", cfg["epochs"]), 500)
            self.assertEqual(cfg["checkpoint_every_epochs"], 50)
            self.assertEqual(pipeline.KS, [5, 10, 15, 20, 30])

    def test_success_runs_all_stages_with_separate_logs(self):
        with TemporaryDirectory() as temporary:
            p = pipeline.paths(temporary)
            calls = []
            def start(command, **kwargs):
                calls.append(command)
                self.assertTrue(kwargs["start_new_session"])
                return FinishedProcess()
            with patch.object(pipeline, "check_dependencies"), patch.object(pipeline.subprocess, "Popen", start):
                pipeline.run(p)
            self.assertEqual(len(calls), 4)
            state = pipeline.read_json(p["jobs"] / "state.json")
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["ks"], pipeline.KS)
            for stage in ("cache", "preflight", "train", "evaluate"):
                self.assertIn(stage, (p["jobs"] / f"{stage}.log").read_text())

    def test_failed_training_never_launches_evaluation(self):
        with TemporaryDirectory() as temporary:
            p = pipeline.paths(temporary)
            calls = []
            def start(command, **kwargs):
                calls.append(command)
                return FinishedProcess(7 if len(calls) == 3 else 0)
            with patch.object(pipeline, "check_dependencies"), patch.object(pipeline.subprocess, "Popen", start), \
                    patch.object(pipeline.os, "killpg"), self.assertRaisesRegex(RuntimeError, "train exited 7"):
                pipeline.run(p)
            self.assertEqual(len(calls), 3)
            self.assertFalse((p["jobs"] / "evaluate.log").exists())
            state = pipeline.read_json(p["jobs"] / "state.json")
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stage"], "train")


if __name__ == "__main__":
    unittest.main()
