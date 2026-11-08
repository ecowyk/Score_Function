"""Process selection and stopping safeguards for retargeting a local continuation."""

import contextlib
import fcntl
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "navsim_dd_continue_launcher", REPOSITORY / "scripts/continue_navsim_dd_aligned.py"
)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def parent_command(output):
    return ["/fixture/env/bin/python3.10", "-u", "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=8", "-m", "score_function.navsim_dd.train",
            "--output", str(output)]


class ContinueLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "dd_score_ddaligned_v1_e250"
        self.lock_path = self.output.with_name(self.output.name + ".launcher.lock")

    def test_matches_only_exact_torchrun_training_entrypoint_and_output(self):
        base = parent_command(self.output)
        self.assertTrue(runner.is_previous_trainer(base, self.output, self.root))
        relative = base[:-2] + ["--output=" + self.output.name]
        self.assertTrue(runner.is_previous_trainer(relative, self.output, self.root))
        cases = {
            "worker": [sys.executable, "-m", "score_function.navsim_dd.train", "--output", str(self.output)],
            "other_output": parent_command(self.root / "another-run"),
            "output_prefix": parent_command(Path(str(self.output) + "-unrelated")),
            "wrong_relative_base": base[:-1] + [self.output.name],
            "shell": ["bash", "-c", " ".join(base)],
            "module_as_other_program_argument": [
                *base[:7], "unrelated.training", "--label", "score_function.navsim_dd.train",
                "--output", str(self.output),
            ],
            "ambiguous_output": [*base, "--output", str(self.root / "another-run")],
        }
        for name, argv in cases.items():
            with self.subTest(name=name):
                cwd = self.root / "different-cwd" if name == "wrong_relative_base" else self.root
                self.assertFalse(runner.is_previous_trainer(argv, self.output, cwd))

    def test_process_inventory_filters_owner_and_ignores_worker_processes(self):
        def process(pid, uid, argv):
            class Process:
                name = str(pid)

                def stat(self):
                    return SimpleNamespace(st_uid=uid)

                def __truediv__(self, name):
                    if name == "cmdline":
                        return SimpleNamespace(read_bytes=lambda: b"\0".join(os.fsencode(a) for a in argv) + b"\0")
                    return SimpleNamespace(resolve=lambda: self_root)

            return Process()

        self_root = self.root
        fixture = [
            process(101, 1234, parent_command(self.output)),
            process(102, 5678, parent_command(self.output)),
            process(103, 1234, [sys.executable, "-m", "score_function.navsim_dd.train", "--output", str(self.output)]),
            process(100, 1234, parent_command(self.output)),
        ]
        actual_path = Path

        def path(value):
            return SimpleNamespace(iterdir=lambda: iter(fixture)) if str(value) == "/proc" else actual_path(value)

        with (patch.object(runner, "Path", side_effect=path),
              patch.object(runner.os, "getuid", return_value=1234),
              patch.object(runner.os, "getpid", return_value=100)):
            self.assertEqual(runner.previous_parents(self.output), [101])

    def test_check_never_signals_or_starts_training_when_previous_parent_is_active(self):
        exp = self.root / "dataset/navsim/exp"
        exp.mkdir(parents=True)
        with (patch.dict(os.environ),
              patch.object(runner, "check_environment") as environment,
              patch.object(runner, "previous_parents", return_value=[12345]),
              patch.object(runner, "terminate_previous") as terminate,
              patch.object(runner, "preflight") as preflight,
              patch.object(runner.os, "execvpe") as execute,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "old 250-epoch job is active"):
                runner.launch(self.root, check=True)
        environment.assert_called_once()
        terminate.assert_not_called()
        preflight.assert_not_called()
        execute.assert_not_called()

    def test_multiple_parents_refuse_to_signal_any_process(self):
        with (patch.object(runner, "previous_parents", return_value=[12345, 23456]),
              patch.object(runner, "terminate_previous") as terminate):
            with self.assertRaisesRegex(RuntimeError, "Multiple old torchrun parents"):
                with runner.previous_guard(self.output, replace=True):
                    self.fail("Ambiguous process selection must not enter the guarded section")
        terminate.assert_not_called()

    def test_guard_waits_for_old_lock_then_holds_it_through_protected_work(self):
        with self.lock_path.open("a") as old_lock:
            fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release_old(_pid, _output):
                fcntl.flock(old_lock, fcntl.LOCK_UN)

            with (patch.object(runner, "previous_parents", side_effect=[[12345], []]),
                  patch.object(runner, "terminate_previous", side_effect=release_old) as terminate,
                  patch.object(runner.time, "sleep")):
                with runner.previous_guard(self.output, replace=True):
                    with self.lock_path.open("a") as competitor:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                terminate.assert_called_once_with(12345, self.output)
            # The guard releases its own descriptor when its scope finishes.
            fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_locked_job_without_identifiable_local_parent_times_out_without_killing(self):
        with self.lock_path.open("a") as old_lock:
            fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (patch.object(runner, "previous_parents", return_value=[]),
                  patch.object(runner, "terminate_previous") as terminate,
                  patch.object(runner.time, "monotonic", side_effect=[0.0, 61.0])):
                with self.assertRaisesRegex(RuntimeError, "Old job/lock is still active"):
                    with runner.previous_guard(self.output, replace=True):
                        self.fail("An unreleased old lock must block a new launch")
            terminate.assert_not_called()

    @unittest.skipUnless(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "requires Linux pidfd")
    def test_pidfd_rechecks_selection_and_only_signals_the_selected_fixture_child(self):
        # These are isolated test children, never real trainers or a process-group kill.
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        actual_signal = signal.pidfd_send_signal
        try:
            with (patch.object(runner, "previous_parents", return_value=[]),
                  patch.object(runner.signal, "pidfd_send_signal", wraps=actual_signal) as send):
                runner.terminate_previous(victim.pid, self.output)
                send.assert_not_called()
                self.assertIsNone(victim.poll())
            with (patch.object(runner, "previous_parents", return_value=[victim.pid]),
                  patch.object(runner.signal, "pidfd_send_signal", wraps=actual_signal) as send,
                  contextlib.redirect_stdout(io.StringIO())):
                runner.terminate_previous(victim.pid, self.output)
                self.assertEqual(victim.wait(timeout=5), -signal.SIGTERM)
                send.assert_called_once()
                self.assertEqual(send.call_args.args[1], signal.SIGTERM)
            self.assertIsNone(outsider.poll(), "The unrelated fixture process must remain alive")
        finally:
            for child in (victim, outsider):
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
