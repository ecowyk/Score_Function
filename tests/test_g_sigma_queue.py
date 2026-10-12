"""Queue safety and final-EMA export, with CPU checkpoints and mocked GPU queries."""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from score_function.utils.config import load_config
from score_function.utils.train_utils import training_signature

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g_sigma_queue", REPO / "scripts/queue_g_sigma_trials.py"
)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        code = root / "code"
        shutil.copytree(
            REPO / "score_function",
            code / "score_function",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        config = json.loads((REPO / "configs/score_function.json").read_text())
        config["paths"]["root"] = str(root)
        config["paths"]["run_dir"] = str(root / "run")
        config["output"] = str(root / "run")
        config_path = root / "config.json"
        queue.write(config_path, config)
        config = load_config(config_path)
        score = root / "run/score"
        score.mkdir(parents=True)
        sources = queue.sources(code)
        signature = training_signature(config, "test_cache_sha256", 4)
        budget = {"planned_epochs": 2, "planned_optimizer_updates": 4}
        state = dict(
            schema_version=1,
            method=config["method"],
            config=config,
            planner={},
            ego_normalizer={"mean": [0] * 4, "std": [1] * 4},
            sigma_score=0.05,
            epoch=2,
            step=4,
            cursor=0,
            ranks=[{}] * 4,
            finished=True,
            termination="epoch_budget_exhausted",
            training_budget=budget,
            signature=signature,
            sources=sources,
            score_branch={"weight": torch.tensor([1.0])},
            ema_branch={"weight": torch.tensor([2.0])},
        )
        torch.save(state, score / "last.pt")
        (score / "best.pt").write_bytes(b"do not change validation-selected best")
        queue.write(
            score / "status.json",
            dict(
                state="complete",
                epoch=2,
                step=4,
                termination="epoch_budget_exhausted",
                training_budget=budget,
            ),
        )
        queue.write(
            score / "provenance.json",
            dict(sources=sources, signature=signature, cache_sha256="test_cache_sha256"),
        )
        queue.write(
            score / "model_info.json",
            dict(
                world_size=4,
                global_batch=config["training"]["batch_size"],
                microbatch_per_rank=config["training"]["microbatch_size"],
            ),
        )
        job = dict(
            id="unit",
            code=str(code),
            config=str(config_path),
            output=str(root / "run"),
            resume=False,
            world_size=4,
            gpus=[0, 1, 2, 3],
            target_epochs=2,
            target_updates=4,
            sources=sources,
        )
        files = {
            str(config_path): queue.digest(config_path),
            **{str(code / "score_function" / name): sha for name, sha in sources.items()},
        }
        return job, dict(jobs=[job], files=files), state

    def test_three_fixed_sigma_jobs_are_valid_without_multi_sigma(self):
        with tempfile.TemporaryDirectory() as directory:
            original, plan, _ = self.fixture(directory)
            plan["jobs"] = []
            for name, sigma, epochs in (
                ("S05-G_plus100", 0.05, 629),
                ("S10-G", 0.10, 529),
                ("S025-G", 0.025, 529),
            ):
                config = queue.read(original["config"])
                output = str(Path(directory) / name)
                config["output"] = config["paths"]["run_dir"] = output
                config["training"]["sigma"] = sigma
                path = Path(directory) / (name + ".json")
                queue.write(path, config)
                plan["files"][str(path)] = queue.digest(path)
                plan["jobs"].append(
                    {
                        **original,
                        "id": name,
                        "config": str(path),
                        "output": output,
                        "target_epochs": epochs,
                        "target_updates": epochs * 462,
                        "resume": name == "S05-G_plus100",
                    }
                )
            queue.verify_plan(plan)
            self.assertEqual(
                [job["id"] for job in plan["jobs"]], ["S05-G_plus100", "S10-G", "S025-G"]
            )

    def test_final_export_requires_checkpoint_and_preserves_best(self):
        with tempfile.TemporaryDirectory() as directory:
            job, plan, state = self.fixture(directory)
            queue.verify_plan(plan)
            result = queue.inspect_checkpoint(job, "plan_hash", final=True)
            final = torch.load(result["final_ema"], weights_only=False)
            self.assertTrue(torch.equal(final["score_branch"]["weight"], torch.tensor([2.0])))
            self.assertEqual(final["weight_kind"], "ema")
            self.assertEqual(
                (Path(job["output"]) / "score/best.pt").read_bytes(),
                b"do not change validation-selected best",
            )
            # Idempotent recheck verifies exported tensors instead of overwriting best/final.
            self.assertEqual(result, queue.inspect_checkpoint(job, "plan_hash", final=True))
            state["step"] = 3
            torch.save(state, Path(job["output"]) / "score/last.pt")
            with self.assertRaisesRegex(RuntimeError, "exact target"):
                queue.inspect_checkpoint(job, "plan_hash", final=True)

    def test_resume_preflight_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            job, plan, state = self.fixture(directory)
            result = queue.inspect_checkpoint(job, "plan_hash")
            self.assertEqual(result["step"], 4)
            command = queue.command_for(job, True)
            self.assertEqual(command[0], sys.executable)
            self.assertIn("--nproc_per_node=4", command)
            self.assertEqual(command[-1], "--resume")
            self.assertNotIn("--resume", queue.command_for(job, False))
            Path(job["config"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                queue.verify_plan(plan)

    def test_gpu_uuid_matching_and_hidden_process_fallback(self):
        inventory = "0, GPU-a, 10, 0\n1, GPU-b, 6000, 80"
        with patch.object(
            queue, "capture", side_effect=[inventory, "GPU-b, 25, 6000\nGPU-a, 99, 1"]
        ):
            busy = queue.gpu_busy([0])
            self.assertEqual(busy[0]["pid"], "99")
            self.assertEqual(len(busy), 1)
        with patch.object(queue, "capture", side_effect=[inventory, OSError("hidden processes")]):
            self.assertEqual(queue.gpu_busy([1])[0]["gpu_index"], 1)
        with patch.object(queue, "capture", side_effect=[inventory, "GPU-b, 25, 6000"]):
            self.assertEqual(queue.gpu_busy([0]), [])

    def test_failed_job_stops_queue_and_existing_last_automatically_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            job, plan, _ = self.fixture(directory)
            root = Path(directory)
            queue.write(root / "plan.json", plan)
            queue.write(Path(job["output"]) / "score/status.json", {"state": "failed"})
            args = SimpleNamespace(
                queue_dir=root, poll_seconds=1, gpu_memory_mib=256, gpu_util_percent=5
            )
            child = SimpleNamespace(pid=123, returncode=7, poll=lambda: 7)
            with (
                patch.object(queue, "gpu_busy", return_value=[]),
                patch.object(queue, "inspect_checkpoint", return_value={"step": 2, "epoch": 1}),
                patch.object(queue.subprocess, "Popen", return_value=child) as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "code 7"):
                    queue.run(args)
            self.assertIn("--resume", launch.call_args.args[0])
            self.assertTrue(launch.call_args.kwargs["pass_fds"])
            saved = queue.read(root / "queue_status.json")
            self.assertEqual(saved["state"], "failed")
            self.assertEqual(saved["jobs"]["unit"]["exit_code"], 7)
            self.assertTrue((root / "logs/unit.launcher.log").is_file())
            self.assertTrue((Path(job["output"]) / "score/last.pt").is_file())

    def test_status_displays_target_and_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            job, _, _ = self.fixture(directory)
            text = queue.progress_line(
                job,
                {"state": "running", "base_step": 0, "started_at": 100, "log": "queue.log"},
                now=110,
            )
            self.assertIn("step=4/4", text)
            self.assertIn("epoch=2/2", text)
            self.assertIn("ETA=0.00h", text)
            self.assertIn("log=queue.log", text)


if __name__ == "__main__":
    unittest.main()
