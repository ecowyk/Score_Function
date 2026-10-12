#!/usr/bin/env python3
"""Freeze three fixed-sigma G trials from the completed run, without starting GPUs."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/home/data/wyk")
ORIGINAL_COMMIT = "d3735d2036aafc7d4667150599f09404af2bee07"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sources(code):
    package = Path(code) / "score_function"
    return {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))}


def clean_env(code, root):
    env = dict(os.environ)
    for key in list(env):
        if key in {
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "RAY_ADDRESS",
        } or key.startswith("TORCHELASTIC_"):
            env.pop(key)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join(
            map(str, (code, root / "Diffusion-Planner", root / "nuplan-devkit"))
        ),
    )
    return env


def restore_original_sources(destination, expected, candidates, git_repositories):
    """Recover exact original bytes; never relabel modified code as the old code."""
    origins = {}
    for name, wanted in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise ValueError(f"Invalid source provenance path: {name}")
        content = None
        for candidate in candidates:
            path = candidate / "score_function" / relative
            if path.is_file() and digest(path) == wanted:
                content, origin = path.read_bytes(), str(path)
                break
        if content is None:
            for repository in git_repositories:
                for revision in ("HEAD", ORIGINAL_COMMIT):
                    result = subprocess.run(
                        ["git", "-C", str(repository), "show", f"{revision}:score_function/{name}"],
                        capture_output=True,
                    )
                    if (
                        result.returncode == 0
                        and hashlib.sha256(result.stdout).hexdigest() == wanted
                    ):
                        content, origin = (
                            result.stdout,
                            f"{repository}@{revision}:score_function/{name}",
                        )
                        break
                if content is not None:
                    break
        if content is None:
            raise ValueError(
                f"Cannot recover original training source {name}; expected SHA256={wanted}"
            )
        target = destination / "score_function" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        origins[name] = origin
    if sources(destination) != expected:
        raise ValueError("Recovered training sources do not match the completed G run")
    return origins


INSPECT = r"""
import json, sys, torch
from pathlib import Path
from score_function.utils.config import load_config
from score_function.utils.training_budget import training_budget
from score_function.model.score_branch import build_model
torch.set_num_threads(2)
c=load_config(sys.argv[1])
if c!=json.loads(Path(sys.argv[1]).read_text()):
    raise ValueError("Prepared configuration does not round-trip unchanged")
m=build_model(c, 'cpu').train()
y=m(torch.zeros(2,80,4),torch.zeros(2,107,192),torch.zeros(2,192))
assert y.shape==(2,80,4) and torch.isfinite(y).all()
y.square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
print(json.dumps({'parameters':sum(p.numel() for p in m.parameters()),
                  'budget':training_budget(c['training'],int(sys.argv[2]))}))
"""


def prepare(args):
    root = args.root.expanduser().resolve()
    queue = (args.queue_dir or root / "outputs/score_aligned_1004/G_sigma_1012_v1").resolve()
    source_run = (
        args.source_run or root / "outputs/score_aligned_1004/after_L_G_DiT_v1/S05-G"
    ).resolve()
    source_code = (args.source_code or root / "Score_Function_aligned_v1").resolve()
    new_code = Path(__file__).resolve().parents[1]
    if queue.exists():
        raise FileExistsError(
            f"Queue already exists: {queue}. Use queue_g_sigma_trials.py run/status for a prepared queue."
        )
    if queue == source_run or source_run in queue.parents or queue in source_run.parents:
        raise ValueError("Use a separate queue outside the source training run")
    score = source_run / "score"
    original, info, provenance, status = (
        read(score / n)
        for n in ("config.json", "model_info.json", "provenance.json", "status.json")
    )
    if status.get("state") != "complete":
        raise ValueError("The source G run must be complete")
    model, cfg = original["model"], original["training"]
    if (
        model.get("architecture") != "temporal_score_function"
        or model.get("parameterization") != "score"
        or not model.get("temporal_attention")
        or model.get("neighbor_future", False)
        or model.get("sigma_conditioning", False)
        or cfg["sigma"] != 0.05
        or cfg.get("early_stopping", True)
        or cfg.get("lr_schedule") != "constant_after_warmup"
        or not cfg.get("data_augmentation", {}).get("enabled")
        or Path(original["output"]).resolve() != source_run
    ):
        raise ValueError("Expected the actual completed aligned fixed-sigma S05-G run")
    world = int(info["world_size"])
    budget = info["training_budget"]
    if (
        world != 4
        or info["global_batch"] != 2048
        or cfg["batch_size"] != 2048
        or info["microbatch_per_rank"] != 256
        or cfg["microbatch_size"] != 256
        or info["gradient_accumulation"] != 2
        or budget["planned_epochs"] != 529
        or budget["planned_optimizer_updates"] != 244398
        or info["train_samples"] != 947912
        or status.get("epoch") != 529
        or status.get("step") != 244398
    ):
        raise ValueError("Source metadata differs from the recorded 4-GPU/529-epoch G protocol")
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    if len(gpu_ids) != world or len(set(gpu_ids)) != world or any(x < 0 for x in gpu_ids):
        raise ValueError(
            "Provide exactly four different physical GPU IDs, matching source world size"
        )
    queue.mkdir(parents=True, exist_ok=False)
    try:
        reference = queue / "reference/score"
        reference.mkdir(parents=True)
        reference_files = {}
        for name in (
            "last.pt",
            "best.pt",
            "config.json",
            "model_info.json",
            "provenance.json",
            "status.json",
            "selection.json",
            "history.json",
            "initial_validation.json",
        ):
            path = score / name
            if not path.is_file():
                raise FileNotFoundError(path)
            before = digest(path)
            shutil.copy2(path, reference / name)
            if digest(path) != before or digest(reference / name) != before:
                raise ValueError(f"Source changed while preserving it: {path}")
            reference_files[name] = before
        fixed_code = queue / "code/fixed"
        candidates = [source_code]
        architecture_plan = source_run.parent / "plan.json"
        if architecture_plan.exists():
            for job in read(architecture_plan).get("jobs", []):
                if job.get("name", job.get("id")) == "S05-G" and job.get("code"):
                    candidates.append(Path(job["code"]))
        origins = restore_original_sources(
            fixed_code, provenance["sources"], candidates, [source_code, new_code]
        )
        jobs = []
        settings = (
            ("S05-G_plus100", 0.05, True, fixed_code),
            ("S10-G", 0.10, False, fixed_code),
            ("S025-G", 0.025, False, fixed_code),
        )
        for name, sigma, resume, code in settings:
            config = copy.deepcopy(original)
            run = queue / "runs" / name
            config["output"] = config["paths"]["run_dir"] = str(run)
            config["training"]["sigma"] = sigma
            if resume:
                config["training"]["max_epochs"] = budget["planned_epochs"] + 100
            restored = copy.deepcopy(config)
            restored["output"] = original["output"]
            restored["paths"]["run_dir"] = original["paths"]["run_dir"]
            restored["training"]["sigma"] = cfg["sigma"]
            restored["training"]["max_epochs"] = cfg["max_epochs"]
            if restored != original:
                raise AssertionError(f"Unintended configuration change in {name}")
            config_path = queue / "configs" / f"{name}.json"
            write(config_path, config)
            check = subprocess.run(
                [sys.executable, "-c", INSPECT, str(config_path), str(info["train_samples"])],
                cwd=code,
                env=clean_env(code, root),
                capture_output=True,
                text=True,
            )
            if check.returncode:
                raise RuntimeError(
                    f"CPU configuration/model preflight failed for {name}:\n{check.stderr}"
                )
            inspection = json.loads(check.stdout)
            target_budget = inspection["budget"]
            expected_epoch = 629 if resume else 529
            if target_budget["planned_epochs"] != expected_epoch:
                raise ValueError(f"Unexpected budget for {name}: {target_budget}")
            if resume:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("fork_g_continuation.py")),
                        "--source-score",
                        str(reference),
                        "--config",
                        str(config_path),
                        "--output",
                        str(run / "score"),
                        "--epochs-extra",
                        "100",
                    ],
                    cwd=code,
                    env=clean_env(code, root),
                    check=True,
                )
            jobs.append(
                dict(
                    id=name,
                    code=str(code),
                    config=str(config_path),
                    output=str(run),
                    resume=resume,
                    world_size=world,
                    gpus=gpu_ids,
                    target_epochs=expected_epoch,
                    target_updates=target_budget["planned_optimizer_updates"],
                    sources=sources(code),
                    parameters=inspection["parameters"],
                    training_budget=target_budget,
                )
            )
            print(
                f"Prepared {name}: {expected_epoch} epochs; {target_budget['planned_optimizer_updates']} updates; GPUs={gpu_ids}",
                flush=True,
            )
        files = {}
        for path in [
            *reference.glob("*"),
            *queue.glob("configs/*.json"),
            *queue.glob("code/*/score_function/**/*.py"),
        ]:
            if path.is_file():
                files[str(path)] = digest(path)
        plan = dict(
            schema_version=1,
            root=str(root),
            queue_dir=str(queue),
            source_run=str(source_run),
            source_reference=reference_files,
            source_recovery=origins,
            protocol="S05-G continuation +100; fresh fixed sigma 0.10, then 0.025",
            execution="sequential, source four-rank/global-batch/microbatch/accumulation retained",
            jobs=jobs,
            files=files,
            preserved_settings="All original scientific/data/runtime settings except explicitly recorded sigma, budget and output changes",
        )
        write(queue / "plan.json", plan)
        (queue / "logs").mkdir()
        (queue / "queue.log").touch()
        print(
            f"READY: {queue}\nOriginal G unchanged. Start queue_g_sigma_trials.py run in tmux.",
            flush=True,
        )
        return plan
    except Exception as exc:
        write(queue / "prepare_failed.json", {"error": repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-code", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
