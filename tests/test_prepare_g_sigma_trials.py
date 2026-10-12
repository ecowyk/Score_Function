"""Prepare real CPU-inspected fixed-sigma configs; only the large fork is mocked."""

import copy
import fcntl
import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from score_function.utils.config import load_config
from score_function.utils.training_budget import training_budget

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_g_sigma_trials", ROOT / "scripts/prepare_g_sigma_trials.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def experiment(tmp_path):
    source_code = tmp_path / "original_code"
    for path in (ROOT / "score_function").rglob("*.py"):
        target = source_code / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    source_run = tmp_path / "original_G"
    score = source_run / "score"
    score.mkdir(parents=True)
    original = load_config(ROOT / "configs/score_function.json", root=tmp_path)
    # The architecture is real; a smaller width keeps CPU preflight inexpensive.
    original["model"].update(temporal_attention=True, hidden_dim=24, num_heads=3)
    original["training"]["microbatch_size"] = 256
    original["output"] = original["paths"]["run_dir"] = str(source_run)
    budget = training_budget(original["training"], 947912)
    assert budget["planned_epochs"] == 529
    info = dict(
        world_size=4, global_batch=2048, microbatch_per_rank=256,
        gradient_accumulation=2, train_samples=947912, training_budget=budget,
    )
    status = dict(state="complete", epoch=529, step=244398)
    provenance = dict(sources=helper.sources(source_code))
    for name, contents in (
        ("config.json", original), ("model_info.json", info),
        ("provenance.json", provenance), ("status.json", status),
        ("selection.json", {}), ("history.json", []), ("initial_validation.json", {}),
    ):
        helper.write(score / name, contents)
    (score / "last.pt").write_bytes(b"Large checkpoint fork is separately tested")
    (score / "best.pt").write_bytes(b"Immutable inherited selection fixture")
    args = SimpleNamespace(
        root=tmp_path, queue_dir=tmp_path / "new_queue", source_run=source_run,
        source_code=source_code, gpus="0,1,2,3",
    )
    return args, original, provenance, score


@pytest.mark.parametrize("world", [4, 8])
def test_three_fixed_sigma_jobs_use_real_budgets_frozen_code_and_readonly_source(
    experiment, monkeypatch, world,
):
    args, original, provenance, score = experiment
    args.gpus = ",".join(map(str, range(world)))
    source_before = {path.name: helper.digest(path) for path in score.iterdir()}
    code_before = helper.sources(args.source_code)
    real_run = subprocess.run
    preflights, forks = [], []

    def run(command, **kwargs):
        if len(command) > 1 and Path(command[1]).name == "fork_g_continuation.py":
            forks.append((command, kwargs))
            output = Path(command[command.index("--output") + 1])
            helper.write(output / "status.json", dict(state="prepared", epoch=529, step=244398))
            return subprocess.CompletedProcess(command, 0)
        if len(command) > 2 and command[1:3] == ["-c", helper.INSPECT]:
            preflights.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(helper.subprocess, "run", run)
    plan = helper.prepare(args)
    assert len(preflights) == 3 and len(forks) == 1
    assert [job["id"] for job in plan["jobs"]] == ["S05-G_plus100", "S10-G", "S025-G"]
    assert [job["resume"] for job in plan["jobs"]] == [True, False, False]
    assert [job["target_epochs"] for job in plan["jobs"]] == [629, 529, 529]
    assert [job["target_updates"] for job in plan["jobs"]] == [290598, 244398, 244398]
    assert not (args.queue_dir / "code/multi").exists()
    assert helper.read(args.queue_dir / "plan.json") == plan
    for job, sigma, epochs in zip(plan["jobs"], [0.05, 0.10, 0.025], [629, 500, 500]):
        expected = copy.deepcopy(original)
        expected["output"] = expected["paths"]["run_dir"] = job["output"]
        expected["training"].update(sigma=sigma, max_epochs=epochs)
        config = helper.read(job["config"])
        assert config == expected
        assert job["training_budget"] == training_budget(config["training"], 947912)
        assert job["world_size"] == world and job["gpus"] == list(range(world))
        assert job["gradient_accumulation"] == 2048 // world // 256
        assert job["sources"] == provenance["sources"] == helper.sources(job["code"])
        assert job["parameters"] > 0
    for _, kwargs in preflights:
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(args.queue_dir / "code/fixed")
    fork_command, fork_kwargs = forks[0]
    assert fork_command[fork_command.index("--epochs-extra") + 1] == "100"
    assert fork_command[fork_command.index("--target-world-size") + 1] == str(world)
    assert fork_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ("0" if world == 8 else "")
    assert fork_command[fork_command.index("--source-score") + 1] == str(
        args.queue_dir / "reference/score"
    )
    assert fork_kwargs["cwd"] == args.queue_dir / "code/fixed"
    assert source_before == {path.name: helper.digest(path) for path in score.iterdir()}
    assert code_before == helper.sources(args.source_code)
    for name, digest in source_before.items():
        assert helper.digest(args.queue_dir / "reference/score" / name) == digest
    for name, digest in provenance["sources"].items():
        frozen = args.queue_dir / "code/fixed/score_function" / name
        assert plan["files"][str(frozen)] == helper.digest(frozen) == digest


def test_unrecoverable_source_hash_is_not_relabelled_or_silently_accepted(
    experiment, monkeypatch,
):
    args, _, provenance, score = experiment
    expected = dict(provenance["sources"])
    first = sorted(expected)[0]
    expected[first] = "0" * 64
    before = {path.name: helper.digest(path) for path in score.iterdir()}
    monkeypatch.setattr(
        helper.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 128, stdout=b""),
    )
    with pytest.raises(ValueError, match="Cannot recover original training source"):
        helper.restore_original_sources(
            args.queue_dir / "frozen", expected, [args.source_code], [args.source_code]
        )
    assert before == {path.name: helper.digest(path) for path in score.iterdir()}


def test_source_run_is_never_used_as_queue_output(experiment):
    args, _, _, score = experiment
    before = {path.name: helper.digest(path) for path in score.iterdir()}
    args.queue_dir = args.source_run / "nested_queue"
    with pytest.raises(ValueError, match="separate queue"):
        helper.prepare(args)
    assert not args.queue_dir.exists()
    assert before == {path.name: helper.digest(path) for path in score.iterdir()}


def test_active_previous_queue_is_rejected_before_new_output_creation(experiment):
    args, _, _, _ = experiment
    previous = args.root / "outputs/score_aligned_1004/G_sigma_1012_v1"
    previous.mkdir(parents=True)
    with (previous / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Previous four-GPU queue is still active"):
            helper.prepare(args)
    assert not args.queue_dir.exists()
