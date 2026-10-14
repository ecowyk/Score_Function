"""Pause ownership, queue interlocks, fixed-noise configuration and real CPU EMA loading."""

import copy
import fcntl
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_g_sigma_1014", ROOT / "scripts/evaluate_g_sigma_1014.py"
)
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


def row(pid, ppid, argv):
    return dict(pid=pid, signal_pid=pid, ppid=ppid, argv=argv, start=str(pid), state="S")


@pytest.fixture
def pause_case(tmp_path, monkeypatch):
    queue = tmp_path / "training"
    jobs = [
        dict(
            id=name,
            config=str(queue / "configs" / (name + ".json")),
            output=str(queue / "runs" / name),
            training_budget=dict(updates_per_epoch=462),
        )
        for name in (*h.MODELS, "S025-G")
    ]
    h.suite.write(queue / "plan.json", dict(jobs=jobs))
    score = Path(jobs[-1]["output"]) / "score"
    h.suite.write(score / "status.json", dict(state="running", step=31450))
    (score / "last.pt").write_bytes(b"existing full checkpoint")
    h.suite.write(
        queue / "queue_status.json", dict(state="running", jobs={"S025-G": dict(state="running")})
    )
    coordinator = row(
        100,
        1,
        ["python", "/code/scripts/queue_g_sigma_trials.py", "run", "--queue-dir", str(queue)],
    )
    launcher = row(
        200, 100, ["python", "-m", "torch.distributed.run", "--config", jobs[-1]["config"]]
    )
    worker = row(
        201, 200, ["python", "-m", "score_function", "train", "--config", jobs[-1]["config"]]
    )
    loader = row(202, 201, ["python", "dataloader"])
    unrelated = row(
        300, 1, ["python", "-m", "torch.distributed.run", "--config", "/unrelated.json"]
    )
    tables = [
        {r["pid"]: r for r in (coordinator, launcher, worker, loader, unrelated)},
        {300: unrelated},
    ]
    monkeypatch.setattr(h, "process_table", lambda: tables.pop(0))
    monkeypatch.setattr(h.training, "verify_plan", lambda plan: None)
    monkeypatch.setattr(
        h.training, "inspect_checkpoint", lambda *a, **k: dict(epoch=68, step=31416, finished=False)
    )
    monkeypatch.setattr(h.processes, "wait_gone", lambda *a: [])
    signals = []
    monkeypatch.setattr(h, "signal_owned", lambda p, s: signals.append((p["pid"], s)))
    return SimpleNamespace(training=queue, wait_save_seconds=0), jobs, score, signals, tables


def test_pause_stops_only_owned_scheduler_and_torchrun_preserving_checkpoint(pause_case):
    args, jobs, score, signals, _ = pause_case
    before = (score / "last.pt").read_bytes()
    h.pause_training(args)
    assert signals == [(100, signal.SIGTERM), (200, signal.SIGTERM)]
    assert (score / "last.pt").read_bytes() == before
    assert h.suite.read(score / "status.json")["state"] == "paused"
    receipt = h.suite.read(args.training / "pause_record.json")
    assert receipt["step"] == 31416 and receipt["checkpoint_sha256"] == h.suite.digest(
        score / "last.pt"
    )


def test_pause_rejects_a_different_active_job_without_signalling(pause_case):
    args, jobs, _, signals, tables = pause_case
    tables[0][200]["argv"][-1] = jobs[0]["config"]
    with pytest.raises(RuntimeError, match="different training job"):
        h.pause_training(args)
    assert not signals


def test_posix_fallback_rechecks_identity_before_signal(monkeypatch):
    target = row(99, 1, ["python"])
    monkeypatch.delattr(h.os, "pidfd_open", raising=False)
    monkeypatch.setattr(h.processes, "same_alive", lambda p: True)
    monkeypatch.setattr(h.processes, "proc", lambda pid: dict(target, start="reused"))
    sent = []
    monkeypatch.setattr(h.os, "kill", lambda *args: sent.append(args))
    h.signal_owned(target, signal.SIGTERM)
    assert not sent
    monkeypatch.setattr(h.processes, "proc", lambda pid: target)
    h.signal_owned(target, signal.SIGTERM)
    assert sent == [(99, signal.SIGTERM)]


@pytest.fixture
def eval_case(tmp_path, monkeypatch):
    old, out, queue = tmp_path / "old", tmp_path / "eval", tmp_path / "training"
    (old / "code/lg").mkdir(parents=True)
    marker = tmp_path / "launches.jsonl"
    stub = tmp_path / "stub.py"
    stub.write_text(
        "import sys,json,os\n"
        + f"with open({str(marker)!r},'a') as f: f.write(json.dumps(dict(args=sys.argv[1:],cuda=os.environ['CUDA_VISIBLE_DEVICES']))+'\\n')\nprint('Results: /fake/result')\n"
    )
    wrapper = old / "code/lg/sim_score_planner.sh"
    wrapper.write_text(f'exec "{sys.executable}" "{stub}" "$@"\n')
    h.suite.write(old / "models/S05-G/config.json", dict(reference_seed=123))
    (old / "models/S05-G/final_ema.pt").write_bytes(b"reference")
    plan = dict(
        root=str(tmp_path),
        tokens=[f"{i:016x}" for i in range(272)],
        gamma=0.1,
        heading_projection=True,
        reference_seed=123,
        jobs=[],
        files={str(wrapper): h.suite.digest(wrapper)},
    )
    h.suite.write(old / "plan.json", plan)
    for job in h.job_list([20])[:2]:
        h.suite.write(
            old / "jobs" / (job["id"] + ".json"),
            dict(job, state="complete", result_dir="/fake/old"),
        )
    jobs = []
    for name in h.MODELS:
        score = queue / "runs" / name / "score"
        h.suite.write(score / "config.json", dict(reference_seed=123))
        (score / "final_ema.pt").write_bytes(name.encode())
        jobs.append(dict(id=name, output=str(score.parent)))
    h.suite.write(queue / "plan.json", dict(jobs=jobs))
    monkeypatch.setattr(h.training, "verify_plan", lambda plan: None)
    monkeypatch.setattr(h.training, "inspect_checkpoint", lambda *a, **k: dict(finished=True))
    monkeypatch.setattr(h.training, "gpu_busy", lambda *a: [])
    monkeypatch.setattr(h.suite, "validate_result", lambda *a: {k: 0.75 for k in h.suite.METRICS})
    real_run = subprocess.run
    monkeypatch.setattr(
        h.subprocess,
        "run",
        lambda cmd, **kw: (
            subprocess.CompletedProcess(cmd, 0)
            if cmd[1:3] == ["-c", h.INSPECT]
            else real_run(cmd, **kw)
        ),
    )
    return SimpleNamespace(source=old, output=out, training=queue, ks=[20]), marker


def test_preparation_preserves_reference_and_runner_launches_only_two_new_jobs(eval_case):
    args, marker = eval_case
    before = (args.source / "plan.json").read_bytes()
    h.prepare(args)
    p = h.suite.read(args.output / "plan.json")
    assert p["reference_seed"] == 123 and p["gamma"] == 0.1 and len(p["tokens"]) == 272
    assert (args.source / "plan.json").read_bytes() == before
    assert [j["id"] for j in p["jobs"]] == ["DP_K0", "S05-G_K20", "S05-G_plus100_K20", "S10-G_K20"]
    h.run(args)
    h.run(args)
    calls = [json.loads(x) for x in marker.read_text().splitlines()]
    assert len(calls) == 2
    for call in calls:
        assert call["cuda"] == "0,1,2,3,4,5,6,7"
        for key, value in (
            ("--split", "test14-hard"),
            ("--steps", "20"),
            ("--gamma", "0.1"),
            ("--gpu-fraction", "0.15"),
            ("--threads", "128"),
        ):
            assert call["args"][call["args"].index(key) + 1] == value
    assert h.suite.read(args.output / "status.json")["state"] == "complete"


def test_running_training_lock_blocks_evaluation(eval_case):
    args, marker = eval_case
    h.prepare(args)
    with (args.training / "queue.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Queue still owns"):
            h.run(args)
    assert not marker.exists()


def test_redirected_code_link_is_rejected(eval_case, tmp_path):
    args, _ = eval_case
    h.prepare(args)
    (args.output / "code").unlink()
    (args.output / "code").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="redirected"):
        h.run(args)


@pytest.fixture
def cpu_models(tmp_path):
    import test_training_pipeline as pipeline

    from score_function.model.score_branch import build_model
    from score_function.utils.planner_utils import planner_identity
    from score_function.utils.train_utils import source_hashes

    torch.set_num_threads(1)
    cfg, _ = pipeline.fixture(tmp_path / "fixture")
    cfg["model"]["temporal_attention"] = True
    model = build_model(cfg, "cpu")
    out, queue = tmp_path / "eval", tmp_path / "training"
    template = dict(
        schema_version=1,
        method=cfg["method"],
        config=cfg,
        planner=planner_identity(cfg),
        ego_normalizer=dict(mean=[0.0] * 4, std=[1.0] * 4),
        weight_kind="ema",
        sigma_score=0.05,
        epoch=529,
        step=244398,
        score_branch=model.state_dict(),
    )
    for name, epoch, step, sigma in (
        ("S05-G", 529, 244398, 0.05),
        ("S05-G_plus100", 629, 290598, 0.05),
        ("S10-G", 529, 244398, 0.1),
    ):
        config = copy.deepcopy(cfg)
        config["training"]["sigma"] = sigma
        state = dict(template, config=config, epoch=epoch, step=step, sigma_score=sigma)
        h.suite.write(out / "models" / name / "config.json", config)
        torch.save(state, out / "models" / name / "final_ema.pt")
        if name != "S05-G":
            score = queue / "runs" / name / "score"
            score.mkdir(parents=True)
            torch.save(
                dict(
                    state,
                    ema_branch=model.state_dict(),
                    finished=True,
                    termination="epoch_budget_exhausted",
                    cursor=0,
                    sources=source_hashes(),
                ),
                score / "last.pt",
            )
    (queue / "reference/score").mkdir(parents=True)
    torch.save(dict(ema_branch=model.state_dict()), queue / "reference/score/last.pt")
    return dict(
        out=str(out),
        training=str(queue),
        code=str(ROOT),
        reference_seed=cfg["reference_seed"],
        gamma=cfg["refinement"]["gamma"],
        heading_projection=cfg["refinement"]["heading_projection"],
    )


def test_cpu_preflight_loads_final_629_and_sigma_010_models(cpu_models):
    result = subprocess.run(
        [sys.executable, "-c", h.INSPECT, json.dumps(cpu_models)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "sigma=0.1" in result.stdout and "epoch=629" in result.stdout


def test_cpu_preflight_rejects_changed_noise_seed(cpu_models):
    path = Path(cpu_models["out"]) / "models/S10-G/config.json"
    cfg = h.suite.read(path)
    cfg["reference_seed"] += 1
    h.suite.write(path, cfg)
    result = subprocess.run(
        [sys.executable, "-c", h.INSPECT, json.dumps(cpu_models)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "AssertionError" in result.stderr


def test_prepare_resolves_hydra_environment_and_restores_parent(eval_case, monkeypatch):
    from omegaconf import OmegaConf

    args, _ = eval_case
    monkeypatch.delenv("NUPLAN_DATA_ROOT", raising=False)
    monkeypatch.setenv("NUPLAN_MAPS_ROOT", "/stale/maps")
    seen = []

    def validate(*unused):
        cfg = OmegaConf.create(
            {
                "data_root": "${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/test",
                "maps": "${oc.env:NUPLAN_MAPS_ROOT}",
                "exp": "${oc.env:NUPLAN_EXP_ROOT}",
            }
        )
        root = args.source.parent / "dataset/nuplan"
        assert cfg.data_root == str(root / "nuplan-v1.1/test")
        assert cfg.maps == str(root / "maps")
        assert cfg.exp == str(root / "exp")
        seen.append(True)
        return {}

    monkeypatch.setattr(h.suite, "validate_result", validate)
    h.prepare(args)
    assert len(seen) == 2
    assert "NUPLAN_DATA_ROOT" not in h.os.environ
    assert h.os.environ["NUPLAN_MAPS_ROOT"] == "/stale/maps"


def test_prepare_retries_partial_without_recopying_models(eval_case, monkeypatch):
    args, _ = eval_case
    validate = h.suite.validate_result
    monkeypatch.setattr(
        h.suite, "validate_result", lambda *a: (_ for _ in ()).throw(ValueError("interrupted"))
    )
    with pytest.raises(ValueError, match="interrupted"):
        h.prepare(args)
    assert not (args.output / "plan.json").exists()
    model = args.output / "models/S10-G/final_ema.pt"
    before = model.stat()
    monkeypatch.setattr(h.suite, "validate_result", validate)
    h.prepare(args)
    assert (model.stat().st_ino, model.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert (args.output / "plan.json").exists()
    h.prepare(args)


def test_partial_preparation_rejects_changed_models(eval_case):
    args, _ = eval_case
    h.suite.write(args.output / "models/S10-G/config.json", {"changed": True})
    with pytest.raises(ValueError, match="differs from source"):
        h.prepare(args)
    assert h.suite.read(args.output / "models/S10-G/config.json") == {"changed": True}
