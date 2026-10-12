"""Queue resume/ownership and real parquet pairing checks; no GPU required."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("g_followup_1012", SCRIPTS / "g_followup_1012.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
suite = helper.suite


@pytest.fixture
def experiment(tmp_path):
    old, out = tmp_path / "old", tmp_path / "new"
    code = old / "code/lg"
    code.mkdir(parents=True)
    model = old / "models/S05-G"
    model.mkdir(parents=True)
    config = dict(reference_seed=736214, refinement=dict(gamma=0.1, heading_projection=True))
    suite.write(model / "config.json", config)
    (model / "final_ema.pt").write_bytes(b"fake-weights-for-launcher-tests")
    marker = tmp_path / "launched.jsonl"
    # The stub exercises the real queue's command/environment, without nuPlan.
    stub = tmp_path / "stub.py"
    stub.write_text(
        "import json, os, sys\n"
        f"with open({str(marker)!r}, 'a') as f:\n"
        " f.write(json.dumps({'args':sys.argv[1:], 'cuda':os.environ['CUDA_VISIBLE_DEVICES']})+'\\n')\n"
        "print('Results: /fake/result', flush=True)\n"
    )
    wrapper = code / "sim_score_planner.sh"
    wrapper.write_text(f'exec "{sys.executable}" "{stub}" "$@"\n')
    tokens = [f"{i:016x}" for i in range(272)]
    plan = dict(
        root=str(tmp_path),
        tokens=tokens,
        gamma=0.1,
        heading_projection=True,
        reference_seed=config["reference_seed"],
        jobs=suite.jobs(),
        files={
            str(p): suite.digest(p)
            for p in (model / "config.json", model / "final_ema.pt", wrapper)
        },
    )
    suite.write(old / "plan.json", plan)
    suite.write(old / "status.json", dict(state="interrupted"))
    return SimpleNamespace(source=old, output=out, marker=marker, plan=plan)


def test_prepare_preserves_seed_model_old_plan_and_rejects_changed_inputs(experiment):
    args = experiment
    original = (args.source / "plan.json").read_bytes()
    helper.prepare(args)
    helper.prepare(args)
    plan = suite.read(args.output / "plan.json")
    assert plan["reference_seed"] == 736214
    assert [j["K"] for j in plan["jobs"]] == [15, 30, 150, 250]
    assert all(j["model"] == "S05-G" for j in plan["jobs"])
    assert (args.output / "models").resolve() == args.source / "models"
    assert (args.source / "plan.json").read_bytes() == original
    (args.source / "models/S05-G/final_ema.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        helper.prepare(args)


def test_resume_rejects_redirected_code(experiment, tmp_path):
    helper.prepare(experiment)
    link = experiment.output / "code"
    link.unlink()
    link.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Unexpected existing path"):
        helper.prepare(experiment)


def test_real_runner_launches_only_four_then_skips_verified_results(experiment, monkeypatch):
    monkeypatch.setattr(suite, "jobs", suite.jobs)
    monkeypatch.setattr(helper, "analyze", lambda args: None)
    monkeypatch.setattr(suite, "validate_result", lambda *args: {k: 0.7 for k in suite.METRICS})
    helper.run(experiment)
    helper.run(experiment)
    launches = [json.loads(line) for line in experiment.marker.read_text().splitlines()]
    assert len(launches) == 4
    assert [int(row["args"][row["args"].index("--steps") + 1]) for row in launches] == [
        15,
        30,
        150,
        250,
    ]
    for row in launches:
        args = row["args"]
        assert row["cuda"] == "0,1,2,3,4,5,6,7"
        assert args[args.index("--split") + 1] == "test14-hard"
        assert args[args.index("--gpu-fraction") + 1] == "0.15"
        assert args[args.index("--threads") + 1] == "128"
        assert args[args.index("--gamma") + 1] == "0.1"
    assert len(pd.read_csv(experiment.output / "summary.csv")) == 4
    assert suite.read(experiment.output / "status.json")["state"] == "complete"


def test_old_lock_blocks_new_launch(experiment):
    with helper.exclusive(experiment.source / ".run.lock"):
        with pytest.raises(RuntimeError, match="Queue still owns"):
            helper.run(experiment)
    assert not experiment.marker.exists()


def make_result(args, k):
    from omegaconf import OmegaConf

    name = "DP_K0" if k == 0 else f"S05-G_K{k}"
    run = args.source / suite.CHALLENGE / name
    (run / "code/hydra").mkdir(parents=True)
    (run / "aggregator_metric").mkdir()
    config, checkpoint = suite.model_paths(args.source, "S05-G")
    cfg = dict(
        planner=dict(
            score_function_planner=dict(
                score_config=str(config),
                score_checkpoint=str(checkpoint),
                steps=k,
                gamma=0.1,
                heading_projection=None,
                trace_dir=None,
                device="cuda",
            )
        ),
        scenario_filter=dict(
            scenario_tokens=args.plan["tokens"],
            limit_total_scenarios=None,
            num_scenarios_per_type=None,
        ),
        scenario_builder=dict(
            data_root=str(Path(args.plan["root"]) / "dataset/nuplan/nuplan-v1.1/test")
        ),
        worker={"_target_": "nuplan.RayDistributed"},
        runner_report_file="runner.parquet",
    )
    OmegaConf.save(OmegaConf.create(cfg), run / "code/hydra/config.yaml")
    rows = [
        dict(
            log_name=f"recording_{i // 4}",
            scenario=token,
            scenario_type="straight" if i % 2 else "turn",
            **{m: 0.75 for m in suite.METRICS},
        )
        for i, token in enumerate(args.plan["tokens"])
    ]
    if k:
        rows[0]["score"] = 0.95
        rows[1]["score"] = 0.65
        rows[1]["no_ego_at_fault_collisions"] = 0.0
    # Deliberately different from plain scenario mean: do not conflate the two.
    rows.append(
        dict(
            log_name=None,
            scenario="final_score",
            scenario_type=None,
            num_scenarios=272,
            **{m: 0.76 if k else 0.75 for m in suite.METRICS},
        )
    )
    pd.DataFrame(rows).sample(frac=1, random_state=k).to_parquet(
        run / "aggregator_metric/scores.parquet"
    )
    pd.DataFrame(
        [
            dict(log_name=r["log_name"], scenario_name=r["scenario"], succeeded=True)
            for r in rows[:-1]
        ]
    ).to_parquet(run / "runner.parquet")
    (run / "result.nuboard").touch()
    suite.write(
        args.source / "jobs" / (name + ".json"),
        dict(
            id=name,
            model="S05-G",
            K=k,
            state="reused" if k in (20, 50, 100) else "complete",
            result_dir=str(run),
        ),
    )
    return run


def test_real_parquet_analysis_pairs_keys_not_row_order(experiment):
    for k in (0, 5, 10, 20, 50, 100, 200):
        make_result(experiment, k)
    helper.analyze(experiment)
    report = experiment.output / "analysis"
    summary = pd.read_csv(report / "paired_summary.csv")
    row = summary.loc[summary.comparison.eq("DP_K0 -> S05-G_K20")].iloc[0]
    assert (row.improved, row.tied, row.worsened) == (1, 270, 1)
    assert row.official_delta_points == pytest.approx(1.0)
    assert row.scene_mean_delta_points == pytest.approx(100 * 0.1 / 272)
    top = pd.read_csv(report / "top_changed.csv")
    assert top.after_nuboard.str.endswith("result.nuboard").all()
    metrics = pd.read_csv(report / "metric_changes.csv")
    collision = metrics.loc[
        metrics.comparison.eq("DP_K0 -> S05-G_K20")
        & metrics.metric.eq("no_ego_at_fault_collisions")
    ].iloc[0]
    assert collision.worsened == 1


def test_pairing_rejects_wrong_recording_and_duplicate_keys(experiment):
    run = make_result(experiment, 0)
    frame = pd.read_parquet(run / "aggregator_metric/scores.parquet")
    detail = helper.scenario_detail(frame, experiment.plan["tokens"])
    other = detail.copy()
    other.loc[other.index[0], "log_name"] = "wrong_recording"
    with pytest.raises(ValueError, match="different log/token"):
        helper.paired(detail, other)
    with pytest.raises(ValueError):
        helper.paired(detail, pd.concat([detail, detail.iloc[:1]]))
    detail.loc[detail.index[0], "score"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        helper.scenario_detail(detail, experiment.plan["tokens"])


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfds required")
def test_pause_stops_only_captured_old_tree(experiment, tmp_path, monkeypatch):
    old_script = tmp_path / "fake_old_suite.py"
    ready = tmp_path / "ready.json"
    status = experiment.source / "status.json"
    old_script.write_text(
        "import fcntl, json, pathlib, signal, subprocess, sys, time\n"
        f"lock=open({str(experiment.source / '.run.lock')!r},'a'); fcntl.flock(lock,fcntl.LOCK_EX)\n"
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)\n"
        "def stop(*args):\n"
        f" pathlib.Path({str(status)!r}).write_text(json.dumps({{'state':'interrupted'}}))\n"
        " sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid':child.pid}}))\n"
        "time.sleep(120)\n"
    )
    monkeypatch.setattr(suite, "__file__", str(old_script))
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    coordinator = subprocess.Popen(
        [sys.executable, str(old_script), "run", "--output", str(experiment.source)]
    )
    child = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child_pid = json.loads(ready.read_text())["pid"]
        identities = [helper.proc(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()]
        child = next(p for p in identities if p and p["signal_pid"] == child_pid)
        helper.pause_old(experiment)
        assert coordinator.wait(timeout=5) == 0
        assert not helper.same_alive(child)
        assert unrelated.poll() is None
    finally:
        for process in (coordinator, unrelated):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
        if child and helper.same_alive(child):
            helper.signal_owned(child, signal.SIGKILL)
