#!/usr/bin/env python3
"""Pause S025-G and evaluate the two finished G trials with the frozen Test14-Hard runner."""

from __future__ import annotations

import argparse
import copy
import errno
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import full_test14hard_suite as suite
import g_followup_1012 as processes
import queue_g_sigma_trials as training

ROOT = Path("/home/data/wyk")
TRAINING = ROOT / "outputs/score_aligned_1004/G_sigma_1012_8gpu_v1"
SOURCE = ROOT / "dataset/nuplan/exp/score_full_k_1009_v1"
OUTPUT = ROOT / "dataset/nuplan/exp/score_G_sigma_closed_loop_1014_v1"
MODELS = ("S05-G_plus100", "S10-G")


def job_list(ks):
    return [dict(id="DP_K0", model="S05-G", K=0), dict(id="S05-G_K20", model="S05-G", K=20)] + [
        dict(id=f"{name}_K{k}", model=name, K=k) for k in ks for name in MODELS
    ]


def verify_code_link(out, old):
    link = out / "code"
    if not link.is_symlink() or link.resolve() != (old / "code").resolve():
        raise ValueError("Frozen evaluation code link was redirected")


def process_table():
    rows = [processes.proc(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()]
    return {row["pid"]: row for row in rows if row and row["state"] != "Z"}


def is_coordinator(row, queue):
    argv = row["argv"]
    scripts = [i for i, value in enumerate(argv) if Path(value).name == "queue_g_sigma_trials.py"]
    if len(scripts) != 1 or argv[scripts[0] + 1 : scripts[0] + 2] != ["run"]:
        return False
    if "--queue-dir" in argv:
        return Path(argv[argv.index("--queue-dir") + 1]).resolve() == queue.resolve()
    # Only this published eight-GPU checkout has the matching default.
    return (
        Path(argv[scripts[0]]).resolve().parent.parent.name == "Score_Function_G_sigma_8gpu_1012"
        and queue.resolve() == TRAINING.resolve()
    )


def is_training_process(row, configs):
    argv = row["argv"]
    if "--config" not in argv:
        return False
    i = argv.index("--config")
    return (
        i + 1 < len(argv)
        and argv[i + 1] in configs
        and ("torch.distributed.run" in argv or "score_function" in argv)
    )


def descendants(table, roots):
    owned = set(roots)
    while True:
        expanded = owned | {pid for pid, row in table.items() if row["ppid"] in owned}
        if expanded == owned:
            return [table[pid] for pid in owned if pid in table]
        owned = expanded


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size


def signal_owned(row, signum):
    """Support the server's Python without pidfd; recheck start time and local PID."""
    if not processes.same_alive(row):
        return
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            fd = os.pidfd_open(row["signal_pid"])
        except ProcessLookupError:
            return
        except OSError as exc:
            if exc.errno not in (errno.ENOSYS, errno.EINVAL):
                raise
        else:
            try:
                if processes.same_alive(row):
                    signal.pidfd_send_signal(fd, signum)
                return
            finally:
                os.close(fd)
    current = processes.proc(row["pid"])
    if (
        current
        and current["start"] == row["start"]
        and current["signal_pid"] == row["signal_pid"]
        and current["state"] != "Z"
    ):
        try:
            os.kill(current["signal_pid"], signum)
        except ProcessLookupError:
            pass


def pause_training(args):
    queue = args.training.resolve()
    plan = suite.read(queue / "plan.json")
    jobs = {job["id"]: job for job in plan["jobs"]}
    target = jobs["S025-G"]
    score = Path(target["output"]) / "score"
    training.verify_plan(plan)
    # Establish that a resumable checkpoint exists before sending any signal.
    training.inspect_checkpoint(target, suite.digest(queue / "plan.json"))
    table = process_table()
    coordinators = [row for row in table.values() if is_coordinator(row, queue)]
    configs = {job["config"] for job in jobs.values()}
    workers = [row for row in table.values() if is_training_process(row, configs)]
    if len(coordinators) > 1:
        raise RuntimeError("Multiple training coordinators found; no process was stopped")
    if any(target["config"] not in row["argv"] for row in workers):
        raise RuntimeError("A different training job is still active; expected only S025-G")
    launchers = [row for row in workers if "torch.distributed.run" in row["argv"]]
    if len(launchers) > 1:
        raise RuntimeError("Multiple S025-G launchers found; no process was stopped")
    owned = descendants(table, [r["pid"] for r in coordinators + workers])
    for row in coordinators:
        signal_owned(row, signal.SIGTERM)
    if processes.wait_gone(coordinators, 5):
        raise RuntimeError("Training coordinator has not exited; evaluation was not started")
    checkpoint = score / "last.pt"
    initial = identity(checkpoint)
    if launchers and args.wait_save_seconds:
        print(
            "Scheduler stopped. Waiting for the next atomic last.pt save (at most "
            f"{args.wait_save_seconds}s); training continues during this wait.",
            flush=True,
        )
        deadline = time.monotonic() + args.wait_save_seconds
        next_notice = time.monotonic() + 15
        while (
            identity(checkpoint) == initial
            and any(processes.same_alive(row) for row in launchers)
            and time.monotonic() < deadline
        ):
            if time.monotonic() >= next_notice:
                print("Waiting for next checkpoint...", flush=True)
                next_notice = time.monotonic() + 15
            time.sleep(1)
        print(
            "New checkpoint observed."
            if identity(checkpoint) != initial
            else "Using the latest already-saved checkpoint.",
            flush=True,
        )
    # torchrun forwards SIGTERM to its ranks; cleanup is restricted to the captured tree.
    for row in launchers:
        signal_owned(row, signal.SIGTERM)
    remaining = processes.wait_gone(owned, 40)
    for row in remaining:
        signal_owned(row, signal.SIGTERM)
    if processes.wait_gone(remaining, 10):
        raise RuntimeError(
            "Some captured training processes are still exiting; do not evaluate yet"
        )
    if any(
        is_coordinator(r, queue) or is_training_process(r, configs)
        for r in process_table().values()
    ):
        raise RuntimeError("Training processes remain; evaluation was not started")
    with processes.exclusive(queue / "queue.lock"):
        saved = training.inspect_checkpoint(target, suite.digest(queue / "plan.json"))
        previous = suite.read(score / "status.json")
        receipt = dict(
            saved,
            checkpoint=str(checkpoint),
            checkpoint_sha256=suite.digest(checkpoint),
            previous_status=previous,
            paused_unix=time.time(),
        )
        suite.write(queue / "pause_record.json", receipt)
        if not saved["finished"]:
            updates = target["training_budget"]["updates_per_epoch"]
            suite.write(
                score / "status.json",
                dict(
                    previous,
                    state="paused",
                    phase="paused",
                    epoch=saved["step"] / updates,
                    step=saved["step"],
                ),
            )
        state = training.read(queue / "queue_status.json", {})
        state.update(state="paused", updated_at=time.time(), paused_checkpoint=receipt)
        if "S025-G" in state.get("jobs", {}):
            state["jobs"]["S025-G"]["state"] = "paused"
        suite.write(queue / "queue_status.json", state)
    print(
        f"PAUSED: S025-G saved epoch={saved['epoch']}, step={saved['step']}. "
        "Resume the SAME training queue later; do not prepare a new run.",
        flush=True,
    )


INSPECT = r"""
import json, sys, torch
from pathlib import Path
from score_function.utils.checkpoint import load_selected
from score_function.utils.planner_utils import planner_identity
from score_function.utils.train_utils import load_tensor
torch.set_num_threads(2)
p=json.loads(sys.argv[1]); out=Path(p['out']); queue=Path(p['training'])
common=json.loads((out/'models/S05-G/config.json').read_text())
reference=load_tensor(out/'models/S05-G/final_ema.pt')
start=load_tensor(queue/'reference/score/last.pt')
def same_weights(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
assert same_weights(reference['score_branch'],start['ema_branch']), 'Original G reference is different'
for name,epoch,step,sigma in [('S05-G_plus100',629,290598,.05),('S10-G',529,244398,.1)]:
    cfg=json.loads((out/'models'/name/'config.json').read_text())
    branch,s=load_selected(cfg,out/'models'/name/'final_ema.pt','cpu')
    last=load_tensor(queue/'runs'/name/'score/last.pt')
    assert s['weight_kind']=='ema' and (s['epoch'],s['step'],s['sigma_score'])==(epoch,step,sigma)
    assert last['finished'] and last['termination']=='epoch_budget_exhausted'
    assert (last['epoch'],last['step'],last['cursor'])==(epoch,step,0)
    assert same_weights(s['score_branch'],last['ema_branch']), 'Not final EMA weights'
    for key in ('planner','ego_normalizer'):
        assert s[key]==reference[key], key+' differs from original G'
    assert cfg['model']==common['model'], 'G architecture changed'
    assert cfg['reference_seed']==common['reference_seed']==p['reference_seed']
    assert cfg['refinement']['gamma']==p['gamma']
    assert cfg['refinement']['heading_projection']==p['heading_projection']
    assert planner_identity(cfg)==s['planner'], 'Frozen DP identity changed'
    for rel in ('model/score_branch.py','model/refined_planner.py','utils/sampling.py',
                'utils/normalizer.py','planner/planner.py'):
        import hashlib
        actual=hashlib.sha256((Path(p['code'])/'score_function'/rel).read_bytes()).hexdigest()
        assert actual==last['sources'][rel], 'Evaluation/training implementation differs: '+rel
    assert all(torch.isfinite(t).all() for t in s['score_branch'].values())
    with torch.no_grad():
        y=branch(torch.zeros(1,80,4),torch.zeros(1,107,192),torch.zeros(1,192))
    assert y.shape==(1,80,4) and torch.isfinite(y).all()
    print(f'VERIFIED {name}: final EMA epoch={epoch}, sigma={sigma}; same DP/noise rule')
"""


@contextmanager
def evaluation_environment(env):
    previous = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def check_partial_preparation(out, old):
    """Only recover known preparation artifacts; never overwrite evaluation output."""
    allowed = {"code", "models", "jobs"}
    allowed.update(f"models/{name}" for name in ("S05-G", *MODELS))
    allowed.update(
        f"models/{name}/{filename}"
        for name in ("S05-G", *MODELS)
        for filename in ("config.json", "final_ema.pt")
    )
    allowed.update(f"jobs/{job['id']}.json" for job in job_list([20])[:2])
    for path in out.rglob("*"):
        rel = path.relative_to(out).as_posix()
        if rel not in allowed or (path.is_symlink() and rel != "code"):
            raise ValueError(f"Unexpected partial preparation artifact: {path}")
    if (out / "code").exists() or (out / "code").is_symlink():
        verify_code_link(out, old)


def prepare(args):
    out, old, queue = args.output.resolve(), args.source.resolve(), args.training.resolve()
    if (out / "plan.json").exists():
        plan = suite.read(out / "plan.json")
        if (
            plan.get("source_suite") != str(old)
            or plan.get("training_queue") != str(queue)
            or plan["jobs"] != job_list(args.ks)
        ):
            raise ValueError("Existing evaluation plan has different inputs or K values")
        verify_code_link(out, old)
        suite.verify(out, plan)
        print("Prepared evaluation already exists; frozen inputs verified.", flush=True)
        return
    original = suite.read(old / "plan.json")
    suite.verify(old, original)
    if len(original["tokens"]) != 272 or len(set(original["tokens"])) != 272:
        raise ValueError("Require the full frozen 272-scenario Test14-Hard suite")
    tp = suite.read(queue / "plan.json")
    training.verify_plan(tp)
    jobs = {j["id"]: j for j in tp["jobs"]}
    for name in MODELS:
        training.inspect_checkpoint(jobs[name], suite.digest(queue / "plan.json"), final=True)
    check_partial_preparation(out, old)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "code").is_symlink():
        (out / "code").symlink_to(old / "code", target_is_directory=True)
    for name in ("S05-G", *MODELS):
        source = old / "models/S05-G" if name == "S05-G" else Path(jobs[name]["output"]) / "score"
        destination = out / "models" / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "final_ema.pt"):
            before = suite.digest(source / filename)
            target = destination / filename
            if target.exists():
                if suite.digest(target) != before:
                    raise ValueError(f"Partial frozen model differs from source: {target}")
            else:
                shutil.copy2(source / filename, target)
            if (
                suite.digest(destination / filename) != before
                or suite.digest(source / filename) != before
            ):
                raise ValueError(f"Model input changed while freezing: {source / filename}")
    code = out / "code/lg"
    payload = dict(
        out=str(out),
        training=str(queue),
        code=str(code),
        **{k: original[k] for k in ("reference_seed", "gamma", "heading_projection")},
    )
    subprocess.run(
        [sys.executable, "-c", INSPECT, json.dumps(payload)],
        cwd=code,
        env=suite.env_for(Path(original["root"]), code, ""),
        check=True,
    )
    plan = copy.deepcopy(original)
    plan.update(
        jobs=job_list(args.ks),
        source_suite=str(old),
        training_queue=str(queue),
        created_unix=time.time(),
        comparison="final G +100 epochs / sigma 0.10; fixed K and gamma",
    )
    for path in [
        old / "plan.json",
        queue / "plan.json",
        Path(__file__).resolve(),
        Path(training.__file__).resolve(),
        Path(processes.__file__).resolve(),
        Path(suite.__file__).resolve(),
        *list((out / "models").rglob("*")),
    ]:
        if path.is_file():
            plan["files"][str(path)] = suite.digest(path)
    for job in job_list(args.ks)[:2]:
        record = suite.read(old / "jobs" / (job["id"] + ".json"))
        if record["state"] not in ("complete", "reused"):
            raise ValueError(f"Original reference is not complete: {job['id']}")
        # Saved Hydra YAML resolves oc.env in THIS process, not the inspector child.
        with evaluation_environment(suite.env_for(Path(original["root"]), code, "")):
            metrics = suite.validate_result(record["result_dir"], job, plan, out)
        suite.write(
            out / "jobs" / (job["id"] + ".json"), dict(record, state="reused", metrics=metrics)
        )
    suite.write(out / "plan.json", plan)
    suite.write(out / "status.json", dict(state="prepared", updated_unix=time.time()))
    (out / "queue.log").touch()
    print(
        f"READY: {len(args.ks) * 2} new full Test14-Hard evaluations; "
        f"K={args.ks}, gamma={plan['gamma']}, seed={plan['reference_seed']}; 8 GPUs per run.",
        flush=True,
    )


def run(args):
    plan = suite.read(args.output / "plan.json")
    if plan.get("training_queue") != str(args.training.resolve()) or plan.get(
        "source_suite"
    ) != str(args.source.resolve()):
        raise ValueError("Requested queue/source differ from the frozen evaluation plan")
    verify_code_link(args.output, args.source)
    with ExitStack() as stack:
        # The training runner uses this same lock, so it cannot resume during evaluation.
        for path in (
            args.training / "queue.lock",
            args.source / ".run.lock",
            Path(plan["root"]) / "dataset/nuplan/exp/score_G_local_k_1012_v1/.run.lock",
        ):
            if path.parent.exists():
                stack.enter_context(processes.exclusive(path))
        busy = training.gpu_busy(list(range(8)))
        if busy:
            raise RuntimeError(
                f"GPUs are still occupied; wait for cleanup before evaluation: {busy}"
            )
        suite.jobs = lambda: plan["jobs"]
        suite.run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pause-training", "prepare", "run", "status"))
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ks", type=int, nargs="+", default=[20])
    parser.add_argument("--wait-save-seconds", type=int, default=180)
    args = parser.parse_args()
    if (
        not args.ks
        or len(set(args.ks)) != len(args.ks)
        or any(k <= 0 for k in args.ks)
        or not 0 <= args.wait_save_seconds <= 600
    ):
        parser.error("Require unique positive K values and a checkpoint wait of 0..600 seconds")
    if args.mode == "pause-training":
        pause_training(args)
    elif args.mode == "prepare":
        prepare(args)
    elif args.mode == "status":
        plan = suite.read(args.output / "plan.json")
        suite.jobs = lambda: plan["jobs"]
        suite.show(args.output)
    else:

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        run(args)


if __name__ == "__main__":
    main()
