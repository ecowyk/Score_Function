#!/usr/bin/env python3
"""Run/status for an immutable prepared G/sigma training queue, sequentially.

No queue is prepared here, no unrelated process is signalled, and no scheduler is
installed. Run `run` in tmux with the desired training Python interpreter.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_QUEUE = Path("/home/data/wyk/outputs/score_aligned_1004/G_sigma_1012_v1")


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def emit(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def sources(code):
    root = Path(code) / "score_function"
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*.py"))}


def clean_env(code, gpus=None):
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
        } or key.startswith("TORCHELASTIC_"):
            env.pop(key)
    env["PYTHONPATH"] = str(Path(code).resolve())
    env["CUDA_VISIBLE_DEVICES"] = "" if gpus is None else ",".join(map(str, gpus))
    env.update(
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        SCORE_FUNCTION_PROGRESS="off",
    )
    return env


def verify_plan(plan):
    if plan.get("hostname") and plan["hostname"] != socket.gethostname():
        raise ValueError("Queue was prepared on a different GPU node")
    files = plan.get("files")
    if not files:
        raise ValueError("Prepared plan must bind immutable files with SHA256")
    checks = (
        files.items()
        if isinstance(files, dict)
        else ((item["path"], item["sha256"]) for item in files)
    )
    bound = {}
    for name, expected in checks:
        path = Path(name).resolve()
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Prepared input missing or changed: {path}")
        bound[str(path)] = expected
    jobs = plan.get("jobs", [])
    if not jobs or len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Queue requires unique nonempty jobs")
    outputs = set()
    for job in jobs:
        if str(Path(job["config"]).resolve()) not in bound:
            raise ValueError(f"Job config is not SHA-bound: {job['id']}")
        actual = sources(job["code"])
        if not actual or job.get("sources", actual) != actual:
            raise ValueError(f"Frozen code changed: {job['id']}")
        for relative, expected in actual.items():
            path = str((Path(job["code"]) / "score_function" / relative).resolve())
            if bound.get(path) != expected:
                raise ValueError(f"Frozen source not SHA-bound: {path}")
        if len(job["gpus"]) != job["world_size"] or len(set(job["gpus"])) != len(job["gpus"]):
            raise ValueError(f"GPU/rank allocation is invalid: {job['id']}")
        if job["target_epochs"] < 1 or job["target_updates"] < 1:
            raise ValueError("Training targets must be positive")
        output = str(Path(job["output"]).resolve())
        if output in outputs or Path(read(job["config"])["output"]).resolve() != Path(output):
            raise ValueError("Jobs must have distinct outputs matching their configs")
        outputs.add(output)


def capture(command):
    return subprocess.check_output(command, text=True, timeout=20).strip()


def gpu_busy(gpus, memory_mib=256, utilization_percent=5):
    """Use physical index→UUID, plus memory/utilization fallback if app query fails.

    Any known compute PID counts as occupied, even with tiny/unknown allocation.
    Memory thresholds cover hidden processes, graphics work, and container NVML
    limitations. Failure to inspect inventory never permits a blind launch.
    """
    text = capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    inventory = {}
    for row in csv.reader(text.splitlines()):
        if len(row) != 4:
            raise RuntimeError("Unrecognized nvidia-smi GPU inventory")
        inventory[int(row[0])] = {
            "uuid": row[1].strip(),
            "memory_mib": float(row[2]),
            "utilization": float(row[3]),
        }
    if not set(gpus).issubset(inventory):
        raise RuntimeError(f"Requested physical GPUs unavailable: {gpus}")
    selected = {inventory[g]["uuid"] for g in gpus}
    busy = []
    try:
        apps = capture(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        for row in csv.reader(apps.splitlines()):
            if not row or row[0].strip() not in selected:
                continue
            if len(row) < 3:
                raise RuntimeError("Unrecognized nvidia-smi compute process row")
            busy.append(
                {
                    "gpu_uuid": row[0].strip(),
                    "pid": row[1].strip(),
                    "used_memory_mib": row[2].strip(),
                }
            )
    except (OSError, subprocess.SubprocessError, RuntimeError):
        # Per-GPU counters above remain available when per-process NVML is hidden.
        pass
    for index in gpus:
        item = inventory[index]
        if item["memory_mib"] > memory_mib or item["utilization"] > utilization_percent:
            busy.append({"gpu_index": index, **item})
    return busy


CHECKPOINT_WORKER = r"""
import json, sys, torch
from pathlib import Path
from score_function.utils.config import load_config, model_config
from score_function.utils.train_utils import load_tensor, atomic_write, file_hash, source_hashes, training_signature
payload=json.load(sys.stdin); j=payload['job']; score=Path(j['output'])/'score'
torch.set_num_threads(2)
c=load_tensor(score/'last.pt'); cfg=load_config(j['config'])
p=json.loads((score/'provenance.json').read_text()); info=json.loads((score/'model_info.json').read_text())
if c['sources'] != source_hashes() or c['sources'] != p['sources']:
    raise ValueError('Checkpoint/source provenance mismatch')
if info['world_size'] != j['world_size'] or len(c['ranks']) != j['world_size']:
    raise ValueError('Checkpoint rank count mismatch')
if info['global_batch'] != cfg['training']['batch_size'] or info['microbatch_per_rank'] != cfg['training']['microbatch_size']:
    raise ValueError('Checkpoint effective batch mismatch')
if c['signature'] != p['signature'] or c['signature'] != training_signature(cfg,p['cache_sha256'],j['world_size']):
    raise ValueError('Checkpoint training signature mismatch')
# Canonical defaults and rank-local CUDA spelling are not scientific changes.
saved=dict(c['config']); saved['model']=model_config(saved['model'])
requested=dict(cfg); requested['model']=model_config(requested['model'])
saved_runtime=dict(saved.get('runtime',{})); requested_runtime=dict(requested.get('runtime',{}))
if str(saved_runtime.get('device','')).startswith('cuda') and str(requested_runtime.get('device','')).startswith('cuda'):
    saved_runtime['device']=requested_runtime['device']='cuda'
saved['runtime']=saved_runtime; requested['runtime']=requested_runtime
if saved != requested:
    raise ValueError('Checkpoint config differs from frozen job config')
result={'epoch':c['epoch'],'step':c['step'],'finished':c.get('finished',False),'elapsed_s':c.get('elapsed_s',0)}
if payload['final']:
    status=json.loads((score/'status.json').read_text())
    if not(c.get('finished') and c.get('termination')=='epoch_budget_exhausted' and status.get('state')=='complete' and status.get('termination')=='epoch_budget_exhausted'):
        raise ValueError('Trainer/checkpoint does not certify full-budget completion')
    if c['epoch'] != j['target_epochs'] or c['step'] != j['target_updates'] or c.get('cursor') != 0:
        raise ValueError('Final checkpoint does not match exact target epoch/update budget')
    if c['epoch'] != status.get('epoch') or c['step'] != status.get('step'):
        raise ValueError('Status/checkpoint counters disagree')
    budget=c['training_budget']
    if budget['planned_epochs'] != j['target_epochs'] or budget['planned_optimizer_updates'] != j['target_updates'] or status.get('training_budget') != budget:
        raise ValueError('Final training budget disagrees')
    last_hash=file_hash(score/'last.pt')
    export=score/'final_ema.pt'
    metadata={'source_last_sha256':last_hash,'queue_plan_sha256':payload['plan_sha256'],'job_id':j['id'],'signature':c['signature'],'sources':c['sources'],'training_budget':budget,'model_info':info,'training_provenance':p}
    final={key:c[key] for key in ('schema_version','method','planner','ego_normalizer','sigma_score','config','epoch','step')}
    final.update(score_branch=c['ema_branch'],weight_kind='ema',cache_sha256=p['cache_sha256'],val_dsm=None,criterion='final_epoch_ema_not_validation_selected',export_provenance=metadata)
    for key in ('sigma_mode','sigma_levels'):
        if key in c or key in cfg['training']:
            final[key]=c.get(key,cfg['training'].get(key))
    if export.exists():
        old=load_tensor(export)
        if old.get('export_provenance') != metadata or old.get('weight_kind')!='ema' or old.get('epoch')!=c['epoch'] or old.get('step')!=c['step']:
            raise ValueError('Existing final_ema.pt has different provenance; refusing overwrite')
        if set(old['score_branch']) != set(c['ema_branch']) or any(not torch.equal(old['score_branch'][k],v) for k,v in c['ema_branch'].items()):
            raise ValueError('Existing final_ema.pt does not contain final EMA weights')
    else:
        atomic_write(export,final,tensor=True)
    result.update(final_ema=str(export),final_ema_sha256=file_hash(export),source_last_sha256=last_hash)
    atomic_write(score/'final_ema_export.json',{**result,**metadata})
print(json.dumps(result,allow_nan=False))
"""


def inspect_checkpoint(job, plan_sha256, final=False):
    result = subprocess.run(
        [sys.executable, "-c", CHECKPOINT_WORKER],
        input=json.dumps({"job": job, "final": final, "plan_sha256": plan_sha256}),
        text=True,
        capture_output=True,
        cwd=job["code"],
        env=clean_env(job["code"]),
        timeout=600,
    )
    if result.returncode:
        raise RuntimeError(f"{job['id']} checkpoint verification failed:\n{result.stderr[-8000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def command_for(job, resume):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={job['world_size']}",
        "-m",
        "score_function",
        "train",
        "--config",
        str(job["config"]),
    ]
    if resume:
        command.append("--resume")
    return command


def progress_line(job, receipt, now=None):
    now = time.time() if now is None else now
    status = read(Path(job["output"]) / "score/status.json", {})
    step = status.get("step", receipt.get("base_step", 0))
    elapsed = max(0.0, receipt.get("completed_at", now) - receipt.get("started_at", now))
    completed = max(0, step - receipt.get("base_step", step))
    rate = completed / elapsed if elapsed else 0
    eta = max(0, job["target_updates"] - step) / rate if rate else None
    eta_text = f"{eta / 3600:.2f}h" if eta is not None else "unknown"
    return (
        f"{job['id']} {receipt.get('state', 'pending')}/{status.get('state', 'not_started')} "
        f"step={step}/{job['target_updates']} epoch={status.get('epoch', '?')}/{job['target_epochs']} "
        f"elapsed={elapsed / 3600:.2f}h ETA={eta_text} "
        f"log={receipt.get('log', str(Path(job['output']) / 'score/console.log'))}"
    )


def show_status(queue, plan):
    status = read(queue / "queue_status.json", {})
    print(f"Queue: {queue} state={status.get('state', 'prepared')}", flush=True)
    receipts = status.get("jobs", {})
    for job in plan["jobs"]:
        emit(progress_line(job, receipts.get(job["id"], {})))
    return status


def run(args):
    queue = args.queue_dir.resolve()
    plan_path = queue / "plan.json"
    with (queue / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "This queue or its training process is still active; use status"
            ) from exc
        plan = read(plan_path)
        verify_plan(plan)
        plan_hash = digest(plan_path)
        identity_path = queue / "queue_identity.json"
        identity = read(identity_path, {})
        if identity and identity.get("plan_sha256") != plan_hash:
            raise ValueError("Prepared plan changed after queue was first armed")
        if not identity:
            write(
                identity_path,
                {"plan_sha256": plan_hash, "python": sys.executable, "created_at": time.time()},
            )
        previous = read(queue / "queue_status.json", {})
        receipts = previous.get("jobs", {})
        for job in plan["jobs"]:
            receipts.setdefault(job["id"], {"state": "pending"})

        def state(value, **extra):
            write(
                queue / "queue_status.json",
                {"state": value, "updated_at": time.time(), "jobs": receipts, **extra},
            )

        for job in plan["jobs"]:
            entry = receipts[job["id"]]
            try:
                verify_plan(plan)
                score = Path(job["output"]) / "score"
                saved = read(score / "status.json", {})
                if saved.get("state") == "complete":
                    result = inspect_checkpoint(job, plan_hash, final=True)
                    entry.update(state="complete", completed_at=time.time(), **result)
                    state("running")
                    emit(f"VERIFIED COMPLETE: {job['id']} final_ema={result['final_ema']}")
                    continue
                last = score / "last.pt"
                resume = last.is_file()
                if job.get("resume") and not resume:
                    raise FileNotFoundError(f"Continuation checkpoint missing: {last}")
                if score.exists() and not resume:
                    raise RuntimeError(
                        f"{score} exists without last.pt; preserving incomplete data, cannot safely resume"
                    )
                check = inspect_checkpoint(job, plan_hash) if resume else {"step": 0, "epoch": 0}
                # Never mkdir score/ here: the trainer owns fresh score directory creation.
                logs = queue / "logs"
                logs.mkdir(exist_ok=True)
                log_path = logs / f"{job['id']}.launcher.log"
                log_path.touch(exist_ok=True)
                entry.update(
                    state="waiting_gpu",
                    log=str(log_path),
                    base_step=check["step"],
                    epoch=check["epoch"],
                )
                while True:
                    occupied = gpu_busy(job["gpus"], args.gpu_memory_mib, args.gpu_util_percent)
                    state("waiting_gpu", current_job=job["id"], busy_gpus=occupied)
                    if not occupied:
                        break
                    emit(f"WAIT {job['id']} GPUs={job['gpus']} busy={occupied}")
                    time.sleep(args.poll_seconds)
                verify_plan(plan)
                if gpu_busy(job["gpus"], args.gpu_memory_mib, args.gpu_util_percent):
                    raise RuntimeError("GPUs became busy just before launch; safe to rerun queue")
                command = command_for(job, resume)
                with log_path.open("a", encoding="utf-8", buffering=1) as log:
                    log.write(
                        f"\n=== QUEUE ATTEMPT {time.strftime('%Y-%m-%d %H:%M:%S')} resume={resume} ===\n"
                    )
                    child = subprocess.Popen(
                        command,
                        cwd=job["code"],
                        env=clean_env(job["code"], job["gpus"]),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(lock.fileno(),),
                    )
                entry.update(
                    state="running",
                    pid=child.pid,
                    started_at=time.time(),
                    command=command,
                    resumed=resume,
                    attempts=entry.get("attempts", 0) + 1,
                )
                state("running", current_job=job["id"])
                emit(f"START {job['id']} pid={child.pid} resume={resume} log={log_path}")
                while child.poll() is None:
                    emit(progress_line(job, entry))
                    state("running", current_job=job["id"])
                    try:
                        child.wait(timeout=args.poll_seconds)
                    except subprocess.TimeoutExpired:
                        pass
                entry["exit_code"] = child.returncode
                if child.returncode != 0:
                    raise RuntimeError(
                        f"Training exited with code {child.returncode}; see {log_path}"
                    )
                result = inspect_checkpoint(job, plan_hash, final=True)
                entry.update(state="complete", completed_at=time.time(), **result)
                state("running")
                emit(
                    f"COMPLETE {job['id']} epoch={result['epoch']} step={result['step']} final_ema={result['final_ema']}"
                )
            except Exception as exc:
                entry.update(
                    state="failed", error=f"{type(exc).__name__}: {exc}", failed_at=time.time()
                )
                state("failed", current_job=job["id"])
                raise
        state("complete")
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--watch", action="store_true", help="Repeat status every polling interval")
    parser.add_argument("--gpu-memory-mib", type=float, default=256)
    parser.add_argument("--gpu-util-percent", type=float, default=5)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.poll_seconds <= 60
        or args.gpu_memory_mib < 0
        or not 0 <= args.gpu_util_percent <= 100
    ):
        parser.error("Require poll seconds 1..60, nonnegative GPU memory and utilization 0..100")
    try:
        if args.action == "run":
            return run(args)
        plan = read(args.queue_dir / "plan.json")
        while True:
            show_status(args.queue_dir, plan)
            if not args.watch:
                return 0
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        emit("Watcher interrupted. Existing training remains running and retains the queue lock.")
        return 130
    except Exception as exc:
        emit(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
