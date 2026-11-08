#!/usr/bin/env python3
"""Run full DD-reader cache preparation, 500-epoch training and paired multi-K evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/navsim_dd_s10_g_reader.json"
KS = [5, 10, 15, 20, 30]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def paths(root):
    root = Path(root).resolve()
    exp = root / "dataset/navsim/exp"
    return dict(root=root, exp=exp, jobs=exp / "dd_score_reader_pipeline",
                source=exp / "dd_score_cache_v1", cache=exp / "dd_score_reader_cache_v1",
                train=exp / "dd_score_reader_v1", evaluation=exp / "dd_score_reader_navtest_v1",
                python=root / "envs/diffusion_drive/bin/python")


def environment(p):
    root = p["root"]
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str, [REPO, root / "DiffusionDrive", root / "nuplan-devkit"])),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               OPENSCENE_DATA_ROOT=str(root / "dataset/navsim"), NAVSIM_DEVKIT_ROOT=str(root / "DiffusionDrive"),
               NAVSIM_EXP_ROOT=str(p["exp"]), NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
               NUPLAN_MAP_VERSION="nuplan-maps-v1.0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def commands(p):
    python, root = str(p["python"]), p["root"]
    weights = root / "DiffusionDrive/checkpoints"
    cache = [python, "-u", "-m", "score_function.navsim_dd.reader_cache",
             "--source-cache", str(p["source"]), "--cache-dir", str(p["cache"]),
             "--checkpoint", str(weights / "diffusiondrive_navsim_88p1_PDMS"),
             "--anchor", str(weights / "kmeans_navsim_traj_20.npy"),
             "--backbone", str(weights / "pytorch_model.bin"),
             "--data-root", str(root / "dataset/navsim"), "--devkit-root", str(root / "DiffusionDrive"),
             "--workers", "8"]
    check = [python, "-u", str(Path(__file__).resolve()), "_preflight", "--root", str(root)]
    train = [python, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
             "-m", "score_function.navsim_dd.train", "--cache-dir", str(p["cache"]),
             "--output", str(p["train"]), "--config", str(CONFIG)]
    if (p["train"] / "run.json").is_file():
        train.append("--resume")
    evaluate = [python, "-u", str(REPO / "scripts/eval_navsim_dd_reader.py"), "run",
                "--root", str(root), "--train-output", str(p["train"]), "--output", str(p["evaluation"])]
    return [("cache", cache), ("preflight", check), ("train", train), ("evaluate", evaluate)]


def check_dependencies(p):
    root = p["root"]
    expected = read_json(CONFIG)
    if (expected.get("epochs") != 500 or expected.get("checkpoint_every_epochs") != 50
            or expected.get("lr_schedule_epochs", 500) != 500
            or expected.get("model", {}).get("architecture") != "dd_scene_reader_score"):
        raise ValueError("Require the full 500-epoch DD reader configuration with 50-epoch snapshots")
    required = [p["python"], CONFIG, p["source"] / "TRAINING_READY.json", p["source"] / "manifest.json",
                root / "DiffusionDrive/navsim", root / "nuplan-devkit",
                root / "dataset/navsim/navsim_logs/trainval", root / "dataset/navsim/sensor_blobs/trainval",
                root / "dataset/navsim/navsim_logs/test", root / "dataset/navsim/sensor_blobs/test",
                root / "dataset/nuplan/maps/nuplan-maps-v1.0.json", p["exp"] / "metric_cache"]
    required += [root / "DiffusionDrive/checkpoints" / name for name in
                 ("diffusiondrive_navsim_88p1_PDMS", "kmeans_navsim_traj_20.npy", "pytorch_model.bin")]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing dependency: {path}")
    if not os.access(p["python"], os.X_OK):
        raise ValueError(f"Python is not executable: {p['python']}")
    baseline = Path((p["exp"] / "dd_score_e200_eval_latest.txt").read_text().strip())
    if not baseline.is_absolute() or read_json(baseline / "summary.json").get("count") != 12146:
        raise ValueError("Require the existing complete 12146-scene K0 evaluation")
    if len(list(baseline.glob("part-*-of-*/complete.json"))) != 8:
        raise ValueError("Existing K0 evaluation is missing complete shard manifests")
    # Validate the downstream experiment before spending time on a full run.
    sys.path.insert(0, str(REPO / "scripts"))
    from eval_navsim_dd_reader import reference_info
    print("Checking all saved K0 records and evaluation assets before training...", flush=True)
    reference = reference_info(root, baseline)
    write_json(p["jobs"] / "k0_preflight.json", reference)
    import torch
    if Path(sys.executable).resolve() != p["python"].resolve():
        raise ValueError(f"Run this pipeline using {p['python']}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError("Require exactly eight visible CUDA GPUs for the full pipeline")


def preflight(p):
    """Exercise the real compact cache and the production AMP DSM update before a long run."""
    import torch
    from score_function.navsim_dd.train import _check_config, _seed, build_training_model, load_training_data
    from score_function.train_epoch import train_epoch, validate_epoch
    from score_function.utils.dataset import collate_cpu
    from score_function.utils.train_utils import file_hash, fingerprint

    cfg = read_json(CONFIG)
    _check_config(cfg)
    metadata, training, validation = load_training_data(p["cache"], cfg)
    ready = read_json(p["cache"] / "TRAINING_READY.json")
    if (ready.get("manifest_sha256") != file_hash(p["cache"] / "manifest.json")
            or ready.get("cache_fingerprint") != fingerprint(metadata)):
        raise ValueError("Reader TRAINING_READY does not validate this manifest")
    if len(training) + len(validation) != 103288 or len(training) < 64 or len(validation) < 1:
        raise ValueError(f"Unexpected full navtrain split sizes: {len(training)}, {len(validation)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError(f"Require eight visible CUDA GPUs; found {torch.cuda.device_count()}")
    for number in range(8):
        value = torch.ones(1, device=f"cuda:{number}")
        if value.sum().item() != 1:
            raise RuntimeError(f"CUDA {number} failed")
        print(f"CUDA {number}: {torch.cuda.get_device_name(number)}", flush=True)
        del value
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    _seed(cfg["seed"])
    model = build_training_model(cfg, metadata, p["cache"], device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Reader experiment has no trainable score weights")
    reference = trainable[-1].detach().clone()
    batch = collate_cpu([training[i] for i in range(64)])
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()
    for attempt in range(8):
        _seed(cfg["seed"] + 1)
        result = next(train_epoch([batch], model, optimizer, cfg, None, device, start_step=0,
                                 start_update=0, updates_per_epoch=1, accumulation=1,
                                 next_lr=cfg["learning_rate"], scaler=scaler))
        if not result["optimizer_step_skipped"]:
            break
    else:
        raise FloatingPointError("Every real-batch FP16 preflight update overflowed")
    if torch.equal(reference, trainable[-1]) or not all(torch.isfinite(p).all() for p in trainable):
        raise FloatingPointError("Preflight did not produce a finite score optimizer update")
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise RuntimeError("Frozen DD parameters received gradients")
    validation_cfg = {**cfg, "num_workers": 0, "validation_repeats": 1}
    from torch.utils.data import Subset
    validation_loss = validate_epoch({"raw": model}, Subset(validation, range(min(64, len(validation)))),
                                     validation_cfg, device)["raw"]
    report = dict(status="complete", train_samples=len(training), val_samples=len(validation),
                  train_dsm=result["train_dsm"], val_dsm=validation_loss,
                  trainable_parameters=sum(p.numel() for p in trainable),
                  frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                  cache_fingerprint=fingerprint(metadata))
    write_json(p["jobs"] / "preflight.json", report)
    print("REAL READER PREFLIGHT OK: " + json.dumps(report), flush=True)


def run(p):
    p["jobs"].mkdir(parents=True, exist_ok=True)
    with (p["jobs"] / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("The DD reader pipeline is already active") from None
        print("Checking dependencies and saved K0 before the full DD reader pipeline...", flush=True)
        check_dependencies(p)
        state = dict(status="running", stage="starting", pid=os.getpid(), host=socket.gethostname(),
                     started=datetime.now(timezone.utc).isoformat(), epochs=500, ks=KS,
                     cache=str(p["cache"]), training=str(p["train"]), evaluation=str(p["evaluation"]))
        previous, child = {}, None
        def interrupted(signum, frame):
            raise InterruptedError(f"Received signal {signum}")
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, interrupted)
            for stage, command in commands(p):
                log = p["jobs"] / f"{stage}.log"
                state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), log=str(log))
                write_json(p["jobs"] / "state.json", state)
                print(f"START {stage}: {log}", flush=True)
                with log.open("a") as stream:
                    stream.write(f"\n=== {state['updated']} {stage} ===\n")
                    stream.flush()
                    child = subprocess.Popen(command, cwd=REPO, env=environment(p), stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    state["child_pid"] = child.pid
                    write_json(p["jobs"] / "state.json", state)
                    heartbeat = time.monotonic()
                    while child.poll() is None:
                        if time.monotonic() - heartbeat >= 30:
                            print(f"RUNNING {stage}; PID={child.pid}; log={log}", flush=True)
                            heartbeat = time.monotonic()
                        time.sleep(1)
                    if child.returncode:
                        raise RuntimeError(f"{stage} exited {child.returncode}; see {log}")
                child = None
                print(f"COMPLETE {stage}", flush=True)
            state.update(status="complete", stage="complete", child_pid=None,
                         finished=datetime.now(timezone.utc).isoformat())
            print(f"ALL COMPLETE: {p['evaluation'] / 'summary.json'}", flush=True)
        except BaseException as exc:
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            state.update(status="failed", detail=f"{type(exc).__name__}: {exc}", child_pid=None)
            raise
        finally:
            write_json(p["jobs"] / "state.json", state)
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def status(p):
    path = p["jobs"] / "state.json"
    print(json.dumps(read_json(path), indent=2, ensure_ascii=False) if path.is_file() else "Pipeline has not started")
    lock_path = p["jobs"] / "pipeline.lock"
    if lock_path.is_file():
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print("pipeline lock: idle")
            except BlockingIOError:
                print("pipeline lock: active")
    if (p["cache"] / "manifest.json").is_file():
        metadata = read_json(p["cache"] / "manifest.json")
        print("Reader cache: " + ", ".join(f"{k}={v['num_records']}" for k, v in metadata["splits"].items()))
    else:
        workers = []
        for worker in sorted(p["cache"].glob("worker*.json")):
            value = read_json(worker)
            workers.append(f"{worker.stem}={value.get('completed', 0)}/{value.get('total', '?')}")
        if workers:
            print("Reader cache extraction: " + ", ".join(workers))
    history_path = p["train"] / "history.json"
    if history_path.is_file():
        history = read_json(history_path)
        if history:
            last = history[-1]
            print(f"Training: {last['epoch']}/500; train DSM={last['train_dsm']:.6f}; val DSM={last['val_dsm']:.6f}")
    summary_path = p["evaluation"] / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        baseline = 100 * summary["by_k"]["0"]["mean"]["score"]
        print(f"Paired scenes: {summary['count']}; reused K0={baseline:.4f}")
        for k in KS:
            arm = summary["by_k"][str(k)]
            score = 100 * arm["mean"]["score"]
            print(f"K={k:2d} PDMS={score:.4f} delta={score-baseline:+.4f}; "
                  f"improved/worsened/tied={arm['improved']}/{arm['worsened']}/{arm['tied']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "_preflight"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    args = parser.parse_args()
    p = paths(args.root)
    {"run": run, "status": status, "_preflight": preflight}[args.action](p)


if __name__ == "__main__":
    main()
