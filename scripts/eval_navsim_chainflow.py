#!/usr/bin/env python3
"""Resume native ChainFlow reproduction or paired Score evaluation on available GPUs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from score_function.utils.train_utils import atomic_write, file_hash, read_json

TOTAL = 12146
KS = [5, 10, 15, 20, 30]


def validate_inherited_lease(args):
    """Only a live parent holding every visible GPU lock can delegate its lease."""
    owner = int(os.environ.get("CF_GPU_LEASE_OWNER_PID", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    budget = float(os.environ.get("CF_CUDA_MEMORY_GB", "nan"))
    lock_root = os.environ.get("CF_GPU_LEASE_LOCK_DIR")
    if (owner != os.getppid() or not lock_root or len(visible) != args.workers
            or len(set(visible)) != len(visible) or any(not v.startswith("GPU-") for v in visible)
            or not math.isfinite(budget) or budget <= 0):
        raise ValueError("--gpu-lease-held requires the live parent's GPU UUID allocation, lock directory and memory cap")
    os.kill(owner, 0)
    for uuid in visible:
        path = Path(lock_root) / f"{uuid}.lock"
        with path.open("r+") as handle:
            record = json.load(handle)
            if (record.get("pid") != owner or record.get("gpu_uuid") != uuid
                    or record.get("host") != socket.gethostname()
                    or record.get("budget_gb_per_worker") != budget
                    or uuid not in record.get("selected_uuids", [])):
                raise ValueError(f"Inherited GPU lease identity differs: {uuid}")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            fcntl.flock(handle, fcntl.LOCK_UN)
            raise ValueError(f"Parent GPU lease is no longer held: {uuid}")


@contextmanager
def evaluation_gpu_lease(args):
    if args.gpu_lease_held:
        validate_inherited_lease(args)
        yield
        return
    from score_function.navsim_chainflow.resources import acquire_gpus, gpu_environment
    def status(state):
        atomic_write(args.output / "gpu_status.json", state)
        print(f"GPU {state['state']}: policy={state['policy']} workers={args.workers} "
              f"wait={state['wait_seconds']:.1f}s; {state['notice']}", flush=True)
    with acquire_gpus(pool=args.gpu_pool, workers=args.workers, budget_gb=args.gpu_budget_gb,
                      reserve_gb=args.gpu_reserve_gb, policy=args.gpu_policy,
                      lock_dir=args.gpu_lock_dir, on_status=status) as selected:
        assigned = gpu_environment(selected, args.gpu_budget_gb)
        previous = {key: os.environ.get(key) for key in assigned}
        os.environ.update(assigned)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def default_paths(root, mode):
    root = Path(root).resolve()
    exp = root / "dataset/navsim/exp"
    return dict(devkit_root=root / "ChainFlow-VLA", data_root=root / "dataset/navsim",
                checkpoint=root / "ChainFlow-VLA/weights/stage_2/stage2_navtrain_9382.ckpt",
                backbone=root / "ChainFlow-VLA/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors",
                vlm_cache=root / "dataset/navsim/chainflow_vlm_cache",
                metric_cache=exp / "metric_cache", baseline=exp / "chainflow_native_navtest_v1",
                score_checkpoint=exp / "chainflow_score_v1/final_ema.pt",
                output=exp / ("chainflow_native_navtest_v1" if mode == "baseline" else "chainflow_score_navtest_v1"))


def child_environment(args):
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str, [REPO, args.devkit_root, args.root / "nuplan-devkit"])),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OPENSCENE_DATA_ROOT=str(args.data_root), NAVSIM_DEVKIT_ROOT=str(args.devkit_root),
               NAVSIM_EXP_ROOT=str(args.root / "dataset/navsim/exp"),
               NAVSIM_METRIC_CACHE_ROOT=str(args.metric_cache.parent),
               NUPLAN_MAPS_ROOT=str(args.root / "dataset/nuplan/maps"), NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
               CHAINFLOW_ENABLE_RAY="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def worker_command(args, index):
    command = [sys.executable, "-u", "-m", "score_function.navsim_chainflow.evaluate",
               "--mode", args.mode, "--device", f"cuda:{index}",
               "--num-shards", str(args.workers), "--shard-index", str(index),
               "--seed", str(args.seed), "--gamma", str(args.gamma), "--ks", *map(str, args.ks)]
    for name in ("devkit_root", "checkpoint", "backbone", "vlm_cache", "data_root", "metric_cache", "output"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.mode == "refine":
        command += ["--score-checkpoint", str(args.score_checkpoint), "--baseline", str(args.baseline)]
    if args.max_scenes is not None:
        command += ["--max-scenes", str(args.max_scenes)]
    return command


def request_identity(args):
    # Bind cached completion to launch arguments without loading any model/GPU.
    from score_function.navsim_chainflow.native import conditioner_identity
    from score_function.navsim_chainflow.evaluate import scoring_identity, validate_native_baseline, baseline_reference
    paths = ("devkit_root", "checkpoint", "backbone", "vlm_cache", "data_root", "metric_cache")
    request = {name: str(getattr(args, name).resolve()) for name in paths}
    request.update(mode=args.mode, workers=args.workers, seed=args.seed, gamma=args.gamma,
                   ks=[0] if args.mode == "baseline" else [0, *sorted(set(args.ks))],
                   max_scenes=args.max_scenes,
                   checkpoint_sha256=file_hash(args.checkpoint), backbone_sha256=file_hash(args.backbone))
    request["native_identity"] = conditioner_identity(args.devkit_root, args.checkpoint, args.backbone, args.vlm_cache)
    request["scoring_identity"] = scoring_identity(args.devkit_root)
    if args.mode == "refine":
        validate_native_baseline(args.baseline, expected_identity=request["native_identity"],
                                 expected_scoring=request["scoring_identity"], expected_count=args.max_scenes or TOTAL)
        request.update(baseline=str(args.baseline.resolve()),
                       baseline_reference=baseline_reference(args.baseline),
                       score_checkpoint=str(args.score_checkpoint.resolve()),
                       score_checkpoint_sha256=file_hash(args.score_checkpoint),
                       expected_epoch=args.expected_epoch, weight_kind=args.weight_kind)
    return request


def require_launch_protocol(protocol, request):
    """A completion marker alone must not bypass the current launch contract."""
    expected = {"mode": request["mode"], "num_shards": request["workers"], "seed": request["seed"],
                "gamma": request["gamma"] if request["mode"] == "refine" else None,
                "ks": request["ks"], "max_scenes": request["max_scenes"],
                "data_root": request["data_root"], "metric_cache": request["metric_cache"],
                "identity": request["native_identity"], "scoring_identity": request["scoring_identity"]}
    if request["mode"] == "refine":
        expected.update(score_checkpoint_sha256=request["score_checkpoint_sha256"],
                        score_epoch=request["expected_epoch"], score_weight_kind=request["weight_kind"])
        if "baseline_reference" in request:
            expected["baseline"] = request["baseline_reference"]
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Completed shard differs from current launch: {key}")


def completed_indices(output, request=None):
    from score_function.navsim_chainflow.evaluate import read_completed_shard
    indices = set()
    for directory in sorted(Path(output).glob("part-*-of-*")):
        if not (directory / "complete.json").exists():
            continue
        manifest, _ = read_completed_shard(directory)
        if request is not None:
            require_launch_protocol(manifest["run"], request)
        indices.add(manifest["shard_index"])
    return indices


def progress_snapshot(output):
    counts = []
    for directory in sorted(Path(output).glob("part-*-of-*")):
        source = directory / "scenes.jsonl"
        if not source.exists():
            source = directory / "scenes.jsonl.tmp"
        count = 0
        if source.exists():
            with source.open("rb") as stream:
                count = sum(block.count(b"\n") for block in iter(lambda: stream.read(1024 * 1024), b""))
        counts.append(count)
    return counts


def print_summary(output):
    summary = read_json(Path(output) / "summary.json")
    print(f"ChainFlow {summary['mode']}: {summary['valid_count']}/{summary['count']} valid scenes", flush=True)
    for k, result in sorted(summary["by_k"].items(), key=lambda pair: int(pair[0])):
        print(f"K={int(k):2d} PDMS={100*result['mean']['score']:.4f}"
              f" delta={100*result['mean_delta']['score']:+.4f};"
              f" improved/worsened/tied={result['improved']}/{result['worsened']}/{result['tied']}", flush=True)
    return summary


def publish_ready(output, summary):
    """Native reproduction can be handed to the Score queue without rerunning K0."""
    if summary["mode"] != "baseline" or summary["count"] != TOTAL or summary["valid_count"] != TOTAL:
        return
    protocol = summary["protocol"]
    atomic_write(Path(output) / "reproduction_ready.json", {
        "schema": "chainflow-native-ready-v1", "baseline_dir": str(Path(output).resolve()),
        "count": summary["count"], "valid_count": summary["valid_count"],
        "pdms": 100 * summary["by_k"]["0"]["mean"]["score"],
        "summary_sha256": file_hash(Path(output) / "summary.json"),
        "native_identity": protocol["identity"], "scoring_identity": protocol["scoring_identity"],
        "token_digest": protocol["token_digest"], "native_seed": protocol["native_seed"],
    })


def validate_score_checkpoint(args):
    import torch
    saved = torch.load(args.score_checkpoint, map_location="cpu", weights_only=True)
    required = {"format": "navsim-chainflow-score-v1", "epoch": args.expected_epoch,
                "weight_kind": args.weight_kind, "sigma": .1}
    for key, value in required.items():
        if saved.get(key) != value:
            raise ValueError(f"Score checkpoint {key} must be {value!r}, got {saved.get(key)!r}")
    if not saved.get("state_dict") or not saved.get("conditioner_identity"):
        raise ValueError("Score checkpoint has no weights or native conditioner identity")


def stop_children(children):
    running = [p for p in children if p.poll() is None]
    for child in running:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for child in running:
        try:
            child.wait(timeout=max(.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def run(args):
    from score_function.navsim_chainflow.evaluate import merge_shards
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This ChainFlow evaluation queue is already active") from None
        for name in ("checkpoint", "backbone", "vlm_cache", "devkit_root", "data_root", "metric_cache"):
            if not getattr(args, name).exists():
                raise FileNotFoundError(f"Missing {name}: {getattr(args, name)}")
        if args.mode == "refine":
            if args.output.resolve() == args.baseline.resolve():
                raise ValueError("Refinement must not overwrite the native baseline")
            validate_score_checkpoint(args)
            from score_function.navsim_chainflow.handoff import is_handoff
            if not is_handoff(args.baseline) and not (args.baseline / "summary.json").is_file():
                raise FileNotFoundError("The native baseline must be complete before Score evaluation")
        request = request_identity(args)
        state_path = args.output / "launcher.json"
        if state_path.exists() and read_json(state_path).get("request") != request:
            raise ValueError("Output queue belongs to another protocol; choose a new output directory")
        complete = completed_indices(args.output, request)
        if not complete <= set(range(args.workers)):
            raise ValueError("Completed shards use a different worker count")
        if len(complete) == args.workers:
            report = merge_shards(args.output)
            publish_ready(args.output, report)
            state = read_json(state_path) if state_path.exists() else {"request": request}
            state.update(status="complete", finished=time.time(), completed_shards=sorted(complete))
            atomic_write(state_path, state)
            print_summary(args.output)
            return report
        atomic_write(state_path, dict(status="waiting_for_gpu", pid=os.getpid(),
                                      host=socket.gethostname(), request=request, started=time.time()))
        try:
            with evaluation_gpu_lease(args):
                return _run_gpu_stage(args, state_path, request, complete)
        except BaseException as exc:
            state = read_json(state_path)
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=f"{type(exc).__name__}: {exc}", updated=time.time())
            atomic_write(state_path, state)
            raise


def _run_gpu_stage(args, state_path, request, complete):
    from score_function.navsim_chainflow.evaluate import merge_shards
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.workers:
        raise ValueError(f"Need {args.workers} visible CUDA GPUs; got {torch.cuda.device_count()}")
    state = dict(status="running", pid=os.getpid(), host=socket.gethostname(), request=request,
                 started=time.time(), completed_shards=sorted(complete))
    atomic_write(state_path, state)
    children, logs, old_handlers = [], [], {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, interrupted)
    try:
        for index in range(args.workers):
            if index in complete:
                continue
            log = (args.output / f"worker{index}.log").open("a", buffering=1)
            logs.append(log)
            print(f"Launching {args.mode} shard {index}: {log.name}", flush=True)
            children.append(subprocess.Popen(worker_command(args, index), env=child_environment(args),
                                              cwd=args.devkit_root, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        last_report = 0.
        while any(child.poll() is None for child in children):
            failed = [p for p in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Evaluation worker exited {failed[0].returncode}; see worker logs")
            if time.monotonic() - last_report >= 15:
                counts = progress_snapshot(args.output)
                print(f"{args.mode}: {sum(counts)}/{args.max_scenes or TOTAL}; shards={counts}", flush=True)
                last_report = time.monotonic()
            time.sleep(1)
        failures = [p.returncode for p in children if p.returncode != 0]
        if failures:
            raise RuntimeError(f"Evaluation workers failed: {failures}")
        report = merge_shards(args.output)
        expected = args.max_scenes or TOTAL
        if report["count"] != expected or report["valid_count"] != expected:
            raise ValueError("Merged evaluation does not cover every expected valid scene")
        publish_ready(args.output, report)
        state.update(status="complete", finished=time.time(), completed_shards=list(range(args.workers)))
        atomic_write(state_path, state)
        print_summary(args.output)
        return report
    except BaseException as exc:
        stop_children(children)
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}", updated=time.time())
        atomic_write(state_path, state)
        raise
    finally:
        for log in logs:
            log.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--mode", choices=("baseline", "refine"), default="baseline")
    for name in ("devkit-root", "checkpoint", "backbone", "vlm-cache", "data-root", "metric-cache", "output", "baseline", "score-checkpoint"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--gpu-policy", choices=("wait-idle", "shared"), default="wait-idle")
    parser.add_argument("--gpu-pool", default="auto")
    parser.add_argument("--gpu-budget-gb", type=float, default=12.)
    parser.add_argument("--gpu-reserve-gb", type=float, default=4.)
    parser.add_argument("--gpu-lock-dir", type=Path)
    parser.add_argument("--gpu-lease-held", action="store_true", help="Internal: validate and reuse the parent queue's active GPU lease")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--ks", nargs="+", type=int, default=KS)
    parser.add_argument("--gamma", type=float, default=.1)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--expected-epoch", type=int, default=500)
    parser.add_argument("--weight-kind", choices=("raw", "ema"), default="ema")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.gpu_lock_dir = (args.gpu_lock_dir or args.root / "dataset/navsim/exp/chainflow_gpu_leases").resolve()
    for key, default in default_paths(args.root, args.mode).items():
        setattr(args, key, (getattr(args, key) or default).resolve())
    if args.workers < 1 or args.expected_epoch < 1 or (args.max_scenes is not None and args.max_scenes < args.workers):
        parser.error("Require positive worker/epoch counts and at least one scene per worker")
    if args.max_scenes is not None and args.max_scenes > TOTAL:
        parser.error(f"navtest has only {TOTAL} canonical scenes")
    if not math.isfinite(args.gamma) or args.gamma < 0 or any(k < 1 for k in args.ks):
        parser.error("Require nonnegative finite gamma and positive refinement K values")
    if (not math.isfinite(args.gpu_budget_gb) or args.gpu_budget_gb <= 0
            or not math.isfinite(args.gpu_reserve_gb) or args.gpu_reserve_gb < 0):
        parser.error("Require positive finite GPU budget and nonnegative reserve")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        try:
            while True:
                state = args.output / "launcher.json"
                if state.exists():
                    print(json.dumps(read_json(state), indent=2), flush=True)
                counts = progress_snapshot(args.output)
                print(f"Completed records: {sum(counts)}/{args.max_scenes or TOTAL}; shards={counts}", flush=True)
                if (args.output / "summary.json").is_file():
                    print_summary(args.output)
                if not args.watch:
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            pass  # Only the monitor stops; running evaluation workers are untouched.


if __name__ == "__main__":
    main()
