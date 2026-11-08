#!/usr/bin/env python3
"""Continue the aligned 100-epoch run to 500, keeping a RAW snapshot every 50 epochs."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SOURCE = "dd_score_ddaligned_v1"
PREVIOUS = "dd_score_ddaligned_v1_e250"
TARGET = "dd_score_ddaligned_v1_e500"


def read_json(path):
    return json.loads(path.read_text())


def locations(root):
    exp = root / "dataset/navsim/exp"
    return exp / SOURCE, exp / TARGET


def select_cache(root):
    explicit = os.environ.get("SCORE_CACHE_DIR")
    if explicit:
        cache = Path(explicit)
        if not cache.is_absolute():
            raise ValueError("SCORE_CACHE_DIR must be an absolute path")
        return cache
    local = Path("/dev/shm/wyk_dd_score_cache_v1")
    return local if (local / "TRAINING_READY.json").is_file() else root / "dataset/navsim/exp/dd_score_cache_v1"


def is_previous_trainer(argv, output, cwd):
    """Match only the exact old torchrun parent, never a worker or another experiment."""
    if not argv or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(argv[0]).name):
        return False
    invocation = argv[1:]
    while invocation and invocation[0] in ("-u", "-B", "-E", "-s", "-S", "-I"):
        invocation = invocation[1:]
    if invocation[:2] != ["-m", "torch.distributed.run"]:
        return False
    modules = [index for index in range(2, len(invocation)) if invocation[index] in ("-m", "--module")]
    if (len(modules) != 1 or modules[0] + 1 >= len(invocation)
            or invocation[modules[0] + 1] != "score_function.navsim_dd.train"):
        return False
    arguments = invocation[modules[0] + 2:]
    destinations = []
    for index, word in enumerate(arguments):
        if word == "--output":
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("-"):
                return False
            destinations.append(arguments[index + 1])
        elif word.startswith("--output="):
            destinations.append(word.split("=", 1)[1])
    if len(destinations) != 1 or not destinations[0]:
        return False
    path = Path(destinations[0])
    return (path if path.is_absolute() else cwd / path).resolve() == output.resolve()


def previous_parents(output):
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            argv = [os.fsdecode(part) for part in (process / "cmdline").read_bytes().split(b"\0") if part]
            if is_previous_trainer(argv, output, (process / "cwd").resolve()):
                matches.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return matches


def terminate_previous(pid, output):
    # A pidfd targets the same process even if its numeric PID is later reused.
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError(f"Stop the old torchrun PID {pid} before retrying; pidfd is unavailable")
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        if pid not in previous_parents(output):
            return
        print(f"Stopping previous 250-epoch torchrun PID {pid}; restoring its latest complete epoch.", flush=True)
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            pass
    finally:
        os.close(descriptor)


@contextmanager
def previous_guard(output, replace):
    """Stop only the superseded local run and prevent it restarting during this run."""
    with output.with_name(output.name + ".launcher.lock").open("a") as lock:
        deadline, sent = time.monotonic() + 60, set()
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                acquired = False
            parents = previous_parents(output)
            if acquired and not parents:
                yield lock
                return
            if not replace:
                raise RuntimeError("The old 250-epoch job is active; run will stop it and keep its completed progress")
            if len(parents) > 1:
                raise RuntimeError("Multiple old torchrun parents found; no process was signalled")
            for pid in parents:
                if pid not in sent:
                    terminate_previous(pid, output)
                    sent.add(pid)
            if time.monotonic() >= deadline:
                raise RuntimeError("Old job/lock is still active; no new training started. Check the old node/job.")
            time.sleep(1)


def check_environment(root, cache):
    import torch
    from score_function.navsim_dd.cache import load_cache_metadata
    from score_function.navsim_dd.continuation import plan_continuation
    from score_function.utils.train_utils import file_hash, fingerprint

    python = root / "envs/diffusion_drive/bin/python"
    if Path(sys.executable).resolve() != python.resolve():
        raise ValueError(f"Use the existing environment: {python}")
    source, output = locations(root)
    original = read_json(source / "run.json")
    expected = read_json(REPO / "configs/navsim_dd_s10_g_ddaligned.json")
    original_config = original["config"]
    if {k: v for k, v in original_config.items() if k != "num_workers"} != {
            k: v for k, v in expected.items() if k != "num_workers"}:
        raise ValueError("Source must be the original DD-aligned 100-epoch configuration")
    if original["protocol"]["world_size"] != 8 or original["protocol"]["device_type"] != "cuda":
        raise ValueError("Source must use the original eight-GPU protocol")
    if original["global_batch"] != 512:
        raise ValueError("Source global batch must be 512")
    ready = read_json(cache / "TRAINING_READY.json")
    metadata = load_cache_metadata(cache)
    if (ready.get("validation_version") != 1
            or ready.get("manifest_sha256") != file_hash(cache / "manifest.json")
            or ready.get("cache_fingerprint") != fingerprint(metadata)
            or metadata["identity"].get("max_per_split") is not None
            or sum(ready.get("records", {}).values()) != 103288):
        raise ValueError("Cache is not the validated full NAVSIM training cache")
    # Validate the full original checkpoint before stopping any previous job.
    plan_continuation(source, cache, additional_epochs=400)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError("Require exactly eight visible CUDA GPUs")
    for index in range(8):
        if torch.ones(1, device=f"cuda:{index}").sum().item() != 1:
            raise ValueError(f"CUDA device {index} check failed")


def preflight(root, cache):
    from score_function.navsim_dd.continuation import plan_retarget_continuation
    from score_function.utils.train_utils import load_tensor

    source, output = locations(root)
    previous = output.parent / PREVIOUS
    if output.exists():
        restore = Path(read_json(output / "continuation.json")["restore_output"])
        if restore not in (source.resolve(), previous.resolve()):
            raise ValueError("Existing continuation has an unexpected restore source")
        previous = None if restore == source.resolve() else previous
    elif not previous.exists():
        previous = None
    if previous is not None and read_json(previous / "run.json")["config"]["epochs"] != 250:
        raise ValueError("Previous output is not the expected 250-epoch continuation")
    plan = plan_retarget_continuation(source, previous, cache, target_epochs=500,
                                     checkpoint_every_epochs=50, checkpoint_start_epoch=150)
    if output.exists():
        run = read_json(output / "run.json")
        if (read_json(output / "continuation.json") != plan["provenance"]
                or run.get("continuation") != plan["provenance"]
                or run.get("config") != plan["config"]
                or run.get("signature") != plan["run"]["signature"]
                or read_json(output / "config.json") != plan["config"]):
            raise ValueError("Existing output is not this exact continuation; nothing overwritten")
        last = load_tensor(output / "last.pt")
        if (last.get("format") != "navsim-dd-train-v1"
                or last.get("signature") != run["signature"]
                or not len(plan["last"]["history"]) <= len(last["history"]) <= 500
                or last["history"][:len(plan["last"]["history"])] != plan["last"]["history"]):
            raise ValueError("Existing continuation has an incompatible recovery checkpoint")
        epoch = len(last["history"])
        if last["step"] != epoch * run["updates_per_epoch"] or last.get("scaler") is None:
            raise ValueError("Existing continuation checkpoint is missing complete step/scaler state")
    else:
        epoch = len(plan["last"]["history"])
    print(f"CHECK OK: completed epoch {epoch}/500; remaining {500-epoch}; "
          "global batch 512; FP16; RAW; epochs 101–500 LR=1e-6; snapshot every 50 epochs", flush=True)
    if not output.exists() and epoch > 150:
        print("Earlier snapshot epochs cannot be reconstructed if the old run did not save their weights.", flush=True)
    return previous


def launch(root, check=False):
    source, output = locations(root)
    cache = select_cache(root)
    # Set CPU thread budgets before importing torch in preflight.
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with output.with_name(output.name + ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("The 500-epoch continuation launcher is already active") from None
        print(f"Source: {source}\nOutput: {output}\nCache: {cache}", flush=True)
        check_environment(root, cache)
        with previous_guard(output.parent / PREVIOUS, replace=not check) as previous_lock:
            previous = preflight(root, cache)
            if check:
                return
            if not output.exists():
                from score_function.navsim_dd.continuation import prepare_retarget_continuation
                prepare_retarget_continuation(source, previous, output, cache, target_epochs=500,
                                              checkpoint_every_epochs=50, checkpoint_start_epoch=150)
            env = os.environ.copy()
            env.update(PYTHONPATH=os.pathsep.join(map(str, [REPO, root / "DiffusionDrive", root / "nuplan-devkit"])),
                       PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
            command = [str(root / "envs/diffusion_drive/bin/python"), "-u", "-m", "torch.distributed.run",
                       "--standalone", "--nproc_per_node=8", "-m", "score_function.navsim_dd.train",
                       "--cache-dir", str(cache), "--output", str(output),
                       "--config", str(output / "config.json"), "--resume"]
            # Keep both launcher locks across exec for the torchrun parent's lifetime.
            for handle in (lock, previous_lock):
                os.set_inheritable(handle.fileno(), True)
            os.chdir(REPO)
            print("Restoring full last.pt state; source runs are preserved.", flush=True)
            os.execvpe(command[0], command, env)


def status(root):
    source, output = locations(root)
    if not (output / "history.json").is_file():
        print(f"Continuation not prepared: {output}\nSource: {source}")
        return
    history = read_json(output / "history.json")
    last = history[-1]
    print(f"Output: {output}\nCompleted: {last['epoch']}/500; "
          f"additional completed: {last['epoch']-100}/400")
    print(f"train DSM {last['train_dsm']:.6f}; val DSM {last['val_dsm']:.6f}; lr {last['lr']:.9g}")
    print(f"final_raw.pt: {'present' if (output / 'final_raw.pt').is_file() else 'not yet generated'}")
    print("Saved snapshots: " + ", ".join(path.name for path in sorted((output / "checkpoints").glob("epoch_*_raw.pt"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "check", "status"])
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        if args.action == "status":
            status(root)
        else:
            launch(root, check=args.action == "check")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
