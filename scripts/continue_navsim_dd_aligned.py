#!/usr/bin/env python3
"""Continue the completed aligned 100-epoch run to 250 at the original LR floor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SOURCE = "dd_score_ddaligned_v1"
TARGET = "dd_score_ddaligned_v1_e250"


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


def preflight(root, cache):
    import torch
    from score_function.navsim_dd.cache import load_cache_metadata
    from score_function.navsim_dd.continuation import plan_continuation
    from score_function.utils.train_utils import file_hash, fingerprint, load_tensor

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
    plan = plan_continuation(source, cache, additional_epochs=150)
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
                or not 100 <= len(last["history"]) <= 250
                or last["history"][:100] != plan["last"]["history"]):
            raise ValueError("Existing continuation has an incompatible recovery checkpoint")
        epoch = len(last["history"])
        if last["step"] != epoch * run["updates_per_epoch"] or last.get("scaler") is None:
            raise ValueError("Existing continuation checkpoint is missing complete step/scaler state")
    else:
        epoch = 100
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError("Require exactly eight visible CUDA GPUs")
    for index in range(8):
        if torch.ones(1, device=f"cuda:{index}").sum().item() != 1:
            raise ValueError(f"CUDA device {index} check failed")
    print(f"CHECK OK: completed epoch {epoch}/250; remaining {250-epoch}; "
          "global batch 512; FP16; RAW; epochs 101–250 LR=1e-6", flush=True)
    return epoch


def launch(root, check=False):
    source, output = locations(root)
    cache = select_cache(root)
    # Set CPU thread budgets before importing torch in preflight.
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with output.with_name(output.name + ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("The 250-epoch continuation launcher is already active") from None
        print(f"Source: {source}\nOutput: {output}\nCache: {cache}", flush=True)
        preflight(root, cache)
        if check:
            return
        if not output.exists():
            from score_function.navsim_dd.continuation import prepare_continuation
            prepare_continuation(source, output, cache, additional_epochs=150)
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join(map(str, [REPO, root / "DiffusionDrive", root / "nuplan-devkit"])),
                   PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        command = [str(root / "envs/diffusion_drive/bin/python"), "-u", "-m", "torch.distributed.run",
                   "--standalone", "--nproc_per_node=8", "-m", "score_function.navsim_dd.train",
                   "--cache-dir", str(cache), "--output", str(output),
                   "--config", str(output / "config.json"), "--resume"]
        # Keep the launcher lock across exec for the torchrun parent's lifetime.
        os.set_inheritable(lock.fileno(), True)
        os.chdir(REPO)
        print("Restoring full last.pt state; source 100-epoch run is preserved.", flush=True)
        os.execvpe(command[0], command, env)


def status(root):
    source, output = locations(root)
    if not (output / "history.json").is_file():
        print(f"Continuation not prepared: {output}\nSource: {source}")
        return
    history = read_json(output / "history.json")
    last = history[-1]
    print(f"Output: {output}\nCompleted: {last['epoch']}/250; "
          f"additional completed: {last['epoch']-100}/150")
    print(f"train DSM {last['train_dsm']:.6f}; val DSM {last['val_dsm']:.6f}; lr {last['lr']:.9g}")
    print(f"final_raw.pt: {'present' if (output / 'final_raw.pt').is_file() else 'not yet generated'}")


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
