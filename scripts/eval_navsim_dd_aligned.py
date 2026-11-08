#!/usr/bin/env python3
"""Evaluate the completed DD-aligned run, reusing the previous E200 K0 results."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
TOTAL = 12146
SHARDS = 8
LATEST = "dd_score_ddaligned_eval_latest.txt"


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def expected_config(epochs=100):
    config = read_json(REPO / "configs/navsim_dd_s10_g_ddaligned.json")
    if epochs == 500:
        config.update(epochs=500, lr_schedule_epochs=100,
                      checkpoint_every_epochs=50, checkpoint_start_epoch=150)
    elif epochs != 100:
        raise ValueError("Supported final checkpoints: 100 or 500 epochs")
    return config


def training_directory(root, epochs=100):
    expected_config(epochs)
    name = "dd_score_ddaligned_v1" if epochs == 100 else "dd_score_ddaligned_v1_e500"
    return root / "dataset/navsim/exp" / name


def latest_pointer(epochs=100):
    expected_config(epochs)
    return LATEST if epochs == 100 else "dd_score_ddaligned_e500_eval_latest.txt"


def validate_training(run, history, epochs=100):
    if len(history) != epochs or [row.get("epoch") for row in history] != list(range(1, epochs + 1)):
        raise ValueError(f"Training is incomplete: require all {epochs} completed epochs")
    config = {key: value for key, value in run.get("config", {}).items() if key != "num_workers"}
    expected = {key: value for key, value in expected_config(epochs).items() if key != "num_workers"}
    if config != expected:
        raise ValueError(f"Training config mismatch for the aligned {epochs}-epoch run")
    if (run.get("protocol", {}).get("world_size") != SHARDS
            or run.get("protocol", {}).get("device_type") != "cuda"
            or run.get("global_batch") != 512):
        raise ValueError("Require the aligned eight-GPU, global-batch-512 training run")


def validate_checkpoint(saved, epochs=100):
    expected = dict(format="navsim-dd-score-v1", epoch=epochs, weight_kind="raw",
                    sigma=0.1, gamma=0.1, training_precision="16-mixed", model=expected_config(epochs)["model"])
    for name, value in expected.items():
        if saved.get(name) != value:
            raise ValueError(f"Checkpoint {name}: expected {value!r}, got {saved.get(name)!r}")


def child_env(root):
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str, [REPO, root / "DiffusionDrive", root / "nuplan-devkit"])),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OPENSCENE_DATA_ROOT=str(root / "dataset/navsim"), NAVSIM_DEVKIT_ROOT=str(root / "DiffusionDrive"),
               NAVSIM_EXP_ROOT=str(root / "dataset/navsim/exp"), NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
               NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def preflight(root, epochs=100, baseline_pointer="dd_score_e200_eval_latest.txt"):
    exp = root / "dataset/navsim/exp"
    training = training_directory(root, epochs)
    checkpoint = training / "final_raw.pt"
    if not checkpoint.is_file() or not (training / "history.json").is_file():
        raise ValueError(f"Training is incomplete: require history epoch {epochs} and {checkpoint}")
    history = read_json(training / "history.json")
    run = read_json(training / "run.json")
    validate_training(run, history, epochs)
    pointer = exp / baseline_pointer
    if not pointer.is_file():
        raise ValueError(f"Missing completed evaluation pointer: {pointer}; K0 will not be recomputed")
    baseline = Path(pointer.read_text().strip())
    if not baseline.is_absolute() or read_json(baseline / "summary.json").get("count") != TOTAL:
        raise ValueError("K0 source must be an absolute, completed 12146-scene evaluation")
    if len(list(baseline.glob("part-*-of-*/complete.json"))) != SHARDS:
        raise ValueError("K0 source is missing its eight complete shard manifests")
    required = [root / "DiffusionDrive/checkpoints" / name for name in
                ("diffusiondrive_navsim_88p1_PDMS", "kmeans_navsim_traj_20.npy", "pytorch_model.bin")]
    required += [root / "dataset/nuplan/maps/nuplan-maps-v1.0.json", exp / "metric_cache",
                 root / "dataset/navsim/navsim_logs/test", root / "dataset/navsim/sensor_blobs/test",
                 root / "nuplan-devkit", root / "DiffusionDrive/navsim"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation dependency: {path}")
    if Path(sys.executable).resolve() != (root / "envs/diffusion_drive/bin/python").resolve():
        raise ValueError(f"Run with {root}/envs/diffusion_drive/bin/python")
    import torch
    validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True), epochs)
    if not torch.cuda.is_available() or torch.cuda.device_count() != SHARDS:
        raise ValueError("Require exactly eight visible CUDA GPUs; check CUDA_VISIBLE_DEVICES")
    return checkpoint, baseline


def worker_command(root, output, checkpoint, baseline, shard):
    weights = root / "DiffusionDrive/checkpoints"
    return [sys.executable, "-m", "score_function.navsim_dd.evaluate",
            "--checkpoint", str(weights / "diffusiondrive_navsim_88p1_PDMS"),
            "--anchor", str(weights / "kmeans_navsim_traj_20.npy"), "--backbone", str(weights / "pytorch_model.bin"),
            "--score-checkpoint", str(checkpoint), "--metric-cache", str(root / "dataset/navsim/exp/metric_cache"),
            "--data-root", str(root / "dataset/navsim"), "--devkit-root", str(root / "DiffusionDrive"),
            "--output", str(output), "--split", "navtest", "--ks", "20", "--gamma", "0.1",
            "--seed", "20260922", "--verify-native", "--reuse-k0-from", str(baseline),
            "--num-shards", str(SHARDS), "--shard-index", str(shard), "--device", f"cuda:{shard}"]


def progress_snapshot(output):
    counts = []
    for shard in range(SHARDS):
        directory = output / f"part-{shard:04d}-of-{SHARDS:04d}"
        count = 0
        for name in ("scenes.jsonl.tmp", "scenes.jsonl"):
            try:
                with (directory / name).open("rb") as stream:
                    count = sum(block.count(b"\n") for block in iter(lambda: stream.read(1024 * 1024), b""))
                break
            except FileNotFoundError:
                continue
        counts.append(count)
    return counts


def print_progress(output):
    counts = progress_snapshot(output)
    print(f"Scenes {sum(counts)}/{TOTAL}; shards " + ", ".join(map(str, counts)), flush=True)


def print_summary(output):
    summary = read_json(output / "summary.json")
    k0, k20 = (summary["by_k"][k] for k in ("0", "20"))
    a, b = k0["mean"]["score"] * 100, k20["mean"]["score"] * 100
    launcher = read_json(output / "launcher.json") if (output / "launcher.json").is_file() else {}
    epochs = launcher.get("epochs", 100)
    print(f"Aligned {epochs} epoch RAW: {summary['count']} scenes; reused K0 {a:.4f}; "
          f"new K20 {b:.4f}; delta {b-a:+.4f} points", flush=True)
    print(f"K20 improved {k20['improved']}, worsened {k20['worsened']}, tied {k20['tied']}", flush=True)
    comparisons = [("dd_score_final_eval_latest.txt", "Previous 100 epoch EMA"),
                   ("dd_score_e200_eval_latest.txt", "Previous 200 epoch EMA")]
    if epochs == 500:
        comparisons.append((LATEST, "Aligned 100 epoch RAW"))
    for pointer, label in comparisons:
        try:
            old_output = Path((output.parent / pointer).read_text().strip())
            if old_output.resolve() == output.resolve():
                continue
            old = read_json(old_output / "summary.json")
            if old["count"] != summary["count"]:
                continue
            x, y = (old["by_k"][k]["mean"]["score"] * 100 for k in ("0", "20"))
            print(f"{label}: K0 {x:.4f}, K20 {y:.4f}; new-minus-old means "
                  f"K0 {a-x:+.4f}, K20 {b-y:+.4f} points (means only, not a paired comparison)", flush=True)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return summary


def stop_children(children):
    live = [child for child in children if child.poll() is None]
    for child in live:
        try:
            child.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for child in live:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run(root, epochs=100):
    exp = root / "dataset/navsim/exp"
    exp.mkdir(parents=True, exist_ok=True)
    with (exp / "dd_score_ddaligned_eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another aligned evaluation runner is active") from None
        checkpoint, baseline = preflight(root, epochs)
        prefix = "dd_score_ddaligned_navtest_" if epochs == 100 else "dd_score_ddaligned_e500_navtest_"
        output = Path(tempfile.mkdtemp(prefix=prefix, dir=exp))
        state = dict(status="running", pid=os.getpid(), host=socket.gethostname(),
                     checkpoint=str(checkpoint), epochs=epochs, baseline=str(baseline), started=time.time())
        atomic_write(output / "launcher.json", json.dumps(state, indent=2) + "\n")
        atomic_write(exp / latest_pointer(epochs), str(output) + "\n")
        children, previous = [], {}
        def interrupted(signum, frame):
            raise InterruptedError(f"Received signal {signum}")
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, interrupted)
            print(f"Evaluation: {output}\nCheckpoint: {checkpoint}\nReused E200 K0: {baseline}", flush=True)
            env = child_env(root)
            active = {}
            for shard in range(SHARDS):
                with (output / f"shard-{shard}.log").open("w") as log:
                    child = subprocess.Popen(worker_command(root, output, checkpoint, baseline, shard),
                                             cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                children.append(child)
                active[shard] = child
            last_report = 0
            while active:
                for shard, child in list(active.items()):
                    code = child.poll()
                    if code is not None:
                        del active[shard]
                        print(f"Shard {shard} exited {code}; log: {output}/shard-{shard}.log", flush=True)
                        if code:
                            raise RuntimeError(f"Shard {shard} failed (exit {code}); merge skipped")
                if time.monotonic() - last_report >= 30:
                    print_progress(output)
                    last_report = time.monotonic()
                if active:
                    time.sleep(1)
            state["status"] = "merging"
            atomic_write(output / "launcher.json", json.dumps(state, indent=2) + "\n")
            merge = subprocess.Popen([sys.executable, "-m", "score_function.navsim_dd.evaluate",
                                      "--merge", "--output", str(output)], cwd=REPO, env=env)
            children.append(merge)
            if merge.wait() != 0:
                raise RuntimeError("Evaluation merge failed")
            if read_json(output / "summary.json").get("count") != TOTAL:
                raise ValueError(f"Merged evaluation must contain {TOTAL} scenes")
            print_summary(output)
            state["status"] = "complete"
        except BaseException as error:
            for signum in previous:
                signal.signal(signum, signal.SIG_IGN)
            stop_children(children)
            state.update(status="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                         error=str(error))
            raise
        finally:
            state["finished"] = time.time()
            atomic_write(output / "launcher.json", json.dumps(state, indent=2) + "\n")
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        return output


def status(root, epochs=100):
    exp = root / "dataset/navsim/exp"
    training = training_directory(root, epochs)
    history = read_json(training / "history.json") if (training / "history.json").is_file() else []
    print(f"Training epoch {history[-1]['epoch'] if history else 0}/{epochs}; "
          f"final_raw.pt: {'present' if (training / 'final_raw.pt').is_file() else 'absent'}")
    pointer = exp / latest_pointer(epochs)
    if not pointer.is_file():
        print("尚未启动评测")
        return
    output = Path(pointer.read_text().strip())
    print(f"Evaluation: {output}")
    if (output / "launcher.json").is_file():
        print(json.dumps(read_json(output / "launcher.json"), indent=2))
    print_progress(output)
    if (output / "summary.json").is_file():
        print_summary(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status"])
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--epochs", type=int, choices=[100, 500], default=100,
                        help="completed aligned training run whose final_raw.pt is evaluated")
    args = parser.parse_args()
    try:
        (run if args.action == "run" else status)(args.root.resolve(), args.epochs)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
