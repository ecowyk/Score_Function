#!/usr/bin/env python3
"""Prepare and detach exactly one of the two authorized eight-GPU queues."""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from nuplan_environment_1025 import nuplan_paths


CODE = Path(__file__).resolve().parent.parent


def commands(role, root, output, setting, retry_failed):
    if role == "baselines":
        python = root / "envs/urbandriver/bin/python"
        script = CODE / "scripts/evaluate_random_baselines_1025.py"
        shared = ["--root", str(root), "--output", str(output)]
        prepare = [str(python), "-u", str(script), "prepare", *shared]
        run = [str(python), "-u", str(script), "run", *shared]
        if retry_failed:
            run.append("--retry-failed")
    else:
        python = root / "envs/diffusion_planner/bin/python"
        prepare = [str(python), "-u", str(CODE / "scripts/prepare_energy_aligned_1025.py"),
                   "--root", str(root), "--queue-dir", str(output), "--setting", setting]
        run = [str(python), "-u", str(CODE / "scripts/run_energy_aligned_1025.py"),
               "run", "--queue-dir", str(output)]
    return python, prepare, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("baselines", "energy"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--energy-setting", choices=("hard_NR", "random_NR"), default="hard_NR")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Baselines only: preserve failed attempt and retry its stage")
    args = parser.parse_args()
    if args.retry_failed and args.role != "baselines":
        parser.error("--retry-failed applies only to the baselines queue")
    root = args.root.expanduser().resolve()
    default = (root / "dataset/nuplan/exp/random_baselines_1025_v1" if args.role == "baselines"
               else root / "outputs/score_aligned_1004/E10_G_aligned_1025_v1")
    output = (args.output or default).expanduser().resolve()
    python, prepare, run = commands(args.role, root, output, args.energy_setting, args.retry_failed)
    if not python.is_file():
        raise FileNotFoundError(f"Required existing environment is missing: {python}")
    # Keep launch bookkeeping outside the queue: prepare rejects a partially
    # populated new queue, and launch files must not look like partial training.
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / (output.name + ".launcher.lock")
    metadata_path = output.parent / (output.name + ".launcher.json")
    log_path = output.parent / (output.name + ".queue.log")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"An existing launcher/queue holds {lock_path}; no duplicate was started.")
            if metadata_path.exists():
                print(metadata_path.read_text())
            return
        env = os.environ.copy()
        # Both preparation controllers resolve saved Hydra configs before
        # spawning GPU workers. Initialize their environment as well.
        env.update(nuplan_paths(root))
        env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                     "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "RAY_ADDRESS"):
            env.pop(name, None)
        for name in list(env):
            if name.startswith("TORCHELASTIC_"):
                env.pop(name, None)
        print(f"Preparing {args.role} on {socket.gethostname()}; GPUs 0–7; output {output}", flush=True)
        subprocess.run(prepare, cwd=CODE, env=env, check=True)
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(run, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        metadata = dict(role=args.role, hostname=socket.gethostname(), pid=process.pid,
                        output=str(output), log=str(log_path), command=run,
                        started_at=dt.datetime.now(dt.timezone.utc).isoformat())
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(metadata_path)
        time.sleep(1)
        code = process.poll()
        if code not in (None, 0):
            raise RuntimeError(f"Queue exited with code {code}; inspect {log_path}")
        print("Queue process detached; closing this terminal will not stop it.", flush=True)
        print(f"PID: {process.pid}\nProgress log: {log_path}\nOutput: {output}", flush=True)
        print("Check the log for GPU readiness, smoke completion and the current stage.", flush=True)


if __name__ == "__main__":
    main()
