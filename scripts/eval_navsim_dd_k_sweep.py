#!/usr/bin/env python3
"""Vary only refinement K for the aligned 500-epoch final RAW checkpoint."""
import argparse
import fcntl
import json
from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

import eval_navsim_dd_aligned as base

sys.path.insert(0, str(base.REPO))
from score_function.navsim_dd.k_sweep import merge_comparison, print_comparison, validate_reference

LATEST = "dd_score_ddaligned_e500_k_sweep_latest.txt"
DEFAULT_KS = (5, 10, 50, 100)


def checked_ks(ks):
    if not ks or any(type(k) is not int or k <= 0 or k == 20 for k in ks):
        raise ValueError("Supply positive K values other than 20; K0 and K20 are reused")
    return sorted(set(ks))


def worker_command(root, output, checkpoint, reference, shard, ks):
    command = base.worker_command(root, output, checkpoint, reference, shard)
    start, end = command.index("--ks") + 1, command.index("--gamma")
    command[start:end] = [str(k) for k in checked_ks(ks)]
    return command


def run(root, ks=DEFAULT_KS):
    ks = checked_ks(ks)
    exp = root / "dataset/navsim/exp"
    exp.mkdir(parents=True, exist_ok=True)
    # Share the lock with the existing eight-GPU evaluator, without changing its outputs.
    with (exp / "dd_score_ddaligned_eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another aligned evaluation runner is active") from None
        checkpoint, reference = base.preflight(root, epochs=500, baseline_pointer=base.latest_pointer(500))
        reference_info = validate_reference(reference, checkpoint)
        output = Path(tempfile.mkdtemp(prefix="dd_score_ddaligned_e500_k_sweep_", dir=exp))
        state = dict(status="running", pid=os.getpid(), host=socket.gethostname(),
                     checkpoint=str(checkpoint), epochs=500, reference=str(reference),
                     reference_validation=reference_info, computed_ks=ks, reused_ks=[0, 20],
                     gamma=0.1, sigma=0.1, seed=20260922, started=time.time())
        def save():
            base.atomic_write(output / "launcher.json", json.dumps(state, indent=2) + "\n")
        save()
        base.atomic_write(exp / LATEST, str(output) + "\n")
        children, previous = [], {}
        def interrupted(signum, frame):
            raise InterruptedError(f"Received signal {signum}")
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, interrupted)
            print(f"K sweep: {output}\nCheckpoint: {checkpoint}\nReused K0/K20: {reference}", flush=True)
            print(f"Compute K={ks}; one shared path through K={max(ks)} per scene; gamma=sigma=0.1", flush=True)
            env, active = base.child_env(root), {}
            for shard in range(base.SHARDS):
                with (output / f"shard-{shard}.log").open("w") as log:
                    child = subprocess.Popen(worker_command(root, output, checkpoint, reference, shard, ks),
                                             cwd=base.REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
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
                    base.print_progress(output)
                    last_report = time.monotonic()
                if active:
                    time.sleep(1)
            state["status"] = "merging"
            save()
            merge = subprocess.Popen([sys.executable, "-m", "score_function.navsim_dd.evaluate",
                                      "--merge", "--output", str(output)], cwd=base.REPO, env=env)
            children.append(merge)
            if merge.wait() != 0:
                raise RuntimeError("Evaluation merge failed")
            if validate_reference(reference, checkpoint) != reference_info:
                raise ValueError("Reference evaluation or checkpoint changed during the K sweep")
            merge_comparison(output, reference)
            print_comparison(output)
            state["status"] = "complete"
        except BaseException as error:
            for signum in previous:
                signal.signal(signum, signal.SIG_IGN)
            base.stop_children(children)
            state.update(status="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed", error=str(error))
            raise
        finally:
            state["finished"] = time.time()
            save()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        return output


def status(root):
    pointer = root / "dataset/navsim/exp" / LATEST
    if not pointer.is_file():
        print("尚未启动 K 对比评测")
        return
    output = Path(pointer.read_text().strip())
    print(f"K sweep: {output}")
    state = base.read_json(output / "launcher.json")
    print(f"State: {state['status']}; computed K={state['computed_ks']}; reused K0/K20")
    if state.get("error"):
        print(state["error"])
    base.print_progress(output)
    if (output / "comparison.json").is_file():
        print_comparison(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status"])
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    args = parser.parse_args()
    try:
        if args.action == "run":
            run(args.root.resolve(), args.ks)
        else:
            status(args.root.resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
