#!/usr/bin/env python3
"""Stop only the active Energy K50/K100 simulation; retain completed results."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import time


DEFAULT_QUEUE = Path("/home/data/wyk/outputs/score_aligned_1004/E10_G_aligned_1025_v1")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def process(pid):
    """Read process identity without trusting a PID from another host/session."""
    proc = Path("/proc") / str(pid)
    stat = proc.joinpath("stat").read_text().rpartition(")")[2].split()
    return dict(pid=pid, ppid=int(stat[1]), pgid=int(stat[2]), sid=int(stat[3]),
                start_ticks=int(stat[19]),
                command=[s.decode() for s in proc.joinpath("cmdline").read_bytes().split(b"\0") if s])


def same_file(left, right):
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def holds_file(pid, path):
    return any(same_file(fd, path) for fd in (Path("/proc") / str(pid) / "fd").iterdir())


def validate_command(queue, evaluation, record):
    """Bind the live shell command to this queue's exact Energy evaluation."""
    k = record.get("K")
    job = f"hard_NR_seed20260922_K{k}"
    require(k in (50, 100) and record.get("id") == job,
            "Refusing to stop a job other than this Energy K50/K100 evaluation")
    require(record.get("model") == "E10-G_seed20260922", "Unexpected Energy model")
    command = record.get("command", [])
    require(len(command) >= 4 and command[0] == "bash" and len(command) % 2 == 0,
            "Unrecognized simulation command")
    require(Path(command[1]).resolve() == Path(evaluation["code"]).resolve() / "sim_score_planner.sh",
            "Simulation script differs from the saved evaluation")
    options = dict(zip(command[2::2], command[3::2]))
    require(len(options) == len(command[2::2]), "Duplicate simulation command options")
    model = queue / "evaluation/models/E10-G_seed20260922"
    expected = {"--root": str(evaluation["root"]), "--steps": str(k), "--gamma": "0.1",
                "--gpus": "0,1,2,3,4,5,6,7", "--gpu-fraction": "1.0",
                "--split": "test14-hard", "--challenge": "closed_loop_nonreactive_agents"}
    require(all(options.get(key) == value for key, value in expected.items()),
            "Simulation settings differ from the intended Energy evaluation")
    for flag, name in (("--config", "config.json"), ("--checkpoint", "final_ema.pt")):
        require(flag in options and Path(options[flag]).resolve() == (model / name).resolve(),
                f"{flag} does not belong to this Energy queue")
    require(options.get("--name", "").startswith(f"E10G_{job}_"), "Unexpected simulation label")
    log = Path(record["log"])
    require(log.parent.resolve() == (queue / "evaluation/logs").resolve(),
            "Simulation log does not belong to this Energy queue")
    return command


def pipeline_idle(queue):
    with (queue / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def stop(queue, wait_seconds=20):
    queue = Path(queue).resolve()
    status = read(queue / "evaluation/status.json")
    if status.get("state") in ("complete", "failed", "interrupted") and pipeline_idle(queue):
        print(f"Energy evaluation is already stopped ({status['state']}); nothing was signalled.", flush=True)
        return
    require(status.get("state") == "running", "Energy is not in a running simulation; no process was stopped")
    record = read(queue / "evaluation/jobs" / (status["job"] + ".json"))
    require(record.get("state") == "running" and record.get("pid") == status.get("pid"),
            "Job changed while reading status; rerun this stop command")
    pipeline = read(queue / "pipeline_status.json")
    require(not pipeline.get("hostname") or pipeline["hostname"] == socket.gethostname(),
            "This queue's controller is on another node; no process was stopped")
    command = validate_command(queue, read(queue / "evaluation/plan.json"), record)
    pid = int(record["pid"])
    current = process(pid)
    require(current["command"] == command and current["pgid"] == pid and current["sid"] == pid,
            "Live PID is not the recorded simulation session; no process was stopped")
    require(same_file(Path("/proc") / str(pid) / "fd/1", record["log"]),
            "Live process output differs from the saved simulation log")
    parent = process(current["ppid"])
    require(parent["command"] and "python" in Path(parent["command"][0]).name
            and holds_file(parent["pid"], queue / ".pipeline.lock"),
            "Parent is not this Energy queue controller; no process was stopped")
    require(process(pid) == current, "Simulation process identity changed; rerun this stop command")
    latest = read(queue / "evaluation/status.json")
    require(latest.get("job") == status["job"] and latest.get("pid") == pid,
            "The current job changed; rerun this stop command")

    print(f"Stopping {status['job']} on {socket.gethostname()} (simulation group {pid})...", flush=True)
    os.killpg(pid, signal.SIGTERM)
    # The existing runner treats nonzero simulation exit as fatal, so it will
    # release the pipeline lock and cannot advance to K100. Do not interrupt its
    # exception handler: it preserves the partial result and refreshes summaries.
    deadline = time.monotonic() + wait_seconds
    while not pipeline_idle(queue):
        if time.monotonic() >= deadline:
            raise RuntimeError("Stop signal sent; the controller is still cleaning up. Do not start a second GPU job yet.")
        time.sleep(.5)
    receipt = dict(hostname=socket.gethostname(), job=status["job"], pid=pid,
                   start_ticks=current["start_ticks"], stopped_unix=time.time(),
                   reason="User stopped K50/K100; retain completed K5/K10/K20")
    path = queue / "evaluation/user_stop_1028.json"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)
    print("Energy controller stopped. Completed K5/K10/K20 and the partial run are preserved; K100 will not be launched.", flush=True)


def stop_evaluation(queue: Path, wait_seconds=20):
    """Entry point for the follow-on baseline launcher."""
    return stop(queue, wait_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--wait-seconds", type=float, default=20)
    args = parser.parse_args()
    stop(args.queue_dir, args.wait_seconds)


if __name__ == "__main__":
    main()
