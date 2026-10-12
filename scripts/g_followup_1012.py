"""Continue the frozen Test14-Hard experiment with G K=15,30,150,250.

The original suite, model, noise seed and all completed runs stay unchanged.
This adapter reuses the original suite's runner and result validation. Its only
runtime override is jobs(), used by the original status/CSV display helpers.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import full_test14hard_suite as suite

KS = (15, 30, 150, 250)
OLD_NAME = "score_full_k_1009_v1"
NEW_NAME = "score_G_local_k_1012_v1"
STATES_DONE = {"complete", "reused"}


def jobs():
    return [dict(id=f"S05-G_K{k}", model="S05-G", K=k) for k in KS]


@contextmanager
def exclusive(path):
    with Path(path).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Queue still owns {path}; do not launch another evaluation"
            ) from exc
        yield


def proc(pid):
    """Read Linux process identity, without touching unrelated processes."""
    try:
        base = Path("/proc") / str(pid)
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        # /proc may be mounted from a parent PID namespace. Translate to the
        # caller's namespace before using pidfd_open; never signal a /proc ID
        # on the assumption it is also a valid local PID.
        own_ids = next(
            line.split()[1:]
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("NSpid:")
        )
        depth = own_ids.index(str(os.getpid()))
        target_ids = next(
            line.split()[1:]
            for line in (base / "status").read_text().splitlines()
            if line.startswith("NSpid:")
        )
        local_pid = int(target_ids[depth])
        argv = [
            s.decode(errors="replace") for s in (base / "cmdline").read_bytes().split(b"\0") if s
        ]
        return dict(
            pid=pid,
            signal_pid=local_pid,
            ppid=int(fields[1]),
            start=fields[19],
            state=fields[0],
            argv=argv,
        )
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def same_alive(identity):
    current = proc(identity["pid"])
    return bool(current and current["start"] == identity["start"] and current["state"] != "Z")


def signal_owned(identity, signum):
    # pidfd targets the captured process even if the numeric PID is later reused.
    if not same_alive(identity):
        return
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            fd = os.pidfd_open(identity["signal_pid"])
        except ProcessLookupError:
            return
        try:
            if same_alive(identity):
                signal.pidfd_send_signal(fd, signum)
        finally:
            os.close(fd)
    else:
        raise RuntimeError("Safe stopping requires Linux pidfd support; use tmux Ctrl+C")


def targets_old(identity, old):
    argv = identity["argv"]
    source = str(Path(suite.__file__).resolve())
    if source not in argv:
        return False
    tail = argv[argv.index(source) + 1 :]
    if not tail or tail[0] != "run":
        return False
    if "--output" in tail:
        return Path(tail[tail.index("--output") + 1]).resolve() == old.resolve()
    root = Path(tail[tail.index("--root") + 1]) if "--root" in tail else Path("/home/data/wyk")
    return (root / "dataset/nuplan/exp" / OLD_NAME).resolve() == old.resolve()


def wait_gone(identities, seconds):
    deadline = time.monotonic() + seconds
    while any(same_alive(p) for p in identities) and time.monotonic() < deadline:
        time.sleep(0.25)
    return [p for p in identities if same_alive(p)]


def pause_old(args):
    old = args.source.resolve()
    processes = [proc(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()]
    processes = {p["pid"]: p for p in processes if p}
    coordinators = [p for p in processes.values() if targets_old(p, old)]
    if len(coordinators) > 1:
        raise RuntimeError("Multiple old coordinators found; refusing an ambiguous stop")
    if not coordinators:
        with exclusive(old / ".run.lock"):
            state = suite.read(old / "status.json")
            if state.get("state") == "running":
                raise RuntimeError(
                    "Old status says running but its coordinator was not found; inspect queue.log"
                )
        print("Old queue is already idle; no process was signalled.", flush=True)
        return
    coordinator = coordinators[0]
    print(
        f"Stopping old coordinator PID={coordinator['signal_pid']}; retaining completed results.",
        flush=True,
    )
    # Briefly suspend the coordinator while capturing its process tree, so it
    # cannot launch the next job between the snapshot and the stop signal.
    signal_owned(coordinator, signal.SIGSTOP)
    try:
        processes = [proc(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()]
        processes = {p["pid"]: p for p in processes if p}
        owned = {coordinator["pid"]}
        while True:
            expanded = owned | {p["pid"] for p in processes.values() if p["ppid"] in owned}
            if expanded == owned:
                break
            owned = expanded
        descendants = [processes[pid] for pid in owned if pid != coordinator["pid"]]
        signal_owned(coordinator, signal.SIGTERM)
    finally:
        signal_owned(coordinator, signal.SIGCONT)
    if wait_gone([coordinator], 45):
        raise RuntimeError("Coordinator has not stopped; inspect it before retrying")
    # Ray can start separate process groups. Clean up only previously captured
    # descendants, with start-time checks and pidfds, never all Ray/Python jobs.
    survivors = wait_gone(descendants, 5)
    for child in survivors:
        signal_owned(child, signal.SIGTERM)
    survivors = wait_gone(survivors, 10)
    for child in survivors:
        signal_owned(child, signal.SIGKILL)
    if wait_gone(survivors, 5):
        raise RuntimeError("Some old workers have not exited; do not start the new queue yet")
    with exclusive(old / ".run.lock"):
        state = suite.read(old / "status.json")
        if state.get("state") == "running":
            # SIGTERM may land between simulations, outside the old runner's
            # poll-loop cleanup. Keep completed records intact in that case.
            if state.get("job"):
                path = old / "jobs" / (state["job"] + ".json")
                record = suite.read(path) if path.exists() else {}
                if record.get("state") == "running":
                    record.update(state="interrupted", error="Paused between coordinator stages")
                    suite.write(path, record)
            suite.write(
                old / "status.json",
                dict(state="interrupted", job=state.get("job"), updated_unix=time.time()),
            )
    print("PAUSED: old coordinator/workers stopped. Completed runs are preserved.", flush=True)


def prepare(args):
    old, out = args.source.resolve(), args.output.resolve()
    if old == out:
        raise ValueError("A separate output directory is required")
    original = suite.read(old / "plan.json")
    suite.verify(old, original)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("models", "code"):
        link, target = out / name, old / name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise ValueError(f"Unexpected existing path: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    if (out / "plan.json").exists():
        plan = suite.read(out / "plan.json")
        if plan.get("source_suite") != str(old) or plan["jobs"] != jobs():
            raise ValueError("Existing follow-up plan belongs to another experiment")
        for key in ("root", "tokens", "gamma", "heading_projection", "reference_seed"):
            if plan[key] != original[key]:
                raise ValueError(f"Follow-up protocol changed: {key}")
        suite.verify(out, plan)
        print("Follow-up inputs verified; completed follow-up jobs will be skipped.", flush=True)
        return
    plan = copy.deepcopy(original)
    plan.update(jobs=jobs(), source_suite=str(old), created_unix=time.time())
    for path in (old / "plan.json", Path(__file__).resolve()):
        plan["files"][str(path)] = suite.digest(path)
    suite.write(out / "plan.json", plan)
    suite.write(out / "status.json", dict(state="prepared", updated_unix=time.time()))
    print(
        f"READY: G K={KS}; 8 GPUs per run; full Test14-Hard; seed={plan['reference_seed']}",
        flush=True,
    )


def scenario_detail(frame, tokens):
    import numpy as np

    detail = frame.loc[frame.log_name.notna()].copy()
    for column in ("log_name", "scenario"):
        detail[column] = detail[column].astype(str)
    if len(detail) != len(tokens) or set(detail.scenario) != set(tokens):
        raise ValueError("Scenario coverage differs from the frozen test set")
    if detail.duplicated(["log_name", "scenario"]).any():
        raise ValueError("Duplicate scenario keys")
    if not np.isfinite(detail[list(suite.METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite per-scenario metrics")
    return detail[["log_name", "scenario", "scenario_type", *suite.METRICS]]


def paired(before, after):
    keys = ["log_name", "scenario"]
    merged = before.merge(
        after,
        on=keys,
        how="outer",
        suffixes=("_before", "_after"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Paired runs have different log/token keys")
    if not merged.scenario_type_before.eq(merged.scenario_type_after).all():
        raise ValueError("Paired scenario types differ")
    merged["scenario_type"] = merged.pop("scenario_type_before")
    merged.drop(columns=["scenario_type_after", "_merge"], inplace=True)
    for metric in suite.METRICS:
        merged[metric + "_delta"] = merged[metric + "_after"] - merged[metric + "_before"]
    merged["score_delta_points"] = 100 * merged.score_delta
    return merged


def analyze(args):
    import pandas as pd

    old, out = args.source.resolve(), args.output.resolve()
    plan = suite.read(old / "plan.json")
    suite.verify(old, plan)
    os.environ.update(suite.env_for(Path(plan["root"]), old / "code/lg", ""))
    selection = [(old, dict(id="DP_K0", model="S05-G", K=0))]
    selection += [
        (old, dict(id=f"S05-G_K{k}", model="S05-G", K=k)) for k in (5, 10, 20, 50, 100, 200)
    ]
    selection += [(out, j) for j in jobs()]
    frames, records, summaries = {}, {}, []
    for origin, job in selection:
        path = origin / "jobs" / (job["id"] + ".json")
        record = suite.read(path) if path.exists() else {}
        if record.get("state") not in STATES_DONE:
            if origin == old:
                raise ValueError(f"Required existing result is not complete: {job['id']}")
            continue
        run = Path(record["result_dir"])
        metrics = suite.validate_result(run, job, plan, old)
        table = pd.read_parquet(next((run / "aggregator_metric").glob("*.parquet")))
        frames[job["id"]] = scenario_detail(table, plan["tokens"])
        records[job["id"]] = dict(record, metrics=metrics)
        summaries.append(
            dict(
                id=job["id"],
                K=job["K"],
                score_points=100 * metrics["score"],
                result_dir=str(run),
                nuboard_files=";".join(map(str, sorted(run.glob("*.nuboard")))),
            )
        )
    summary = pd.DataFrame(summaries).sort_values("K")
    baseline = records["DP_K0"]["metrics"]["score"]
    summary["delta_vs_DP_points"] = summary.score_points - 100 * baseline
    comparisons = [("DP_K0", name) for name in summary.id if name != "DP_K0"]
    comparisons += [
        ("S05-G_K5", "S05-G_K20"),
        ("S05-G_K20", "S05-G_K50"),
        ("S05-G_K50", "S05-G_K200"),
    ]
    rows, metric_rows, details, top = [], [], [], []
    for before, after in comparisons:
        table = paired(frames[before], frames[after])
        label = f"{before} -> {after}"
        table.insert(0, "comparison", label)
        for side, name in (("before", before), ("after", after)):
            run = Path(records[name]["result_dir"])
            table[side + "_result_dir"] = str(run)
            table[side + "_nuboard"] = ";".join(map(str, sorted(run.glob("*.nuboard"))))
        delta = table.score_delta
        rows.append(
            dict(
                comparison=label,
                n=len(table),
                improved=int((delta > 1e-8).sum()),
                tied=int((delta.abs() <= 1e-8).sum()),
                worsened=int((delta < -1e-8).sum()),
                official_delta_points=100
                * (records[after]["metrics"]["score"] - records[before]["metrics"]["score"]),
                scene_mean_delta_points=100 * delta.mean(),
            )
        )
        for metric in suite.METRICS:
            difference = table[metric + "_delta"]
            metric_rows.append(
                dict(
                    comparison=label,
                    metric=metric,
                    improved=int((difference > 1e-8).sum()),
                    worsened=int((difference < -1e-8).sum()),
                    mean_delta_raw=difference.mean(),
                )
            )
        details.append(table)
        for kind, subset in (
            ("largest_improvements", table.loc[delta > 1e-8].nlargest(10, "score_delta")),
            ("largest_regressions", table.loc[delta < -1e-8].nsmallest(10, "score_delta")),
        ):
            subset = subset.copy()
            subset.insert(1, "change_group", kind)
            top.append(subset)
    report = out / "analysis"
    report.mkdir(parents=True, exist_ok=True)
    pair_summary = pd.DataFrame(rows)
    for filename, frame in (
        ("result_summary.csv", summary),
        ("paired_summary.csv", pair_summary),
        ("metric_changes.csv", pd.DataFrame(metric_rows)),
        ("paired_scenes.csv", pd.concat(details, ignore_index=True)),
        ("top_changed.csv", pd.concat(top, ignore_index=True)),
    ):
        temp = report / (filename + ".tmp")
        frame.to_csv(temp, index=False)
        temp.replace(report / filename)
    print(summary[["id", "score_points", "delta_vs_DP_points"]].to_string(index=False), flush=True)
    print("\nPaired scene counts (positive score change = improvement):", flush=True)
    print(pair_summary.to_string(index=False), flush=True)
    print(
        f"\nSaved: {report}\nOfficial aggregate differences and plain scene means are reported separately.\n"
        "Metric changes describe co-occurring changes, not causal attribution.",
        flush=True,
    )


def run(args):
    # Retain the original queue lock for the WHOLE follow-up, so restarting the
    # old queue cannot unexpectedly launch a competing eight-GPU evaluation.
    with exclusive(args.source / ".run.lock"):
        old_state = suite.read(args.source / "status.json")
        if old_state.get("state") == "running":
            raise RuntimeError("Old queue still says running; run pause-old first")
        prepare(args)
        suite.jobs = jobs
        suite.run(args)
    analyze(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pause-old", "prepare", "run", "status", "analyze"))
    base = Path("/home/data/wyk/dataset/nuplan/exp")
    parser.add_argument("--source", type=Path, default=base / OLD_NAME)
    parser.add_argument("--output", type=Path, default=base / NEW_NAME)
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.mode == "pause-old":
            pause_old(args)
        elif args.mode == "prepare":
            prepare(args)
        elif args.mode == "run":
            run(args)
        elif args.mode == "analyze":
            analyze(args)
        else:
            suite.jobs = jobs
            suite.show(args.output)
    except KeyboardInterrupt:
        print("Interrupted; verified completed runs are preserved.", flush=True)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
