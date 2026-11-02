#!/usr/bin/env python3
"""Flow CFG1.8 hard/random NR, then eight-baseline Val NR / hard R / random R / Val R."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from uuid import uuid4

import evaluate_random_baselines_1025 as old
import baseline_extra_planners_1028 as extra
from baseline_job_cleanup_1028 import cleanup_job
from baseline_extension_config_1028 import compose_template, select_cohort, order_metric_callbacks

SETTINGS = (
    ("hard_NR", "test14-hard", "closed_loop_nonreactive_agents"),
    ("random_NR", "test14-random", "closed_loop_nonreactive_agents"),
    ("val_NR", "val14", "closed_loop_nonreactive_agents"),
    ("hard_R", "test14-hard", "closed_loop_reactive_agents"),
    ("random_R", "test14-random", "closed_loop_reactive_agents"),
    ("val_R", "val14", "closed_loop_reactive_agents"),
)
HERE = Path(__file__).resolve()
VARIANTS = ("diffusion_planner", "flow_planner", *old.VARIANTS)


def variants_for(setting):
    return ("flow_planner",) if setting in ("hard_NR", "random_NR") else VARIANTS


TOTAL = sum(len(variants_for(setting)) * 2 for setting, _, _ in SETTINGS)


def environment(plan, name):
    env = old.environment(plan, name if name in old.VARIANTS else "urban_driver")
    return extra.extra_environment(plan, name, env) if name in extra.EXTRA_VARIANTS else env


def validate_stage(directory, cfg, cohort, name, k):
    if name in extra.EXTRA_VARIANTS:
        return extra.validate_extra(name, directory, cfg, cohort)
    return old.validate_stage(directory, cfg, cohort, name, k)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def state(out, **values):
    write(out / "status.json", dict(updated_unix=time.time(), host=socket.gethostname(), **values))


def bootstrap(root):
    env = dict(DP_DIR=str(root / "Diffusion-Planner"),
        NUPLAN_DEVKIT_ROOT=str(root / "nuplan-devkit"),
        NUPLAN_DATA_ROOT=str(root / "dataset/nuplan"),
        NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
        NUPLAN_EXP_ROOT=str(root / "dataset/nuplan/exp"),
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.environ.update(env)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "RAY_ADDRESS", "NUPLAN_HYDRA_CONFIG_PATH"):
        os.environ.pop(key, None)
    for path in (root / "nuplan-devkit", root / "Diffusion-Planner"):
        sys.path.insert(0, str(path))


def verify(plan, source):
    old.require(plan["source_plan_sha256"] == old.digest(source / "plan.json"),
                "The source Random-NR plan changed")
    old.require(plan["settings"] == [list(x) for x in SETTINGS] and plan["ks"] == [0, 20]
                and plan["variants"] == list(VARIANTS) and plan["gpus"] == list(range(8))
                and plan.get("flow_cfg_weight") == 1.8,
                "Extension axes changed")
    for path, expected in plan["inputs"].items():
        old.require(old.digest(path) == expected, f"Prepared input changed: {path}")


def prepare(args):
    out, source = args.output, args.source
    if (out / "plan.json").exists():
        plan = read(out / "plan.json")
        old.require(plan["host"] == socket.gethostname(), "Resume on the original evaluation node")
        old.require(plan["seed"] == args.seed, "Prepared seed differs; use a new --output")
        verify(plan, source)
        return plan
    prior = read(source / "plan.json")
    old.require(read(source / "status.json").get("state") == "complete",
                "The source Random-NR baseline queue must be complete")
    old.require(prior["variants"] == list(old.VARIANTS) and prior["ks"] == [0, 20]
                and prior["gamma"] == .1 and prior["sigma"] == .1,
                "Unexpected source baseline protocol")
    root = Path(prior["root"])
    old.require(root == args.root, "Source root differs from --root")
    seed = args.seed
    # Reuse exact successful native/refined planner configurations and weights.
    plan = {key: copy.deepcopy(prior[key]) for key in (
        "root", "seed", "gamma", "sigma", "ks", "variants", "frozen_score_code",
        "score_config", "score_checkpoint", "sources")}
    plan.update(schema_version=4, host=socket.gethostname(), gpus=list(range(8)), workers=8,
        flow_cfg_weight=1.8,
        seed=seed, source_seed=prior["seed"], variants=list(VARIANTS),
        source=str(source), source_plan_sha256=old.digest(source / "plan.json"),
        settings=[list(x) for x in SETTINGS], cohorts={}, configs={}, jobs=[])
    score_settings = copy.deepcopy(read(prior["score_config"]))
    score_settings["reference_seed"] = seed
    score_cfg = out / "models" / f"S10-G_seed{seed}" / "config.json"
    write(score_cfg, score_settings)
    plan["score_config"] = str(score_cfg)
    plan["sources"].update(extra.resolve_extra_sources(root, prior,
        flow_adapter=args.flow_adapter, flow_code=args.flow_code,
        flow_config=args.flow_config, flow_checkpoint=args.flow_checkpoint))
    matrix = read(prior["matrix"])
    templates = {}
    for setting, split, challenge in SETTINGS:
        print(f"CONFIG {setting}: official {split} / {challenge}", flush=True)
        templates[setting] = compose_template(root, split, challenge, out / "templates" / setting, seed)
    # Select Val14 once with its actual trainval builder; NR/R share these identities.
    print("Selecting the official Val14 cohort once (no DP simulation)", flush=True)
    val = old.keys(select_cohort(templates["val_NR"], seed))
    for setting, split, challenge in SETTINGS:
        cohort = val if split == "val14" else old.keys(matrix["scenario_keys"][split])
        plan["cohorts"][setting] = cohort
        plan["configs"][setting] = {}
        for name in variants_for(setting):
            plan["configs"][setting][name] = {}
            for k in (0, 20):
                target = out / setting / name / ("native" if k == 0 else "S10G_K20")
                cfg = old.shared().execution_config(templates[setting], target, 8)
                if name in extra.EXTRA_VARIANTS:
                    cfg["planner"] = extra.configure_extra(name, plan["sources"][name],
                        score_cfg, plan["score_checkpoint"], seed, k, target)
                    cfg.pop("model", None)
                else:
                    original = prior["configs"][name][str(k)]
                    cfg["planner"] = old.shared().relocate(copy.deepcopy(original["planner"]),
                                                           original["output_dir"], str(target))
                    if k:
                        next(iter(cfg["planner"].values()))["score_config"] = str(score_cfg)
                    if "model" in original:
                        cfg["model"] = copy.deepcopy(original["model"])
                cfg.update(seed=seed, job_name=f"{setting}_{name}_K{k}",
                    experiment_uid=f"{split}/{name}/K{k}", enable_simulation_progress_bar=True)
                cfg["scenario_filter"].update(scenario_tokens=sorted({t for _, t in cohort}),
                    log_names=sorted({log for log, _ in cohort}), num_scenarios_per_type=None,
                    limit_total_scenarios=None, shuffle=False)
                cfg = order_metric_callbacks(cfg)
                plan["configs"][setting][name][str(k)] = cfg
                plan["jobs"].append(dict(id=f"{setting}_{name}_K{k}", setting=setting,
                                         variant=name, K=k, output=str(target)))
        print(f"READY {setting}: {len(cohort)} scenarios, {2 * len(variants_for(setting))} evaluations", flush=True)
    tracked = {HERE, HERE.with_name("baseline_extension_config_1028.py"),
               Path(old.__file__).resolve(), HERE.with_name("evaluate_native_score_queue.py"),
               HERE.with_name("baseline_extra_planners_1028.py"), HERE.with_name("baseline_job_cleanup_1028.py"),
               Path(prior["score_config"]),
               Path(plan["score_config"]), Path(plan["score_checkpoint"])}
    tracked.update(Path(value["checkpoint"]) for value in plan["sources"].values())
    for name in extra.EXTRA_VARIANTS:
        tracked.update(extra.input_files(plan["sources"][name]))
    for directory in (Path(plan["frozen_score_code"]) / "score_function",
                      HERE.parent.parent / "urban_driver_score", HERE.parent.parent / "native_planner_score",
                      HERE.parent.parent / "plantf_score", HERE.parent.parent / "pluto_score"):
        tracked.update(directory.rglob("*.py"))
    plan["inputs"] = {str(p): old.digest(p) for p in sorted(tracked)}
    write(out / "plan.json", plan)
    state(out, state="prepared", full_evaluations=TOTAL)
    return plan


def receipt_path(out, job):
    return out / "jobs" / (job["id"] + ".json")


def worker(args):
    plan = read(args.output / "plan.json")
    job = next(j for j in plan["jobs"] if j["id"] == args.job)
    setting, name, k = job["setting"], job["variant"], job["K"]
    cfg = order_metric_callbacks(plan["configs"][setting][name][str(k)])
    cohort = old.keys(plan["cohorts"][setting])
    target = Path(job["output"])
    if target.exists():
        try:
            result = validate_stage(target, cfg, cohort, name, k)
            print("REUSE verified", job["id"], flush=True)
        except (ValueError, FileNotFoundError, KeyError):
            if not args.retry_failed:
                raise RuntimeError(f"Incomplete output preserved: {target}; use --retry-failed")
            archive = target.with_name(target.name + f".interrupted.{time.time_ns()}")
            target.rename(archive)
            print("PRESERVED", archive, flush=True)
            result = None
    else:
        result = None
    if result is None:
        old.shared().run_simulation(cfg, target, Path(plan["sources"][name]["runtime"]["entry"]),
                                    environment(plan, name))
        result = validate_stage(target, cfg, cohort, name, k)
    record = dict(job, state="complete", score_100=result[3], scenarios=len(cohort),
                  result_dir=str(target), finished_unix=time.time())
    write(receipt_path(args.output, job), record)
    print(f"COMPLETE {job['id']}: {result[3]:.4f}; {len(cohort)}/{len(cohort)}", flush=True)


def pair_results(out, plan, setting, name):
    """Pair only two independently successful stages, including after a K0 retry."""
    paths = [out / "jobs" / f"{setting}_{name}_K{k}.json" for k in (0, 20)]
    records = [read(path) if path.exists() else {} for path in paths]
    pair_path = out / setting / name / "pairing.json"
    if any(record.get("state") != "complete" for record in records):
        write(pair_path, dict(state="pending", reason="Waiting for successful K0 and K20"))
        return False
    identity = [old.digest(path) for path in paths]
    csv_path = pair_path.parent / "paired_scores.csv"
    if pair_path.exists() and csv_path.exists():
        previous = read(pair_path)
        if previous.get("state") == "complete" and previous.get("receipts") == identity:
            return True
    cohort = old.keys(plan["cohorts"][setting])
    results = [old.shared().read_result(record["result_dir"], len(cohort)) for record in records]
    old.require(all(result[2] == cohort for result in results), "Pairing scenario identities differ")
    paired = results[0][1][["log_name", "scenario", "score"]].merge(
        results[1][1][["log_name", "scenario", "score"]], on=["log_name", "scenario"],
        suffixes=("_native", "_K20"), validate="one_to_one")
    paired["delta_100"] = 100 * (paired.score_K20.astype(float) - paired.score_native.astype(float))
    temporary = csv_path.with_suffix(".csv.tmp")
    paired.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    write(pair_path, dict(state="complete", receipts=identity, scenarios=len(paired)))
    return True


def try_pair(out, plan, job):
    try:
        pair_results(out, plan, job["setting"], job["variant"])
    except Exception as error:
        write(out / job["setting"] / job["variant"] / "pairing.json",
              dict(state="failed", error=f"{type(error).__name__}: {error}"))
        print(f"PAIR FAILED {job['setting']}/{job['variant']}: {error}; continuing", flush=True)


def counts(out, plan):
    records = [read(receipt_path(out, job)) if receipt_path(out, job).exists() else {}
               for job in plan["jobs"]]
    return dict(completed=sum(r.get("state") == "complete" for r in records),
                failed=sum(r.get("state") in ("failed", "interrupted") for r in records),
                pair_failures=sum(read(p).get("state") == "failed" for p in out.glob("*/*/pairing.json")),
                total=len(plan["jobs"]))


def summarize(out, plan):
    rows, failures = [], []
    for setting, _, _ in SETTINGS:
        for name in variants_for(setting):
            row = dict(setting=setting, variant=name, seed=plan["seed"], native_100=None, K20_100=None)
            for k, column in ((0, "native_100"), (20, "K20_100")):
                p = out / "jobs" / f"{setting}_{name}_K{k}.json"
                r = read(p) if p.exists() else dict(state="pending")
                row[column + "_state"] = r["state"]
                if r["state"] == "complete":
                    row[column] = r["score_100"]
                elif r["state"] in ("failed", "interrupted"):
                    failures.append(dict(job=f"{setting}_{name}_K{k}", kind="evaluation",
                                         error=r.get("error", ""), log=r.get("log", "")))
            row["delta_100"] = (row["K20_100"] - row["native_100"]
                if row["K20_100"] is not None and row["native_100"] is not None else None)
            pair_path = out / setting / name / "pairing.json"
            pair = read(pair_path) if pair_path.exists() else dict(state="pending")
            row["paired_state"] = pair["state"]
            if pair["state"] == "failed":
                failures.append(dict(job=f"{setting}_{name}", kind="pairing",
                                     error=pair.get("error", ""), log=str(pair_path)))
            rows.append(row)
    for filename, values, fields in (("summary.csv", rows, list(rows[0])),
                                     ("failed_jobs.csv", failures, ["job", "kind", "error", "log"])):
        target = out / filename
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
        temporary.replace(target)
    write(out / "summary.json", rows)
    write(out / "failed_jobs.json", failures)


def wait_gpu(out):
    while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        state(out, state="waiting_gpu")
        print("WAIT GPUs 0–7 to become free", flush=True)
        time.sleep(15)


def execute_job(args, plan, job, index):
    log = args.output / "logs" / (job["id"] + ".log")
    log.parent.mkdir(exist_ok=True)
    runtime = plan["sources"][job["variant"]]["runtime"]
    command = [runtime["python"], "-u", str(HERE), "worker", "--root", str(args.root),
               "--source", str(args.source), "--output", str(args.output), "--job", job["id"],
               "--seed", str(plan["seed"])]
    if args.retry_failed:
        command.append("--retry-failed")
    token = uuid4().hex
    env = environment(plan, job["variant"])
    env.update(SCORE_BASELINE_QUEUE=str(args.output.resolve()), SCORE_BASELINE_JOB_TOKEN=token)
    record = dict(job, state="running", log=str(log), started_unix=time.time(), token=token)
    path = receipt_path(args.output, job)
    if path.exists():
        write(args.output / "retry_history" / f"{job['id']}_{time.time_ns()}.json", read(path))
    write(path, record)
    print(f"START {index+1}/{len(plan['jobs'])} {job['id']}; log={log}", flush=True)
    child = None
    try:
        with log.open("a") as stream:
            child = subprocess.Popen(command, cwd=runtime["cwd"], env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state(args.output, state="running", job=job["id"], index=index+1, total=len(plan["jobs"]),
                  pid=child.pid, log=str(log), started_unix=record["started_unix"], token=token)
            code = child.wait()
        if code:
            raise RuntimeError(f"Simulation exited {code}; log={log}")
        if read(path).get("state") != "complete":
            raise RuntimeError(f"Worker exited without a validated completion receipt; log={log}")
    finally:
        # Ray may detach children into other sessions; match the exact inherited
        # queue+attempt token, never terminate another experiment's GPU process.
        # Cleanup must not replace a manual interrupt or invalidate a completed score.
        try:
            cleanup = cleanup_job(args.output, token, process_group=child.pid if child else None)
        except Exception as error:
            cleanup = dict(errors=[f"{type(error).__name__}: {error}"])
            print(f"CLEANUP warning {job['id']}: {error}", flush=True)
        try:
            write(args.output / "cleanup" / (job["id"] + ".json"), cleanup)
        except Exception as error:
            print(f"CLEANUP log warning {job['id']}: {error}", flush=True)
        try:
            if child is not None and child.poll() is None:
                # Fallback touches only the child created by this invocation.
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        except Exception as error:
            print(f"CHILD cleanup warning {job['id']}: {error}", flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "queue.lock").open("a") as lock, Path("/tmp/score_random_baselines_1025.lock").open("a") as node_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            state(args.output, state="preparing", pid=os.getpid())
            plan = prepare(args)
            write(args.output / "process.json", dict(host=socket.gethostname(), pid=os.getpid()))
            for index, job in enumerate(plan["jobs"]):
                path = receipt_path(args.output, job)
                previous = read(path) if path.exists() else {}
                if previous.get("state") == "complete":
                    print("SKIP complete", job["id"], flush=True)
                    try_pair(args.output, plan, job)
                    continue
                if previous.get("state") in ("failed", "interrupted") and not args.retry_failed:
                    print("SKIP failed (use --retry-failed)", job["id"], flush=True)
                    continue
                wait_gpu(args.output)
                try:
                    execute_job(args, plan, job, index)
                except Exception as error:
                    log = args.output / "logs" / (job["id"] + ".log")
                    record = dict(job, state="failed", error=f"{type(error).__name__}: {error}",
                                  log=str(log), finished_unix=time.time())
                    write(path, record)
                    print(f"FAILED {job['id']}: {error}; continuing to next evaluation", flush=True)
                try_pair(args.output, plan, job)
                summarize(args.output, plan)
            summarize(args.output, plan)
            totals = counts(args.output, plan)
            outcome = "complete_with_errors" if totals["failed"] or totals["pair_failures"] else "complete"
            state(args.output, state=outcome, **totals)
            print(f"{outcome.upper()}: {totals['completed']}/{totals['total']} succeeded, "
                  f"{totals['failed']} evaluations failed, {totals['pair_failures']} pairing failures; "
                  f"summary={args.output / 'failed_jobs.csv'}", flush=True)
        except BaseException as error:
            previous = read(args.output / "status.json")
            if isinstance(error, KeyboardInterrupt) and previous.get("job"):
                path = args.output / "jobs" / (previous["job"] + ".json")
                if path.exists():
                    record = read(path)
                    if record.get("state") == "running":
                        record.update(state="interrupted", error="Stopped by user", finished_unix=time.time())
                        write(path, record)
            previous.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            write(args.output / "status.json", previous)
            raise


def launch(args):
    if args.stop_energy:
        from stop_energy_evaluation_1028 import stop_evaluation
        stop_evaluation(args.root / "outputs/score_aligned_1004/E10_G_aligned_1025_v1")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = args.output / "queue.log"
        command = [str(args.root / "envs/urbandriver/bin/python"), "-u", str(HERE), "run",
                   "--root", str(args.root), "--source", str(args.source), "--output", str(args.output)]
        command += ["--seed", str(args.seed)]
        for flag in ("flow_adapter", "flow_code", "flow_config", "flow_checkpoint"):
            value = getattr(args, flag)
            if value is not None:
                command += ["--" + flag.replace("_", "-"), str(value)]
        if args.retry_failed:
            command.append("--retry-failed")
        with log.open("a") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(lock.fileno(),))
        write(args.output / "launcher.json", dict(pid=child.pid, host=socket.gethostname(), command=command, log=str(log)))
        print(f"Launched PID {child.pid}; GPUs 0–7\nQueue log: {log}", flush=True)


def status(args):
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        try:
            s = read(args.output / "status.json")
            records = [read(p) for p in (args.output / "jobs").glob("*.json")]
            done = sum(r.get("state") == "complete" for r in records)
            failed = sum(r.get("state") in ("failed", "interrupted") for r in records)
            print(f"{time.strftime('%H:%M:%S')}  {done}/{TOTAL} complete  {failed} failed  {s.get('state')}  {s.get('job', '')}")
            if s.get("log") and Path(s["log"]).exists():
                log = Path(s["log"])
                with log.open("rb") as f:
                    f.seek(max(0, log.stat().st_size - 1024 * 1024))
                    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", f.read().decode(errors="replace"))
                lines = [x.strip() for x in text.splitlines() if x.strip()]
                bars = [x for x in lines if "Ray objects:" in x]
                if bars:
                    print(bars[-1])
                print("\n".join(lines[-4:]))
            if s.get("error"):
                print(s["error"])
            if not args.watch or s.get("state") in ("complete", "complete_with_errors", "failed", "interrupted"):
                return
        except (FileNotFoundError, ValueError):
            print("Waiting for queue status")
            if not args.watch:
                return
        time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "run", "worker", "status"))
    p.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    p.add_argument("--source", type=Path)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--flow-adapter", type=Path)
    p.add_argument("--flow-code", type=Path)
    p.add_argument("--flow-config", type=Path)
    p.add_argument("--flow-checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--job")
    p.add_argument("--retry-failed", action="store_true", help="Retry failed/interrupted jobs; keep successful results")
    p.add_argument("--stop-energy", action="store_true")
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        p.error("--seed must be in [0, 2**32)")
    args.root = args.root.resolve()
    args.source = (args.source or args.root / "dataset/nuplan/exp/random_baselines_1025_v1").resolve()
    args.output = (args.output or args.root / f"dataset/nuplan/exp/baselines8_flowcfg18_seed{args.seed}_1028_v4").resolve()
    if args.action != "status":
        bootstrap(args.root)
    if args.action == "run":
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Stopped by signal {signum}")
        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
    try:
        {"launch": launch, "run": run, "worker": worker, "status": status}[args.action](args)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
