"""Launch the frozen eight-model training matrix; no benchmark selection or evaluation."""

import argparse
import copy
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--root", required=True, help="Workspace containing repositories/data/output"
    )
    result.add_argument("--database-dir")
    result.add_argument("--maps-dir")
    result.add_argument("--planner-dir")
    result.add_argument("--devkit-dir")
    result.add_argument("--checkpoint")
    result.add_argument("--planner-args")
    result.add_argument(
        "--data-output", help="Shared preprocessing/cache directory; preferably SSD"
    )
    result.add_argument("--run-name", default="score_matrix_1m")
    result.add_argument("--total-scenarios", type=int, help="Global selection cap; default 1000000")
    result.add_argument(
        "--reuse-features-from",
        help="Old data_output directory; selected NPZs are reused read-only",
    )
    result.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    result.add_argument("--base-config", default=str(PROJECT / "configs/score_function.json"))
    result.add_argument("--suite-config", default=str(PROJECT / "configs/experiment_suite.json"))
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    # Handled by the Bash bootstrap; retained for a single shared command line.
    result.add_argument("--python")
    result.add_argument("--env-name", default="score_function_wyk")
    result.add_argument("--foreground", action="store_true")
    return result


def make_plan(args):
    root = Path(args.root).expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("run-name may contain only letters, digits, underscores and hyphens")
    gpus = args.gpus.split(",")
    if len(gpus) != 8 or len(set(gpus)) != 8 or any(not x.isdigit() for x in gpus):
        raise ValueError("Supply exactly eight distinct physical GPU indices")
    base, registry = read(args.base_config), read(args.suite_config)
    if len(registry["experiments"]) != 8:
        raise ValueError("This launcher requires eight experiments")
    output = root / "outputs" / args.run_name
    data = (
        Path(args.data_output).expanduser().resolve()
        if args.data_output
        else (root / "score_data" / args.run_name)
    )
    paths = base["paths"]
    paths["root"] = str(root)
    replacements = {
        "planner_dir": args.planner_dir,
        "devkit_dir": args.devkit_dir,
        "database_dir": args.database_dir,
        "maps_dir": args.maps_dir,
        "planner_checkpoint": args.checkpoint,
        "planner_args": args.planner_args,
    }
    for key, value in replacements.items():
        if value is not None:
            paths[key] = str(Path(value).expanduser().resolve())
    for key in list(paths):
        if key != "root":
            path = Path(paths[key]).expanduser()
            paths[key] = str(path if path.is_absolute() else root / path)
    if args.planner_dir:
        for key, filename in (
            ("planner_args", "checkpoints/args.json"),
            ("planner_checkpoint", "checkpoints/model.pth"),
            ("train_log_allowlist", "nuplan_train.json"),
        ):
            if key == "train_log_allowlist" or replacements.get(key) is None:
                paths[key] = str(Path(paths["planner_dir"]) / filename)
    paths.update(
        data_output=str(data),
        manifest=str(data / "features_manifest.json"),
        cache=str(data / "cache/index.json"),
        neighbor_cache=str(data / "neighbors/index.json"),
    )
    if args.reuse_features_from:
        paths["reuse_features_from"] = str(Path(args.reuse_features_from).expanduser().resolve())
        if Path(paths["reuse_features_from"]) == data:
            raise ValueError(
                "Use a new data-output; reuse-features-from must point to the old directory"
            )
    for section in ("training", "runtime", "data"):
        base[section].update(registry[section])
    base["runtime"]["device"] = "cuda:0"
    # Full raw source coverage, with the official GLOBAL scenario budget.
    base["data"].update(
        timestamp_spacing_s=None,
        max_scenarios_per_db=None,
        expand_scenarios=True,
        remove_invalid_goals=False,
    )
    if args.total_scenarios is not None:
        base["data"]["total_scenarios"] = args.total_scenarios
    if (
        not isinstance(base["data"]["total_scenarios"], int)
        or isinstance(base["data"]["total_scenarios"], bool)
        or base["data"]["total_scenarios"] < 1
    ):
        raise ValueError("total-scenarios must be a positive integer")
    entries, identifiers = [], set()
    for spec, gpu in zip(registry["experiments"], gpus):
        identifier, variant = spec["id"], spec["variant"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in identifiers:
            raise ValueError("Invalid or duplicate experiment identifier")
        if variant not in ("local", "wide", "global", "neighbors"):
            raise ValueError(f"Unknown model variant: {variant}")
        identifiers.add(identifier)
        config = copy.deepcopy(base)
        config["model"].update(
            pre_dilations=[1, 2],
            post_dilations=[4, 12] if variant in ("wide", "neighbors") else [2, 4],
            temporal_attention=variant == "global",
            neighbor_future=variant == "neighbors",
        )
        config["training"]["sigma"] = spec["sigma"]
        config["paths"]["run_dir"] = str(output / identifier)
        entries.append(
            {
                **spec,
                "gpu": gpu,
                "config": config,
                "config_path": str(output / "configs" / f"{identifier}.json"),
            }
        )
    files = list((PROJECT / "score_function").rglob("*.py")) + list(
        (PROJECT / "scripts").glob("*.sh")
    )
    sources = {
        p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }
    return {
        "name": registry["name"],
        "output": str(output),
        "sources": sources,
        "experiments": entries,
    }


class Suite:
    def __init__(self, plan, resume):
        self.plan, self.resume = plan, resume
        self.output = Path(plan["output"])
        self.lock = threading.Lock()
        self.children = {}
        self.cancelled = threading.Event()
        self.status = {"state": "running", "stage": "preflight", "experiments": {}}

    def publish(self, **values):
        with self.lock:
            self.status.update(values)
            write(self.output / "status.json", self.status)

    def job_status(self, identifier, **values):
        with self.lock:
            self.status["experiments"][identifier] = values
            write(self.output / "status.json", self.status)

    def stop(self, signum, frame):
        self.cancelled.set()
        with self.lock:
            processes = list(self.children.values())
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def command(self, arguments, gpu, log, label):
        env = os.environ.copy()
        # Each training process is an independent single-GPU experiment.
        for name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
        ):
            env.pop(name, None)
        env.update(
            CUDA_VISIBLE_DEVICES=gpu,
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            PYTHONUNBUFFERED="1",
            TQDM_DISABLE="1",
        )
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            print(f"[{label}] log: {log}", flush=True)
            with self.lock:
                if self.cancelled.is_set():
                    raise RuntimeError("Suite interrupted")
                process = subprocess.Popen(
                    [sys.executable, *arguments],
                    env=env,
                    cwd=PROJECT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.children[label] = process
            try:
                code = process.wait()
            finally:
                with self.lock:
                    self.children.pop(label, None)
            if code:
                raise RuntimeError(f"{label} exited {code}; inspect {log}")

    def stage(self, entry, command, *, all_gpus=False, extra=()):
        argv = ["-m", "score_function", command, "--config", entry["config_path"], *extra]
        gpu = entry["gpu"]
        if all_gpus:
            gpu = ",".join(row["gpu"] for row in self.plan["experiments"])
            if command == "cache":
                argv = [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    "--nproc_per_node=8",
                    *argv,
                ]
        log = self.output / entry["id"] / f"{command}.log"
        self.command(argv, gpu, log, f"{entry['id']}/{command}")

    def worker(self, entry):
        identifier = entry["id"]
        started = time.time()
        try:
            if entry["config"]["model"]["neighbor_future"]:
                self.job_status(
                    identifier,
                    state="running",
                    stage="cache-neighbors",
                    gpu=entry["gpu"],
                    started=started,
                )
                self.stage(entry, "cache-neighbors")
            directory = self.output / identifier / "score"
            extra = ()
            if directory.exists():
                if not self.resume or not (directory / "last.pt").is_file():
                    raise RuntimeError(
                        f"{directory} has no resumable checkpoint; use a new run-name"
                    )
                if (directory / "status.json").is_file():
                    previous = read(directory / "status.json")
                    if previous.get("state") == "complete" and (directory / "best.pt").is_file():
                        result = {
                            "state": "complete",
                            "reused": True,
                            **read(directory / "selection.json"),
                        }
                        self.job_status(identifier, **result)
                        return result
                extra = ("--resume",)
            self.job_status(
                identifier, state="running", stage="train", gpu=entry["gpu"], started=started
            )
            self.stage(entry, "train", extra=extra)
            final = read(directory / "status.json")
            if final["state"] != "complete" or not (directory / "best.pt").is_file():
                raise RuntimeError("Training exited without complete status and selected weights")
            result = {
                "state": "complete",
                "gpu": entry["gpu"],
                "training": final,
                "selection": read(directory / "selection.json"),
                "elapsed_s": time.time() - started,
            }
        except Exception as error:
            result = {"state": "failed", "gpu": entry["gpu"], "error": str(error)}
        self.job_status(identifier, **result)
        return result

    def run(self):
        entries = self.plan["experiments"]
        self.publish(stage="preflight")
        self.stage(entries[0], "preflight", all_gpus=True, extra=("--gpus", "8"))
        # Real configured forward/backward on each GPU before costly preprocessing.
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [pool.submit(self.stage, row, "check-ddp") for row in entries]
            for job in jobs:
                job.result()
        self.publish(stage="prepare")
        self.stage(entries[0], "prepare")
        self.publish(stage="shared_scene_cache")
        self.stage(entries[0], "cache", all_gpus=True)
        self.publish(stage="training")
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [pool.submit(self.worker, row) for row in entries]
            results = {row["id"]: job.result() for row, job in zip(entries, jobs)}
        success = all(row["state"] == "complete" for row in results.values())
        write(
            self.output / "summary.json",
            {
                "state": "complete" if success else "failed",
                "experiments": results,
                "evaluation_launched": False,
            },
        )
        self.publish(state="complete" if success else "failed", stage="finished")
        return 0 if success else 1


def main(argv=None):
    args = parser().parse_args(argv)
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    import fcntl

    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process already owns this experiment suite") from error
        frozen = output / "suite.json"
        if frozen.exists():
            if not args.resume:
                raise ValueError("Existing suite: use --resume or a new --run-name")
            if read(frozen) != plan:
                raise ValueError(
                    "Frozen suite/config/source/GPU assignment changed; use a new run-name"
                )
        else:
            if args.resume:
                raise ValueError("No existing suite to resume")
            write(frozen, plan)
        for row in plan["experiments"]:
            write(row["config_path"], row["config"])
        suite = Suite(plan, args.resume)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, suite.stop)
        try:
            data_root = Path(plan["experiments"][0]["config"]["paths"]["data_output"])
            data_root.mkdir(parents=True, exist_ok=True)
            with (data_root / "suite_cache.lock").open("a") as cache_lock:
                try:
                    fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(
                        "Another suite owns the shared data/cache directory"
                    ) from error
                return suite.run()
        except Exception as error:
            suite.stop(None, None)
            suite.publish(state="failed", error=str(error))
            raise


if __name__ == "__main__":
    sys.exit(main())
