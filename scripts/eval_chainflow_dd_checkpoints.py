#!/usr/bin/env python3
"""Compare best and final/last EMA checkpoints for five trained Score models on the same DD navtest starts."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import eval_chainflow_score_on_dd as legacy
from scripts import run_navsim_chainflow as pipeline

OUTPUT = "chainflow_score_b64_e40_sweep_dd_k20"
WORKER = "score_function.navsim_chainflow.dd_checkpoint_matrix"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "report"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--plan", type=Path, help="B64/E40 reference plan.json")
    parser.add_argument("--dd-source", type=Path, help="Original paired DD run directory or pointer")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--gpu-policy", choices=("shared", "wait-idle"), default="wait-idle")
    parser.add_argument("--gpu-pool", default="auto")
    parser.add_argument("--gpu-budget-gb", type=float, default=12)
    parser.add_argument("--gpu-reserve-gb", type=float, default=4)
    parser.add_argument("--watch", action="store_true", help="Refresh status every ten seconds")
    args = parser.parse_args(argv)
    if args.watch and args.command != "status":
        parser.error("--watch applies to status")
    args.root = args.root.expanduser().resolve()
    exp = args.root / "dataset/navsim/exp"
    args.plan = (args.plan or exp / "chainflow_score_pipeline_b64_e40/plan.json").expanduser().resolve()
    args.output = exp / OUTPUT
    return args


def prior_result(directory, base):
    """Reuse the scored E40 arm only when the completed legacy run used the same inputs."""
    marker_path = directory / "launcher.json"
    if not marker_path.is_file():
        return None
    try:
        marker = pipeline.read_json(marker_path)
        if marker.get("status") != "complete":
            return None
        summary_path = directory / "summary.json"
        if pipeline.sha256(summary_path) != marker["summary_sha256"]:
            raise ValueError("summary hash changed")
        summary = pipeline.read_json(summary_path)
        old = pipeline.read_json(directory / "request.json")
        same = {**base, "workers": old["workers"]}
        if (old != same or summary["protocol"] != old or summary["count"] != legacy.TOTAL
                or summary["valid_count"] != legacy.TOTAL or summary["failed_count"] != 0):
            raise ValueError("different DD transfer protocol or scene coverage")
        for i in range(old["workers"]):
            shard = legacy.shard_path(directory, i, old["workers"])
            complete = pipeline.read_json(shard / "complete.json")
            if pipeline.sha256(shard / "scenes.jsonl") != complete["scenes_sha256"]:
                raise ValueError(f"scene records changed in {shard}")
        arm = summary["by_k"]["20"]
        if arm["improved"] + arm["worsened"] + arm["tied"] != legacy.TOTAL:
            raise ValueError("incomplete K20 results")
        return dict(output=str(directory), by_k20=arm, summary_sha256=marker["summary_sha256"])
    except (KeyError, OSError, ValueError) as exc:
        print(f"Scoring reference/final again ({exc})", flush=True)
        return None


def report(output):
    state = pipeline.read_json(output / "launcher.json")
    path = output / "summary.json"
    if state.get("status") != "complete" or state["summary_sha256"] != pipeline.sha256(path):
        raise ValueError("No verified checkpoint comparison yet")
    summary = pipeline.read_json(path)
    req = pipeline.read_json(output / "request.json")
    baseline = 100 * summary["k0"]
    print(f"DD K0 {baseline:.4f} | K20, paired navtest scenes {summary['valid_count']:,}")
    print("group               EMA               epoch  val_DSM    DD_K20     ΔK0     improved/worsened/tied")
    for key, meta in req["checkpoints"].items():
        arm = summary["by_checkpoint"][key]
        print(f"{key.split('/')[0]:19s} {meta['selection']:17s} {meta['epoch']:4d}  "
              f"{meta['val_dsm']:.6f}  {100 * arm['mean']['score']:8.4f}  "
              f"{100 * arm['mean_delta']['score']:+8.4f}   "
              f"{arm['improved']}/{arm['worsened']}/{arm['tied']}")
    print(path, flush=True)


def run_workers(args, pending, env):
    from scripts.eval_navsim_chainflow import stop_children
    children = {}
    with ExitStack() as stack:
        try:
            for device, idx in enumerate(pending):
                log = stack.enter_context((args.output / f"worker{idx}.log").open("a", buffering=1))
                cmd = [str(args.python), "-u", "-m", WORKER, "--request", str(args.output / "request.json"),
                       "--shard-index", str(idx), "--device", f"cuda:{device}"]
                children[idx] = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
            next_report = time.monotonic() + 30
            while children:
                for idx, child in list(children.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"Shard {idx} exited {code}: {legacy.last_line(args.output / f'worker{idx}.log')}")
                    del children[idx]
                if time.monotonic() >= next_report:
                    legacy.print_progress(args.output, args.workers)
                    next_report = time.monotonic() + 30
                if children:
                    time.sleep(1)
        finally:
            stop_children(list(children.values()))


def run(args):
    from score_function.navsim_chainflow.dd_checkpoint_matrix import merge
    from score_function.navsim_chainflow.sweep_checkpoints import prepare_checkpoints
    exp = args.output.parent
    with pipeline.pipeline_lock(args.output), ExitStack() as stack:
        state_path = args.output / "launcher.json"
        req_path = args.output / "request.json"
        prev = pipeline.read_json(state_path) if state_path.is_file() else {}
        existing = pipeline.read_json(req_path) if req_path.is_file() else None
        if existing is not None:
            if existing["workers"] != args.workers:
                raise ValueError(f"Resume this run with --workers {existing['workers']}")
            if args.dd_source is None:
                args.dd_source = Path(existing["base"]["dd_source"])

        def status(value, **fields):
            pipeline.atomic_json(state_path, dict(status=value, workers=args.workers, updated=pipeline.now(),
                                                  host=socket.gethostname(), pid=os.getpid(), **fields))

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGINT):
            stack.callback(signal.signal, sig, signal.signal(sig, interrupted))
        try:
            status("verifying")
            plan = legacy.load_plan(args)
            env = pipeline.environment(args)
            sys.path[:0] = [str(args.devkit_root), str(args.root / "nuplan-devkit")]
            for name, value in env.items():
                if name.startswith(("OPENSCENE_", "NUPLAN_", "NAVSIM_", "CHAINFLOW_")):
                    os.environ[name] = value
            base = legacy.build_request(args, plan)
            if existing is None:
                checkpoints = prepare_checkpoints(plan, exp / "chainflow_score_b64_e40_sweep", args.output)
            else:
                checkpoints = existing["checkpoints"]
                for key, record in checkpoints.items():
                    if pipeline.sha256(record["path"]) != record["sha256"]:
                        raise ValueError(f"Saved checkpoint snapshot changed: {key}")
            if checkpoints["reference/final"]["sha256"] != base["score_checkpoint_sha256"]:
                raise ValueError("Reference final differs from the original B64 final EMA")
            reuse = None
            if existing is None or existing["reuse_reference_final"] is not None:
                reuse = prior_result(exp / legacy.OUTPUT_NAME, base)
            files = [Path(__file__), REPO / "score_function/navsim_chainflow/dd_checkpoint_matrix.py",
                     REPO / "score_function/navsim_chainflow/sweep_checkpoints.py"]
            req = dict(schema="chainflow-dd-checkpoints-k20-v1", base=base, workers=args.workers,
                       checkpoints=checkpoints, reuse_reference_final=reuse,
                       source_sha256={p.name: pipeline.sha256(p) for p in files})
            if existing is not None and req != existing:
                raise ValueError("Sources/checkpoints changed since this comparison started; the existing results are preserved")
            if existing is None and any(args.output.glob("part-*-of-*")):
                raise ValueError("Found shards without their original request")
            pipeline.atomic_json(req_path, req)
            for key, meta in checkpoints.items():
                print(f"{key:26s} {meta['selection']} epoch {meta['epoch']}  {meta['path']}", flush=True)
            print("Reference final K20:", "verified prior result" if reuse else "evaluate here", flush=True)
            if prev.get("status") == "complete" and (args.output / "summary.json").is_file():
                if pipeline.sha256(args.output / "summary.json") == prev.get("summary_sha256"):
                    pipeline.atomic_json(state_path, prev)
                    report(args.output)
                    return
            pending = [idx for idx in range(args.workers)
                       if not (legacy.shard_path(args.output, idx, args.workers) / "complete.json").exists()]
            if pending:
                def on_gpu(snapshot):
                    status("waiting_gpu" if snapshot["state"] == "waiting" else "running",
                           resource=snapshot["state"], selected_uuids=snapshot["selected_uuids"])
                    print(f"GPU {snapshot['state']} ({snapshot['policy']})", flush=True)
                status("waiting_gpu")
                n = next(n for n in (1, 2, 4, 8) if n >= len(pending))
                with legacy.acquire_gpus(pool=args.gpu_pool, workers=n, policy=args.gpu_policy,
                                         budget_gb=args.gpu_budget_gb, reserve_gb=args.gpu_reserve_gb,
                                         lock_dir=args.gpu_lock_dir, on_status=on_gpu) as selected:
                    env.update(legacy.gpu_environment(selected, args.gpu_budget_gb))
                    run_workers(args, pending, env)
            status("merging")
            summary = merge(args.output, req)
            if summary["valid_count"] != legacy.TOTAL or set(summary["by_checkpoint"]) != set(checkpoints):
                raise ValueError("Incomplete checkpoint comparison")
            status("complete", summary_sha256=pipeline.sha256(args.output / "summary.json"))
            report(args.output)
        except BaseException as exc:
            if prev.get("status") == "complete":
                pipeline.atomic_json(state_path, prev)
            else:
                status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "report":
        report(args.output)
    else:
        while True:
            path = args.output / "launcher.json"
            state = pipeline.read_json(path) if path.is_file() else {"status": "not_started"}
            print(state.get("updated", ""), state["status"], state.get("error", ""), flush=True)
            legacy.print_progress(args.output, state.get("workers", 0))
            if state["status"] == "complete":
                report(args.output)
            if not args.watch or state["status"] in {"complete", "failed", "interrupted"}:
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
