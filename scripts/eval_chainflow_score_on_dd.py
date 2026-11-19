#!/usr/bin/env python3
"""Evaluate the B64 epoch-40 ChainFlow conditional score on saved DiffusionDrive navtest trajectories."""
from __future__ import annotations

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
from scripts import run_navsim_chainflow as pipeline
from score_function.navsim_chainflow.resources import acquire_gpus, gpu_environment

KS = [0, 5, 10, 15, 20, 30]
TOTAL = 12146
OUTPUT_NAME = "chainflow_score_b64_e40_on_dd_navtest"
# These files supplied the score weights' interpretation, features and update rule in B64.
SCORE_SOURCES = (
    "score_function/navsim_chainflow/model.py", "score_function/navsim_chainflow/runtime.py",
    "score_function/navsim_chainflow/native.py", "score_function/navsim_chainflow/evaluate.py",
    "score_function/model/score_branch.py", "score_function/model/module/attention.py",
    "score_function/model/module/temporal.py", "score_function/model/module/future_condition.py",
    "score_function/utils/normalizer.py", "score_function/utils/train_utils.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "report"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--plan", type=Path, help="Completed B64/e40 pipeline plan.json")
    parser.add_argument("--dd-source", type=Path, help="Completed paired DD eval directory or latest pointer")
    parser.add_argument("--workers", type=int, default=8, choices=(1, 2, 4, 8))
    parser.add_argument("--gpu-policy", default="wait-idle", choices=("wait-idle", "shared"))
    parser.add_argument("--gpu-pool", default="auto")
    parser.add_argument("--gpu-budget-gb", type=float, default=12)
    parser.add_argument("--gpu-reserve-gb", type=float, default=4)
    parser.add_argument("--watch", action="store_true", help="Refresh status every ten seconds")
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    exp = args.root / "dataset/navsim/exp"
    args.plan = (args.plan or exp / "chainflow_score_pipeline_b64_e40/plan.json").expanduser().resolve()
    args.output = exp / OUTPUT_NAME
    if args.watch and args.command != "status":
        parser.error("--watch applies to status")
    return args


def load_plan(args):
    plan = pipeline.read_json(args.plan)
    if (plan.get("schema") != "chainflow-score-pipeline-v1" or plan.get("recipe") != "b64-e40"
            or plan.get("final_weights") != "ema" or plan.get("primary_k") != 20):
        raise ValueError(f"Expected the completed B64/e40 training plan at {args.plan}")
    paths = plan["paths"]
    for key in ("python", "devkit_root", "data_root", "metric_cache", "training", "evaluation",
                "navtest_vlm_cache", "checkpoint", "backbone", "gpu_lock_dir"):
        path = Path(paths[key])
        if not path.is_absolute():
            raise ValueError(f"The B64 plan needs an absolute {key} path: {path}")
        # Preserve the conda interpreter path, including venv-style symlinks.
        setattr(args, key, path if key == "python" else path.resolve())
    args.score_checkpoint = args.training / "final_ema.pt"
    return plan


def dd_input(args):
    exp = args.root / "dataset/navsim/exp"
    if args.dd_source is not None:
        return args.dd_source.expanduser()
    # Use the first existing pointer; corruption in an existing source must surface.
    for name in ("dd_score_e200_eval_latest.txt", "dd_score_ddaligned_e500_eval_latest.txt",
                 "dd_score_ddaligned_eval_latest.txt", "dd_score_final_eval_latest.txt"):
        pointer = exp / name
        if pointer.exists():
            return pointer
    raise FileNotFoundError("No completed DD eval pointer. Supply --dd-source with a completed paired DD eval directory.")


def score_sha(args, plan):
    published = pipeline.read_json(args.training / "TRAINING_COMPLETE.json")
    run = pipeline.read_json(args.training / "run.json")
    if (published.get("epochs") != 40 or not published.get("signature")
            or published["signature"] != run.get("signature") or run["config"] != plan["config"]
            or run.get("global_batch") != 64 or run["protocol"].get("world_size") != 8
            or run.get("planned_optimizer_updates") != 53160):
        raise ValueError("The training completion record does not match B64/e40")
    checksum = pipeline.sha256(args.score_checkpoint)
    if checksum != published.get("final_ema_sha256"):
        raise ValueError("final_ema.pt differs from the B64/e40 completion record")
    return checksum


def build_request(args, plan):
    from score_function.navsim_chainflow.cross_reference import load_reference
    from score_function.navsim_chainflow.dd_baseline import load_dd_baseline, resolve_dd_source
    from score_function.navsim_chainflow.evaluate import scoring_identity
    from score_function.navsim_chainflow.native import conditioner_identity

    source = resolve_dd_source(dd_input(args))
    checksum = score_sha(args, plan)
    hashes = {p: pipeline.sha256(REPO / p) for p in SCORE_SOURCES}
    for name, value in hashes.items():
        if plan["source_sha256"].get(name) != value:
            raise ValueError(f"Score inference source differs from the completed B64 run: {name}")
    print(f"Checking legacy DD trajectories at {source}", flush=True)
    _, dd = load_dd_baseline(source, needed=set())
    if abs(100 * dd["mean_k0_score"] - 88.0337) > .00005:
        raise ValueError("Expected the paired 88.0337 DD K0 trajectories")
    print(f"Checking the matching ChainFlow evaluation at {args.evaluation}", flush=True)
    _, summary, reference = load_reference(args.evaluation, needed=set())
    condition = conditioner_identity(args.devkit_root, args.checkpoint, args.backbone, args.navtest_vlm_cache)
    scoring = scoring_identity(args.devkit_root)
    cf = summary["protocol"]
    data_paths = dict(data_root=str(args.data_root), metric_cache=str(args.metric_cache))
    for key, value in data_paths.items():
        if cf.get(key) != value or dd["protocol"].get(key) != value:
            raise ValueError(f"Source experiments and current {key} must agree")
    expected = dict(mode="refine", split="navtest", count=TOTAL, max_scenes=None, ks=KS,
                    seed=20260922, native_seed=27, score_epoch=40, score_weight_kind="ema", gamma=.1,
                    score_checkpoint_sha256=checksum, identity=condition, scoring_identity=scoring)
    for key, value in expected.items():
        if cf.get(key) != value:
            raise ValueError(f"ChainFlow reference does not match this run: {key}")
    if dd["protocol"]["token_digest"] != cf["token_digest"]:
        raise ValueError("DD and ChainFlow runs have different NAVSIM token sets")
    sources = (Path(__file__), REPO / "score_function/navsim_dd/runtime.py", REPO / "score_function/navsim_dd/evaluate.py",
               *(REPO / "score_function/navsim_chainflow" / name for name in
                 ("cross_dd.py", "dd_baseline.py", "cross_reference.py")))
    hashes.update({str(p.relative_to(REPO)): pipeline.sha256(p) for p in sources})
    return dict(schema="chainflow-score-on-dd-request-v1", **data_paths, devkit_root=str(args.devkit_root),
                checkpoint=str(args.checkpoint), backbone=str(args.backbone), vlm_cache=str(args.navtest_vlm_cache),
                score_checkpoint=str(args.score_checkpoint), score_checkpoint_sha256=checksum,
                dd_source=str(source), dd_baseline=dd, reference_evaluation=str(args.evaluation),
                reference_summary_sha256=reference["summary_sha256"], reference_provenance=reference,
                workers=args.workers, ks=KS, primary_k=20, gamma=.1, sigma=.1, seed=20260922,
                native_identity=condition, scoring_identity=scoring,
                source_sha256=hashes)


def shard_path(output, i, workers):
    return output / f"part-{i:04d}-of-{workers:04d}"


def last_line(path):
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 4096))
        lines = stream.read().splitlines()
    return lines[-1].decode(errors="replace")[-400:] if lines else ""


def print_progress(output, workers):
    for i in range(workers):
        marker = shard_path(output, i, workers) / "complete.json"
        info = f"{pipeline.read_json(marker)['count']} records saved" if marker.is_file() else last_line(output / f"worker{i}.log")
        if info:
            print(f"shard {i}: {info}", flush=True)


def print_report(output):
    state = pipeline.read_json(output / "launcher.json")
    summary_file = output / "summary.json"
    if state["status"] != "complete" or state["summary_sha256"] != pipeline.sha256(summary_file):
        raise ValueError("This cross-planner evaluation has no verified final summary yet")
    s = pipeline.read_json(summary_file)
    print(f"{s['valid_count']}/{s['count']} valid navtest scenes; primary K=20; final EMA epoch 40")
    print(" K   DD PDMS     ΔDD K0     ΔCF K0    ΔDD−ΔCF    improved / worsened / tied")
    for k in KS:
        r = s["by_k"][str(k)]
        dd = 100 * r["mean_delta"]["score"]
        cf = 100 * s["chainflow_reference_by_k"][str(k)]["mean_delta"]["score"]
        print(f"{k:2d}  {100*r['mean']['score']:8.4f}  {dd:+9.4f}  {cf:+9.4f}  {dd-cf:+9.4f}"
              f"    {r['improved']} / {r['worsened']} / {r['tied']}")
    main = s["by_k"]["20"]["mean_delta"]
    print("K20 paired metric deltas (×100):", {k: round(100*v, 4) for k, v in main.items()})
    print(f"Summary: {summary_file}", flush=True)


def launch_workers(args, pending, env):
    from scripts.eval_navsim_chainflow import stop_children
    children = {}
    with ExitStack() as stack:
        try:
            for device_index, i in enumerate(pending):
                log = stack.enter_context((args.output / f"worker{i}.log").open("a", buffering=1))
                cmd = [str(args.python), "-u", "-m", "score_function.navsim_chainflow.cross_dd",
                       "--request", str(args.output / "request.json"), "--shard-index", str(i),
                       "--device", f"cuda:{device_index}"]
                children[i] = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
            next_report = time.monotonic() + 15
            while children:
                for i, child in list(children.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    if code != 0:
                        raise RuntimeError(f"Shard {i} exited {code}; see {args.output / f'worker{i}.log'}")
                    del children[i]
                if time.monotonic() >= next_report:
                    print_progress(args.output, args.workers)
                    next_report = time.monotonic() + 30
                if children:
                    time.sleep(1)
        finally:
            stop_children(list(children.values()))


def run(args):
    from score_function.navsim_chainflow.cross_dd import merge
    with pipeline.pipeline_lock(args.output):
        state_path = args.output / "launcher.json"
        previous = pipeline.read_json(state_path) if state_path.is_file() else {}
        req_path = args.output / "request.json"
        existing = pipeline.read_json(req_path) if req_path.is_file() else None
        if existing is not None:
            if existing["workers"] != args.workers:
                raise ValueError(f"Existing shards use --workers {existing['workers']}; use the same count to resume")
            if args.dd_source is None:
                args.dd_source = Path(existing["dd_source"])
        state = dict(pid=os.getpid(), host=socket.gethostname(), workers=args.workers, started=pipeline.now())

        def status(value, **fields):
            state.update(status=value, updated=pipeline.now(), **fields)
            pipeline.atomic_json(state_path, state)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        with ExitStack() as stack:
            for sig in (signal.SIGINT, signal.SIGTERM):
                stack.callback(signal.signal, sig, signal.signal(sig, interrupted))
            try:
                status("verifying_sources")
                plan = load_plan(args)
                env = pipeline.environment(args)
                env["PYTHONPATH"] = os.pathsep.join(map(str, (REPO, args.devkit_root, args.root / "nuplan-devkit")))
                # Configure the launcher's CPU-side official imports (workers get env explicitly).
                for path in (args.root / "nuplan-devkit", args.devkit_root):
                    if str(path) in sys.path:
                        sys.path.remove(str(path))
                    sys.path.insert(0, str(path))
                for name in ("OPENSCENE_DATA_ROOT", "NUPLAN_MAPS_ROOT", "NUPLAN_MAP_VERSION", "NAVSIM_DEVKIT_ROOT",
                             "NAVSIM_EXP_ROOT", "NAVSIM_METRIC_CACHE_ROOT"):
                    os.environ[name] = env[name]
                request = build_request(args, plan)
                if existing is not None and existing != request:
                    raise ValueError("This output belongs to a different request; check source paths, checkpoint and worker count")
                if existing is None and any(args.output.glob("part-*-of-*")):
                    raise ValueError("Existing shards have no request.json; verify the output directory")
                pipeline.atomic_json(req_path, request)
                pending = [i for i in range(args.workers)
                           if not (shard_path(args.output, i, args.workers) / "complete.json").is_file()]
                if pending:
                    def on_gpu(snapshot):
                        status("waiting_gpu" if snapshot["state"] == "waiting" else "running",
                               resource=snapshot["state"], selected_uuids=snapshot["selected_uuids"])
                        print(f"GPU {snapshot['state']} ({snapshot['policy']}); wait={snapshot['wait_seconds']:.0f}s",
                              flush=True)

                    status("waiting_gpu")
                    gpu_count = next(n for n in (1, 2, 4, 8) if n >= len(pending))
                    with acquire_gpus(pool=args.gpu_pool, workers=gpu_count, budget_gb=args.gpu_budget_gb,
                                      reserve_gb=args.gpu_reserve_gb, policy=args.gpu_policy,
                                      lock_dir=args.gpu_lock_dir, on_status=on_gpu) as selected:
                        env.update(gpu_environment(selected, args.gpu_budget_gb))
                        launch_workers(args, pending, env)
                status("merging")
                summary = merge(args.output, request)
                if summary["count"] != TOTAL or summary["valid_count"] != TOTAL:
                    raise ValueError("Expected all 12,146 paired cross-planner scores")
                status("complete", finished=pipeline.now(), summary_sha256=pipeline.sha256(args.output / "summary.json"))
                print_report(args.output)
            except BaseException as exc:
                if previous.get("status") == "complete" and state["status"] == "verifying_sources":
                    # A mistaken relaunch must leave the previous verified result accessible.
                    pipeline.atomic_json(state_path, previous)
                else:
                    status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                           error=f"{type(exc).__name__}: {exc}")
                raise


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "report":
        print_report(args.output)
    else:
        path = args.output / "launcher.json"
        try:
            while True:
                state = pipeline.read_json(path) if path.is_file() else {"status": "not_started"}
                print(f"{state.get('updated', '')} {state['status']}: {state.get('error', '')}", flush=True)
                print_progress(args.output, state.get("workers", 0))
                if state["status"] == "complete":
                    print_report(args.output)
                if not args.watch or state["status"] in ("complete", "failed", "interrupted"):
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
