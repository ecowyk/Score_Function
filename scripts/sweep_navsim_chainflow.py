#!/usr/bin/env python3
"""Train B64/E40 ChainFlow Score hyperparameter comparisons on the existing cache."""
import argparse
import copy
import signal
from pathlib import Path

import run_navsim_chainflow as pipeline


# Each name lists the factors changed relative to the completed B64/E40 baseline.
CORE = ("lr_wd_dropout", "lr", "wd", "dropout")
PAIRS = ("lr_wd", "lr_dropout", "wd_dropout")


def variant(reference, name):
    cfg = copy.deepcopy(reference)
    changes = name.split("_")
    if "lr" in changes:
        cfg["learning_rate"] = 1e-4
    if "wd" in changes:
        cfg["weight_decay"] = 0.05
    if "dropout" in changes:
        cfg["model"]["dropout"] = 0.20
    return cfg


def report(name, training, output):
    state_file = output / "state.json"
    history_file = training / "history.json"
    state = pipeline.read_json(state_file)["status"] if state_file.exists() else "pending"
    if (training / "TRAINING_COMPLETE.json").is_file():
        state = "complete"
    history = pipeline.read_json(history_file) if history_file.exists() else []
    if not history:
        print(f"{name:16} {state:12}", flush=True)
        return
    best = min(history, key=lambda row: row["val_dsm_ema"])
    last = history[-1]
    print(f"{name:16} {state:12} {last['epoch']:>3}     {best['epoch']:>3}       "
          f"{best['val_dsm_ema']:.6f}    {last['val_dsm_ema']:.6f}   "
          f"{last['diagnostic_train_dsm_ema']:.6f}", flush=True)


def run_trial(base, sweep, name, config):
    directory = sweep / name
    args = copy.copy(base)
    args.config = directory / "config.json"
    args.training = directory / "training"
    args.output = directory
    args.evaluation = directory / "evaluation"
    with pipeline.pipeline_lock(directory):
        pipeline.bind_plan(args.config, config)
        if pipeline.training_complete(args):
            print(f"{name}: reusing completed training", flush=True)
            return
        queue = pipeline.Queue(args)
        try:
            world = pipeline.cached_world(args, "train")
            queue.update()
            queue.gpu("train", lambda n: pipeline.train_command(args, n), workers=world)
            if not pipeline.training_complete(args):
                raise RuntimeError(f"{name}: trainer exited before verified completion")
            queue.update(status="complete", stage="complete", finished=pipeline.now())
        except BaseException as exc:
            queue.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=str(exc), finished=pipeline.now())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--root", type=Path, default=Path("/home/data/wyk"))
    parser.add_argument("--baseline-plan", type=Path, help="Completed B64/E40 plan.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Also train all three two-factor combinations")
    group.add_argument("--only", nargs="+", choices=CORE + PAIRS, help="Run specific groups")
    options = parser.parse_args(argv)
    paths = pipeline.paths_for(options.root, "b64-e40")
    plan_file = options.baseline_plan or paths["output"] / "plan.json"
    plan = pipeline.read_json(plan_file)
    if plan.get("recipe") != "b64-e40":
        parser.error("Use the completed B64/E40 pipeline plan")
    reference = plan["config"]
    base = pipeline.parse_args(["run", "--recipe", "b64-e40", "--root", str(options.root),
                                "--assets-from-plan", str(plan_file), "--loader-workers", str(reference["num_workers"])])
    reference_training = Path(plan["paths"]["training"])
    sweep = paths["output"].parent / "chainflow_score_b64_e40_sweep"
    names = options.only or (CORE + PAIRS if options.full else CORE)
    if (reference["learning_rate"], reference["weight_decay"], reference["model"]["dropout"]) != (2e-4, 0.01, 0.1):
        parser.error("The reference should use LR=2e-4, WD=0.01 and dropout=0.10")
    failures = []
    if options.command == "run":
        base.training = reference_training
        if pipeline.effective_config(base) != reference or not pipeline.training_complete(base):
            parser.error("The completed training and B64/E40 reference configuration must match")
        for name in names:
            print(f"\n{name}: {sweep / name}", flush=True)
            try:
                run_trial(base, sweep, name, variant(reference, name))
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(name)
                print(f"{name}: {exc}", flush=True)
        if failures:
            print(f"Groups requiring attention: {', '.join(failures)}", flush=True)
    print("group            state        epoch  best_epoch  best_val    latest_val  latest_train (EMA DSM)", flush=True)
    report("reference", reference_training, Path(plan["paths"]["output"]))
    for name in names:
        report(name, sweep / name / "training", sweep / name)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    main()
