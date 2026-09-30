"""Explicit independent stages. No automatic benchmark or parameter sweep."""

import argparse
import json
import os

from score_function.utils.config import configure_runtime, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score Function: time-independent ego score refinement"
    )
    parser.add_argument(
        "command",
        choices=(
            "check-config",
            "preflight",
            "check-ddp",
            "prepare",
            "cache",
            "cache-neighbors",
            "smoke",
            "train",
            "evaluate",
            "evaluate-planner",
            "visualize-refinement",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--root")
    parser.add_argument("--device")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Existing dotted.key=JSON override; repeatable",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--progress",
        choices=("auto", "on", "off"),
        help="Terminal bars (auto/on), or periodic plain progress lines (off)",
    )
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--steps", type=int, help="Long-refinement diagnostic update budget")
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="+",
        help="Long-refinement trajectory snapshot iterations",
    )
    parser.add_argument(
        "--initializations",
        nargs="+",
        choices=("planner", "expert_noise"),
        help="Long-refinement starting trajectories; defaults to both",
    )
    parser.add_argument("--seed", type=int, help="Long-refinement reproducibility seed")
    parser.add_argument("--gamma", type=float, help="Long-refinement update step factor")
    args = parser.parse_args(argv)
    if args.progress is not None:
        os.environ["SCORE_FUNCTION_PROGRESS"] = args.progress
    if args.command == "train" and int(os.environ.get("RANK", "0")) == 0:
        print(f"Loading training config: {args.config}", flush=True)
    config = load_config(args.config, args.root, args.overrides)
    if args.device:
        config["runtime"]["device"] = args.device
    if args.resume and args.command != "train":
        parser.error(
            "--resume applies only to training; prepare/cache resume completed shards automatically"
        )
    if (
        args.command in ("evaluate", "evaluate-planner", "visualize-refinement")
        and not args.checkpoint
    ):
        parser.error("Evaluation requires --checkpoint score/best.pt")
    diagnostic_options = (
        args.steps,
        args.snapshot_steps,
        args.initializations,
        args.seed,
        args.gamma,
    )
    if args.command != "visualize-refinement" and any(
        value is not None for value in diagnostic_options
    ):
        parser.error(
            "--steps/--snapshot-steps/--initializations/--seed/--gamma require visualize-refinement"
        )
    if args.command == "check-config":
        print(json.dumps(config, indent=2))
        return
    if args.command == "preflight":
        from score_function.tools.check_environment import check

        check(config, args.gpus)
        return
    if args.command == "train" and int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Initializing {config['runtime']['device']} | "
            f"parameterization={config['model']['parameterization']} | "
            f"output={config['output']}",
            flush=True,
        )
    evaluation = args.command in ("evaluate", "evaluate-planner", "visualize-refinement")
    configure_runtime(
        config,
        require_cuda=args.command != "prepare"
        and (not evaluation or str(config["runtime"]["device"]).startswith("cuda")),
    )
    if args.command == "check-ddp":
        from score_function.tools.check_environment import check_ddp

        check_ddp(config)
    elif args.command == "prepare":
        from score_function.data_process.data_processor import prepare

        if config["data"]["prepare_raw"]:
            prepare(config)
        else:
            print("Reusing the explicit read-only NPZ manifest; raw preprocessing skipped.")
    elif args.command == "cache":
        from score_function.data_process.feature_cache import build_cache

        build_cache(config)
    elif args.command == "smoke":
        from score_function.tools.check_small_batch import run_smoke

        run_smoke(config)
    elif args.command == "cache-neighbors":
        from score_function.data_process.neighbor_cache import build_neighbor_cache

        build_neighbor_cache(config)
    elif args.command == "train":
        from score_function.train import train

        train(config, resume=args.resume)
    elif args.command == "visualize-refinement":
        from score_function.evaluation.long_refinement import diagnose_refinement

        diagnose_refinement(
            config,
            args.checkpoint,
            split=args.split,
            output=args.output,
            max_samples=args.max_samples,
            steps=args.steps,
            snapshot_steps=args.snapshot_steps,
            initializations=args.initializations,
            seed=args.seed,
            gamma=args.gamma,
        )
    else:
        from score_function.evaluation.diagnostics import evaluate
        from score_function.evaluation.planner_evaluation import evaluate_planner

        action = evaluate if args.command == "evaluate" else evaluate_planner
        action(
            config,
            args.checkpoint,
            split=args.split,
            output=args.output,
            max_samples=args.max_samples,
        )
