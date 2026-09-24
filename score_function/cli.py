"""Explicit independent stages. No automatic benchmark or parameter sweep."""

import argparse
import json

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
            "smoke",
            "train",
            "evaluate",
            "evaluate-planner",
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
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config, args.root, args.overrides)
    if args.device:
        config["runtime"]["device"] = args.device
    if args.resume and args.command != "train":
        parser.error(
            "--resume applies only to training; prepare/cache resume completed shards automatically"
        )
    if args.command in ("evaluate", "evaluate-planner") and not args.checkpoint:
        parser.error("Evaluation requires --checkpoint score/best.pt")
    if args.command == "check-config":
        print(json.dumps(config, indent=2))
        return
    if args.command == "preflight":
        from score_function.tools.check_environment import check

        check(config, args.gpus)
        return
    configure_runtime(config, require_cuda=args.command != "prepare")
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
    elif args.command == "train":
        from score_function.train import train

        train(config, resume=args.resume)
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
