#!/usr/bin/env python3
"""Derive a DiT comparison config without resetting the current training recipe."""

import argparse
import copy
import json
from pathlib import Path


def prepare(source, destination, run_dir, *, root=None):
    source, destination = (
        Path(source).expanduser().resolve(),
        Path(destination).expanduser().resolve(),
    )
    if source == destination:
        raise ValueError("Choose a new config path; the source experiment is retained")
    original = json.loads(source.read_text(encoding="utf-8"))
    if original["model"].get("parameterization", "score") != "score":
        raise ValueError("Use a direct-score source config for an architecture-only comparison")
    if original["model"].get("neighbor_future", False):
        raise ValueError("Use an ego-only source config; DiT does not consume neighbor futures")
    if original["model"]["architecture"] not in ("temporal_score_function", "dit_score_function"):
        raise ValueError("Unsupported source model architecture")
    config = copy.deepcopy(original)
    defaults = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/score_function_dit.json").read_text(
            encoding="utf-8"
        )
    )
    config["model"] = copy.deepcopy(defaults["model"])
    # Retain common dimensions/dropout, and existing DiT tuning when re-deriving.
    for key in config["model"]:
        if key not in ("architecture", "parameterization") and key in original["model"]:
            config["model"][key] = original["model"][key]
    if root is not None:
        config["paths"]["root"] = str(Path(root).expanduser().resolve())
    workspace = Path(config["paths"]["root"]).expanduser().resolve()
    target = Path(run_dir).expanduser()
    target = target.resolve() if target.is_absolute() else (workspace / target).resolve()
    old_run = Path(original["paths"]["run_dir"]).expanduser()
    old_root = Path(original["paths"]["root"]).expanduser().resolve()
    old_run = old_run.resolve() if old_run.is_absolute() else (old_root / old_run).resolve()
    if target == old_run:
        raise ValueError("Choose a new run directory; do not replace the source experiment")
    if target.exists():
        raise FileExistsError(f"Choose a new run directory: {target}")
    config["paths"]["run_dir"] = str(target)
    config.pop("output", None)
    config.pop("cache", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--root", help="Same workspace root used by the source run")
    args = parser.parse_args()
    result = prepare(args.source_config, args.output, args.run_dir, root=args.root)
    print(f"Fresh DiT config: {result}")
    print("Training, sigma, data, frozen conditioning and evaluation settings retained.")
    print("Start a new run without --resume; this command does not launch training.")


if __name__ == "__main__":
    main()
