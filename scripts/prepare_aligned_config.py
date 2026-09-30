#!/usr/bin/env python3
"""Create a fresh DP-aligned training config while retaining an existing model/data setup."""

import argparse
import copy
import json
from pathlib import Path


def prepare(source, destination, run_dir, *, root=None, microbatch=256, gpus=8):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Choose a new config path; the source experiment is retained")
    original = json.loads(source.read_text(encoding="utf-8"))
    defaults = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/score_function.json").read_text(
            encoding="utf-8"
        )
    )
    global_batch = defaults["training"]["batch_size"]
    if gpus < 1 or microbatch < 1 or global_batch % (gpus * microbatch):
        raise ValueError("2048 must be divisible by GPU count times microbatch")
    config = copy.deepcopy(original)
    settings = copy.deepcopy(defaults["training"])
    # Keep the scientific target and reproducible scene/noise choices of the source.
    for key in ("sigma", "seed", "validation_seed", "validation_repeats", "num_workers"):
        if key in original["training"]:
            settings[key] = original["training"][key]
    settings["microbatch_size"] = microbatch
    config["training"] = settings
    config["long_refinement"] = copy.deepcopy(defaults["long_refinement"])
    if root is not None:
        config["paths"]["root"] = str(Path(root).expanduser().resolve())
    workspace = Path(config["paths"]["root"]).expanduser().resolve()
    target = Path(run_dir).expanduser()
    target = target.resolve() if target.is_absolute() else (workspace / target).resolve()
    if target.exists():
        raise FileExistsError(f"Choose a new run directory: {target}")
    old_run = Path(original["paths"]["run_dir"]).expanduser()
    old_root = Path(original["paths"]["root"]).expanduser().resolve()
    old_run = old_run.resolve() if old_run.is_absolute() else (old_root / old_run).resolve()
    if target == old_run:
        raise ValueError("Choose a new run directory; do not replace the source experiment")
    config["paths"]["run_dir"] = str(target)
    # Resolved runtime fields are recomputed by load_config when training starts.
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
    parser.add_argument("--root")
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--gpus", type=int, default=8, help="Number of training processes")
    args = parser.parse_args()
    result = prepare(
        args.source_config,
        args.output,
        args.run_dir,
        root=args.root,
        microbatch=args.microbatch,
        gpus=args.gpus,
    )
    print(f"Fresh training config: {result}")
    print("Model, sigma and data paths retained; DP training budget and augmentation applied.")
    print("This command does not train, resume, prepare data or run simulation.")


if __name__ == "__main__":
    main()
