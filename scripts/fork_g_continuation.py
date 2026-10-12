#!/usr/bin/env python3
"""Fork a completed S05-G run without resetting its optimizer or random streams.

Invoke with PYTHONPATH pointing at the ORIGINAL, hash-verified training package.
This file intentionally never inserts the current checkout into sys.path. The
original run is read-only; only a previously nonexistent output is created.
"""

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from score_function.utils.config import load_config
from score_function.utils.planner_utils import planner_identity
from score_function.utils.train_utils import (
    atomic_write,
    file_hash,
    fingerprint,
    load_tensor,
    read_json,
    resolve_path,
    source_hashes,
    stable_seed,
    training_signature,
)
from score_function.utils.training_budget import training_budget

SOURCE_FILES = (
    "last.pt", "config.json", "model_info.json", "provenance.json", "status.json",
    "best.pt", "selection.json", "initial_validation.json", "history.json",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cache_identity(config, provenance, state):
    """Verify the published cache/augmentation identity without loading all NPZs."""
    index_path = Path(config["cache"])
    _require(index_path.resolve() == resolve_path(config, "cache").resolve(),
             "Source cache paths disagree")
    index = read_json(index_path)
    _require(index["metadata"]["planner"] == state["planner"],
             "Cache frozen planner identity changed")
    _require(index["metadata"]["ego_normalizer"] == state["ego_normalizer"],
             "Cache normalizer changed")
    clean_hash = file_hash(index_path)
    augmentation = config["training"].get("data_augmentation", {})
    candidates = [clean_hash]
    if augmentation.get("enabled"):
        from score_function.data_process import online_training

        protocol = dict(
            clean_cache=clean_hash,
            raw_manifest_fallback=None,
            protocol="official_state_perturbation_before_normalization_v1",
            implementation=file_hash(online_training.__file__),
            augmentation=augmentation,
        )
        candidates = [fingerprint(protocol)]
        manifest = resolve_path(config, "manifest")
        if manifest.is_file():
            protocol["raw_manifest_fallback"] = file_hash(manifest)
            candidates.append(fingerprint(protocol))
    _require(provenance["cache_sha256"] in candidates,
             "Actual cache/manifest/augmentation identity changed")
    return provenance["cache_sha256"]


def _expanded_rank_states(state, config, target_world):
    """Retain old RNG streams and seed new ranks without touching global RNGs."""
    old_world = len(state["ranks"])
    uses_cuda = [rank["rng"]["cuda"] is not None for rank in state["ranks"]]
    _require(len(set(uses_cuda)) == 1, "Source ranks disagree about CUDA RNG state")
    if target_world > old_world and uses_cuda[0]:
        _require(torch.cuda.is_available(),
                 "Expanding CUDA rank RNGs requires an available CUDA device; expose one GPU "
                 "to this preparation subprocess with CUDA_VISIBLE_DEVICES")
    ranks = list(state["ranks"])
    seeds = {}
    for rank in range(old_world, target_world):
        seed = stable_seed(config["training"]["seed"], "world_size_expansion",
                           state["epoch"], rank)
        rng = {
            "torch": torch.Generator(device="cpu").manual_seed(seed).get_state(),
            "cuda": (torch.Generator(device="cuda:0").manual_seed(seed).get_state()
                     if uses_cuda[0] else None),
            "python": random.Random(seed).getstate(),
            "numpy": np.random.RandomState(seed % (2 ** 32)).get_state(),
        }
        ranks.append({"rng": rng, "epoch_loss": 0.0, "epoch_count": 0})
        seeds[str(rank)] = {"seed": seed, "numpy_seed": seed % (2 ** 32)}
    return ranks, seeds


def fork_checkpoint(source_score, config_path, output, epochs_extra=100, *,
                    expected_world_size=4, target_world_size=None):
    """Prepare a complete-state continuation; world-size override is for CPU tests."""
    source = Path(source_score).resolve()
    output = Path(output).resolve()
    config_path = Path(config_path).resolve()
    _require(isinstance(epochs_extra, int) and not isinstance(epochs_extra, bool)
             and epochs_extra > 0, "epochs_extra must be a positive integer")
    _require(output.name == "score", "Output must be the new RUN/score directory")
    _require(not output.exists(), f"Output already exists: {output}")
    _require(not output.is_relative_to(source.parent)
             and not source.is_relative_to(output.parent),
             "Continuation output must be separate from the entire source run")
    original_hashes = {name: file_hash(source / name) for name in SOURCE_FILES}
    config_file_hash = file_hash(config_path)
    state = load_tensor(source / "last.pt")
    old = read_json(source / "config.json")
    info = read_json(source / "model_info.json")
    provenance = read_json(source / "provenance.json")
    status = read_json(source / "status.json")
    new = load_config(config_path)
    _require(state["config"] == old, "Source checkpoint/config.json disagree")
    _require(state["sources"] == provenance["sources"] == source_hashes(),
             "Original training source hashes do not match the imported package")
    _require(old["model"].get("temporal_attention") is True
             and not old["model"].get("neighbor_future", False)
             and old["model"].get("parameterization", "score") == "score"
             and old["training"]["sigma"] == state["sigma_score"] == 0.05,
             "Source must be the fixed-sigma S05-G score branch")
    _require(old["training"].get("early_stopping") is False
             and old["training"].get("lr_schedule") == "constant_after_warmup",
             "Continuation requires the original constant-LR, no-early-stop protocol")
    _require(state["planner"] == planner_identity(old), "Frozen DP identity changed")
    cache_hash = _cache_identity(old, provenance, state)
    world = info["world_size"]
    _require(world == expected_world_size and len(state["ranks"]) == world,
             "Original world size/rank states differ from the required continuation")
    target_world = world if target_world_size is None else target_world_size
    _require(isinstance(target_world, int) and not isinstance(target_world, bool)
             and target_world in (world, 2 * world),
             "Target world size must retain the source size or expand it exactly twofold")
    cfg = old["training"]
    _require(info["global_batch"] == cfg["batch_size"]
             and info["microbatch_per_rank"] == cfg["microbatch_size"]
             and cfg["batch_size"] % (world * cfg["microbatch_size"]) == 0
             and info["gradient_accumulation"]
             == cfg["batch_size"] // world // cfg["microbatch_size"],
             "Original batch/world-size/accumulation metadata disagree")
    _require(cfg["batch_size"] % (target_world * cfg["microbatch_size"]) == 0,
             "Global batch must divide target world size and unchanged microbatch exactly")
    signature = training_signature(old, cache_hash, world)
    _require(signature == state["signature"] == provenance["signature"],
             "Source training signature does not match its config/data/world size")
    old_budget = training_budget(cfg, info["train_samples"])
    _require(old_budget == state["training_budget"] == info["training_budget"]
             == status["training_budget"], "Source training budgets disagree")
    _require(state["finished"] is True and status["state"] == "complete"
             and state["termination"] == status["termination"] == "epoch_budget_exhausted"
             and state["cursor"] == 0
             and state["epoch"] == status["epoch"] == old_budget["planned_epochs"]
             and state["step"] == status["step"] == old_budget["planned_optimizer_updates"]
             and state["sample_presentations"] == status["sample_presentations"]
             == old_budget["planned_sample_presentations"],
             "Source is not a complete, consistent epoch-budget checkpoint")
    _require(all(rank["epoch_loss"] == 0 and rank["epoch_count"] == 0
                 and {"torch", "cuda", "python", "numpy"}.issubset(rank["rng"])
                 for rank in state["ranks"]), "Invalid epoch-boundary rank/RNG state")
    _require(state["history"] == read_json(source / "history.json"),
             "Checkpoint and published validation history disagree")
    best = load_tensor(source / "best.pt")
    selection = read_json(source / "selection.json")
    _require(best["val_dsm"] == selection["val_dsm"] == state["best"]
             and all(best[key] == selection[key]
                     for key in ("epoch", "step", "weight_kind", "criterion"))
             and best["planner"] == state["planner"]
             and best["config"]["model"] == old["model"]
             and best["sigma_score"] == state["sigma_score"]
             and best["cache_sha256"] == cache_hash,
             "Inherited best checkpoint/selection/provenance disagree")

    target_epochs = state["epoch"] + epochs_extra
    allowed = copy.deepcopy(old)
    allowed["output"] = str(output.parent)
    allowed["paths"]["run_dir"] = str(output.parent)
    allowed["training"]["max_epochs"] = target_epochs
    _require(new == allowed,
             "Only output, paths.run_dir and training.max_epochs may change for continuation")
    budget = training_budget(new["training"], info["train_samples"])
    _require(budget["planned_epochs"] == target_epochs,
             "New budget does not add exactly epochs_extra epochs")
    fork = dict(state)
    new_signature = training_signature(new, cache_hash, target_world)
    ranks, new_rank_seeds = _expanded_rank_states(state, new, target_world)
    fork.update(config=new, signature=new_signature, training_budget=budget,
                finished=False, termination=None, ranks=ranks)
    migration = dict(
        source_world_size=world, target_world_size=target_world,
        preserved_ranks=list(range(world)), new_rank_seeds=new_rank_seeds,
        exact_original_world_size_continuation=(target_world == world),
        note=("Original rank RNG streams retained."
              if target_world == world else
              "World size expanded at a completed epoch boundary. Optimizer and model state "
              "are retained; rank sample partitioning and new RNG streams change, so this "
              "is not a bitwise continuation of the original world size."),
    )
    metadata = dict(
        schema_version=1, kind="full_state_epoch_budget_continuation",
        created_unix=time.time(), source_score=str(source), output_score=str(output),
        source_files_sha256=original_hashes, config_file=str(config_path),
        config_file_sha256=config_file_hash, source_signature=signature,
        signature=new_signature, world_size=target_world, epochs_extra=epochs_extra,
        world_size_migration=migration,
        start_epoch=state["epoch"], start_step=state["step"],
        original_training_budget=old_budget, training_budget=budget,
        allowed_config_changes={
            "output": [old["output"], new["output"]],
            "paths.run_dir": [old["paths"]["run_dir"], new["paths"]["run_dir"]],
            "training.max_epochs": [cfg["max_epochs"], target_epochs],
        },
        inherited_best=dict(selection),
        note="Optimizer, raw/EMA weights, all rank RNGs, LR, history and counters retained. "
             "Inherited best.pt may precede the final epoch; export final EMA from last.pt.",
    )
    _require(all(file_hash(source / name) == digest
                 for name, digest in original_hashes.items())
             and file_hash(config_path) == config_file_hash,
             "Source or requested config changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    try:
        atomic_write(output / "last.pt", fork, tensor=True)
        for name in ("best.pt", "selection.json", "initial_validation.json", "history.json"):
            shutil.copy2(source / name, output / name)
            _require(file_hash(output / name) == original_hashes[name],
                     f"Inherited artifact changed while copying: {name}")
        atomic_write(output / "config.json", new)
        atomic_write(output / "model_info.json", {
            **info, "training_budget": budget, "world_size": target_world,
            "gradient_accumulation": cfg["batch_size"] // target_world // cfg["microbatch_size"],
        })
        atomic_write(output / "provenance.json", {
            **provenance, "signature": new_signature, "world_size_migration": migration,
        })
        atomic_write(output / "status.json", {
            "state": "prepared", "phase": "continuation", "epoch": state["epoch"],
            "step": state["step"], "sample_presentations": state["sample_presentations"],
            "training_budget": budget, "termination": None,
        })
        metadata["prepared_checkpoint_sha256"] = file_hash(output / "last.pt")
        _require(all(file_hash(source / name) == digest
                     for name, digest in original_hashes.items()),
                 "Source changed while the continuation was being written")
        atomic_write(output / "continuation.json", metadata)
    except BaseException:
        shutil.rmtree(output)
        raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-score", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs-extra", type=int, default=100)
    parser.add_argument("--target-world-size", type=int, default=None)
    args = parser.parse_args()
    metadata = fork_checkpoint(args.source_score, args.config, args.output, args.epochs_extra,
                               target_world_size=args.target_world_size)
    print(json.dumps(metadata, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
