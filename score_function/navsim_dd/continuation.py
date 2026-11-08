"""Explicit full-state continuation of a completed raw-weight cosine run."""

from collections.abc import Mapping
from contextlib import contextmanager
import copy
import fcntl
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import numpy as np
import torch

from score_function.navsim_dd.cache import NavsimScoreDataset, load_cache_metadata
from score_function.navsim_dd.runtime import CHECKPOINT_FORMAT
from score_function.navsim_dd.train import _check_config
from score_function.utils.train_utils import atomic_write, file_hash, fingerprint, load_tensor, read_json
from score_function.utils.training_budget import learning_rate_at_step


@contextmanager
def _locked_source(source):
    source = Path(source).resolve(strict=True)
    # Every trainer creates this lock. Open the existing file read/write without
    # truncating or writing: NFS exclusive flock requires a writable descriptor.
    with (source / ".train.lock").open("r+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Source training is active: {source}") from exc
        yield source


def _same_weights(left, right):
    return (
        isinstance(left, Mapping) and isinstance(right, Mapping) and bool(left)
        and left.keys() == right.keys()
        and all(
            isinstance(left[key], torch.Tensor) and isinstance(right[key], torch.Tensor)
            and left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
            and torch.equal(left[key], right[key])
            for key in left
        )
    )


def _validate_rng(states, world, device_type):
    if not isinstance(states, (list, tuple)) or len(states) != world:
        raise ValueError("Source checkpoint must preserve RNG for every training rank")
    for state in states:
        if not isinstance(state, Mapping) or set(state) != {"torch", "cuda", "python", "numpy"}:
            raise ValueError("Incomplete source rank RNG state")
        try:
            torch.Generator(device="cpu").set_state(state["torch"])
            random.Random().setstate(state["python"])
            np.random.RandomState().set_state(state["numpy"])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Invalid source rank RNG state") from exc
        if device_type == "cuda" and (
            not isinstance(state["cuda"], torch.Tensor)
            or state["cuda"].dtype != torch.uint8 or not state["cuda"].numel()
        ):
            raise ValueError("Missing CUDA RNG state in source checkpoint")
        if device_type == "cpu" and state["cuda"] is not None:
            raise ValueError("CPU source unexpectedly contains CUDA RNG state")


def _plan_locked(source, cache_dir, additional_epochs):
    if type(additional_epochs) is not int or additional_epochs <= 0:
        raise ValueError("additional_epochs must be a positive integer")
    original = read_json(source / "run.json")
    cfg = original["config"]
    _check_config(cfg)
    if cfg["lr_schedule"] != "dd_warmup_cosine" or cfg["ema_decay"] != 0:
        raise ValueError("Continuation requires a completed cosine run with raw weights")
    protocol = original["protocol"]
    world, device_type = protocol.get("world_size"), protocol.get("device_type")
    if type(world) is not int or world < 1 or device_type not in ("cpu", "cuda"):
        raise ValueError("Invalid source training world size or device")
    metadata = load_cache_metadata(cache_dir)
    expected_protocol = {
        "config": {key: value for key, value in cfg.items() if key != "num_workers"},
        "world_size": world, "device_type": device_type,
        "cache_fingerprint": fingerprint(metadata),
    }
    if protocol != expected_protocol or original.get("signature") != fingerprint(expected_protocol):
        raise ValueError("Source protocol/signature does not match its config and the selected cache")
    training = NavsimScoreDataset(cache_dir, "train")
    validation = NavsimScoreDataset(cache_dir, "val")
    batch = cfg["microbatch_size"] * world
    updates = (len(training) // batch if cfg.get("drop_last", True)
               else math.ceil(math.ceil(len(training) / world) / cfg["microbatch_size"]))
    if (updates < 1 or not len(validation) or original.get("updates_per_epoch") != updates
            or original.get("global_batch") != batch or original.get("train_samples") != len(training)
            or original.get("val_samples") != len(validation)):
        raise ValueError("Source training counts or update budget differ from the selected cache")
    # last.pt is the trusted local training checkpoint, including Python/NumPy
    # RNG and AdamW state. The portable final_raw.pt alone cannot resume training.
    last = load_tensor(source / "last.pt")
    fields = {"raw", "ema", "optimizer", "rng_by_rank", "best_weights", "history", "scaler", "step"}
    if (last.get("format") != "navsim-dd-train-v1" or not fields <= last.keys()
            or last.get("signature") != original["signature"]):
        raise ValueError("Source last.pt is incomplete or belongs to a different protocol")
    history = read_json(source / "history.json")
    epochs = cfg["epochs"]
    if (not isinstance(history, list) or len(history) != epochs or history != last["history"]
            or last["step"] != epochs * updates):
        raise ValueError("Source must contain a completed, matching history and optimizer step count")
    for epoch, row in enumerate(history):
        expected_lr = learning_rate_at_step(cfg, epoch * updates, updates, cfg["learning_rate"])
        if (row.get("epoch") != epoch + 1 or row.get("step") != (epoch + 1) * updates
                or not all(math.isfinite(row.get(key, float("nan"))) for key in ("train_dsm", "val_dsm", "lr"))
                or not math.isclose(row["lr"], expected_lr, rel_tol=1e-12, abs_tol=0)):
            raise ValueError("Source history is not contiguous or does not follow its recorded cosine schedule")
    if not _same_weights(last["raw"], last["ema"]):
        raise ValueError("Raw source checkpoint has inconsistent raw/EMA compatibility weights")
    if not isinstance(last["best_weights"], Mapping) or last["best_weights"].keys() != last["raw"].keys():
        raise ValueError("Source checkpoint is missing its best weights")
    optimizer = last["optimizer"]
    if not isinstance(optimizer, Mapping) or not optimizer.get("param_groups") or not optimizer.get("state"):
        raise ValueError("Source checkpoint is missing AdamW optimizer state")
    parameter_ids = [pid for group in optimizer["param_groups"] for pid in group.get("params", [])]
    if (not parameter_ids or len(set(parameter_ids)) != len(parameter_ids)
            or not set(optimizer["state"]) <= set(parameter_ids)
            or any(not {"step", "exp_avg", "exp_avg_sq"} <= state.keys()
                   for state in optimizer["state"].values())):
        raise ValueError("Source AdamW optimizer state is incomplete")
    _validate_rng(last["rng_by_rank"], world, device_type)
    precision = cfg.get("precision", "32-true")
    if last["scaler"] is not None and not isinstance(last["scaler"], Mapping):
        raise ValueError("Invalid source GradScaler state")
    if precision == "16-mixed" and not last["scaler"]:
        raise ValueError("Mixed-precision source is missing its GradScaler state")
    final = torch.load(source / "final_raw.pt", map_location="cpu", weights_only=True)
    final_metadata = {
        "format": CHECKPOINT_FORMAT, "epoch": epochs, "weight_kind": "raw",
        "training_precision": precision, "model": cfg["model"],
        "sigma": cfg["sigma"], "gamma": cfg["gamma"], "normalizer": metadata["normalizer"],
        "conditioner_identity": metadata["conditioner_identity"], "val_dsm": history[-1]["val_dsm"],
    }
    if any(final.get(key) != value for key, value in final_metadata.items()) or not _same_weights(
        last["raw"], final.get("state_dict")
    ):
        raise ValueError("Source final_raw.pt is inconsistent with its completed training checkpoint")
    cfg = copy.deepcopy(cfg)
    horizon = cfg.get("lr_schedule_epochs", epochs)
    cfg.update(epochs=epochs + additional_epochs, lr_schedule_epochs=horizon)
    _check_config(cfg)
    provenance = {
        "version": 1, "policy": "preserve_cosine_horizon_then_hold_minimum",
        "source_output": str(source), "source_epochs": epochs,
        "additional_epochs": additional_epochs, "target_epochs": cfg["epochs"],
        "lr_schedule_epochs": horizon, "source_signature": original["signature"],
        "source_sha256": {name: file_hash(source / name) for name in
                          ("run.json", "last.pt", "history.json", "final_raw.pt")},
    }
    new_protocol = {**protocol, "config": {key: value for key, value in cfg.items() if key != "num_workers"}}
    signature = fingerprint(new_protocol)
    new_run = {**copy.deepcopy(original), "config": cfg, "protocol": new_protocol,
               "signature": signature, "continuation": provenance}
    new_last = {**last, "signature": signature, "continuation": provenance}
    return {"config": cfg, "run": new_run, "last": new_last, "provenance": provenance}


def plan_continuation(source, cache_dir, additional_epochs=150):
    """Validate a completed source without writes; return config/run/last/provenance.

    The returned ``last`` retains all training tensors and rank RNG states.
    This operation does not initialize CUDA or alter the process RNG streams.
    """
    with _locked_source(source) as source:
        return _plan_locked(source, cache_dir, additional_epochs)


def prepare_continuation(source, output, cache_dir, additional_epochs=150):
    """Atomically create a new resume directory, preserving the source verbatim.

    Return the same dictionary as :func:`plan_continuation`. The new directory
    contains config.json, run.json, last.pt, history.json and continuation.json.
    Launch the normal trainer with its config.json and ``--resume``; the original
    strict resume signature remains in force for every subsequent restart.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Continuation output already exists: {output}")
    output = output.resolve()
    with _locked_source(source) as source:
        if output == source or source in output.parents:
            raise ValueError("Continuation output must be outside the source run")
        plan = _plan_locked(source, cache_dir, additional_epochs)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{output.name}.continuation-", dir=output.parent) as temporary:
            stage = Path(temporary) / "run"
            stage.mkdir()
            atomic_write(stage / "config.json", plan["config"])
            atomic_write(stage / "run.json", plan["run"])
            atomic_write(stage / "history.json", plan["last"]["history"])
            atomic_write(stage / "continuation.json", plan["provenance"])
            atomic_write(stage / "last.pt", plan["last"], tensor=True)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Continuation output already exists: {output}")
            stage.rename(output)
        return plan
