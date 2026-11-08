"""Train an independent direct score on cached DiffusionDrive conditions and NAVSIM poses."""

from __future__ import annotations

import argparse
import copy
import fcntl
import math
import os
import random
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler

from score_function.model.score_branch import build_model
from score_function.navsim_dd.cache import NavsimScoreDataset, load_cache_metadata
from score_function.navsim_dd.runtime import CHECKPOINT_FORMAT
from score_function.train_epoch import train_epoch, validate_epoch
from score_function.utils import ddp
from score_function.utils.dataset import EpochBatchSampler, build_data_loader
from score_function.utils.progress import report
from score_function.utils.train_utils import (
    atomic_write,
    capture_rank_rng,
    fingerprint,
    load_tensor,
    read_json,
    restore_rank_rng,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "navsim_dd_s10_g.json"
READER_ARCHITECTURE = "dd_scene_reader_score"
READER_TRAIN_FORMAT = "navsim-dd-reader-train-v1"


def is_reader_config(cfg):
    return cfg["model"].get("architecture") == READER_ARCHITECTURE


def load_training_data(cache_dir, cfg):
    """Dispatch by the declared model, so old compressed caches cannot train the reader."""
    if is_reader_config(cfg):
        from score_function.navsim_dd.reader_cache import DDReaderDataset, load_reader_cache_metadata

        metadata = load_reader_cache_metadata(cache_dir)
        return metadata, DDReaderDataset(cache_dir, "train"), DDReaderDataset(cache_dir, "val")
    metadata = load_cache_metadata(cache_dir)
    return metadata, NavsimScoreDataset(cache_dir, "train"), NavsimScoreDataset(cache_dir, "val")


def build_training_model(cfg, metadata, cache_dir, device):
    """Load fixed DD reader weights once, before creating the trainable optimizer."""
    if not is_reader_config(cfg):
        return build_model(cfg, device)
    from score_function.navsim_dd.reader import build_reader_score
    from score_function.utils.train_utils import file_hash

    root = Path(cache_dir).resolve()
    record = metadata["reader_assets"]
    path = (root / record["path"]).resolve()
    if root not in path.parents or file_hash(path) != record["sha256"]:
        raise ValueError("Frozen DD reader assets changed after cache publication")
    assets = load_tensor(path)
    if assets.get("spec") != record["spec"]:
        raise ValueError("Frozen DD reader architecture differs from cache metadata")
    return build_reader_score(cfg["model"], metadata["normalizer"], assets, device)


def _check_config(cfg):
    model = cfg["model"]
    if model.get("architecture") not in ("temporal_score_function", READER_ARCHITECTURE):
        raise ValueError("Unsupported NAVSIM score architecture")
    for key, value in {
        "parameterization": "score",
        "future_len": 8,
        "input_dim": 4,
        "context_dim": 256,
        "temporal_attention": True,
        "neighbor_future": False,
    }.items():
        if model.get(key) != value:
            raise ValueError(f"NAVSIM S10-G requires model.{key}={value!r}")
    for key in ("epochs", "microbatch_size", "validation_repeats"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num_workers", "warmup_epochs"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("sigma", "learning_rate", "warmup_learning_rate"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("gamma", "weight_decay", "gradient_clip"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 <= cfg["ema_decay"] < 1 or cfg["lr_schedule"] not in (
        "constant_after_warmup", "dd_warmup_cosine",
    ):
        raise ValueError("Invalid EMA decay or learning-rate schedule")
    if cfg.get("precision", "32-true") not in ("32-true", "16-mixed"):
        raise ValueError("precision must be '32-true' or '16-mixed'")
    if type(cfg.get("drop_last", True)) is not bool:
        raise ValueError("drop_last must be a boolean")
    if "checkpoint_every_epochs" in cfg:
        interval = cfg["checkpoint_every_epochs"]
        if type(interval) is not int or interval <= 0:
            raise ValueError("checkpoint_every_epochs must be a positive integer")
        start = cfg.get("checkpoint_start_epoch", interval)
        if type(start) is not int or start <= 0:
            raise ValueError("checkpoint_start_epoch must be a positive integer")
    elif "checkpoint_start_epoch" in cfg:
        raise ValueError("checkpoint_start_epoch requires checkpoint_every_epochs")
    if cfg["lr_schedule"] == "dd_warmup_cosine":
        minimum = cfg.get("min_learning_rate")
        if minimum is None or not math.isfinite(minimum) or not 0 <= minimum <= cfg["learning_rate"]:
            raise ValueError("min_learning_rate must be finite and between zero and learning_rate")
        horizon = cfg.get("lr_schedule_epochs", cfg["epochs"])
        if type(horizon) is not int or not 0 < cfg["warmup_epochs"] < horizon <= cfg["epochs"]:
            raise ValueError("DD cosine requires 0 < warmup_epochs < lr_schedule_epochs <= epochs")
    elif "lr_schedule_epochs" in cfg:
        raise ValueError("lr_schedule_epochs is only valid for dd_warmup_cosine")


@contextmanager
def _lock_output(output):
    """Protect one shared output directory from competing rank-zero writers."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another trainer owns {output}") from exc
        yield


def _rank_zero(action):
    """Propagate a filesystem failure before other ranks enter their next collective."""
    error = None
    if ddp.rank() == 0:
        try:
            action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = ddp.gather_objects(error)
    if errors[0]:
        raise RuntimeError(errors[0])


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def train(cache_dir, output, cfg, device, *, resume=False):
    """Call after initializing torchrun. Recovery starts at the latest durable epoch."""
    _check_config(cfg)
    precision = cfg.get("precision", "32-true")
    if precision == "16-mixed" and device.type != "cuda":
        raise ValueError("16-mixed training requires CUDA; use 32-true for CPU runs")
    output = Path(output)
    rank, world = ddp.rank(), ddp.world_size()
    metadata, training, validation = load_training_data(cache_dir, cfg)
    reader = is_reader_config(cfg)
    train_format = READER_TRAIN_FORMAT if reader else "navsim-dd-train-v1"
    batch_size = cfg["microbatch_size"] * world
    drop_last = cfg.get("drop_last", True)
    updates = (
        len(training) // batch_size if drop_last
        else math.ceil(math.ceil(len(training) / world) / cfg["microbatch_size"])
    )
    if updates < 1 or not len(validation):
        raise ValueError("Need nonempty train/validation splits and at least one training batch. "
                         "With drop_last=True, reduce --batch-size for a small cache.")
    # Cosine depends on the complete training horizon. Legacy constant-LR runs
    # may extend epochs, and absent precision stays absent in their signature.
    excluded = {"num_workers"}
    if cfg["lr_schedule"] == "constant_after_warmup":
        excluded.add("epochs")
    protocol = {
        "config": {k: v for k, v in cfg.items() if k not in excluded},
        "world_size": world,
        "device_type": device.type,
        "cache_fingerprint": fingerprint(metadata),
    }
    signature = fingerprint(protocol)
    last_path = output / "last.pt"

    def prepare_output():
        if resume:
            if not (output / "run.json").is_file():
                raise ValueError("No run.json in resume output")
            if read_json(output / "run.json")["signature"] != signature:
                raise ValueError("Resume rejected: training config, cache, device or world size changed")
            if not last_path.exists() and (output / "history.json").exists():
                raise ValueError("Run contains trained epochs but its resume checkpoint is missing")
        else:
            if any(p.name != ".train.lock" for p in output.iterdir()):
                raise ValueError("Training output is not empty. Use --resume or choose a new output.")
            atomic_write(output / "run.json", {
                "signature": signature, "config": cfg, "protocol": protocol,
                "updates_per_epoch": updates, "global_batch": batch_size,
                "train_samples": len(training), "val_samples": len(validation),
                "drop_last": drop_last,
                "train_presentations_per_epoch": (
                    updates * batch_size if drop_last else math.ceil(len(training) / world) * world
                ),
                "sigma_in_physical_channels": [cfg["sigma"] * s for s in metadata["normalizer"]["std"]],
            })

    with ExitStack() as stack:
        _rank_zero(lambda: stack.enter_context(_lock_output(output)))
        _rank_zero(prepare_output)
        _seed(cfg["seed"])
        model = build_training_model(cfg, metadata, cache_dir, device)
        wrapped = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        ) if world > 1 else None
        # With decay zero the validated/exported model is the raw model. No
        # separate averaged network is needed for the DD-aligned experiment.
        ema = copy.deepcopy(model).eval().requires_grad_(False) if cfg["ema_decay"] else None
        validation_model = model if ema is None else ema
        weight_kind = "raw" if ema is None else "ema"
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        scaler = torch.cuda.amp.GradScaler() if precision == "16-mixed" else None
        history, step, best_weights = [], 0, None
        _seed(cfg["seed"] + rank + 1)
        if resume and last_path.exists():
            last = load_tensor(last_path)
            if last.get("format") != train_format or last["signature"] != signature:
                raise ValueError("Resume checkpoint does not match this NAVSIM experiment")
            history, step, best_weights = last["history"], last["step"], last["best_weights"]
            model.load_state_dict(last["raw"])
            if ema is not None:
                ema.load_state_dict(last["ema"])
            optimizer.load_state_dict(last["optimizer"])
            if scaler is not None:
                if last.get("scaler") is None:
                    raise ValueError("Mixed-precision resume checkpoint is missing GradScaler state")
                scaler.load_state_dict(last["scaler"])
            restore_rank_rng(last["rng_by_rank"][rank], device)
            del last
        if len(history) > cfg["epochs"]:
            raise ValueError("--epochs is smaller than the number of completed epochs")

        def refresh_run_metadata():
            # Record an extended constant-LR target only after checkpoint
            # validation succeeds. The compatible protocol/signature is stable.
            record = read_json(output / "run.json")
            record.update(config=cfg, protocol=protocol, signature=signature)
            atomic_write(output / "run.json", record)

        if resume:
            _rank_zero(refresh_run_metadata)

        def publish_exports(final=False, repair_best=False):
            """Derive deployment files from the committed epoch, also after a failed write."""
            epoch = len(history)
            if not epoch:
                return
            best_row = min(history, key=lambda row: row["val_dsm"])
            common = {
                "format": CHECKPOINT_FORMAT, "model": cfg["model"],
                "sigma": cfg["sigma"], "gamma": cfg["gamma"],
                "normalizer": metadata["normalizer"], "conditioner_identity": metadata["conditioner_identity"],
                "weight_kind": weight_kind, "training_precision": precision,
            }
            if reader:
                from score_function.navsim_dd.reader_runtime import CHECKPOINT_FORMAT as READER_CHECKPOINT_FORMAT

                common.update(format=READER_CHECKPOINT_FORMAT, reader_spec=model.reader_spec)
            atomic_write(output / "history.json", history)
            if repair_best or best_row["epoch"] == epoch:
                best_export = {
                    **common, "state_dict": best_weights, "epoch": best_row["epoch"],
                    "val_dsm": best_row["val_dsm"],
                }
                atomic_write(output / "best_ema.pt", best_export, tensor=True)
                if ema is None:
                    atomic_write(output / "best_raw.pt", best_export, tensor=True)
            interval = cfg.get("checkpoint_every_epochs")
            start = cfg.get("checkpoint_start_epoch", interval)
            periodic = interval is not None and epoch >= start and (epoch - start) % interval == 0
            if final or periodic:
                current_export = {
                    **common, "state_dict": validation_model.state_dict(), "epoch": epoch,
                    "val_dsm": history[-1]["val_dsm"],
                }
            if periodic:
                # last.pt has already committed this epoch. This same path runs
                # on resume, repairing a snapshot interrupted after that commit.
                atomic_write(output / "checkpoints" / f"epoch_{epoch:04d}_{weight_kind}.pt",
                             current_export, tensor=True)
            if final:
                atomic_write(output / "final_ema.pt", current_export, tensor=True)
                if ema is None:
                    atomic_write(output / "final_raw.pt", current_export, tensor=True)

        if resume and history:
            # Republish even when last.pt was committed just before the job stopped.
            _rank_zero(lambda: publish_exports(final=len(history) == cfg["epochs"], repair_best=True))
        if len(history) < cfg["epochs"]:
            _rank_zero(lambda: (output / "final_ema.pt").unlink(missing_ok=True))
            if ema is None:
                _rank_zero(lambda: (output / "final_raw.pt").unlink(missing_ok=True))
        report(f"NAVSIM score train: {len(training)} train, {len(validation)} val, "
               f"world={world}, global_batch={batch_size}, updates/epoch={updates}, "
               f"precision={precision}, validation_weights={weight_kind}")
        for epoch in range(len(history), cfg["epochs"]):
            epoch_started = time.perf_counter()
            if drop_last:
                sampler = EpochBatchSampler(
                    len(training), batch_size, cfg["microbatch_size"], rank, world,
                    cfg["seed"], epoch,
                )
            else:
                # Match Lightning's DDP sampler padding and keep the final
                # partial per-rank batch, as in native DiffusionDrive training.
                distributed_sampler = DistributedSampler(
                    training, num_replicas=world, rank=rank, shuffle=True,
                    seed=cfg["seed"], drop_last=False,
                )
                distributed_sampler.set_epoch(epoch)
                sampler = BatchSampler(distributed_sampler, cfg["microbatch_size"], drop_last=False)
            loader = build_data_loader(training, cfg, batch_sampler=sampler)
            loss_sum = 0.
            sample_count = 0
            skipped_updates = 0
            for result in train_epoch(
                loader, model, optimizer, cfg, ema, device, start_step=step, start_update=0,
                updates_per_epoch=updates, accumulation=1, next_lr=cfg["learning_rate"],
                distributed_model=wrapped, scaler=scaler,
            ):
                step = result["step"]
                if drop_last:
                    loss_sum += result["train_dsm"]
                else:
                    local_samples = result["sample_presentations"] // world
                    loss_sum += result["train_dsm"] * local_samples
                    sample_count += local_samples
                skipped_updates += int(result.get("optimizer_step_skipped", False))
            loss_sum = ddp.sum_tensor(torch.tensor(loss_sum, dtype=torch.float64, device=device)).item()
            denominator = world * updates
            if not drop_last:
                denominator = int(ddp.sum_tensor(
                    torch.tensor(sample_count, dtype=torch.int64, device=device)
                ).item())
            if scaler is not None:
                skipped_updates = int(ddp.sum_tensor(
                    torch.tensor(skipped_updates, dtype=torch.int64, device=device)
                ).item())
                if skipped_updates == world * updates:
                    raise FloatingPointError("Every optimizer update was skipped by AMP in this epoch")
            # Keep validation in FP32 for comparable DSM across training precision.
            validation_started = time.perf_counter()
            val = validate_epoch({weight_kind: validation_model}, validation, cfg, device)[weight_kind]
            validation_seconds = time.perf_counter() - validation_started
            row = {
                "epoch": epoch + 1, "step": step,
                "train_dsm": loss_sum / denominator, "val_dsm": val, "lr": result["lr"],
            }
            if scaler is not None:
                row.update(amp_skipped_rank_updates=skipped_updates, amp_scale=scaler.get_scale())
            if not history or val < min(item["val_dsm"] for item in history):
                best_weights = {k: v.detach().cpu().clone() for k, v in validation_model.state_dict().items()}
            history.append(row)
            rank_rng = ddp.gather_objects(capture_rank_rng(device))

            def save():
                atomic_write(last_path, {
                    "format": train_format, "signature": signature, "step": step,
                    "raw": model.state_dict(), "ema": validation_model.state_dict(),
                    "optimizer": optimizer.state_dict(), "rng_by_rank": rank_rng,
                    "best_weights": best_weights, "history": history,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                }, tensor=True)
                publish_exports(final=len(history) == cfg["epochs"])

            _rank_zero(save)
            amp_status = (
                f"  AMP skipped rank-updates {skipped_updates}/{world * updates}, "
                f"scale {row['amp_scale']:g}" if scaler is not None else ""
            )
            report(f"epoch {row['epoch']}  train DSM {row['train_dsm']:.6f}  val DSM {val:.6f}{amp_status}"
                   f"  elapsed {time.perf_counter() - epoch_started:.1f}s"
                   f"  train {validation_started - epoch_started:.1f}s"
                   f"  validation {validation_seconds:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Samples per rank and update (default 64)")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume from last complete epoch")
    args = parser.parse_args()
    cfg = read_json(args.config)
    for name, value in (
        ("epochs", args.epochs), ("microbatch_size", args.batch_size), ("num_workers", args.workers),
    ):
        if value is not None:
            cfg[name] = value
    device = torch.device(args.device)
    if device.type == "cuda":
        index = int(os.environ.get("LOCAL_RANK", device.index or 0))
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        parser.error("--device must be cuda or cpu")
    ddp.setup(device)
    try:
        train(args.cache_dir, args.output, cfg, device, resume=args.resume)
    finally:
        ddp.close()


if __name__ == "__main__":
    main()
