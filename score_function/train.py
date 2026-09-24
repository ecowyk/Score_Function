"""Training orchestration: data, optimizer state, checkpoint selection, and early stopping."""

import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from score_function.model.score_branch import build_model
from score_function.train_epoch import train_epoch, validate_epoch
from score_function.utils import ddp
from score_function.utils.config import METHOD
from score_function.utils.dataset import (
    EpochBatchSampler,
    ShardedDataset,
    build_data_loader,
)
from score_function.utils.lr_schedule import ValidationPlateau
from score_function.utils.planner_utils import planner_identity
from score_function.utils.train_utils import (
    atomic_write,
    capture_rank_rng,
    load_tensor,
    restore_rank_rng,
    source_hashes,
    training_signature,
)


def train(config, resume=False, device_override=None, stop_after_updates=None, model_factory=None):
    """stop_after_updates is used only by isolated integration tests, not the CLI."""
    device = torch.device(device_override or config["runtime"]["device"])
    ddp.setup(device)
    output = Path(config["output"]) / "score"
    try:
        if ddp.rank() == 0:
            if resume:
                if not (output / "last.pt").is_file():
                    raise FileNotFoundError("--resume requires an existing score/last.pt")
            else:
                output.mkdir(parents=True, exist_ok=False)
            atomic_write(output / "status.json", {"state": "initializing", "resume": resume})
        ddp.barrier()
        return _train(config, output, device, resume, stop_after_updates, model_factory)
    except Exception as exc:
        if output.exists():
            atomic_write(output / f"failure_rank{ddp.rank()}.json", {"error": repr(exc)})
            if ddp.rank() == 0:
                atomic_write(output / "status.json", {"state": "failed", "error": repr(exc)})
        raise
    finally:
        ddp.close()


def _train(config, output, device, resume, stop_after_updates, model_factory):
    cfg = config["training"]
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    training, validation = (
        ShardedDataset(config["cache"], "train"),
        ShardedDataset(config["cache"], "val"),
    )
    rank, world = ddp.rank(), ddp.world_size()
    batch_size, microbatch = cfg["batch_size"], cfg["microbatch_size"]
    sampler_check = EpochBatchSampler(len(training), batch_size, microbatch, rank, world)
    updates_per_epoch = sampler_check.updates
    accumulation = batch_size // world // microbatch
    signature = training_signature(config, training.sha256, world)
    state = load_tensor(output / "last.pt") if resume else None
    if state and (state["signature"] != signature or state["sources"] != source_hashes()):
        raise ValueError("Resume configuration, data, world size or source code changed")
    identity = planner_identity(config)
    metadata = training.metadata
    if metadata["planner"] != identity:
        raise ValueError("Cache and current frozen Planner/normalizers do not match")
    normalizer = metadata["ego_normalizer"]
    if (
        len(normalizer["mean"]) != 4
        or len(normalizer["std"]) != 4
        or any(x <= 0 or not math.isfinite(x) for x in normalizer["std"])
    ):
        raise ValueError("Invalid cached ego normalizer")
    if state and state["planner"] != identity:
        raise ValueError("Frozen Planner changed since checkpoint")
    saved_config = copy.deepcopy(config)
    model = (model_factory or build_model)(config, device).to(device)
    if state:
        model.load_state_dict(state["score_branch"], strict=True)
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if world > 1
        else model
    )
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    plateau = ValidationPlateau(**cfg["plateau"])
    epoch, cursor, step = 0, 0, 0
    next_lr, best, history = cfg["learning_rate"], math.inf, []
    epoch_loss, epoch_count = 0.0, 0
    elapsed_before = 0.0
    if state:
        averaged.load_state_dict(state["ema_branch"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        plateau.__dict__.update(state["plateau"])
        epoch, cursor, step = state["epoch"], state["cursor"], state["step"]
        next_lr, best, history = state["next_lr"], state["best"], state["history"]
        local = state["ranks"][rank]
        epoch_loss, epoch_count = local["epoch_loss"], local["epoch_count"]
        restore_rank_rng(local["rng"], device)
        elapsed_before = state["elapsed_s"]
        if state["finished"]:
            if rank == 0:
                atomic_write(
                    output / "status.json",
                    {
                        "state": "complete",
                        "step": step,
                        "epoch": epoch,
                        "termination": state["termination"],
                        "best_val_dsm": best,
                    },
                )
            return
    else:
        # Parameters are synchronized; dropout/noise streams differ by rank.
        torch.manual_seed(cfg["seed"] + rank + 1)
        random.seed(cfg["seed"] + rank + 1)
        np.random.seed(cfg["seed"] + rank + 1)
    if rank == 0:
        atomic_write(output / "config.json", saved_config)
        atomic_write(
            output / "model_info.json",
            {
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_score_parameters": sum(p.numel() for p in model.parameters()),
                "base_planner_in_training_graph": False,
                "sigma_physical_coordinate_std": [cfg["sigma"] * x for x in normalizer["std"]],
                "world_size": world,
                "global_batch": batch_size,
                "microbatch_per_rank": microbatch,
                "gradient_accumulation": accumulation,
                "updates_per_epoch": updates_per_epoch,
                "train_samples": len(training),
                "val_samples": len(validation),
                "precision": "float32",
            },
        )
        atomic_write(
            output / "provenance.json",
            {
                "sources": source_hashes(),
                "signature": signature,
                "cache_sha256": training.sha256,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "selection_uses_test": False,
            },
        )
    start_time = time.monotonic()

    def save(finished=False, termination=None):
        ranks = ddp.gather_objects(
            {"rng": capture_rank_rng(device), "epoch_loss": epoch_loss, "epoch_count": epoch_count}
        )
        if rank == 0:
            atomic_write(
                output / "last.pt",
                {
                    "schema_version": 1,
                    "method": METHOD,
                    "planner": identity,
                    "ego_normalizer": normalizer,
                    "sigma_score": cfg["sigma"],
                    "signature": signature,
                    "sources": source_hashes(),
                    "config": saved_config,
                    "score_branch": model.state_dict(),
                    "ema_branch": averaged.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "plateau": vars(plateau),
                    "epoch": epoch,
                    "cursor": cursor,
                    "step": step,
                    "next_lr": next_lr,
                    "best": best,
                    "history": history,
                    "ranks": ranks,
                    "elapsed_s": elapsed_before + time.monotonic() - start_time,
                    "finished": finished,
                    "termination": termination,
                },
                tensor=True,
            )
        ddp.barrier()

    def publish_best(kind, value):
        if rank != 0:
            return
        selection = {
            "epoch": epoch,
            "step": step,
            "weight_kind": kind,
            "val_dsm": value,
            "criterion": config["checkpoint_selection"],
        }
        atomic_write(
            output / "best.pt",
            {
                "schema_version": 1,
                "method": METHOD,
                "planner": identity,
                "ego_normalizer": normalizer,
                "sigma_score": cfg["sigma"],
                "config": saved_config,
                "score_branch": (averaged if kind == "ema" else model).state_dict(),
                **selection,
                "cache_sha256": training.sha256,
            },
            tensor=True,
        )
        atomic_write(output / "selection.json", selection)

    if not state:
        initial = validate_epoch({"raw": model, "ema": averaged}, validation, cfg, device)
        best = initial["raw"]
        publish_best("initial", best)
        if rank == 0:
            atomic_write(output / "initial_validation.json", initial)
        save()
    termination = "epoch_budget_exhausted"
    while epoch < cfg["max_epochs"]:
        sampler = EpochBatchSampler(
            len(training), batch_size, microbatch, rank, world, cfg["seed"], epoch, cursor
        )
        batches = build_data_loader(training, cfg, batch_sampler=sampler)
        lr = optimizer.param_groups[0]["lr"]
        for progress in train_epoch(
            batches,
            model,
            optimizer,
            cfg,
            averaged,
            device,
            start_step=step,
            start_update=cursor,
            updates_per_epoch=updates_per_epoch,
            accumulation=accumulation,
            next_lr=next_lr,
            distributed_model=wrapped,
        ):
            step, update_loss, lr = progress["step"], progress["train_dsm"], progress["lr"]
            epoch_loss += update_loss
            epoch_count += 1
            cursor = progress["cursor"]
            if step % cfg["log_every_updates"] == 0:
                loss_sum = ddp.sum_tensor(
                    torch.tensor(update_loss, dtype=torch.float64, device=device)
                )
                if rank == 0:
                    progress = {
                        "state": "running",
                        "epoch": epoch + cursor / updates_per_epoch,
                        "step": step,
                        "train_dsm": loss_sum.item() / world,
                        "lr": lr,
                        "elapsed_s": elapsed_before + time.monotonic() - start_time,
                        "peak_allocated_gib_rank0": torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else None,
                    }
                    atomic_write(output / "status.json", progress)
                    print(progress, flush=True)
            if step % cfg["checkpoint_every_updates"] == 0:
                save()
            if stop_after_updates is not None and step >= stop_after_updates:
                save()
                return
        del batches
        epoch += 1
        cursor = 0
        sums = ddp.sum_tensor(
            torch.tensor([epoch_loss, epoch_count], dtype=torch.float64, device=device)
        )
        decision = None
        if epoch % cfg["validate_every_epochs"] == 0 or epoch == cfg["max_epochs"]:
            losses = validate_epoch({"raw": model, "ema": averaged}, validation, cfg, device)
            kind = "ema"
            selected = losses[kind]
            if epoch >= cfg["warmup_epochs"]:
                decision = plateau.observe(selected, next_lr)
                next_lr = decision["learning_rate"]
            row = {
                "epoch": epoch,
                "step": step,
                "train_dsm": (sums[0] / sums[1]).item(),
                "val_dsm_raw": losses["raw"],
                "val_dsm_ema": losses["ema"],
                "lr": lr,
                "plateau": decision,
                "elapsed_s": elapsed_before + time.monotonic() - start_time,
            }
            history.append(row)
            if selected < best:
                best = selected
                publish_best(kind, selected)
            if rank == 0:
                atomic_write(output / "history.json", history)
                print(row, flush=True)
        epoch_loss, epoch_count = 0.0, 0
        stopped = decision and decision["stop"] and epoch >= cfg["minimum_epochs"]
        if stopped:
            termination = "early_stopped_validation_plateau"
        finished = bool(stopped or epoch == cfg["max_epochs"])
        save(finished, termination if finished else None)
        if finished:
            break
    if rank == 0:
        atomic_write(
            output / "status.json",
            {
                "state": "complete",
                "epoch": epoch,
                "step": step,
                "termination": termination,
                "best_val_dsm": best,
                "elapsed_s": elapsed_before + time.monotonic() - start_time,
            },
        )
