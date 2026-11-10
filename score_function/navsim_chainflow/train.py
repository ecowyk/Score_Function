"""Train a conditional trajectory density score on frozen ChainFlow/VLM features.

Fixed global batches are split across ranks and accumulation microbatches. The
native planner is never in the training graph. All validations use FP32, eval
mode and fixed per-token perturbations, including a fixed clean training subset.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import math
import os
import random
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from score_function.navsim_chainflow.model import ChainFlowScore, ARCHITECTURE
from score_function.navsim_chainflow.runtime import CHECKPOINT_FORMAT
from score_function.utils import ddp
from score_function.utils.dataset import EpochBatchSampler
from score_function.utils.ema import update_ema
from score_function.utils.progress import Progress, report
from score_function.utils.train_utils import (atomic_write, capture_rank_rng, file_hash, fingerprint,
                                             load_tensor, read_json, restore_rank_rng, sample_noise)

TRAIN_FORMAT = "navsim-chainflow-train-v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/navsim_chainflow_s10g.json"


def check_config(cfg):
    if cfg["model"].get("architecture") != ARCHITECTURE:
        raise ValueError("Expected a ChainFlow conditional score model")
    for key in ("epochs", "microbatch_size", "global_batch_size", "validation_repeats", "checkpoint_every_epochs"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num_workers", "warmup_epochs", "diagnostic_train_samples"):
        if type(cfg.get(key)) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not cfg["warmup_epochs"] < cfg["epochs"]:
        raise ValueError("warmup_epochs must be less than epochs")
    for key in ("sigma", "learning_rate", "warmup_learning_rate"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("gamma", "weight_decay", "gradient_clip", "min_learning_rate"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["min_learning_rate"] > cfg["learning_rate"] or cfg["warmup_learning_rate"] > cfg["learning_rate"]:
        raise ValueError("Warmup/minimum learning rates must not exceed peak")
    if not 0 < cfg["ema_decay"] < 1:
        raise ValueError("This experiment requires EMA decay in (0,1)")
    if cfg.get("lr_schedule") != "linear_warmup_cosine" or cfg.get("precision") != "32-true":
        raise ValueError("Require linear_warmup_cosine and FP32 training")
    if cfg.get("drop_last") is not True:
        raise ValueError("Require complete global batches (drop_last=true)")
    if cfg.get("condition_augmentation", "none") != "none":
        raise ValueError("Cached frozen conditions do not support geometric augmentation")
    start = cfg.get("checkpoint_start_epoch", cfg["checkpoint_every_epochs"])
    if type(start) is not int or start < 1:
        raise ValueError("checkpoint_start_epoch must be positive")


_check_config = check_config


def collate_rows(rows):
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    values = {}
    for key, shape in (("scene", (64, 256)), ("ego", (256,)), ("target", (8, 4))):
        tensors = [row[key] for row in rows]
        if any(tuple(t.shape) != shape or not torch.isfinite(t).all() for t in tensors):
            raise ValueError(f"Invalid/nonfinite {key} cache tensor")
        values[key] = torch.stack(tensors).float()
    lengths = [row["vlm"].shape[0] for row in rows]
    if any(row["vlm"].ndim != 2 or row["vlm"].shape[1] != 1536 or length < 1
           or not torch.isfinite(row["vlm"]).all() for row, length in zip(rows, lengths)):
        raise ValueError("Require finite nonempty VLM token sequences [L,1536]")
    values["vlm"] = torch.zeros(len(rows), max(lengths), 1536, dtype=torch.float32)
    values["vlm_mask"] = torch.zeros(len(rows), max(lengths), dtype=torch.bool)
    for index, (row, length) in enumerate(zip(rows, lengths)):
        values["vlm"][index, :length] = row["vlm"].float()
        values["vlm_mask"][index, :length] = True
    values["tokens"] = [row["record"]["token"] for row in rows]
    return values


def device_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_loader(dataset, cfg, *, indices=None, sampler=None):
    if indices is not None:
        dataset = Subset(dataset, indices)
    options = dict(num_workers=cfg["num_workers"], collate_fn=collate_rows,
                   pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(1947))
    if cfg["num_workers"]:
        options.update(prefetch_factor=1, persistent_workers=False, multiprocessing_context="spawn")
    if sampler is not None:
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=cfg["microbatch_size"], shuffle=False, **options)


def load_training_data(cache_dir, cfg=None):
    from score_function.navsim_chainflow.cache import ChainFlowDataset, validate_training_ready
    metadata = validate_training_ready(cache_dir)
    return metadata, ChainFlowDataset(cache_dir, "train"), ChainFlowDataset(cache_dir, "val")


def build_training_model(cfg, metadata=None, cache_dir=None, device="cpu"):
    return ChainFlowScore(cfg["model"]).to(device)


def learning_rate(cfg, step, updates_per_epoch):
    warmup = cfg["warmup_epochs"] * updates_per_epoch
    total = cfg["epochs"] * updates_per_epoch
    if step < warmup:
        phase = step / max(1, warmup - 1)
        return cfg["warmup_learning_rate"] + phase * (cfg["learning_rate"] - cfg["warmup_learning_rate"])
    phase = min(1., max(0., (step - warmup) / max(1, total - warmup - 1)))
    return cfg["min_learning_rate"] + .5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (1 + math.cos(math.pi * phase))


def predict(model, noisy, batch):
    return model(noisy, batch["scene"], batch["ego"], batch["vlm"], batch["vlm_mask"])


def run_training_epoch(loader, model, optimizer, ema, cfg, device, *, epoch, step,
                       updates_per_epoch, accumulation, distributed_model=None):
    model.train()
    wrapped = distributed_model if distributed_model is not None else model
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    batches = iter(loader)
    with Progress(f"Train epoch {epoch + 1}", total=updates_per_epoch, unit="update") as progress:
        for update in range(updates_per_epoch):
            lr = learning_rate(cfg, step, updates_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            for micro in range(accumulation):
                batch = device_batch(next(batches), device)
                noise = torch.randn_like(batch["target"])
                sync = wrapped.no_sync() if distributed_model is not None and micro < accumulation - 1 else nullcontext()
                with sync:
                    output = predict(wrapped, batch["target"] + cfg["sigma"] * noise, batch)
                    error = (cfg["sigma"] * output + noise).square()
                    loss = error.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training DSM")
                    (loss / accumulation).backward()
                totals[0] += error[..., :2].detach().double().sum()
                totals[1] += error[..., 2:].detach().double().sum()
                totals[2] += error[..., :2].numel()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"] or float("inf"))
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite score gradient norm")
            optimizer.step()
            update_ema(ema, model, cfg["ema_decay"])
            step += 1
            progress(update + 1)
    ddp.sum_tensor(totals)
    xy, heading = (totals[:2] / totals[2]).tolist()
    return {"step": step, "lr": lr, "train_dsm": (xy + heading) / 2., "train_xy": xy, "train_heading": heading}


@torch.no_grad()
def validate_models(models, dataset, cfg, device, *, indices=None, namespace="validation", description="Validation"):
    for model in models.values():
        model.eval()
    selected = list(range(len(dataset))) if indices is None else list(indices)
    if not selected:
        raise ValueError("Validation/diagnostic set must be nonempty")
    loader = make_loader(dataset, cfg, indices=selected[ddp.rank()::ddp.world_size()])
    totals = torch.zeros(len(models), 3, dtype=torch.float64, device=device)
    with Progress(description, total=len(loader) * cfg["validation_repeats"] * len(models), unit="forward") as progress:
        count = 0
        for raw in loader:
            batch = device_batch(raw, device)
            # The condition is independent of the noisy trajectory. Prepare it
            # once per model/batch, while preserving native VLM token masking.
            prepared = {name: model.prepare_condition(batch["scene"], batch["ego"], batch["vlm"], batch["vlm_mask"])
                        for name, model in models.items()}
            for repeat in range(cfg["validation_repeats"]):
                noise = sample_noise((8, 4), batch["tokens"], cfg["validation_seed"], f"{namespace}_{repeat}").to(device)
                noisy = batch["target"] + cfg["sigma"] * noise
                for index, (name, model) in enumerate(models.items()):
                    error = (cfg["sigma"] * model.score_prepared(noisy, prepared[name]) + noise).square()
                    totals[index, 0] += error[..., :2].double().sum()
                    totals[index, 1] += error[..., 2:].double().sum()
                    totals[index, 2] += error[..., :2].numel()
                    count += 1
                    progress(count)
    ddp.sum_tensor(totals)
    if not torch.isfinite(totals).all() or not (totals[:, 2] > 0).all():
        raise FloatingPointError("Nonfinite/empty validation")
    result = {}
    for i, name in enumerate(models):
        xy, heading = (totals[i, :2] / totals[i, 2]).tolist()
        result[name] = {"dsm": (xy + heading) / 2., "xy": xy, "heading": heading}
    return result


@contextmanager
def _lock_output(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".train.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another trainer owns {output}") from exc
        yield


def _rank_zero(action):
    error = None
    if ddp.rank() == 0:
        try:
            action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = ddp.gather_objects(error)[0]
    if error:
        raise RuntimeError(error)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def source_identity():
    root = Path(__file__).parents[1]
    paths = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("runtime.py")]
    paths += sorted((root / "model").rglob("*.py"))
    paths += [root / "utils" / name for name in ("normalizer.py", "dataset.py", "train_utils.py", "ema.py", "ddp.py")]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def train(cache_dir, output, cfg, device, *, resume=False):
    check_config(cfg)
    device = torch.device(device)
    output = Path(output)
    rank, world = ddp.rank(), ddp.world_size()
    global_batch, microbatch = cfg["global_batch_size"], cfg["microbatch_size"]
    if global_batch % (world * microbatch):
        raise ValueError("global_batch_size must be divisible by world_size * microbatch_size")
    accumulation = global_batch // (world * microbatch)
    metadata, training, validation = load_training_data(cache_dir, cfg)
    updates = len(training) // global_batch
    if updates < 1 or len(validation) < 1:
        raise ValueError("Need nonempty validation and at least one complete global training batch")
    diagnostic_indices = torch.randperm(len(training), generator=torch.Generator().manual_seed(cfg["validation_seed"])).tolist()
    diagnostic_indices = diagnostic_indices[:min(len(training), cfg["diagnostic_train_samples"])]
    protocol = {"config": {k: v for k, v in cfg.items() if k != "num_workers"},
                "world_size": world, "device_type": device.type, "cache_fingerprint": fingerprint(metadata),
                "gradient_accumulation": accumulation, "source_identity": source_identity()}
    signature = fingerprint(protocol)
    last_path = output / "last.pt"

    def prepare_output():
        if resume:
            if not (output / "run.json").is_file() or read_json(output / "run.json").get("signature") != signature:
                raise ValueError("Resume rejected: training config, code, cache, device or world size changed")
            if not last_path.is_file() and (output / "history.json").is_file():
                raise ValueError("Training history exists but its resumable checkpoint is missing")
        else:
            if any(p.name != ".train.lock" for p in output.iterdir()):
                raise ValueError("Training output is not empty; use --resume or a new output")
            atomic_write(output / "run.json", {
                "signature": signature, "protocol": protocol, "config": cfg, "global_batch": global_batch,
                "microbatch_per_rank": microbatch, "gradient_accumulation": accumulation,
                "updates_per_epoch": updates, "train_samples": len(training), "val_samples": len(validation),
                "train_presentations_per_epoch": updates * global_batch,
                "dropped_samples_per_epoch": len(training) - updates * global_batch,
                "planned_optimizer_updates": updates * cfg["epochs"], "drop_last": True,
                "sigma_in_physical_channels": [cfg["sigma"] * s for s in metadata["normalizer"]["std"]],
                "diagnostic_train_indices": diagnostic_indices,
                "condition_augmentation": "none; scene/VLM encoders frozen and cached",
                "cuda_allocator_budget_gb": os.getenv("CF_CUDA_MEMORY_GB"),
            })

    with ExitStack() as stack:
        _rank_zero(lambda: stack.enter_context(_lock_output(output)))
        _rank_zero(prepare_output)
        _seed(cfg["seed"])
        model = build_training_model(cfg, metadata, cache_dir, device)
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                          broadcast_buffers=False) if world > 1 else None
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        history, step, best = [], 0, {}
        _seed(cfg["seed"] + rank + 1)
        if resume and last_path.is_file():
            saved = load_tensor(last_path)
            if saved.get("format") != TRAIN_FORMAT or saved.get("signature") != signature:
                raise ValueError("Resume checkpoint belongs to another ChainFlow experiment")
            model.load_state_dict(saved["raw"], strict=True)
            ema.load_state_dict(saved["ema"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            history, step, best = saved["history"], saved["step"], saved["best"]
            if [r["epoch"] for r in history] != list(range(1, len(history) + 1)) or step != len(history) * updates:
                raise ValueError("Invalid resume history/step cursor")
            restore_rank_rng(saved["rng_by_rank"][rank], device)

        common = {"format": CHECKPOINT_FORMAT, "model": cfg["model"], "normalizer": metadata["normalizer"],
                  "conditioner_identity": metadata["conditioner_identity"], "sigma": cfg["sigma"], "gamma": cfg["gamma"],
                  "training_precision": cfg["precision"], "config": cfg, "cache_fingerprint": fingerprint(metadata),
                  "training_signature": signature}

        def export(path, state, epoch, kind, val):
            atomic_write(path, {**common, "state_dict": state, "epoch": epoch,
                                "weight_kind": kind, "val_dsm": val}, tensor=True)

        def publish():
            atomic_write(output / "history.json", history)
            if not history:
                return
            epoch = history[-1]["epoch"]
            for kind, module in (("raw", model), ("ema", ema)):
                selected = best[kind]
                export(output / f"best_{kind}.pt", selected["state_dict"], selected["epoch"], kind, selected["val_dsm"])
                interval = cfg["checkpoint_every_epochs"]
                start = cfg.get("checkpoint_start_epoch", interval)
                if epoch >= start and epoch % interval == 0:
                    export(output / f"epoch_{epoch:04d}_{kind}.pt", module.state_dict(), epoch, kind, history[-1][f"val_dsm_{kind}"])
                if epoch == cfg["epochs"]:
                    export(output / f"final_{kind}.pt", module.state_dict(), epoch, kind, history[-1][f"val_dsm_{kind}"])
            if epoch == cfg["epochs"]:
                atomic_write(output / "TRAINING_COMPLETE.json", {"signature": signature, "epochs": epoch,
                    "final_raw_sha256": file_hash(output / "final_raw.pt"), "final_ema_sha256": file_hash(output / "final_ema.pt")})

        if history:
            _rank_zero(publish)  # Repair exports if interrupted after authoritative last.pt.
        report(f"ChainFlow Score train: {len(training)} train, {len(validation)} val; world={world}, "
               f"global_batch={global_batch}, microbatch={microbatch}, accumulation={accumulation}, "
               f"updates/epoch={updates}; FP32, RAW+EMA, fixed train diagnostic={len(diagnostic_indices)}")
        for epoch in range(len(history), cfg["epochs"]):
            started = time.perf_counter()
            sampler = EpochBatchSampler(len(training), global_batch, microbatch, rank, world, cfg["seed"], epoch)
            result = run_training_epoch(make_loader(training, cfg, sampler=sampler), model, optimizer, ema, cfg, device,
                epoch=epoch, step=step, updates_per_epoch=updates, accumulation=accumulation, distributed_model=wrapped)
            step = result["step"]
            models = {"raw": model, "ema": ema}
            val = validate_models(models, validation, cfg, device)
            diagnostic = validate_models(models, training, cfg, device, indices=diagnostic_indices,
                                          namespace="validation", description="Train diagnostic") if diagnostic_indices else None
            row = {"epoch": epoch + 1, **result}
            for kind in models:
                row.update({f"val_{key}_{kind}": value for key, value in val[kind].items()})
                if diagnostic:
                    row.update({f"diagnostic_train_{key}_{kind}": value for key, value in diagnostic[kind].items()})
                if kind not in best or val[kind]["dsm"] < best[kind]["val_dsm"]:
                    best[kind] = {"epoch": epoch + 1, "val_dsm": val[kind]["dsm"],
                                  "state_dict": {k: v.detach().cpu().clone() for k, v in models[kind].state_dict().items()}}
            history.append(row)
            rng_by_rank = ddp.gather_objects(capture_rank_rng(device))

            def save():
                atomic_write(last_path, {"format": TRAIN_FORMAT, "signature": signature, "history": history,
                    "step": step, "raw": model.state_dict(), "ema": ema.state_dict(), "best": best,
                    "optimizer": optimizer.state_dict(), "rng_by_rank": rng_by_rank}, tensor=True)
                publish()
            _rank_zero(save)
            report(f"epoch {epoch + 1}/{cfg['epochs']} train DSM {row['train_dsm']:.6f} "
                   f"val RAW {row['val_dsm_raw']:.6f} EMA {row['val_dsm_ema']:.6f}; "
                   f"lr={row['lr']:.8g}, elapsed={time.perf_counter() - started:.1f}s")
        for dataset in (training, validation):
            if hasattr(dataset, "close"):
                dataset.close()


def configure_device_memory(device):
    value = os.getenv("CF_CUDA_MEMORY_GB")
    if value is None or device.type != "cuda":
        return
    budget = float(value) * 1024**3
    total = torch.cuda.get_device_properties(device).total_memory
    if not math.isfinite(budget) or not 0 < budget <= total:
        raise ValueError("CF_CUDA_MEMORY_GB must fit within one GPU's total memory")
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    report(f"CUDA allocator budget {float(value):g} GiB; this excludes driver/non-PyTorch allocations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch-size", "--batch-size", dest="microbatch_size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.microbatch_size is not None:
        cfg["microbatch_size"] = args.microbatch_size
    if args.workers is not None:
        cfg["num_workers"] = args.workers
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.getenv("LOCAL_RANK", device.index or 0)))
        torch.cuda.set_device(device)
        configure_device_memory(device)
    elif device.type != "cpu":
        parser.error("--device must be cuda or cpu")
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))
    ddp.setup(device)
    try:
        train(args.cache_dir, args.output, cfg, device, resume=args.resume)
    finally:
        ddp.close()


if __name__ == "__main__":
    main()
