"""Train an independent direct score on cached DiffusionDrive conditions and NAVSIM poses."""

from __future__ import annotations

import argparse
import copy
import fcntl
import math
import os
import random
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

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


def _check_config(cfg):
    model = cfg["model"]
    for key, value in {
        "architecture": "temporal_score_function",
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
    for key in ("sigma", "learning_rate", "warmup_learning_rate", "gradient_clip"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("gamma", "weight_decay"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 <= cfg["ema_decay"] < 1 or cfg["lr_schedule"] != "constant_after_warmup":
        raise ValueError("Invalid EMA decay or learning-rate schedule")


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
    output = Path(output)
    rank, world = ddp.rank(), ddp.world_size()
    metadata = load_cache_metadata(cache_dir)
    training = NavsimScoreDataset(cache_dir, "train")
    validation = NavsimScoreDataset(cache_dir, "val")
    batch_size = cfg["microbatch_size"] * world
    updates = len(training) // batch_size
    if updates < 1 or not len(validation):
        raise ValueError("Need a nonempty validation split and a complete global training batch. "
                         "Use a smaller --batch-size for a small cache.")
    # Each full globally shuffled batch is partitioned evenly among workers.
    protocol = {
        "config": {k: v for k, v in cfg.items() if k not in ("epochs", "num_workers")},
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
                "sigma_in_physical_channels": [cfg["sigma"] * s for s in metadata["normalizer"]["std"]],
            })

    with ExitStack() as stack:
        _rank_zero(lambda: stack.enter_context(_lock_output(output)))
        _rank_zero(prepare_output)
        _seed(cfg["seed"])
        model = build_model(cfg, device)
        wrapped = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        ) if world > 1 else None
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        history, step, best_weights = [], 0, None
        _seed(cfg["seed"] + rank + 1)
        if resume and last_path.exists():
            last = load_tensor(last_path)
            if last.get("format") != "navsim-dd-train-v1" or last["signature"] != signature:
                raise ValueError("Resume checkpoint does not match this NAVSIM experiment")
            history, step, best_weights = last["history"], last["step"], last["best_weights"]
            model.load_state_dict(last["raw"])
            ema.load_state_dict(last["ema"])
            optimizer.load_state_dict(last["optimizer"])
            restore_rank_rng(last["rng_by_rank"][rank], device)
            del last
        if len(history) > cfg["epochs"]:
            raise ValueError("--epochs is smaller than the number of completed epochs")

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
            }
            atomic_write(output / "history.json", history)
            if repair_best or best_row["epoch"] == epoch:
                atomic_write(output / "best_ema.pt", {
                    **common, "state_dict": best_weights, "epoch": best_row["epoch"],
                    "val_dsm": best_row["val_dsm"],
                }, tensor=True)
            if final:
                atomic_write(output / "final_ema.pt", {
                    **common, "state_dict": ema.state_dict(), "epoch": epoch,
                    "val_dsm": history[-1]["val_dsm"],
                }, tensor=True)

        if resume and history:
            # Republish even when last.pt was committed just before the job stopped.
            _rank_zero(lambda: publish_exports(final=len(history) == cfg["epochs"], repair_best=True))
        if len(history) < cfg["epochs"]:
            _rank_zero(lambda: (output / "final_ema.pt").unlink(missing_ok=True))
        report(f"NAVSIM score train: {len(training)} train, {len(validation)} val, "
               f"world={world}, global_batch={batch_size}, updates/epoch={updates}")
        for epoch in range(len(history), cfg["epochs"]):
            sampler = EpochBatchSampler(
                len(training), batch_size, cfg["microbatch_size"], rank, world,
                cfg["seed"], epoch,
            )
            loader = build_data_loader(training, cfg, batch_sampler=sampler)
            loss_sum = 0.
            for result in train_epoch(
                loader, model, optimizer, cfg, ema, device, start_step=step, start_update=0,
                updates_per_epoch=updates, accumulation=1, next_lr=cfg["learning_rate"],
                distributed_model=wrapped,
            ):
                step = result["step"]
                loss_sum += result["train_dsm"]
            loss_sum = ddp.sum_tensor(torch.tensor(loss_sum, dtype=torch.float64, device=device)).item()
            val = validate_epoch({"ema": ema}, validation, cfg, device)["ema"]
            row = {
                "epoch": epoch + 1, "step": step,
                "train_dsm": loss_sum / (world * updates), "val_dsm": val, "lr": result["lr"],
            }
            if not history or val < min(item["val_dsm"] for item in history):
                best_weights = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
            history.append(row)
            rank_rng = ddp.gather_objects(capture_rank_rng(device))

            def save():
                atomic_write(last_path, {
                    "format": "navsim-dd-train-v1", "signature": signature, "step": step,
                    "raw": model.state_dict(), "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(), "rng_by_rank": rank_rng,
                    "best_weights": best_weights, "history": history,
                }, tensor=True)
                publish_exports(final=len(history) == cfg["epochs"])

            _rank_zero(save)
            report(f"epoch {row['epoch']}  train DSM {row['train_dsm']:.6f}  val DSM {val:.6f}")


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
