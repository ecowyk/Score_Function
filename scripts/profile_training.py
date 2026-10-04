#!/usr/bin/env python3
"""Read-only real-data throughput comparison for the DP-aligned trainer.

Run from the retained training checkout with PYTHONPATH pointing to that checkout.
Only this standalone script is needed; production modules/configs/checkpoints are
never edited. Stop competing GPU jobs before benchmarking. Results are written to
a new directory outside the training run. No trained weights are saved.

Example (use the same GPU count as training):
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
    /path/to/profile_training.py --config CONFIG --root ROOT \
    --checkpoint RUN/score/last.pt --output NEW_PROFILE_DIRECTORY

64 -> 256 -> 64 uses identical scene tokens per rank. The repeated baseline helps
expose filesystem-cache/order bias. Augmentation distributions are unchanged, but
different encoding chunk sizes consume random draws in a different order, so the
augmented tensors/losses are not an exact numerical equivalence test.
"""

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel

from score_function.data_process.online_training import OnlineTrainingProvider, RawTrainingDataset
from score_function.model.score_branch import build_model
from score_function.utils import ddp
from score_function.utils.config import configure_runtime, load_config
from score_function.utils.dataset import EpochBatchSampler, ShardedDataset, build_data_loader
from score_function.utils.planner_utils import planner_identity
from score_function.utils.train_utils import atomic_write, file_hash, load_tensor, source_hashes

epoch_module = importlib.import_module("score_function.train_epoch")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Meter:
    """Synchronize only during the separate diagnostic steps, not throughput steps."""

    def __init__(self, device):
        self.device = device
        self.enabled = False
        self.parts = {}

    def call(self, name, function, *args):
        if not self.enabled:
            return function(*args)
        synchronize(self.device)
        started = time.perf_counter()
        value = function(*args)
        synchronize(self.device)
        self.parts[name] = self.parts.get(name, 0.0) + time.perf_counter() - started
        return value


class MeasuredBatches:
    def __init__(self, iterator, meter):
        self.iterator, self.meter = iterator, meter
        self.collect = False
        self.digest = hashlib.sha256()
        self.samples = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.meter.call("data_wait_s", next, self.iterator)
        if self.collect:
            self.digest.update(json.dumps(batch["tokens"], separators=(",", ":")).encode())
            self.samples += len(batch["tokens"])
        return batch


def summarize(rows, encoding_batch_size, measured_steps):
    seconds = max(row["throughput_elapsed_s"] for row in rows) / measured_steps
    total_samples = sum(row["measured_samples"] for row in rows)
    detailed = [step for row in rows for step in row["diagnostic_steps"]]
    return {
        "encoding_batch_size": encoding_batch_size,
        "world_size": len(rows),
        "measured_updates": measured_steps,
        "seconds_per_update": seconds,
        "samples_per_second": total_samples / max(row["throughput_elapsed_s"] for row in rows),
        "peak_allocated_gib_max": max(row["peak_allocated_gib"] for row in rows),
        "peak_reserved_gib_max": max(row["peak_reserved_gib"] for row in rows),
        "synchronized_stage_mean_ms": {
            key: 1000 * statistics.mean(step[key] for step in detailed)
            for key in (
                "data_wait_s",
                "transfer_s",
                "augmentation_encoding_s",
                "score_optimizer_ddp_s",
                "total_s",
            )
        },
        "token_sha256_by_rank": [row["token_sha256"] for row in rows],
        "ranks": rows,
    }


def run_case(config, dataset, device, encoding_batch_size, warmup, steps, timing_steps, state=None):
    """Exercise the actual train_epoch generator with disposable model/AdamW/EMA state."""
    config = copy.deepcopy(config)
    cfg = config["training"]
    cfg["data_augmentation"]["encoding_batch_size"] = encoding_batch_size
    rank, world = ddp.rank(), ddp.world_size()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    provider = OnlineTrainingProvider(config, device)
    model = build_model(config, device)
    if state is not None:
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
    if state is not None and "ema_branch" in state:
        averaged.load_state_dict(state["ema_branch"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    sampler = EpochBatchSampler(
        len(dataset), cfg["batch_size"], cfg["microbatch_size"], rank, world, cfg["seed"], 0
    )
    if warmup + steps + timing_steps > sampler.updates:
        raise ValueError("The requested profile exceeds one epoch; reduce profile step counts")
    loader = build_data_loader(dataset, cfg, batch_sampler=sampler)
    meter = Meter(device)
    batches = MeasuredBatches(iter(loader), meter)
    original_transfer = epoch_module.device_batch
    epoch_module.device_batch = lambda batch, target: meter.call(
        "transfer_s", original_transfer, batch, target
    )
    random.seed(cfg["seed"] + rank + 1)
    np.random.seed(cfg["seed"] + rank + 1)
    torch.manual_seed(cfg["seed"] + rank + 1)
    updates = epoch_module.train_epoch(
        batches,
        model,
        optimizer,
        cfg,
        averaged,
        device,
        start_step=0,
        start_update=0,
        updates_per_epoch=sampler.updates,
        accumulation=cfg["batch_size"] // world // cfg["microbatch_size"],
        next_lr=cfg["learning_rate"],
        distributed_model=wrapped,
        batch_transform=lambda batch: meter.call("augmentation_encoding_s", provider, batch),
    )
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            next(updates)
        synchronize(device)
        ddp.barrier()
        synchronize(device)
        batches.collect = True
        started = time.perf_counter()
        for _ in range(steps):
            next(updates)
        synchronize(device)
        elapsed = time.perf_counter() - started
        batches.collect = False
        # These deliberately serialized steps locate bottlenecks, but their
        # timings must NOT replace the above unsplit throughput measurement.
        meter.enabled = True
        detailed = []
        for _ in range(timing_steps):
            synchronize(device)
            meter.parts = {}
            started = time.perf_counter()
            next(updates)
            synchronize(device)
            total = time.perf_counter() - started
            parts = dict(meter.parts)
            parts["score_optimizer_ddp_s"] = max(0.0, total - sum(parts.values()))
            parts["total_s"] = total
            detailed.append(parts)
        row = {
            "rank": rank,
            "throughput_elapsed_s": elapsed,
            "measured_samples": batches.samples,
            "token_sha256": batches.digest.hexdigest(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else 0.0,
            "diagnostic_steps": detailed,
        }
    finally:
        epoch_module.device_batch = original_transfer
        updates.close()
    rows = ddp.gather_objects(row)
    return summarize(rows, encoding_batch_size, steps)


def prepare_output(output, training_output):
    output, training_output = Path(output).resolve(), Path(training_output).resolve()
    if output == training_output or output.is_relative_to(training_output):
        raise ValueError("Profile output must be outside the retained training run")
    output.mkdir(parents=True, exist_ok=False)
    return output


@record
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--root")
    parser.add_argument(
        "--checkpoint", help="Read-only last.pt/best.pt; default: fresh temporary model"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoding-batches", nargs="+", type=int, default=[64, 256, 64])
    parser.add_argument(
        "--warmup", type=int, default=5, help="Discarded benchmark updates, not LR warmup"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Normally executed updates timed as one block"
    )
    parser.add_argument(
        "--timing-steps", type=int, default=5, help="Separate synchronized diagnostic updates"
    )
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.timing_steps, *args.encoding_batches) < 1:
        parser.error("All step counts and encoding batches must be positive")
    config = load_config(args.config, args.root)
    if not config["training"].get("data_augmentation", {}).get("enabled"):
        parser.error("This comparison requires the aligned online-augmentation config")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    cfg = config["training"]
    if cfg["batch_size"] % (world * cfg["microbatch_size"]):
        parser.error("Global batch must divide world size times microbatch")
    if max(args.encoding_batches) > cfg["microbatch_size"]:
        parser.error("Encoding batch cannot exceed the configured per-rank microbatch")
    if world == 1:
        print(
            "NOTE: single-process results cannot measure the original 8-GPU data/communication load.",
            flush=True,
        )
    output = Path(args.output).expanduser().resolve()
    if output == Path(config["output"]).resolve() or output.is_relative_to(
        Path(config["output"]).resolve()
    ):
        parser.error("Choose a new profile output outside the training run")
    hashes_before = {
        "config": file_hash(args.config),
        "checkpoint": file_hash(args.checkpoint) if args.checkpoint else None,
        "production_sources": source_hashes(),
    }
    device = configure_runtime(config)
    ddp.setup(device, timeout_seconds=1800)
    dataset = None
    try:
        if ddp.rank() == 0:
            prepare_output(output, config["output"])
            print(
                "Indexing the original training split (startup excluded from timing)...", flush=True
            )
        ddp.barrier()

        def progress(done, total):
            if ddp.rank() == 0 and (done % 250 == 0 or done == total):
                print(f"Cache index: {done}/{total}", flush=True)

        cached = ShardedDataset(config["cache"], "train", progress=progress)
        if cached.metadata["planner"] != planner_identity(config):
            raise ValueError("Cache and configured frozen planner do not match")
        dataset = RawTrainingDataset(cached, config)
        state = load_tensor(args.checkpoint) if args.checkpoint else None
        provenance = {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "config_values": config,
            "hashes_before": hashes_before,
            "profile_script_sha256": file_hash(__file__),
            "clean_cache_sha256": cached.sha256,
            "train_samples": len(dataset),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "world_size": world,
            "encoding_batches": args.encoding_batches,
            "warmup_updates": args.warmup,
            "measured_updates": args.steps,
            "diagnostic_updates": args.timing_steps,
            "notes": [
                "No production configuration/checkpoint/weights are written.",
                "Only encoding_batch_size differs between cases; temporary AdamW is reset.",
                "Scene tokens match; augmentation random draw ordering can differ across chunk sizes.",
                "Throughput excludes startup, validation, checkpoint saving and normal training logging.",
                "Stage times synchronize CUDA and are diagnostic only; DDP waits are included in score time.",
                "Repeated scenes can warm filesystem caches; compare both baseline passes.",
            ],
        }
        if ddp.rank() == 0:
            atomic_write(output / "provenance.json", provenance)
        results = []
        for number, size in enumerate(args.encoding_batches, 1):
            gc.collect()
            torch.cuda.empty_cache()
            ddp.barrier()
            if ddp.rank() == 0:
                print(
                    f"Case {number}/{len(args.encoding_batches)}: encoding_batch_size={size}",
                    flush=True,
                )
            result = run_case(
                config, dataset, device, size, args.warmup, args.steps, args.timing_steps, state
            )
            if results and result["token_sha256_by_rank"] != results[0]["token_sha256_by_rank"]:
                raise RuntimeError("The compared cases used different scene tokens")
            result["case"] = number
            results.append(result)
            if ddp.rank() == 0:
                atomic_write(output / "summary.json", {"cases": results, "complete": False})
                brief = {
                    k: v for k, v in result.items() if k not in ("ranks", "token_sha256_by_rank")
                }
                print(json.dumps(brief, indent=2), flush=True)
        hashes_after = {
            "config": file_hash(args.config),
            "checkpoint": file_hash(args.checkpoint) if args.checkpoint else None,
            "production_sources": source_hashes(),
        }
        if hashes_before != hashes_after:
            raise RuntimeError("Config, checkpoint or production sources changed during profiling")
        if ddp.rank() == 0:
            atomic_write(
                output / "summary.json",
                {
                    "cases": results,
                    "complete": True,
                    "input_files_unchanged": True,
                },
            )
            print("PROFILE COMPLETE | input config/checkpoint/code unchanged", flush=True)
            print(f"Results: {output / 'summary.json'}", flush=True)
    finally:
        if dataset is not None:
            dataset.close()
        ddp.close()


if __name__ == "__main__":
    main()
