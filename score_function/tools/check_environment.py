"""Check the real official checkpoint and score-branch backward before an expensive cache build."""

import importlib
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from score_function.data_process.data_processor import allowed_databases
from score_function.model.score_branch import build_model
from score_function.utils import ddp
from score_function.utils.config import configure_runtime
from score_function.utils.train_utils import read_json, resolve_path


def synthetic_conditions(size, device):
    return (torch.randn(size, 107, 192, device=device), torch.randn(size, 192, device=device))


def check_ddp(config):
    device = torch.device(config["runtime"]["device"])
    ddp.setup(device, timeout_seconds=600)
    try:
        size = config["training"]["microbatch_size"]
        if config["training"]["batch_size"] % (ddp.world_size() * size):
            raise ValueError("Global batch must divide world size times microbatch")
        probe = ddp.sum_tensor(torch.tensor(ddp.rank() + 1.0, device=device))
        if probe.item() != ddp.world_size() * (ddp.world_size() + 1) / 2:
            raise RuntimeError("Distributed sum failed")
        model = build_model(config, device)
        wrapped = (
            DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
            if ddp.world_size() > 1
            else model
        )
        x = torch.randn(size, 80, 4, device=device)
        noise = torch.randn_like(x)
        sigma = config["training"]["sigma"]
        score = wrapped(x + sigma * noise, *synthetic_conditions(size, device))
        (sigma * score + noise).square().mean().backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Missing/nonfinite score gradient")
        print(
            json.dumps(
                {
                    "state": "passed",
                    "rank": ddp.rank(),
                    "microbatch": size,
                    "score_parameters": sum(p.numel() for p in model.parameters()),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                }
            ),
            flush=True,
        )
        ddp.barrier()
    finally:
        ddp.close()


def check(config, expected_gpus):
    if (
        expected_gpus < 1
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < expected_gpus
    ):
        raise RuntimeError(f"Require {expected_gpus} visible GPUs")
    if expected_gpus > 1 and not torch.distributed.is_nccl_available():
        raise RuntimeError("Require Linux NCCL PyTorch for multi-GPU training")
    if config["training"]["batch_size"] % (expected_gpus * config["training"]["microbatch_size"]):
        raise ValueError("Invalid global batch / GPUs / microbatch")
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError("Use numpy<2 with the official PyTorch2.0 environment")
    for key in (
        "planner_dir",
        "devkit_dir",
        "planner_args",
        "planner_checkpoint",
        "train_log_allowlist",
    ):
        if not resolve_path(config, key).exists():
            raise FileNotFoundError(f"Missing paths.{key}: {resolve_path(config, key)}")
    configure_runtime(config)
    for name in (
        "diffusion_planner.data_process.data_processor",
        "diffusion_planner.model.diffusion_planner",
    ):
        importlib.import_module(name)
    if config["data"]["prepare_raw"]:
        version = resolve_path(config, "maps_dir") / (config["data"]["map_version"] + ".json")
        if not version.is_file():
            raise FileNotFoundError(version)
        files, excluded = allowed_databases(config)
        data = {"official_train_dbs": len(files), "excluded_dbs": excluded}
    else:
        rows = read_json(resolve_path(config, "manifest"))
        if not rows:
            raise ValueError("Empty reused feature manifest")
        data = {"reused_npz_frames": len(rows)}
    from score_function.utils.planner_utils import ego_normalizer_metadata, load_frozen_planner

    base, planner_args = load_frozen_planner(config, config["runtime"]["device"])
    normalizer = ego_normalizer_metadata(planner_args)
    del base
    torch.cuda.empty_cache()
    # check-ddp tests the configured trainable branch microbatch on every GPU.
    print(
        json.dumps(
            {
                "state": "dependencies_passed",
                "data": data,
                "sigma_physical_coordinate_std": [
                    config["training"]["sigma"] * v for v in normalizer["std"]
                ],
                "gpus": [torch.cuda.get_device_name(i) for i in range(expected_gpus)],
            },
            indent=2,
        ),
        flush=True,
    )
