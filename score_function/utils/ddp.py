"""Single-node torchrun setup, shared by cache construction and DSM training."""

import os
from datetime import timedelta

import torch.distributed as dist


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def setup(device, timeout_seconds=7200):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(seconds=timeout_seconds),
        )


def barrier():
    if dist.is_initialized():
        dist.barrier()


def sum_tensor(value):
    if dist.is_initialized():
        dist.all_reduce(value)
    return value


def gather_objects(value):
    if not dist.is_initialized():
        return [value]
    result = [None] * world_size()
    dist.all_gather_object(result, value)
    return result


def close():
    if dist.is_initialized():
        dist.destroy_process_group()
