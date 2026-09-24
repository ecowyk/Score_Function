"""One independent, explicit protocol for clean-space score learning."""

import copy
import json
import math
import os
import sys
from pathlib import Path

from score_function.utils.train_utils import read_json, resolve_path

METHOD = "score_function_v1"


def load_config(path, root=None, overrides=()):
    config = copy.deepcopy(read_json(path))
    for expression in overrides:
        key, value = expression.split("=", 1)
        parts, node = key.split("."), config
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown configuration key: {key}")
        node[parts[-1]] = json.loads(value)
    if config.get("schema_version") != 1 or config.get("method") != METHOD:
        raise ValueError("Require this project's configs/score_function.json")
    config["paths"]["root"] = str(Path(root or config["paths"]["root"]).expanduser().resolve())
    model, cfg = config["model"], config["training"]
    if (
        model["architecture"] != "temporal_score_function"
        or model["future_len"] != 80
        or model["input_dim"] != 4
        or model["context_dim"] != 192
    ):
        raise ValueError("Require clean ego80x4 and official context width192")
    if (
        model["hidden_dim"] <= 0
        or model["num_heads"] <= 0
        or model["hidden_dim"] % model["num_heads"]
        or not 0 <= model["dropout"] < 1
    ):
        raise ValueError("Invalid hidden width/heads/dropout")
    for key in ("pre_dilations", "post_dilations"):
        if not model[key] or any(not isinstance(x, int) or x < 1 for x in model[key]):
            raise ValueError(f"Invalid model.{key}")
    for key in ("temporal_attention", "neighbor_future"):
        if not isinstance(model.get(key, False), bool):
            raise ValueError(f"model.{key} must be boolean")
    if model.get("neighbor_future") and (
        not config["paths"].get("neighbor_cache")
        or config["data"].get("neighbor_batch_size", 0) < 1
    ):
        raise ValueError(
            "Neighbor branch requires paths.neighbor_cache and data.neighbor_batch_size"
        )
    for key in (
        "sigma",
        "max_epochs",
        "batch_size",
        "microbatch_size",
        "learning_rate",
        "warmup_learning_rate",
        "validation_repeats",
        "validate_every_epochs",
        "checkpoint_every_updates",
        "log_every_updates",
        "gradient_clip",
    ):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid training.{key}")
    if (
        not 0 <= cfg["warmup_epochs"] < cfg["max_epochs"]
        or not 1 <= cfg["minimum_epochs"] <= cfg["max_epochs"]
    ):
        raise ValueError("Invalid warmup/minimum/max epochs")
    if (
        cfg["batch_size"] % cfg["microbatch_size"]
        or cfg["num_workers"] < 0
        or cfg["weight_decay"] < 0
    ):
        raise ValueError("Invalid batch/workers/weight decay")
    if not 0 <= cfg["ema_decay"] < 1 or not 0 < config["runtime"]["memory_fraction"] <= 1:
        raise ValueError("Invalid EMA/memory fraction")
    if (
        cfg["sampling"] != "uniform_frame_without_replacement"
        or config["checkpoint_selection"] != "minimum validation DSM among initial and EMA branches"
    ):
        raise ValueError("Unsupported sampling/checkpoint selection policy")
    for key in ("shard_size", "preprocess_workers", "encoding_batch_size"):
        if config["data"][key] < 1:
            raise ValueError(f"Invalid data.{key}")
    ref = config["refinement"]
    if (
        not math.isfinite(ref["gamma"])
        or ref["gamma"] < 0
        or not isinstance(ref["steps"], int)
        or ref["steps"] < 0
    ):
        raise ValueError("Invalid refinement gamma/steps")
    if not isinstance(ref["heading_projection"], bool):
        raise ValueError("heading_projection must be boolean")
    if (
        config["smoke"]["frames"] < 1
        or config["smoke"]["updates"] < 1
        or config["smoke"]["learning_rate"] <= 0
    ):
        raise ValueError("Invalid fixed-corruption smoke settings")
    config["output"], config["cache"] = (str(resolve_path(config, k)) for k in ("run_dir", "cache"))
    return config


def configure_runtime(config, require_cuda=True):
    import torch

    device = torch.device(config["runtime"]["device"])
    if device.type == "cuda" and "LOCAL_RANK" in os.environ:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        config["runtime"]["device"] = str(device)
    if require_cuda:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Production cache/train requires CUDA")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["memory_fraction"], device)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for key in ("planner_dir", "devkit_dir"):
        path = str(resolve_path(config, key))
        if path not in sys.path:
            sys.path.insert(0, path)
    return device
