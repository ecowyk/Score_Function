"""Configuration, provenance, atomic writes, and reproducible random streams."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_write(path, value, tensor=False):
    """Replace only complete artifacts; temporary files stay on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    try:
        if tensor:
            torch.save(value, temporary)
        else:
            Path(temporary).write_text(
                json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def load_tensor(path):
    """Load trusted local research artifacts only; PyTorch files may contain pickle."""
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_path(config, key):
    path = Path(config["paths"][key]).expanduser()
    return path if path.is_absolute() else Path(config["paths"]["root"]) / path


def stable_seed(seed, *parts):
    """Seed identity is independent of batch ordering, size, and Python hash randomization."""
    return int(fingerprint([seed, *parts])[:15], 16)


def sample_noise(shape, identifiers, seed, namespace, dtype=torch.float32):
    """CPU-generated per-scenario Gaussian noise, shared across devices and methods."""
    return torch.stack(
        [
            torch.randn(
                shape,
                generator=torch.Generator().manual_seed(stable_seed(seed, namespace, key)),
                dtype=dtype,
            )
            for key in identifiers
        ]
    )


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.py"))}


def planner_source_hashes(config):
    """Hash actual official Python sources, including uncommitted local modifications."""
    root = resolve_path(config, "planner_dir") / "diffusion_planner"
    files = sorted(root.rglob("*.py"))
    if not files:
        raise FileNotFoundError(f"Planner sources not found: {root}")
    return {path.relative_to(root).as_posix(): file_hash(path) for path in files}


def capture_rank_rng(device):
    # Do not initialize contexts on the other seven GPUs from every DDP rank.
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rank_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def training_signature(config, cache_hash, world_size):
    training = {
        k: v
        for k, v in config["training"].items()
        if k not in ("num_workers", "log_every_updates", "checkpoint_every_updates")
    }
    return fingerprint(
        {
            "model": config["model"],
            "method": config["method"],
            "training": training,
            "cache": cache_hash,
            "world_size": world_size,
            "selection": config["checkpoint_selection"],
        }
    )
