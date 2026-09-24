"""Portable, memory-mapped tensor shards. Only requested samples enter RAM/GPU."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from score_function.utils.train_utils import atomic_write, file_hash, fingerprint, read_json

FIELDS = ("target", "context", "route")
SHAPES = {"target": (80, 4), "context": (107, 192), "route": (192,)}


def write_shard(directory, rows, identity):
    """A completion marker is published last; interrupted shards are safe to rebuild."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for key in FIELDS:
        tensor = torch.stack([row[key] for row in rows]).cpu()
        if tuple(tensor.shape[1:]) != SHAPES[key] or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid cache tensor: {key}")
        path = directory / f"{key}.npy"
        with path.open("wb") as stream:
            np.save(stream, tensor.numpy(), allow_pickle=False)
        hashes[path.name] = file_hash(path)
    records = [row["record"] for row in rows]
    atomic_write(directory / "records.json", records)
    hashes["records.json"] = file_hash(directory / "records.json")
    atomic_write(
        directory / "complete.json",
        {
            "identity": identity,
            "count": len(rows),
            "hashes": hashes,
        },
    )


def validate_shard(directory, identity=None, checksums=True):
    directory = Path(directory)
    state = read_json(directory / "complete.json")
    if identity is not None and state["identity"] != identity:
        raise ValueError(f"Cache provenance changed: {directory}")
    for name, digest in state["hashes"].items():
        path = directory / name
        if not path.is_file() or (checksums and file_hash(path) != digest):
            raise ValueError(f"Missing/corrupt cache file: {path}")
    return state


class ShardedDataset(Dataset):
    def __init__(self, index_path, split=None, max_open_shards=16, neighbor_index=None):
        self.index_path = Path(index_path)
        index = read_json(self.index_path)
        if index.get("schema_version") != 1 or index.get("method") != "score_function_v1":
            raise ValueError(
                "Use the score-function context cache, not a cache from either previous method"
            )
        self.metadata = index["metadata"]
        self.sha256 = file_hash(self.index_path)
        self.shards = index["shards"]
        self.neighbor_root = None
        if neighbor_index is not None:
            neighbor_index = Path(neighbor_index)
            neighbors = read_json(neighbor_index)
            if neighbors.get("source_cache_sha256") != self.sha256 or len(
                neighbors["shards"]
            ) != len(self.shards):
                raise ValueError("Neighbor cache does not match the shared scene cache")
            self.neighbor_root = neighbor_index.parent.resolve()
            self.neighbor_shards = neighbors["shards"]
            self.sha256 = fingerprint([self.sha256, file_hash(neighbor_index)])
            self.neighbor_identity = neighbors["identity"]
        self.records, self.locations = [], []
        seen, recordings = set(), {}
        root = self.index_path.parent.resolve()
        for number, item in enumerate(self.shards):
            directory = (root / item["path"]).resolve()
            if not directory.is_relative_to(root):
                raise ValueError("Shard path escapes cache directory")
            marker = validate_shard(directory, self.metadata["identity"], checksums=False)
            if file_hash(directory / "complete.json") != item["marker_sha256"]:
                raise ValueError("Shard marker changed since index publication")
            rows = read_json(directory / "records.json")
            if self.neighbor_root is not None:
                neighbor_item = self.neighbor_shards[number]
                neighbor_dir = (self.neighbor_root / neighbor_item["path"]).resolve()
                if not neighbor_dir.is_relative_to(self.neighbor_root):
                    raise ValueError("Neighbor shard escapes cache directory")
                marker_n = validate_shard(neighbor_dir, self.neighbor_identity, checksums=False)
                if (
                    file_hash(neighbor_dir / "complete.json") != neighbor_item["marker_sha256"]
                    or marker_n["count"] != len(rows)
                    or marker_n["source_marker_sha256"] != item["marker_sha256"]
                ):
                    raise ValueError("Neighbor shard alignment/provenance mismatch")
            if file_hash(directory / "records.json") != marker["hashes"]["records.json"]:
                raise ValueError("Shard records changed since cache construction")
            if len(rows) != marker["count"]:
                raise ValueError("Shard record count mismatch")
            for offset, record in enumerate(rows):
                key = (record["recording"], record["start_time_us"])
                token = record["token"]
                if key in seen or ("token", token) in seen:
                    raise ValueError("Duplicate frame/token in cache")
                seen.update((key, ("token", token)))
                group, part = record["recording"], record["split"]
                if part not in ("train", "val", "test"):
                    raise ValueError("Invalid split")
                if group in recordings and recordings[group] != part:
                    raise ValueError("Recording leakage across splits")
                recordings[group] = part
                if split is None or part == split:
                    self.records.append(record)
                    self.locations.append((number, offset))
        if len(seen) // 2 != self.metadata["samples"] or not self.records:
            raise ValueError(f"Count mismatch or empty split: {split}")
        self.max_open_shards = max_open_shards
        self._maps = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_maps"] = OrderedDict()
        return state

    def close(self):
        for arrays in self._maps.values():
            for array in arrays.values():
                array._mmap.close()
        self._maps.clear()

    def __getitem__(self, index):
        shard, offset = self.locations[index]
        if shard not in self._maps:
            directory = self.index_path.parent / self.shards[shard]["path"]
            self._maps[shard] = {
                key: np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                for key in FIELDS
            }
            if self.neighbor_root is not None:
                neighbor_dir = self.neighbor_root / self.neighbor_shards[shard]["path"]
                self._maps[shard].update(
                    {
                        key: np.load(neighbor_dir / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                        for key in ("neighbor_future", "neighbor_valid")
                    }
                )
            if len(self._maps) > self.max_open_shards:
                _, arrays = self._maps.popitem(last=False)
                for array in arrays.values():
                    array._mmap.close()
        self._maps.move_to_end(shard)
        row = {
            key: torch.from_numpy(np.array(value[offset], copy=True))
            for key, value in self._maps[shard].items()
        }
        row["record"] = self.records[index]
        return row


def collate_cpu(rows):
    return {
        **{key: torch.stack([row[key] for row in rows]) for key in rows[0] if key != "record"},
        "tokens": [row["record"]["token"] for row in rows],
    }


def device_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class EpochBatchSampler(Sampler):
    """One global shuffle, disjoint ranks, exact update-boundary resume.

    Like the official trainer, incomplete global batches are dropped. Microbatches
    split a global update without changing its membership or weighting.
    """

    def __init__(
        self, size, global_batch, microbatch, rank=0, world_size=1, seed=0, epoch=0, start_update=0
    ):
        if global_batch % world_size or (global_batch // world_size) % microbatch:
            raise ValueError("Global batch must divide world size and microbatch exactly")
        if size < global_batch or not 0 <= rank < world_size:
            raise ValueError("Require at least one complete global batch and a valid rank")
        self.size, self.global_batch, self.microbatch = size, global_batch, microbatch
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch, self.start_update = epoch, start_update
        self.updates = size // global_batch

    def __iter__(self):
        order = torch.randperm(
            self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)
        ).tolist()
        local = self.global_batch // self.world_size
        for update in range(self.start_update, self.updates):
            start = update * self.global_batch + self.rank * local
            for offset in range(0, local, self.microbatch):
                yield order[start + offset : start + offset + self.microbatch]

    def __len__(self):
        return (self.updates - self.start_update) * (
            self.global_batch // self.world_size // self.microbatch
        )


def build_data_loader(dataset, cfg, *, batch_sampler=None, indices=None):
    if indices is not None:
        dataset = Subset(dataset, indices)
    kwargs = {
        "num_workers": cfg["num_workers"],
        "collate_fn": collate_cpu,
        "pin_memory": True,
        "generator": torch.Generator().manual_seed(1947),
    }
    if cfg["num_workers"]:
        kwargs.update(persistent_workers=False, prefetch_factor=2, multiprocessing_context="spawn")
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=cfg["microbatch_size"], shuffle=False, **kwargs)
