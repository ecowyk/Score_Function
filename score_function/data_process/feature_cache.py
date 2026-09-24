"""Encode clean observations once, independently of DSM sigma and planner generation."""

from collections import Counter
from pathlib import Path

import torch

from score_function.utils import ddp
from score_function.utils.config import METHOD
from score_function.utils.dataset import ShardedDataset, validate_shard, write_shard
from score_function.utils.feature_dataset import read_feature_batch
from score_function.utils.planner_utils import (
    ego_normalizer_metadata,
    encode_conditions,
    load_frozen_planner,
    planner_identity,
)
from score_function.utils.train_utils import (
    atomic_write,
    file_hash,
    fingerprint,
    read_json,
    resolve_path,
)


def build_cache(config):
    device = torch.device(config["runtime"]["device"])
    ddp.setup(device)
    index = resolve_path(config, "cache")
    root = index.parent
    root.mkdir(parents=True, exist_ok=True)
    try:
        records = read_json(resolve_path(config, "manifest"))
        allowed = {
            Path(name).stem for name in read_json(resolve_path(config, "train_log_allowlist"))
        }
        if not records or any(row["log"] not in allowed for row in records):
            raise ValueError(
                "Manifest is empty or contains logs outside official training allowlist"
            )
        official = planner_identity(config)
        size, batch_size = config["data"]["shard_size"], config["data"]["encoding_batch_size"]
        identity = fingerprint(
            {
                "method": METHOD,
                "official": official,
                "shard_size": size,
                "manifest": file_hash(resolve_path(config, "manifest")),
                "implementation": {
                    name: file_hash(Path(__file__).resolve().parents[1] / name)
                    for name in (
                        "data_process/feature_cache.py",
                        "utils/planner_utils.py",
                        "utils/feature_dataset.py",
                        "utils/normalizer.py",
                        "utils/dataset.py",
                    )
                },
            }
        )
        if ddp.rank() == 0:
            marker = root / "build.json"
            if marker.exists() and read_json(marker)["identity"] != identity:
                raise ValueError("Cache inputs changed; choose a new score-function cache directory")
            atomic_write(marker, {"identity": identity})
        ddp.barrier()
        planner, planner_args = load_frozen_planner(config, device)
        metadata = {
            "samples": len(records),
            "identity": identity,
            "planner": official,
            "ego_normalizer": ego_normalizer_metadata(planner_args),
            "split_counts": dict(Counter(row["split"] for row in records)),
            "condition_source": "frozen scene encoder and route encoder; no diffusion sampling",
        }
        count = (len(records) + size - 1) // size
        for shard in range(ddp.rank(), count, ddp.world_size()):
            directory = root / f"shard_{shard:06d}"
            if (directory / "complete.json").exists():
                validate_shard(directory, identity)
                continue
            chunk, rows = records[shard * size : (shard + 1) * size], []
            for start in range(0, len(chunk), batch_size):
                subset = chunk[start : start + batch_size]
                for record in subset:
                    if file_hash(record["feature"]) != record["feature_sha256"]:
                        raise ValueError(f"Changed NPZ: {record['feature']}")
                inputs, target = read_feature_batch(subset, planner_args, device)
                context, route = encode_conditions(planner, inputs)
                for offset, record in enumerate(subset):
                    rows.append(
                        {
                            "record": {
                                key: record[key]
                                for key in (
                                    "token",
                                    "recording",
                                    "split",
                                    "start_time_us",
                                    "feature",
                                    "feature_sha256",
                                )
                            },
                            "target": target[offset].cpu(),
                            "context": context[offset].cpu(),
                            "route": route[offset].cpu(),
                        }
                    )
            write_shard(directory, rows, identity)
            progress = {
                "state": "running",
                "rank": ddp.rank(),
                "last_shard": shard,
                "total_shards": count,
            }
            atomic_write(root / f"status_rank{ddp.rank()}.json", progress)
            print(progress, flush=True)
        ddp.barrier()
        if ddp.rank() == 0:
            shards = [
                {
                    "path": f"shard_{i:06d}",
                    "marker_sha256": file_hash(root / f"shard_{i:06d}/complete.json"),
                }
                for i in range(count)
            ]
            atomic_write(
                index,
                {"schema_version": 1, "method": METHOD, "metadata": metadata, "shards": shards},
            )
            ShardedDataset(index).close()
            atomic_write(root / "status.json", {"state": "complete", "samples": len(records)})
    except Exception as exc:
        atomic_write(
            root / f"status_rank{ddp.rank()}.json", {"state": "failed", "error": repr(exc)}
        )
        raise
    finally:
        ddp.close()
