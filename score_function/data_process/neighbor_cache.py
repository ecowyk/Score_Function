"""Cache one frozen-planner joint sample per frame; only its neighbors condition DSM."""

from pathlib import Path

import numpy as np
import torch

from score_function.utils import ddp
from score_function.utils.dataset import ShardedDataset, validate_shard
from score_function.utils.feature_dataset import read_feature_batch
from score_function.utils.neighbor import prediction_neighbors
from score_function.utils.planner_utils import load_frozen_planner, planner_identity
from score_function.utils.train_utils import (
    atomic_write,
    file_hash,
    fingerprint,
    read_json,
    resolve_path,
    stable_seed,
)


@torch.no_grad()
def build_neighbor_cache(config):
    device = torch.device(config["runtime"]["device"])
    ddp.setup(device)
    index = resolve_path(config, "neighbor_cache")
    root = index.parent
    root.mkdir(parents=True, exist_ok=True)
    try:
        source_path = resolve_path(config, "cache")
        source = read_json(source_path)
        source_hash = file_hash(source_path)
        official = planner_identity(config)
        if source["metadata"]["planner"] != official:
            raise ValueError("Shared cache and frozen planner differ")
        batch_size = config["data"]["neighbor_batch_size"]
        identity = fingerprint(
            {
                "source_cache_sha256": source_hash,
                "seed": config["reference_seed"],
                "batch_size": batch_size,
                "source": file_hash(__file__),
                "neighbor_util": file_hash(Path(__file__).parents[1] / "utils/neighbor.py"),
                "planner": official,
            }
        )
        marker = root / "build.json"
        if ddp.rank() == 0:
            if marker.exists() and read_json(marker)["identity"] != identity:
                raise ValueError("Neighbor cache protocol changed; choose a new cache directory")
            atomic_write(marker, {"identity": identity})
        ddp.barrier()
        planner, args = load_frozen_planner(config, device)
        for number in range(ddp.rank(), len(source["shards"]), ddp.world_size()):
            item = source["shards"][number]
            directory = root / item["path"]
            if (directory / "complete.json").exists():
                validate_shard(directory, identity)
                continue
            source_dir = source_path.parent / item["path"]
            validate_shard(source_dir, source["metadata"]["identity"])
            records = read_json(source_dir / "records.json")
            context = np.load(source_dir / "context.npy", mmap_mode="r", allow_pickle=False)
            futures, masks = [], []
            for start in range(0, len(records), batch_size):
                subset = records[start : start + batch_size]
                for row in subset:
                    if file_hash(row["feature"]) != row["feature_sha256"]:
                        raise ValueError(f"Changed input feature: {row['feature']}")
                inputs, _ = read_feature_batch(subset, args, device)
                # Match the official online observation adapter's fixed current ego state.
                current = (
                    inputs["ego_current_state"]
                    .new_tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                    .expand(len(subset), -1)
                )
                inputs["ego_current_state"] = args.observation_normalizer(
                    {"ego_current_state": current}
                )["ego_current_state"]
                encoding = torch.from_numpy(
                    np.array(context[start : start + len(subset)], copy=True)
                ).to(device)
                # Fixed batch membership is part of the cache identity. Changing GPU
                # assignment or resuming shards does not change the sampled futures.
                seed = stable_seed(
                    config["reference_seed"], "neighbor_cache", [row["token"] for row in subset]
                )
                with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                    torch.manual_seed(seed)
                    output = planner.decoder({"encoding": encoding}, inputs)
                condition = prediction_neighbors(
                    output["prediction"], inputs, args.state_normalizer
                )
                futures.append(condition["neighbor_future"].cpu().numpy())
                masks.append(condition["neighbor_valid"].cpu().numpy())
            context._mmap.close()
            directory.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for key, values in (("neighbor_future", futures), ("neighbor_valid", masks)):
                path = directory / f"{key}.npy"
                with path.open("wb") as stream:
                    np.save(stream, np.concatenate(values), allow_pickle=False)
                hashes[path.name] = file_hash(path)
            atomic_write(
                directory / "complete.json",
                {
                    "identity": identity,
                    "count": len(records),
                    "hashes": hashes,
                    "source_marker_sha256": item["marker_sha256"],
                },
            )
            progress = {
                "state": "running",
                "last_shard": number,
                "total_shards": len(source["shards"]),
                "rank": ddp.rank(),
            }
            atomic_write(root / f"status_rank{ddp.rank()}.json", progress)
            print(progress, flush=True)
        ddp.barrier()
        if ddp.rank() == 0:
            atomic_write(
                index,
                {
                    "schema_version": 1,
                    "identity": identity,
                    "source_cache_sha256": source_hash,
                    "conditioning": "frozen DP predicted neighbors; no expert neighbor futures",
                    "seed": config["reference_seed"],
                    "batch_size": batch_size,
                    "shards": [
                        {
                            "path": item["path"],
                            "marker_sha256": file_hash(root / item["path"] / "complete.json"),
                        }
                        for item in source["shards"]
                    ],
                },
            )
            ShardedDataset(source_path, neighbor_index=index).close()
            atomic_write(
                root / "status.json",
                {
                    "state": "complete",
                    "samples": source["metadata"]["samples"],
                },
            )
    except Exception as error:
        atomic_write(
            root / f"status_rank{ddp.rank()}.json", {"state": "failed", "error": repr(error)}
        )
        raise
    finally:
        ddp.close()
