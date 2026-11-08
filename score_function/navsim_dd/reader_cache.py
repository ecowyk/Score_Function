"""Prepare compact frozen-DD backbone features for the full trajectory reader.

The original keyval cache remains authoritative for scene membership, targets,
split assignment and normalization. This cache has a separate schema and path.
Only scene features are cached: the noisy/current trajectory is read online.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import Dataset

from score_function.navsim_dd import cache as source_cache
from score_function.utils.dataset import validate_shard
from score_function.utils.train_utils import atomic_write, file_hash, fingerprint, read_json

SCHEMA = "navsim_dd_reader_cache_v1"
CONDITION = "dd-backbone512-status-reader-v1"
ASSETS_FILE = "reader_assets.pt"
FIELDS = {
    "context": ((512, 8, 8), np.float16),
    "route": ((256,), np.float16),
    "target": ((8, 4), np.float32),
}
REQUIRED_FILES = {f"{name}.npy" for name in FIELDS} | {"records.json"}


def _inside(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(root).resolve() not in path.parents:
        raise ValueError(f"Cache path escapes root: {relative}")
    return path


def source_tasks(source_dir):
    """Read exact existing shard membership, detecting leakage and stale markers."""
    source_dir = Path(source_dir).resolve()
    metadata = source_cache.load_cache_metadata(source_dir)
    tasks, seen, log_splits = [], set(), {}
    for split in ("train", "val"):
        dataset = source_cache.NavsimScoreDataset(source_dir, split)
        for item, directory in zip(metadata["splits"][split]["shards"], dataset._directories):
            mark = validate_shard(directory, fingerprint(metadata["identity"]), checksums=False)
            for name in ("records.json", "target.npy", "route.npy"):
                if name not in mark["hashes"] or file_hash(directory / name) != mark["hashes"][name]:
                    raise ValueError(f"Source cache changed: {directory / name}")
            records = read_json(directory / "records.json")
            logs = {r["log_name"] for r in records}
            if len(logs) != 1:
                raise ValueError("Source shard must contain exactly one log")
            log_name = next(iter(logs))
            if log_splits.setdefault(log_name, split) != split:
                raise ValueError(f"Source log occurs in both splits: {log_name}")
            for record in records:
                token = record["token"]
                if token in seen or record["split"] != split:
                    raise ValueError(f"Duplicate token or split mismatch: {token}")
                seen.add(token)
            target = np.load(directory / "target.npy", mmap_mode="r", allow_pickle=False)
            route = np.load(directory / "route.npy", mmap_mode="r", allow_pickle=False)
            if (target.shape != (len(records), 8, 4) or target.dtype != np.float32
                    or route.shape != (len(records), 256) or route.dtype != np.float16
                    or not np.isfinite(target).all() or not np.isfinite(route).all()):
                raise ValueError(f"Invalid source targets/status: {directory}")
            tasks.append({"path": item["path"], "split": split, "log_name": log_name,
                          "count": len(records), "records": records,
                          "source_marker_sha256": item["marker_sha256"],
                          "target_sha256": mark["hashes"]["target.npy"]})
    return metadata, tasks


def make_identity(source_dir, source_metadata, asset_sha256):
    return {
        **source_metadata["identity"],
        "condition": CONDITION, "schema": SCHEMA,
        "source_manifest_sha256": file_hash(Path(source_dir) / "manifest.json"),
        "source_cache_fingerprint": fingerprint(source_metadata),
        "source_normalizer": source_metadata["normalizer"],
        "reader_assets_sha256": asset_sha256,
        "cache_dtype": "float16", "latent_shape": [512, 8, 8],
    }


def _conditioner_identity(identity):
    return {key: identity[key] for key in (*source_cache.CONDITIONER_KEYS, "reader_assets_sha256")}


def _validate_chunk(directory, identity, task, *, checksums=True):
    directory = Path(directory)
    mark = validate_shard(directory, fingerprint(identity), checksums=checksums)
    if (set(mark["hashes"]) != REQUIRED_FILES or mark["count"] != task["count"]
            or mark.get("source_marker_sha256") != task["source_marker_sha256"]
            or mark["hashes"].get("target.npy") != task["target_sha256"]
            or read_json(directory / "records.json") != task["records"]):
        raise ValueError(f"Reader cache scene/target provenance mismatch: {directory}")
    for name, (shape, dtype) in FIELDS.items():
        value = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if value.shape != (task["count"], *shape) or value.dtype != np.dtype(dtype):
            raise ValueError(f"Reader cache shape/dtype mismatch: {directory}/{name}.npy")
        if checksums and not np.isfinite(value).all():
            raise ValueError(f"Reader cache nonfinite values: {directory}/{name}.npy")
    return mark


def write_chunk(cache_dir, source_dir, identity, task, encode):
    """Resume only complete matching shards; incomplete temp directories are invisible."""
    destination = _inside(cache_dir, task["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A per-shard lock also protects against orphan workers from a killed coordinator.
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            _validate_chunk(destination, identity, task)
            return False
        source = _inside(source_dir, task["path"])
        if file_hash(source / "complete.json") != task["source_marker_sha256"]:
            raise ValueError(f"Source marker changed: {source}")
        if file_hash(source / "target.npy") != task["target_sha256"]:
            raise ValueError(f"Source targets changed: {source}")
        targets = np.load(source / "target.npy", allow_pickle=False)
        with TemporaryDirectory(prefix=".building-reader-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "shard"
            stage.mkdir()
            arrays = {name: np.empty((task["count"], *shape), dtype=dtype)
                      for name, (shape, dtype) in FIELDS.items() if name != "target"}
            for offset, record in enumerate(task["records"]):
                latent, status = encode(record, targets[offset])
                for name, value in (("context", latent), ("route", status)):
                    shape, dtype = FIELDS[name]
                    arrays[name][offset] = source_cache._array(value, shape, dtype)
            for name, array in arrays.items():
                np.save(stage / f"{name}.npy", array, allow_pickle=False)
            # Preserve exact target-file bytes, not a round trip through normalized values.
            (stage / "target.npy").write_bytes((source / "target.npy").read_bytes())
            atomic_write(stage / "records.json", task["records"])
            hashes = {name: file_hash(stage / name) for name in REQUIRED_FILES}
            if hashes["target.npy"] != task["target_sha256"]:
                raise ValueError("Source target file changed during copy")
            atomic_write(stage / "complete.json", {
                "identity": fingerprint(identity), "count": task["count"],
                "source_marker_sha256": task["source_marker_sha256"], "hashes": hashes,
            })
            _validate_chunk(stage, identity, task)
            stage.rename(destination)
    return True


def publish_cache(cache_dir, source_dir, identity, tasks, source_metadata, assets_spec):
    """Coordinator-only publication after all exact source scenes have completed."""
    cache_dir = Path(cache_dir)
    if file_hash(Path(source_dir) / "manifest.json") != identity["source_manifest_sha256"]:
        raise ValueError("Source manifest changed during cache preparation")
    if file_hash(cache_dir / ASSETS_FILE) != identity["reader_assets_sha256"]:
        raise ValueError("Frozen reader assets changed during cache preparation")
    splits = {split: {"num_records": 0, "shards": []} for split in ("train", "val")}
    tokens, log_splits = set(), {}
    for task in tasks:
        path = _inside(cache_dir, task["path"])
        _validate_chunk(path, identity, task)
        for record in task["records"]:
            if record["token"] in tokens or log_splits.setdefault(record["log_name"], record["split"]) != record["split"]:
                raise ValueError("Duplicate token or cross-split log in reader cache")
            tokens.add(record["token"])
        entry = splits[task["split"]]
        entry["num_records"] += task["count"]
        entry["shards"].append({"path": task["path"], "count": task["count"],
                                "marker_sha256": file_hash(path / "complete.json"),
                                "source_marker_sha256": task["source_marker_sha256"],
                                "target_sha256": task["target_sha256"]})
    if any(splits[s]["num_records"] != source_metadata["splits"][s]["num_records"]
           or not splits[s]["num_records"] for s in splits):
        raise ValueError("Reader cache does not exactly cover the source train/val splits")
    metadata = {
        "schema": SCHEMA, "identity": identity, "splits": splits,
        "normalizer": source_metadata["normalizer"],
        "conditioner_identity": _conditioner_identity(identity),
        "reader_assets": {"path": ASSETS_FILE, "sha256": identity["reader_assets_sha256"], "spec": assets_spec},
    }
    atomic_write(cache_dir / "manifest.json", metadata)
    atomic_write(cache_dir / "TRAINING_READY.json", {
        "validation_version": 1, "schema": SCHEMA,
        "manifest_sha256": file_hash(cache_dir / "manifest.json"),
        "cache_fingerprint": fingerprint(metadata),
        "records": {s: splits[s]["num_records"] for s in splits},
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "reader_assets_sha256": identity["reader_assets_sha256"],
    })
    return metadata


def load_cache_metadata(cache_dir):
    cache_dir = Path(cache_dir)
    data = read_json(cache_dir / "manifest.json")
    identity = data.get("identity", {})
    if (data.get("schema") != SCHEMA or identity.get("schema") != SCHEMA
            or identity.get("condition") != CONDITION
            or read_json(cache_dir / "build.json") != identity
            or data.get("conditioner_identity") != _conditioner_identity(identity)
            or data.get("normalizer") != identity.get("source_normalizer")):
        raise ValueError("Invalid full DD reader cache schema/provenance")
    asset = data.get("reader_assets", {})
    if (asset.get("path") != ASSETS_FILE or asset.get("sha256") != identity["reader_assets_sha256"]
            or file_hash(cache_dir / ASSETS_FILE) != asset["sha256"]):
        raise ValueError("Frozen reader assets are missing or changed")
    mean = source_cache._array(data["normalizer"]["mean"], (4,), np.float64)
    std = source_cache._array(data["normalizer"]["std"], (4,), np.float64)
    if np.any(std <= 0) or np.any(mean[2:] != 0) or np.any(std[2:] != 1):
        raise ValueError("Invalid reader cache normalization")
    return data


load_reader_cache_metadata = load_cache_metadata


class ReaderScoreDataset(Dataset):
    """Read compact scene latents on demand; never cache trajectory-dependent reads."""

    def __init__(self, cache_dir, split):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        self.root = Path(cache_dir).resolve()
        self.metadata = load_cache_metadata(self.root)
        self.sha256 = file_hash(self.root / "manifest.json")
        self.mean = torch.tensor(self.metadata["normalizer"]["mean"], dtype=torch.float32)
        self.std = torch.tensor(self.metadata["normalizer"]["std"], dtype=torch.float32)
        self.records, self._locations, self._directories = [], [], []
        self._maps = OrderedDict()
        entry = self.metadata["splits"][split]
        seen, log_splits = set(), {}
        # Validate both splits' record membership so even a one-split reader detects leakage.
        for part in ("train", "val"):
            for item in self.metadata["splits"][part]["shards"]:
                directory = _inside(self.root, item["path"])
                if file_hash(directory / "complete.json") != item["marker_sha256"]:
                    raise ValueError(f"Changed reader completion marker: {directory}")
                mark = validate_shard(directory, fingerprint(self.metadata["identity"]), checksums=False)
                if set(mark["hashes"]) != REQUIRED_FILES or file_hash(directory / "records.json") != mark["hashes"]["records.json"]:
                    raise ValueError(f"Changed reader records: {directory}")
                records = read_json(directory / "records.json")
                task = {**item, "records": records}
                _validate_chunk(directory, self.metadata["identity"], task, checksums=False)
                for record in records:
                    if (record["split"] != part or record["token"] in seen
                            or log_splits.setdefault(record["log_name"], part) != part):
                        raise ValueError("Duplicate token or train/val leakage in reader cache")
                    seen.add(record["token"])
                if part == split:
                    self._locations.extend((len(self._directories), i) for i in range(len(records)))
                    self.records.extend(records)
                    self._directories.append(directory)
        if not self.records or len(self.records) != entry["num_records"]:
            raise ValueError(f"Incomplete reader cache {split} split")

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_maps": OrderedDict()}

    def __getitem__(self, index):
        number, offset = self._locations[index]
        if number not in self._maps:
            directory = self._directories[number]
            self._maps[number] = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                                  for name in FIELDS}
            if len(self._maps) > 16:
                self._maps.popitem(last=False)
        self._maps.move_to_end(number)
        row = {key: torch.from_numpy(np.array(array[offset], dtype=np.float32))
               for key, array in self._maps[number].items()}
        row["target"] = (row["target"] - self.mean) / self.std
        row["record"] = self.records[index]
        return row


DDReaderDataset = ReaderScoreDataset


def _native_worker(args):
    from dataclasses import replace
    from navsim.common.dataloader import SceneLoader
    from score_function.navsim_dd.conditioner import features_from_input, load_diffusiondrive
    from score_function.navsim_dd.reader import verify_native_reader

    root = source_cache.verified_devkit_root(args.devkit_root)
    identity = read_json(args.cache_dir / "build.json")
    source_metadata, tasks = source_tasks(args.source_cache)
    if file_hash(args.source_cache / "manifest.json") != identity["source_manifest_sha256"]:
        raise ValueError("Source cache changed before worker startup")
    # Group consecutive shards for the same log onto one worker to avoid repeated log loading.
    groups = list(dict.fromkeys((t["split"], t["log_name"]) for t in tasks))
    assigned = set(groups[args.rank::args.workers])
    tasks = [t for t in tasks if (t["split"], t["log_name"]) in assigned]
    device = torch.device("cuda", args.rank)
    torch.cuda.set_device(device)
    agent = load_diffusiondrive(args.checkpoint, args.anchor, args.backbone, device)
    model = agent._transfuser_model
    assets = torch.load(args.cache_dir / ASSETS_FILE, map_location="cpu", weights_only=True)
    filters = {s: source_cache.build_navtrain_filter(s, root) for s in ("train", "val")}
    sensors = replace(agent.get_sensor_config(), cam_l1=False, cam_l2=False,
                      cam_r1=False, cam_r2=False, cam_b0=False)
    current_group, loader, checked = None, None, False
    completed = 0
    for task in tasks:
        destination = _inside(args.cache_dir, task["path"])
        if destination.exists():
            _validate_chunk(destination, identity, task)
            completed += task["count"]
            print(f"REUSED {task['path']} ({task['count']} scenes)", flush=True)
            continue
        group = (task["split"], task["log_name"])
        if group != current_group:
            loader = SceneLoader(args.data_root / "navsim_logs/trainval",
                                 args.data_root / "sensor_blobs/trainval",
                                 replace(filters[task["split"]], log_names=[task["log_name"]]), sensors)
            current_group = group

        @torch.no_grad()
        def encode(record, target):
            nonlocal checked
            scene = loader.get_scene_from_token(record["token"])
            if (scene.scene_metadata.log_name != record["log_name"]
                    or scene.scene_metadata.initial_token != record["token"]):
                raise ValueError("Native scene identity differs from original cache")
            poses = np.asarray(scene.get_future_trajectory(8).poses, dtype=np.float32)
            expected = np.column_stack((poses[:, :2], np.cos(poses[:, 2]), np.sin(poses[:, 2])))
            if not np.allclose(expected, target, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Native expert target differs from source cache: {record['token']}")
            features = features_from_input(agent, scene.get_agent_input(), device)
            _, latent, _ = model._backbone(features["camera_feature"], features["lidar_feature"])
            status = model._status_encoding(features["status_feature"])
            if not checked:
                xy = torch.as_tensor(poses[None, :, :2], dtype=torch.float32, device=device)
                # Verify actual copied weights and coordinate handling before publishing a shard.
                report = verify_native_reader(agent, latent, status, xy, assets=assets)
                atomic_write(args.cache_dir / f"native_check_rank{args.rank}.json", report)
                print(f"NATIVE READER CHECK OK {report}", flush=True)
                checked = True
            return latent[0], status[0]

        write_chunk(args.cache_dir, args.source_cache, identity, task, encode)
        completed += task["count"]
        atomic_write(args.cache_dir / f"worker{args.rank}.json", {
            "state": "running", "completed": completed,
            "total": sum(t["count"] for t in tasks), "last_shard": task["path"],
        })
        print(f"READY {task['path']} ({completed}/{sum(t['count'] for t in tasks)} scenes)", flush=True)
    atomic_write(args.cache_dir / f"worker{args.rank}.json", {
        "state": "complete", "completed": completed, "total": completed,
    })


def _run_workers(args):
    processes, streams = [], []
    try:
        for rank in range(args.workers):
            command = [sys.executable, "-u", "-m", "score_function.navsim_dd.reader_cache",
                       "--source-cache", str(args.source_cache), "--cache-dir", str(args.cache_dir),
                       "--checkpoint", str(args.checkpoint), "--anchor", str(args.anchor),
                       "--backbone", str(args.backbone), "--data-root", str(args.data_root),
                       "--devkit-root", str(args.devkit_root), "--workers", str(args.workers),
                       "--rank", str(rank)]
            stream = (args.cache_dir / f"worker{rank}.log").open("a", buffering=1)
            streams.append(stream)
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL))
        last_report = 0.
        while any(process.poll() is None for process in processes):
            failures = [(rank, p.returncode) for rank, p in enumerate(processes)
                        if p.poll() is not None and p.returncode != 0]
            if failures:
                raise RuntimeError(f"Cache worker failed {failures}; inspect {args.cache_dir}/worker*.log. "
                                   "Rerun the same command to resume complete shards.")
            if time.monotonic() - last_report > 30:
                done = sum(1 for split in ("train", "val") for _ in (args.cache_dir / split).glob("*/complete.json"))
                print(f"Reader cache: {done} completed shards; worker logs: {args.cache_dir}/worker*.log", flush=True)
                last_report = time.monotonic()
            time.sleep(1)
        failures = [(rank, p.returncode) for rank, p in enumerate(processes) if p.returncode]
        if failures:
            raise RuntimeError(f"Cache workers failed: {failures}; inspect worker logs")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()


def prepare(args):
    source_dir, cache_dir = args.source_cache.resolve(), args.cache_dir.resolve()
    if source_dir == cache_dir or source_dir in cache_dir.parents or cache_dir in source_dir.parents:
        raise ValueError("Reader cache must be separate from the original score cache")
    args.source_cache, args.cache_dir = source_dir, cache_dir
    source_metadata, tasks = source_tasks(source_dir)
    counts = {s: source_metadata["splits"][s]["num_records"] for s in ("train", "val")}
    if source_metadata["identity"].get("max_per_split") is not None or sum(counts.values()) != 103288:
        raise ValueError(f"Require the complete official source cache, found {counts}")
    original_assets = source_cache.conditioner_identity(args.checkpoint, args.anchor, args.backbone)
    if source_metadata["conditioner_identity"] != original_assets:
        raise ValueError("DD checkpoint, anchor or backbone differs from the original cache")
    args.devkit_root = source_cache.verified_devkit_root(args.devkit_root)
    args.data_root = args.data_root.resolve()
    if str(args.data_root) != source_metadata["identity"]["data_root"]:
        raise ValueError("Data root differs from the original score cache")
    log_dir = args.data_root / "navsim_logs/trainval"
    inventory = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(log_dir.glob("*.pkl"))]
    if fingerprint(inventory) != source_metadata["identity"].get("log_inventory_sha256"):
        raise ValueError("NAVSIM metadata inventory changed since original cache construction")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / ".build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another coordinator is preparing {cache_dir}") from exc
        if args.verify_only and not (cache_dir / "manifest.json").is_file():
            raise ValueError("Reader cache is not complete")
        assets_path = cache_dir / ASSETS_FILE
        if not assets_path.is_file():
            if (cache_dir / "build.json").exists():
                raise ValueError("Reader assets disappeared from an existing build")
            from score_function.navsim_dd.conditioner import load_diffusiondrive
            from score_function.navsim_dd.reader import export_reader_assets
            # Export on CPU; the coordinator must not retain a CUDA context during workers.
            agent = load_diffusiondrive(args.checkpoint, args.anchor, args.backbone, "cpu")
            atomic_write(assets_path, export_reader_assets(agent), tensor=True)
            del agent
        assets = torch.load(assets_path, map_location="cpu", weights_only=True)
        identity = make_identity(source_dir, source_metadata, file_hash(assets_path))
        build_file = cache_dir / "build.json"
        if build_file.exists():
            if read_json(build_file) != identity:
                raise ValueError("Reader cache build identity changed; use a separate cache directory")
        else:
            atomic_write(build_file, identity)
        if not (cache_dir / "manifest.json").is_file():
            if not torch.cuda.is_available() or torch.cuda.device_count() < args.workers:
                raise ValueError(f"Need {args.workers} visible GPUs for cache preparation")
            print(f"Preparing full reader cache: {counts}; {args.workers} GPU workers; compact latents.", flush=True)
            _run_workers(args)
        result = publish_cache(cache_dir, source_dir, identity, tasks, source_metadata, assets["spec"])
        print(f"READER CACHE READY: {cache_dir}; {counts}; source targets and splits unchanged.", flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--devkit-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.rank is not None and not 0 <= args.rank < args.workers):
        parser.error("Invalid worker count/rank")
    if args.rank is not None:
        _native_worker(args)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
