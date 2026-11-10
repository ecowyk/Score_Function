"""Resumable frozen ChainFlow conditions, exact existing NAVSIM split and expert targets.

VLM sequences remain complete: ragged FP16 shards use offsets rather than truncation
or mean pooling. Only selected rows enter RAM/GPU during Score training.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import time
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import Dataset

from score_function.navsim_dd.reader_cache import source_tasks
from score_function.utils.dataset import validate_shard
from score_function.utils.train_utils import atomic_write, file_hash, fingerprint, read_json
from .native import (CAMERAS, CONDITION, activate_devkit, conditioner_identity, encode_scene,
                     load_agent, make_scene_loader)

SCHEMA = "navsim_chainflow_cache_v1"
FILES = {"scene.npy", "ego.npy", "vlm.npy", "offsets.npy", "target.npy", "records.json", "vlm_sources.json"}


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Path escapes cache: {relative}")
    return path


def validate_normalizer(stats):
    mean, std = (np.asarray(stats[k], dtype=np.float64) for k in ("mean", "std"))
    if (mean.shape != (4,) or std.shape != (4,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()
            or (mean[2:] != 0).any() or (std[2:] != 1).any()):
        raise ValueError("Invalid four-channel train-only normalizer")


def make_identity(source_dir, source_metadata, native_identity):
    validate_normalizer(source_metadata["normalizer"])
    return {"schema": SCHEMA, "condition": CONDITION, "native": native_identity,
            "source_manifest_sha256": file_hash(Path(source_dir) / "manifest.json"),
            "source_cache_fingerprint": fingerprint(source_metadata),
            "source_normalizer": source_metadata["normalizer"], "storage_dtype": "float16"}


def validate_chunk(directory, identity, task=None, checksums=True):
    directory = Path(directory)
    mark = validate_shard(directory, fingerprint(identity), checksums=checksums)
    if set(mark.get("hashes", {})) != FILES:
        raise ValueError(f"Unexpected ChainFlow shard members: {directory}")
    records = read_json(directory / "records.json")
    if len(records) != mark["count"] or not records:
        raise ValueError("Empty or inconsistent shard records")
    if file_hash(directory / "records.json") != mark["hashes"]["records.json"]:
        raise ValueError("Changed cached scene membership")
    if file_hash(directory / "vlm_sources.json") != mark["hashes"]["vlm_sources.json"]:
        raise ValueError("Changed VLM source provenance")
    sources = read_json(directory / "vlm_sources.json")
    if len(sources) != len(records) or any(
        src.get("token") != rec["token"] or src.get("log_name") != rec["log_name"]
        or len(src.get("sha256", "")) != 64 for src, rec in zip(sources, records)
    ):
        raise ValueError("Missing VLM source content provenance")
    if task is not None and (records != task["records"] or mark["count"] != task["count"]
                            or mark.get("source_marker_sha256") != task["source_marker_sha256"]
                            or mark["hashes"]["target.npy"] != task["target_sha256"]):
        raise ValueError("ChainFlow cache does not match original DD split/target provenance")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("scene", "ego", "vlm", "offsets", "target")}
    n = len(records)
    for key, shape, dtype in (("scene", (n, 64, 256), np.float16), ("ego", (n, 256), np.float16),
                              ("target", (n, 8, 4), np.float32), ("offsets", (n + 1,), np.int64)):
        if arrays[key].shape != shape or arrays[key].dtype != np.dtype(dtype):
            raise ValueError(f"Invalid {key} shape/dtype in {directory}")
    offsets, vlm = arrays["offsets"], arrays["vlm"]
    if (vlm.ndim != 2 or vlm.shape[1] != 1536 or vlm.dtype != np.float16
            or offsets[0] != 0 or offsets[-1] != len(vlm) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Invalid ragged VLM offsets or empty/truncated sequences")
    if checksums:
        for key, value in arrays.items():
            # Chunked scanning avoids allocating a corpus-sized isfinite array.
            for start in range(0, len(value), 1024):
                if not np.isfinite(value[start:start + 1024]).all():
                    raise ValueError(f"Nonfinite {key} in {directory}")
    return mark


def _vlm_header(stream, rows):
    """Reserve a constant-size NumPy v2 header, allowing bounded-memory appends."""
    header = repr({"descr": "<f2", "fortran_order": False, "shape": (int(rows), 1536)}).encode("ascii")
    header_length = 244  # 12-byte prefix + header = 256, aligned to 64 bytes.
    if len(header) >= header_length:
        raise ValueError("VLM array header exceeds reserved space")
    stream.seek(0)
    stream.write(np.lib.format.magic(2, 0))
    stream.write(struct.pack("<I", header_length))
    stream.write(header + b" " * (header_length - len(header) - 1) + b"\n")


def write_chunk(cache_dir, source_dir, identity, task, encode, vlm_source):
    destination = inside(cache_dir, task["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            validate_chunk(destination, identity, task)
            return False
        source = inside(source_dir, task["path"])
        if (file_hash(source / "complete.json") != task["source_marker_sha256"]
                or file_hash(source / "target.npy") != task["target_sha256"]):
            raise ValueError("Source target shard changed")
        targets = np.load(source / "target.npy", mmap_mode="r", allow_pickle=False)
        if targets.shape != (task["count"], 8, 4) or len(task["records"]) != task["count"]:
            raise ValueError("Source target shape/count disagrees with task")
        with TemporaryDirectory(prefix=destination.name + ".tmp-", dir=destination.parent) as temp:
            stage = Path(temp)
            arrays = {name: np.lib.format.open_memmap(stage / f"{name}.npy", mode="w+", dtype=np.float16,
                        shape=(task["count"], *shape))
                      for name, shape in (("scene", (64, 256)), ("ego", (256,)))}
            offsets, provenance = [0], []
            with (stage / "vlm.npy").open("w+b") as stream:
                _vlm_header(stream, 0)
                for row, (record, target) in enumerate(zip(task["records"], targets)):
                    origin = Path(vlm_source(record))
                    origin_hash = file_hash(origin)
                    condition = encode(record, target)
                    values = {name: torch.as_tensor(condition[name]).detach().cpu()
                              for name in ("scene", "ego", "vlm")}
                    if (values["scene"].shape != (64, 256) or values["ego"].shape != (256,)
                            or values["vlm"].ndim != 2 or values["vlm"].shape[1] != 1536
                            or values["vlm"].shape[0] < 1):
                        raise ValueError(f"Bad native condition shape for {record['token']}")
                    for name, value in values.items():
                        value = value.half().numpy()
                        if not np.isfinite(value).all():
                            raise ValueError(f"Condition overflows FP16: {record['token']} {name}")
                        values[name] = value
                    if file_hash(origin) != origin_hash:
                        raise ValueError("VLM source changed while encoding")
                    needed = sum(value.nbytes for value in values.values()) + 16 * 1024**2
                    if shutil.disk_usage(stage).free < needed:
                        raise OSError(f"Insufficient reported free space for next cache sample: {stage}")
                    for name in arrays:
                        arrays[name][row] = values[name]
                    stream.write(values["vlm"].astype("<f2", copy=False).tobytes(order="C"))
                    offsets.append(offsets[-1] + len(values["vlm"]))
                    provenance.append({"token": record["token"], "log_name": record["log_name"], "sha256": origin_hash})
                _vlm_header(stream, offsets[-1])
                stream.flush()
                os.fsync(stream.fileno())
            for array in arrays.values():
                array.flush()
            arrays.clear()
            np.save(stage / "offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
            shutil.copyfile(source / "target.npy", stage / "target.npy")
            atomic_write(stage / "records.json", task["records"])
            atomic_write(stage / "vlm_sources.json", provenance)
            hashes = {name: file_hash(stage / name) for name in FILES}
            if hashes["target.npy"] != task["target_sha256"]:
                raise ValueError("Expert targets changed while copying")
            atomic_write(stage / "complete.json", {
                "identity": fingerprint(identity), "count": task["count"],
                "source_marker_sha256": task["source_marker_sha256"], "hashes": hashes})
            validate_chunk(stage, identity, task)
            stage.rename(destination)
    return True


def publish_cache(cache_dir, source_dir, identity, tasks, source_metadata):
    cache_dir = Path(cache_dir)
    if file_hash(Path(source_dir) / "manifest.json") != identity["source_manifest_sha256"]:
        raise ValueError("Source membership changed during cache build")
    splits = {s: {"num_records": 0, "shards": []} for s in ("train", "val")}
    seen, logs = set(), {}
    for task in tasks:
        directory = inside(cache_dir, task["path"])
        validate_chunk(directory, identity, task)
        for r in task["records"]:
            if r["token"] in seen or logs.setdefault(r["log_name"], r["split"]) != r["split"]:
                raise ValueError("Duplicate token or train/val log leakage")
            seen.add(r["token"])
        entry = splits[task["split"]]
        entry["num_records"] += task["count"]
        entry["shards"].append({"path": task["path"], "count": task["count"],
                                 "marker_sha256": file_hash(directory / "complete.json")})
    if any(splits[s]["num_records"] != source_metadata["splits"][s]["num_records"]
           or not splits[s]["num_records"] for s in splits):
        raise ValueError("Incomplete train/validation cache")
    metadata = {"schema": SCHEMA, "identity": identity, "splits": splits,
                "normalizer": source_metadata["normalizer"], "conditioner_identity": identity["native"]}
    atomic_write(cache_dir / "manifest.json", metadata)
    atomic_write(cache_dir / "TRAINING_READY.json", {
        "schema": SCHEMA, "manifest_sha256": file_hash(cache_dir / "manifest.json"),
        "cache_fingerprint": fingerprint(metadata),
        "counts": {s: splits[s]["num_records"] for s in splits}})
    return metadata


def load_cache_metadata(cache_dir):
    root = Path(cache_dir)
    metadata = read_json(root / "manifest.json")
    identity = metadata.get("identity", {})
    if (metadata.get("schema") != SCHEMA or identity.get("schema") != SCHEMA
            or identity.get("condition") != CONDITION or read_json(root / "build.json") != identity
            or metadata.get("conditioner_identity") != identity.get("native")
            or metadata.get("normalizer") != identity.get("source_normalizer")):
        raise ValueError("Invalid ChainFlow cache identity/schema")
    validate_normalizer(metadata["normalizer"])
    return metadata


def validate_training_ready(cache_dir):
    root = Path(cache_dir)
    metadata = load_cache_metadata(root)
    ready = read_json(root / "TRAINING_READY.json")
    if (ready.get("schema") != SCHEMA or ready.get("manifest_sha256") != file_hash(root / "manifest.json")
            or ready.get("cache_fingerprint") != fingerprint(metadata)
            or ready.get("counts") != {s: metadata["splits"][s]["num_records"] for s in ("train", "val")}):
        raise ValueError("TRAINING_READY does not validate this ChainFlow manifest")
    return metadata


class ChainFlowDataset(Dataset):
    def __init__(self, cache_dir, split):
        if split not in ("train", "val"):
            raise ValueError("Expected train or val")
        self.root = Path(cache_dir).resolve()
        self.metadata = validate_training_ready(self.root)
        self.sha256 = file_hash(self.root / "manifest.json")
        self.mean = torch.tensor(self.metadata["normalizer"]["mean"], dtype=torch.float32)
        self.std = torch.tensor(self.metadata["normalizer"]["std"], dtype=torch.float32)
        self.records, self.locations, self.directories = [], [], []
        self._maps = OrderedDict()
        seen, log_splits = set(), {}
        for part in ("train", "val"):
            count = 0
            for shard in self.metadata["splits"][part]["shards"]:
                directory = inside(self.root, shard["path"])
                if file_hash(directory / "complete.json") != shard["marker_sha256"]:
                    raise ValueError("Changed ChainFlow shard completion marker")
                mark = validate_chunk(directory, self.metadata["identity"], checksums=False)
                records = read_json(directory / "records.json")
                if shard["count"] != mark["count"]:
                    raise ValueError("Shard count disagrees with manifest")
                count += len(records)
                for record in records:
                    if (record["split"] != part or record["token"] in seen
                            or log_splits.setdefault(record["log_name"], part) != part):
                        raise ValueError("Duplicate scene, incorrect split or train/val leakage")
                    seen.add(record["token"])
                if part == split:
                    self.locations.extend((len(self.directories), i) for i in range(len(records)))
                    self.records.extend(records)
                    self.directories.append(directory)
            if count != self.metadata["splits"][part]["num_records"] or count < 1:
                raise ValueError("Incomplete split count")

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_maps": OrderedDict()}

    def __getitem__(self, index):
        number, row = self.locations[index]
        if number not in self._maps:
            directory = self.directories[number]
            self._maps[number] = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                                  for name in ("scene", "ego", "target", "offsets", "vlm")}
            if len(self._maps) > 8:
                self._maps.popitem(last=False)
        self._maps.move_to_end(number)
        arrays = self._maps[number]
        values = {key: torch.from_numpy(np.array(arrays[key][row], dtype=np.float32))
                  for key in ("scene", "ego", "target")}
        begin, end = arrays["offsets"][row:row + 2]
        values["vlm"] = torch.from_numpy(np.array(arrays["vlm"][begin:end], dtype=np.float32))
        values["target"] = (values["target"] - self.mean) / self.std
        if any(not torch.isfinite(v).all() for v in values.values()):
            raise ValueError(f"Nonfinite cached condition at {self.records[index]['token']}")
        values["record"] = self.records[index]
        return values


def check_data(args, tasks):
    """CPU-only coverage check: all source scenes, four current cameras and VLM files.

    Inspect log dictionaries instead of loading/decoding all camera images.
    """
    activate_devkit(args.devkit_root)
    groups = defaultdict(set)
    for task in tasks:
        groups[(task["split"], task["log_name"])].update(r["token"] for r in task["records"])
    missing, counts = [], {"train": 0, "val": 0, "navtest": 0}
    missing_count = 0
    vlm_bytes = {"train": 0, "val": 0, "navtest": 0}

    def record_missing(kind, value):
        nonlocal missing_count
        missing_count += 1
        if len(missing) < 100:
            missing.append({"kind": kind, "path_or_token": str(value)})

    def inspect(loader, split, requested=None):
        if requested is not None:
            for token in requested - set(loader.tokens):
                record_missing("metadata", token)
        part = "test" if split == "navtest" else "trainval"
        sensor_root = Path(args.data_root) / "sensor_blobs" / part
        for token, frames in loader.scene_frames_dicts.items():
            frame = frames[loader._scene_filter.num_history_frames - 1]
            counts[split] += 1
            cameras = {key.lower(): value for key, value in frame["cams"].items()}
            for camera in CAMERAS:
                path = sensor_root / cameras[camera]["data_path"] if camera in cameras else None
                if path is None or not path.is_file():
                    record_missing(camera, path or f"{token}/{camera}")
            cache = Path(args.vlm_cache) / frame["log_name"] / token / "internvl_feature.gz"
            if not cache.is_file() or cache.stat().st_size == 0:
                record_missing("VLM", cache)
            else:
                vlm_bytes[split] += cache.stat().st_size

    for (split, log), tokens in sorted(groups.items()):
        loader = make_scene_loader(args.data_root, args.devkit_root, split, [log], tokens, True)
        inspect(loader, split, tokens)
    loader = make_scene_loader(args.data_root, args.devkit_root, "navtest", load_image_path=True)
    inspect(loader, "navtest")
    expected_test = set(loader._scene_filter.tokens or loader.tokens)
    for token in expected_test - set(loader.tokens):
        record_missing("navtest_metadata", token)
    complete = missing_count == 0 and counts == {"train": 85109, "val": 18179, "navtest": 12146}
    report = {"state": "ready" if complete else "failed", "counts": counts,
              "missing_count": missing_count, "missing_examples": missing,
              "cameras": list(CAMERAS), "history_images_required": False,
              "compressed_vlm_bytes": vlm_bytes,
              "cache_storage": "Full uncompressed FP16 VLM sequences; additional space required. "
                               "Compressed bytes and filesystem free-space reports are not a guarantee.",
              "reported_free_bytes": shutil.disk_usage(args.cache_dir).free}
    atomic_write(Path(args.cache_dir) / "coverage.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not complete:
        raise ValueError("ChainFlow sensor/VLM/metadata coverage is incomplete; see coverage.json")
    return report


def worker(args):
    torch.set_num_threads(1)
    activate_devkit(args.devkit_root)
    metadata, tasks = source_tasks(args.source_cache)
    identity = read_json(Path(args.cache_dir) / "build.json")
    agent = load_agent(args.devkit_root, args.checkpoint, args.backbone, args.vlm_cache, f"cuda:{args.rank}")
    if agent._score_bridge_identity != identity["native"]:
        raise ValueError("Native assets changed after cache launch")
    current, loader = None, None
    for task in tasks[args.rank::args.workers]:
        destination = inside(args.cache_dir, task["path"])
        if destination.exists():
            validate_chunk(destination, identity, task)
            print(f"REUSED {task['path']} {task['count']} scenes", flush=True)
            continue
        key = (task["split"], task["log_name"])
        if key != current:
            loader = make_scene_loader(args.data_root, args.devkit_root, task["split"], [task["log_name"]])
            current = key

        def encode(record, target):
            scene = loader.get_scene_from_token(record["token"])
            if scene.scene_metadata.log_name != record["log_name"]:
                raise ValueError("Native scene log differs from original training record")
            poses = np.asarray(scene.get_future_trajectory(8).poses, dtype=np.float32)
            actual = np.column_stack((poses[:, :2], np.cos(poses[:, 2]), np.sin(poses[:, 2])))
            if not np.allclose(actual, target, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Expert target mismatch: {record['token']}")
            conditions = encode_scene(agent, scene, f"cuda:{args.rank}")
            return {name: conditions[name][0] for name in ("scene", "ego", "vlm")}

        def vlm_source(record):
            return Path(args.vlm_cache) / record["log_name"] / record["token"] / "internvl_feature.gz"

        write_chunk(args.cache_dir, args.source_cache, identity, task, encode, vlm_source)
        print(f"READY {task['path']} {task['count']} scenes", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-cache", "cache-dir", "checkpoint", "backbone", "vlm-cache", "data-root", "devkit-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check-data", action="store_true", help="CPU-only coverage audit, no model allocation")
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.rank is not None:
        return worker(args)
    with (args.cache_dir / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata, tasks = source_tasks(args.source_cache)
        if args.check_data:
            return check_data(args, tasks)
        native = conditioner_identity(args.devkit_root, args.checkpoint, args.backbone, args.vlm_cache)
        identity = make_identity(args.source_cache, metadata, native)
        path = args.cache_dir / "build.json"
        if path.exists() and read_json(path) != identity:
            raise ValueError("Existing ChainFlow cache was built with different assets/split/normalizer")
        atomic_write(path, identity)
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.workers:
            raise ValueError(f"Need at least {args.workers} visible CUDA devices")
        processes, streams = [], []
        try:
            for rank in range(args.workers):
                command = [sys.executable, "-u", "-m", "score_function.navsim_chainflow.cache"]
                for name in ("source_cache", "cache_dir", "checkpoint", "backbone", "vlm_cache", "data_root", "devkit_root"):
                    command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
                command += ["--workers", str(args.workers), "--rank", str(rank)]
                stream = (args.cache_dir / f"worker{rank}.log").open("a")
                streams.append(stream)
                processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
            while True:
                codes = [process.poll() for process in processes]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError(f"Cache worker failure {codes}; see worker*.log")
                if all(code is not None for code in codes):
                    break
                time.sleep(1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
            for stream in streams:
                stream.close()
        publish_cache(args.cache_dir, args.source_cache, identity, tasks, metadata)
        print("CHAINFLOW CACHE READY: train=85109 val=18179", flush=True)


if __name__ == "__main__":
    main()
