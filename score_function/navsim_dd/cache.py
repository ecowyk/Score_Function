"""Build and read the frozen DiffusionDrive conditioning cache on NAVSIM v1 navtrain."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import Dataset

from score_function.utils.dataset import validate_shard
from score_function.utils.train_utils import atomic_write, file_hash, fingerprint, read_json

DD_COMMIT = "9b52ed0ec06b073d82d6f392ab084c7b301c8681"
# Digest of the pinned navsim Python/YAML tree. Works for git clones and source archives.
DD_SOURCE_SHA256 = "ddf0438046ffc9bcc3607690dd8b20cf1199f406276c7981f92536398ebfc9a7"
SCHEMA = "navsim_dd_keyval_score_v1"
CONDITIONER_KEYS = ("condition", "dd_commit", "checkpoint_sha256", "anchor_sha256", "backbone_sha256")
FILTER_CONFIG = Path("navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
LOG_CONFIG = Path("navsim/planning/script/config/training/default_train_val_test_log_split.yaml")
FIELDS = {"context": ((64, 256), np.float16), "route": ((256,), np.float16), "target": ((8, 4), np.float32)}


def verified_devkit_root(devkit_root=None):
    """Make import/PYTHONPATH mistakes explicit. The archive need not include .git."""
    import navsim

    installed = Path(navsim.__file__).resolve().parent.parent
    root = Path(devkit_root or os.environ.get("NAVSIM_DEVKIT_ROOT") or installed).resolve()
    if installed != root:
        raise ValueError(f"Imported navsim from {installed}, expected {root}; check PYTHONPATH")
    package = root / "navsim"
    digest = hashlib.sha256()
    for path in sorted(p for p in package.rglob("*") if p.suffix in {".py", ".yaml"} and p.is_file()):
        contents = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(path.relative_to(package).as_posix().encode() + b"\0" + hashlib.sha256(contents).digest())
    if digest.hexdigest() != DD_SOURCE_SHA256:
        raise ValueError(f"Expected unmodified DiffusionDrive navsim source from {DD_COMMIT} in {root}")
    return root


def conditioner_identity(checkpoint, anchor, backbone):
    return {
        "condition": "dd-keyval64-status-v1",
        "dd_commit": DD_COMMIT,
        "checkpoint_sha256": file_hash(checkpoint),
        "anchor_sha256": file_hash(anchor),
        "backbone_sha256": file_hash(backbone),
    }


def build_navtrain_filter(split, devkit_root=None):
    """Match DiffusionDrive's training splits (official log sets intersected with navtrain)."""
    import yaml
    from navsim.common.dataclasses import SceneFilter

    if split not in ("train", "val"):
        raise ValueError(f"Expected train or val, got {split!r}")
    root = verified_devkit_root(devkit_root)
    with (root / FILTER_CONFIG).open() as stream:
        options = yaml.safe_load(stream)
    with (root / LOG_CONFIG).open() as stream:
        log_config = yaml.safe_load(stream)
    train, val = (set(log_config[name]) for name in ("train_logs", "val_logs"))
    if train & val or (train | val) & set(log_config.get("test_logs", [])):
        raise ValueError("Official log split overlaps")
    navtrain = set(options["log_names"])
    options["log_names"] = sorted(navtrain & (train if split == "train" else val))
    if navtrain - (train | val) or not options["log_names"]:
        raise ValueError("Unassigned or empty navtrain log split")
    return SceneFilter(**{k: v for k, v in options.items() if not k.startswith("_")})


def _array(value, shape, dtype):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    value = np.asarray(value, dtype=dtype)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite {shape}, got {value.shape}")
    return value


def _write_chunk(directory, identity, records, encode):
    """A renamed directory is always complete. An interrupted writer leaves no visible shard."""
    if directory.exists():
        mark = validate_shard(directory, identity, checksums=True)
        if read_json(directory / "records.json") != records or mark["count"] != len(records):
            raise ValueError(f"Scene selection changed for {directory}")
        return
    with TemporaryDirectory(dir=directory.parent, prefix=".building-") as temporary:
        stage = Path(temporary) / "shard"
        stage.mkdir()
        arrays = {k: np.empty((len(records), *shape), dtype) for k, (shape, dtype) in FIELDS.items()}
        for i, record in enumerate(records):
            poses, context, status = encode(record["token"])
            poses = _array(poses, (8, 3), np.float32)
            target = np.column_stack((poses[:, :2], np.cos(poses[:, 2]), np.sin(poses[:, 2])))
            values = {"context": context, "route": status, "target": target}
            for k, (shape, dtype) in FIELDS.items():
                arrays[k][i] = _array(values[k], shape, dtype)
        for name, array in arrays.items():
            np.save(stage / f"{name}.npy", array, allow_pickle=False)
        atomic_write(stage / "records.json", records)
        hashes = {p.name: file_hash(p) for p in stage.iterdir()}
        atomic_write(stage / "complete.json", {"count": len(records), "identity": identity, "hashes": hashes})
        stage.rename(directory)


def _publish_cache(directory, identity, batches, expected_tokens=None):
    """Produce both splits, including train-only statistics. Resume completed compatible shards."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is building {directory}") from exc
        build_file = directory / "build.json"
        if build_file.exists() and read_json(build_file) != identity:
            raise ValueError(f"Cache identity differs; choose a separate directory: {directory}")
        if not build_file.exists():
            if (directory / "manifest.json").exists():
                raise ValueError("Found cache manifest without its build identity")
            atomic_write(build_file, identity)
        identity_hash = fingerprint(identity)
        splits = {k: {"num_records": 0, "shards": []} for k in ("train", "val")}
        manifest = {"schema": SCHEMA, "identity": identity, "splits": splits,
                    "conditioner_identity": {key: identity[key] for key in CONDITIONER_KEYS}}
        seen_tokens, log_splits = set(), {}
        xy_sum, xy_square_sum, num_points = np.zeros(2), np.zeros(2), 0

        for split, log_name, tokens, encode in batches:
            entry = splits[split]
            if log_splits.setdefault(log_name, split) != split:
                raise ValueError(f"Log present in both splits: {log_name}")
            if seen_tokens.intersection(tokens) or len(set(tokens)) != len(tokens) or not tokens:
                raise ValueError(f"Duplicate or empty token chunk: {log_name}")
            seen_tokens.update(tokens)
            records = [{"token": t, "log_name": log_name, "split": split} for t in tokens]
            folder = directory / split / f"{len(entry['shards']):06d}"
            folder.parent.mkdir(exist_ok=True)
            _write_chunk(folder, identity_hash, records, encode)
            if split == "train":
                xy = np.load(folder / "target.npy", allow_pickle=False)[..., :2].astype(np.float64)
                xy_sum += xy.sum(axis=(0, 1))
                xy_square_sum += (xy * xy).sum(axis=(0, 1))
                num_points += xy.shape[0] * xy.shape[1]
            entry["num_records"] += len(records)
            entry["shards"].append({"path": str(folder.relative_to(directory)), "count": len(tokens),
                                    "marker_sha256": file_hash(folder / "complete.json")})
        if not all(part["num_records"] for part in splits.values()):
            raise ValueError("Both train and validation must contain scenes")
        if expected_tokens is not None and seen_tokens != expected_tokens:
            missing = sorted(expected_tokens - seen_tokens)
            raise ValueError(f"Official navtrain coverage mismatch; {len(missing)} missing tokens e.g. {missing[:5]}")
        mean = xy_sum / num_points
        std = np.sqrt(np.maximum(0., xy_square_sum / num_points - mean**2))
        # A stationary-only smoke sample can have zero spread on an axis.
        std[std < 1e-6] = 1.
        manifest["normalizer"] = {"mean": [*mean.tolist(), 0., 0.], "std": [*std.tolist(), 1., 1.]}
        atomic_write(directory / "manifest.json", manifest)
        return manifest


def load_cache_metadata(cache_dir):
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No completed NAVSIM score cache at {path}")
    data = read_json(path)
    if (data.get("schema") != SCHEMA or read_json(path.parent / "build.json") != data.get("identity")
            or data.get("conditioner_identity") != {k: data["identity"].get(k) for k in CONDITIONER_KEYS}):
        raise ValueError("Invalid NAVSIM score cache schema or provenance")
    mean = _array(data["normalizer"]["mean"], (4,), np.float64)
    std = _array(data["normalizer"]["std"], (4,), np.float64)
    if np.any(std <= 0) or np.any(mean[2:] != 0) or np.any(std[2:] != 1):
        raise ValueError("Invalid NAVSIM score normalization")
    return data


class NavsimScoreDataset(Dataset):
    """Map shards on demand; DataLoader workers only copy selected rows."""
    def __init__(self, cache_dir, split):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        self.root = Path(cache_dir).resolve()
        self.metadata = load_cache_metadata(self.root)
        self.sha256 = file_hash(self.root / "manifest.json")
        normalizer = self.metadata["normalizer"]
        self.mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
        self.std = torch.tensor(normalizer["std"], dtype=torch.float32)
        self.records, self._locations, self._directories = [], [], []
        self._maps = OrderedDict()
        entry = self.metadata["splits"][split]
        identity_hash = fingerprint(self.metadata["identity"])
        for item in entry["shards"]:
            directory = (self.root / item["path"]).resolve()
            if self.root not in directory.parents or file_hash(directory / "complete.json") != item["marker_sha256"]:
                raise ValueError(f"Invalid shard directory or completion marker: {directory}")
            mark = validate_shard(directory, identity_hash, checksums=False)
            if file_hash(directory / "records.json") != mark["hashes"]["records.json"]:
                raise ValueError(f"Invalid records in {directory}")
            records = read_json(directory / "records.json")
            if not len(records) == item["count"] == mark["count"] or any(r["split"] != split for r in records):
                raise ValueError(f"Invalid split/count for {directory}")
            self._locations.extend((len(self._directories), i) for i in range(len(records)))
            self.records.extend(records)
            self._directories.append(directory)
        if not self.records or len(self.records) != entry["num_records"]:
            raise ValueError(f"Incomplete {split} split")

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_maps": OrderedDict()}

    def __getitem__(self, index):
        number, offset = self._locations[index]
        if number not in self._maps:
            directory = self._directories[number]
            arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in FIELDS}
            count = self.metadata["splits"][self.records[index]["split"]]["shards"][number]["count"]
            for name, (shape, dtype) in FIELDS.items():
                if arrays[name].shape != (count, *shape) or arrays[name].dtype != np.dtype(dtype):
                    raise ValueError(f"Invalid array metadata: {directory}/{name}.npy")
            self._maps[number] = arrays
            if len(self._maps) > 16:
                self._maps.popitem(last=False)
        self._maps.move_to_end(number)
        row = {key: torch.from_numpy(np.array(array[offset], dtype=np.float32))
               for key, array in self._maps[number].items()}
        row["target"] = (row["target"] - self.mean) / self.std
        row["record"] = self.records[index]
        return row


def _native_batches(data_root, filters, agent, conditioner, device, shard_size, max_per_split):
    from dataclasses import replace
    from navsim.common.dataloader import SceneLoader
    from score_function.navsim_dd.conditioner import features_from_input

    log_dir = data_root / "navsim_logs/trainval"
    sensor_dir = data_root / "sensor_blobs/trainval"
    # Native DD reads only cam_l0/f0/r0 and lidar at history index 3. Avoid decoding unused image blobs.
    sensors = replace(agent.get_sensor_config(), cam_l1=False, cam_l2=False,
                      cam_r1=False, cam_r2=False, cam_b0=False)
    for split, scene_filter in filters.items():
        count = 0
        for log_name in scene_filter.log_names:
            if not (log_dir / (log_name + ".pkl")).is_file():
                if max_per_split is not None:
                    continue
                raise FileNotFoundError(f"Missing required navtrain log: {log_name}")
            loader = SceneLoader(log_dir, sensor_dir, replace(scene_filter, log_names=[log_name]), sensors)
            tokens = sorted(loader.tokens)
            if max_per_split is not None:
                tokens = tokens[:max_per_split - count]

            def encode(token):
                scene = loader.get_scene_from_token(token)
                if scene.scene_metadata.log_name != log_name or scene.scene_metadata.initial_token != token:
                    raise ValueError(f"Scene/log token mismatch: {log_name} / {token}")
                poses = scene.get_future_trajectory(8).poses
                features = features_from_input(agent, scene.get_agent_input(), device)
                c, g = conditioner.encode_features(features)
                return poses, c[0], g[0]

            for start in range(0, len(tokens), shard_size):
                yield split, log_name, tokens[start:start + shard_size], encode
            count += len(tokens)
            if max_per_split and count >= max_per_split:
                break
        if max_per_split is not None and count < max_per_split:
            raise ValueError(f"Requested {max_per_split} {split} scenes, only found {count}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="default: OPENSCENE_DATA_ROOT")
    parser.add_argument("--devkit-root", type=Path, help="default: NAVSIM_DEVKIT_ROOT / imported DiffusionDrive checkout")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-per-split", type=int, help="small smoke cache; exact count per train and val split")
    args = parser.parse_args(argv)
    if args.shard_size < 1 or (args.max_per_split is not None and args.max_per_split < 1):
        parser.error("--shard-size and --max-per-split must be positive")
    data_root = args.data_root or os.environ.get("OPENSCENE_DATA_ROOT")
    if not data_root:
        parser.error("set OPENSCENE_DATA_ROOT or pass --data-root")
    data_root = Path(data_root).resolve()
    log_dir = data_root / "navsim_logs/trainval"
    if not log_dir.is_dir() or not (data_root / "sensor_blobs/trainval").is_dir():
        parser.error(f"missing trainval data beneath {data_root}")
    root = verified_devkit_root(args.devkit_root)
    filters = {s: build_navtrain_filter(s, root) for s in ("train", "val")}
    log_inventory = [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in sorted(log_dir.glob("*.pkl"))]
    identity = {
        **conditioner_identity(args.checkpoint, args.anchor, args.backbone),
        "schema": SCHEMA, "data_root": str(data_root), "log_inventory_sha256": fingerprint(log_inventory),
        "filter_sha256": file_hash(root / FILTER_CONFIG), "split_sha256": file_hash(root / LOG_CONFIG),
        "shard_size": args.shard_size, "max_per_split": args.max_per_split,
    }
    from score_function.navsim_dd.conditioner import DDConditioner, load_diffusiondrive

    if (args.cache_dir / "manifest.json").is_file():
        existing = load_cache_metadata(args.cache_dir)
        if existing["identity"] != identity:
            raise ValueError("Existing cache describes different inputs; choose a fresh --cache-dir")
        for split in filters:
            dataset = NavsimScoreDataset(args.cache_dir, split)
            for directory in dataset._directories:
                validate_shard(directory, fingerprint(identity), checksums=True)
        print(f"Verified completed cache {args.cache_dir}; all shards reusable.")
        return
    agent = load_diffusiondrive(args.checkpoint, args.anchor, args.backbone, args.device)
    batches = _native_batches(data_root, filters, agent, DDConditioner(agent), args.device, args.shard_size, args.max_per_split)
    expected_tokens = set(filters["train"].tokens) if args.max_per_split is None else None
    manifest = _publish_cache(args.cache_dir, identity, batches, expected_tokens)
    print(f"Published {args.cache_dir}/manifest.json: "
          + ", ".join(f"{s}={v['num_records']}" for s, v in manifest["splits"].items()))
    print(f"Normalizer (train expert only): {manifest['normalizer']}")


if __name__ == "__main__":
    main()
