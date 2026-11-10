import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from score_function.navsim_chainflow import cache
from score_function.utils.train_utils import atomic_write, file_hash, read_json


@pytest.fixture
def fixture_cache(tmp_path):
    source, output, vlm = (tmp_path / name for name in ("source", "cache", "vlm"))
    for root in (source, output, vlm):
        root.mkdir()
    metadata = {"normalizer": {"mean": [10., 1., 0., 0.], "std": [2., 3., 1., 1.]},
                "splits": {"train": {"num_records": 2}, "val": {"num_records": 1}}}
    atomic_write(source / "manifest.json", metadata)
    identity = cache.make_identity(source, metadata, {"checkpoint_sha256": "a" * 64})
    atomic_write(output / "build.json", identity)
    tasks, conditions = [], {}
    for split, count in (("train", 2), ("val", 1)):
        relative = f"{split}/log-{split}/part-0000"
        directory = source / relative
        directory.mkdir(parents=True)
        records = [{"token": f"{split}-{i}", "log_name": f"log-{split}", "split": split} for i in range(count)]
        target = np.zeros((count, 8, 4), dtype=np.float32)
        target[..., 0] = 14
        target[..., 1] = 7
        target[..., 2] = 1
        np.save(directory / "target.npy", target)
        atomic_write(directory / "complete.json", {"count": count})
        task = {"path": relative, "split": split, "log_name": f"log-{split}", "count": count,
                "records": records, "source_marker_sha256": file_hash(directory / "complete.json"),
                "target_sha256": file_hash(directory / "target.npy")}
        tasks.append(task)
        for i, record in enumerate(records):
            token = record["token"]
            (vlm / token).write_bytes(token.encode())
            conditions[token] = {"scene": torch.ones(64, 256) * (i + 0.25), "ego": torch.ones(256),
                                 "vlm": torch.arange((2 + i * 3) * 1536).reshape(2 + i * 3, 1536).float() / 1000}
    def encode(record, target):
        return conditions[record["token"]]
    def origin(record):
        return vlm / record["token"]
    return source, output, metadata, identity, tasks, conditions, encode, origin


def build_all(fixture):
    source, output, metadata, identity, tasks, conditions, encode, origin = fixture
    for task in tasks:
        assert cache.write_chunk(output, source, identity, task, encode, origin)
    cache.publish_cache(output, source, identity, tasks, metadata)
    return cache.ChainFlowDataset(output, "train")


def test_full_ragged_sequences_and_exact_targets_survive_atomic_cache(fixture_cache):
    source, output, metadata, identity, tasks, conditions, encode, origin = fixture_cache
    dataset = build_all(fixture_cache)
    assert len(dataset) == 2
    for i, length in enumerate((2, 5)):
        sample = dataset[i]
        assert sample["vlm"].shape == (length, 1536)
        assert torch.equal(sample["vlm"], conditions[f"train-{i}"]["vlm"].half().float())
        assert sample["scene"].shape == (64, 256)
        assert sample["ego"].shape == (256,)
        assert torch.equal(sample["target"], torch.tensor([2., 2., 1., 0.]).repeat(8, 1))
    task = tasks[0]
    assert file_hash(output / task["path"] / "target.npy") == task["target_sha256"]
    provenance = read_json(output / task["path"] / "vlm_sources.json")
    assert provenance[0]["sha256"] == file_hash(origin(task["records"][0]))
    ready = read_json(output / "TRAINING_READY.json")
    assert ready["counts"] == {"train": 2, "val": 1}
    assert not cache.write_chunk(output, source, identity, task, lambda *args: pytest.fail("must reuse"), origin)


def test_failed_write_publishes_no_partial_shard(fixture_cache):
    source, output, _, identity, tasks, _, encode, origin = fixture_cache
    calls = []
    def bad_encode(record, target):
        calls.append(record["token"])
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return encode(record, target)
    with pytest.raises(RuntimeError, match="interrupted"):
        cache.write_chunk(output, source, identity, tasks[0], bad_encode, origin)
    assert not (output / tasks[0]["path"]).exists()
    assert cache.write_chunk(output, source, identity, tasks[0], encode, origin)


def test_source_target_mutation_and_condition_overflow_are_rejected(fixture_cache):
    source, output, _, identity, tasks, conditions, encode, origin = fixture_cache
    task = tasks[0]
    conditions["train-0"]["scene"][0, 0] = 1e8
    with pytest.raises(ValueError, match="FP16"):
        cache.write_chunk(output, source, identity, task, encode, origin)
    np.save(source / task["path"] / "target.npy", np.ones((2, 8, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Source target"):
        cache.write_chunk(output, source, identity, task, encode, origin)


def test_corrupted_completed_cache_is_not_silently_reused(fixture_cache):
    source, output, _, identity, tasks, _, encode, origin = fixture_cache
    build_all(fixture_cache)
    path = output / tasks[0]["path"] / "offsets.npy"
    offsets = np.load(path)
    offsets[1] = 0
    np.save(path, offsets)
    with pytest.raises(ValueError, match="corrupt"):
        cache.write_chunk(output, source, identity, tasks[0], encode, origin)
    with pytest.raises(ValueError, match="offsets"):
        cache.ChainFlowDataset(output, "train")


def test_manifest_or_ready_tampering_rejected(fixture_cache):
    _, output, _, _, _, _, _, _ = fixture_cache
    build_all(fixture_cache)
    ready = read_json(output / "TRAINING_READY.json")
    ready["counts"]["train"] = 999
    atomic_write(output / "TRAINING_READY.json", ready)
    with pytest.raises(ValueError, match="TRAINING_READY"):
        cache.ChainFlowDataset(output, "train")


def test_duplicate_tokens_cannot_publish(fixture_cache):
    source, output, metadata, identity, tasks, conditions, encode, origin = fixture_cache
    tasks[1]["records"][0]["token"] = "train-0"
    for task in tasks:
        cache.write_chunk(output, source, identity, task, encode, origin)
    with pytest.raises(ValueError, match="Duplicate token"):
        cache.publish_cache(output, source, identity, tasks, metadata)


def test_path_escape_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        cache.inside(tmp_path, "../other")


def test_cpu_coverage_reports_missing_camera_before_gpu_work(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root, vlm, output = (tmp_path / p for p in ("data", "vlm", "cache"))
    output.mkdir()
    loaders = {}
    for split, part in (("train", "trainval"), ("navtest", "test")):
        token, log = f"{split}-token", f"{split}-log"
        cameras = {}
        for camera in cache.CAMERAS:
            relative = f"{log}/{camera}.jpg"
            path = root / "sensor_blobs" / part / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not (split == "navtest" and camera == "cam_r0"):
                path.write_bytes(b"image")
            cameras[camera.upper()] = {"data_path": relative}
        feature = vlm / log / token / "internvl_feature.gz"
        feature.parent.mkdir(parents=True)
        feature.write_bytes(b"feature")
        frame = {"log_name": log, "cams": cameras}
        loaders[split] = SimpleNamespace(tokens=[token], scene_frames_dicts={token: [None] * 3 + [frame]},
                                         _scene_filter=SimpleNamespace(num_history_frames=4, tokens=[token]))
    monkeypatch.setattr(cache, "activate_devkit", lambda path: path)
    monkeypatch.setattr(cache, "make_scene_loader", lambda data, devkit, split, *args, **kw: loaders[split])
    args = SimpleNamespace(data_root=root, vlm_cache=vlm, cache_dir=output, devkit_root=tmp_path)
    tasks = [{"split": "train", "log_name": "train-log", "records": [{"token": "train-token"}]}]
    with pytest.raises(ValueError, match="coverage is incomplete"):
        cache.check_data(args, tasks)
    report = read_json(output / "coverage.json")
    assert report["state"] == "failed"
    assert report["missing_count"] == 1
    assert report["missing_examples"][0]["kind"] == "cam_r0"
    assert report["history_images_required"] is False
    assert report["compressed_vlm_bytes"] == {"train": 7, "val": 0, "navtest": 7}


def test_disk_space_failure_does_not_publish_completed_cache(fixture_cache, monkeypatch):
    from types import SimpleNamespace
    source, output, _, identity, tasks, _, encode, origin = fixture_cache
    monkeypatch.setattr(cache.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="Insufficient reported free space"):
        cache.write_chunk(output, source, identity, tasks[0], encode, origin)
    assert not (output / tasks[0]["path"]).exists()
