"""Full-state budget continuation using a real tiny CPU score training run."""

import copy
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
import test_training_pipeline as pipeline
import torch

from score_function.train import train
from score_function.utils.train_utils import atomic_write, file_hash, load_tensor, read_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fork_g_continuation.py"
SPEC = importlib.util.spec_from_file_location("fork_g_continuation", SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def completed(tmp_path):
    config, _ = pipeline.fixture(tmp_path)
    config["paths"]["cache"] = config["cache"]
    config["paths"]["run_dir"] = config["output"]
    config["training"].update(lr_schedule="constant_after_warmup", early_stopping=False)
    config["model"]["temporal_attention"] = True
    train(config, device_override="cpu")
    source = Path(config["output"]) / "score"
    new = copy.deepcopy(config)
    new["output"] = str(tmp_path / "continued")
    new["paths"]["run_dir"] = new["output"]
    new["training"]["max_epochs"] = 3
    path = tmp_path / "continued.json"
    atomic_write(path, new)
    return source, path, Path(new["output"]) / "score", new


def assert_identical(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_identical(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for left, right in zip(first, second):
            assert_identical(left, right)
    else:
        assert first == second


def test_complete_fork_preserves_state_and_matches_uninterrupted_training(completed, tmp_path):
    source, path, output, new = completed
    before = {name: file_hash(source / name) for name in helper.SOURCE_FILES}
    metadata = helper.fork_checkpoint(source, path, output, 1, expected_world_size=1)
    original, fork = load_tensor(source / "last.pt"), load_tensor(output / "last.pt")
    changed = {"config", "signature", "training_budget", "finished", "termination"}
    assert original.keys() == fork.keys()
    for key in original.keys() - changed:
        assert_identical(original[key], fork[key])
    assert fork["finished"] is False and fork["termination"] is None
    assert fork["epoch"] == 2 and fork["cursor"] == 0 and fork["step"] == 4
    assert metadata["training_budget"]["planned_epochs"] == 3
    assert metadata["training_budget"]["planned_optimizer_updates"] == 6
    assert read_json(output / "continuation.json") == metadata
    for name in ("best.pt", "selection.json", "initial_validation.json", "history.json"):
        assert file_hash(output / name) == before[name]
    assert read_json(output / "provenance.json")["signature"] == fork["signature"]
    assert read_json(output / "model_info.json")["training_budget"] == fork["training_budget"]
    train(new, resume=True, device_override="cpu")
    full_config = copy.deepcopy(new)
    full_config["output"] = str(tmp_path / "full")
    full_config["paths"]["run_dir"] = full_config["output"]
    train(full_config, device_override="cpu")
    full = load_tensor(Path(full_config["output"]) / "score/last.pt")
    continued = load_tensor(output / "last.pt")
    pipeline.PipelineTests()._assert_state_equal(full, continued)
    assert before == {name: file_hash(source / name) for name in helper.SOURCE_FILES}


@pytest.mark.parametrize("field,value", [
    ("sigma", 0.1), ("learning_rate", 0.001), ("seed", 11), ("microbatch_size", 2),
    ("max_epochs", 4),
])
def test_rejects_any_unapproved_training_change(completed, field, value):
    source, path, output, new = completed
    new["training"][field] = value
    atomic_write(path, new)
    with pytest.raises(ValueError, match="Only output"):
        helper.fork_checkpoint(source, path, output, 1, expected_world_size=1)
    assert not output.exists()


def test_rejects_incomplete_checkpoint_and_changed_original_sources(completed, monkeypatch):
    source, path, output, _ = completed
    original = load_tensor(source / "last.pt")
    broken = dict(original, finished=False)
    atomic_write(source / "last.pt", broken, tensor=True)
    with pytest.raises(ValueError, match="not a complete"):
        helper.fork_checkpoint(source, path, output, 1, expected_world_size=1)
    atomic_write(source / "last.pt", original, tensor=True)
    monkeypatch.setattr(helper, "source_hashes", lambda: {"changed.py": "bad"})
    with pytest.raises(ValueError, match="source hashes"):
        helper.fork_checkpoint(source, path, output, 1, expected_world_size=1)
    assert not output.exists()


def test_production_default_requires_original_four_rank_state(completed):
    source, path, output, _ = completed
    with pytest.raises(ValueError, match="world size"):
        helper.fork_checkpoint(source, path, output, 1)
    assert not output.exists()


def test_existing_output_is_never_overwritten(completed):
    source, path, output, _ = completed
    output.mkdir(parents=True)
    sentinel = output / "keep"
    sentinel.write_text("existing output")
    with pytest.raises(ValueError, match="already exists"):
        helper.fork_checkpoint(source, path, output, 1, expected_world_size=1)
    assert sentinel.read_text() == "existing output"


def test_inconsistent_metadata_rejected_before_creating_output(completed):
    source, path, output, _ = completed
    provenance = read_json(source / "provenance.json")
    provenance["signature"] = "changed"
    atomic_write(source / "provenance.json", provenance)
    with pytest.raises(ValueError, match="training signature"):
        helper.fork_checkpoint(source, path, output, 1, expected_world_size=1)
    assert not output.exists()


def test_reference_snapshot_can_keep_original_output_in_its_config(completed, tmp_path):
    source, path, output, _ = completed
    reference = tmp_path / "reference" / "score"
    shutil.copytree(source, reference)
    assert Path(read_json(reference / "config.json")["output"]) != reference.parent
    metadata = helper.fork_checkpoint(reference, path, output, 1, expected_world_size=1)
    assert metadata["source_score"] == str(reference)
    assert read_json(output / "status.json")["state"] == "prepared"
