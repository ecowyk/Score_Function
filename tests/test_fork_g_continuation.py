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
from score_function.utils.train_utils import (
    atomic_write,
    file_hash,
    load_tensor,
    read_json,
    stable_seed,
    training_signature,
)

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


def test_twofold_expansion_preserves_optimizer_and_old_rng_and_seeds_new_rank(completed):
    source, path, output, new = completed
    original = load_tensor(source / "last.pt")
    global_rng = torch.get_rng_state().clone()
    metadata = helper.fork_checkpoint(
        source, path, output, 1, expected_world_size=1, target_world_size=2
    )
    expanded = load_tensor(output / "last.pt")
    for key in original.keys() - {
        "config", "signature", "training_budget", "finished", "termination", "ranks",
    }:
        assert_identical(original[key], expanded[key])
    assert len(expanded["ranks"]) == 2
    assert_identical(original["ranks"][0], expanded["ranks"][0])
    added = expanded["ranks"][1]
    assert added["epoch_loss"] == 0 and added["epoch_count"] == 0
    assert added["rng"]["cuda"] is None
    assert not torch.equal(added["rng"]["torch"], original["ranks"][0]["rng"]["torch"])
    expected_seed = stable_seed(new["training"]["seed"], "world_size_expansion", 2, 1)
    assert torch.equal(
        added["rng"]["torch"], torch.Generator().manual_seed(expected_seed).get_state()
    )
    assert_identical(
        added["rng"]["numpy"], np.random.RandomState(expected_seed % (2 ** 32)).get_state()
    )
    assert torch.equal(global_rng, torch.get_rng_state())
    migration = metadata["world_size_migration"]
    assert migration["source_world_size"] == 1 and migration["target_world_size"] == 2
    assert migration["preserved_ranks"] == [0]
    assert migration["new_rank_seeds"]["1"]["seed"] == expected_seed
    assert migration["exact_original_world_size_continuation"] is False
    info = read_json(output / "model_info.json")
    assert info["world_size"] == 2 and info["gradient_accumulation"] == 2
    assert info["global_batch"] == 4 and info["microbatch_per_rank"] == 1
    provenance = read_json(output / "provenance.json")
    assert provenance["world_size_migration"] == migration
    assert expanded["signature"] == training_signature(new, provenance["cache_sha256"], 2)
    assert original["ranks"] and len(load_tensor(source / "last.pt")["ranks"]) == 1


def test_cuda_rank_expansion_requires_cuda_for_new_generator_states(completed, monkeypatch):
    source, path, output, _ = completed
    state = load_tensor(source / "last.pt")
    state["ranks"][0]["rng"]["cuda"] = torch.zeros(16, dtype=torch.uint8)
    atomic_write(source / "last.pt", state, tensor=True)
    monkeypatch.setattr(helper.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires an available CUDA"):
        helper.fork_checkpoint(
            source, path, output, 1, expected_world_size=1, target_world_size=2
        )
    assert not output.exists()


def test_world_size_change_can_only_double_at_epoch_boundary(completed):
    source, path, output, _ = completed
    with pytest.raises(ValueError, match="exactly twofold"):
        helper.fork_checkpoint(
            source, path, output, 1, expected_world_size=1, target_world_size=3
        )
    assert not output.exists()
