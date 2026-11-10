from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from score_function.navsim_chainflow import native


def test_checkpoint_prefix_normalization_and_strict_loading():
    layer = torch.nn.Linear(2, 1)
    state = {"agent." + k: v.clone() for k, v in layer.state_dict().items()}
    normalized = native.normalize_checkpoint_state({"state_dict": state})
    layer.load_state_dict(normalized, strict=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        layer.load_state_dict({"weight": normalized["weight"]}, strict=True)
    with pytest.raises(ValueError, match="duplicate"):
        native.normalize_checkpoint_state({"state_dict": {"agent.x": torch.zeros(1), "x": torch.zeros(1)}})
    with pytest.raises(ValueError, match="state_dict"):
        native.normalize_checkpoint_state({"weights": state})


def test_full_variable_length_vlm_is_retained():
    value = torch.arange(7 * 1536).reshape(1, 7, 1536).float()
    assert torch.equal(native.normalize_vlm(value), value[0])
    assert native.normalize_vlm(torch.zeros(1536)).shape == (1, 1536)
    for invalid in (torch.zeros(0, 1536), torch.zeros(3, 768), torch.zeros(2, 7, 1536),
                    torch.full((2, 1536), float("nan"))):
        with pytest.raises(ValueError):
            native.normalize_vlm(invalid)


def fake_features():
    return {"ego_status": torch.zeros(1, 4, 11), "image": torch.ones(1, 4, 3, 2, 2),
            "vlm_hidden_state": torch.arange(5 * 1536).reshape(1, 5, 1536).float(),
            "scenario_token": ["scene-token"]}


class NativeModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.encoder_calls = 0

    def scene_encoder(self, features):
        self.encoder_calls += 1
        return {"ego_token": torch.ones(1, 1, 256), "scene_features": torch.ones(1, 64, 256)}

    def forward(self, features):
        self.scene_encoder(features)
        assert features["scenario_token"] == ["scene-token"]
        if self.fail:
            raise RuntimeError("native failure")
        proposals = torch.arange(3 * 8 * 3).reshape(1, 3, 8, 3).float()
        scores = torch.tensor([[0.1, 0.9, 0.4]])
        return {"trajectory": proposals[:, 1], "proposals": proposals, "pdm_score": scores}


class NativeAgent(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self._chainflow_vla_model = NativeModel(fail)

    def forward(self, features):
        return self._chainflow_vla_model(features)


def test_native_prediction_uses_official_forward_once_and_preserves_selection(monkeypatch):
    monkeypatch.setattr(native, "build_features", lambda *args: fake_features())
    agent = NativeAgent().eval().requires_grad_(False)
    original = agent._chainflow_vla_model.scene_encoder
    trajectory, condition = native.native_predict(agent, None, "cpu")
    assert agent._chainflow_vla_model.encoder_calls == 1
    assert agent._chainflow_vla_model.scene_encoder == original
    assert torch.equal(trajectory, torch.arange(24, 48).reshape(1, 8, 3).float())
    assert condition["vlm"].shape == (1, 5, 1536)
    assert condition["vlm_mask"].dtype == torch.bool and condition["vlm_mask"].all()
    assert condition["scene"].shape == (1, 64, 256)
    assert condition["ego"].shape == (1, 256)


def test_native_observer_restored_after_failure(monkeypatch):
    monkeypatch.setattr(native, "build_features", lambda *args: fake_features())
    agent = NativeAgent(fail=True).eval().requires_grad_(False)
    original = agent._chainflow_vla_model.scene_encoder
    with pytest.raises(RuntimeError, match="native failure"):
        native.native_predict(agent, None, "cpu")
    assert agent._chainflow_vla_model.scene_encoder == original


def test_build_features_injects_native_scenario_seed_key_and_full_sequence(monkeypatch):
    module = ModuleType("navsim.planning.training.dataset")
    value = torch.randn(9, 1536)
    module._load_vlm_cache_features = lambda spec, log, token: {"vlm_hidden_state": value}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    builder = SimpleNamespace(compute_features=lambda inp: {"image": torch.ones(4, 3, 2, 2)})
    agent = SimpleNamespace(get_feature_builders=lambda: [builder], _score_vlm_spec=None)
    scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token="abc", log_name="log"),
                            get_agent_input=lambda: None)
    features = native.build_features(agent, scene, "cpu")
    assert features["scenario_token"] == ["abc"]
    assert torch.equal(features["vlm_hidden_state"], value.unsqueeze(0))
    builder.compute_features = lambda inp: {"image": torch.ones(3, 3, 2, 2)}
    with pytest.raises(ValueError, match="four current cameras"):
        native.build_features(agent, scene, "cpu")


def test_allocator_cap_uses_visible_device_before_allocation(monkeypatch):
    calls = []
    monkeypatch.setenv("CF_CUDA_MEMORY_GB", "8")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=32 * 1024**3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: calls.append((fraction, str(device))))
    native.apply_memory_limit("cuda:1")
    assert calls == [(0.25, "cuda:1")]
    monkeypatch.setenv("CF_CUDA_MEMORY_GB", "40")
    with pytest.raises(ValueError, match="no larger"):
        native.apply_memory_limit("cuda:1")


def test_release_identity_is_pinned_to_85k_and_rejects_other_weights(tmp_path):
    from score_function.utils.train_utils import atomic_write, file_hash
    path = tmp_path / "release_manifest.json"
    release = {"schema": "chainflow_release_manifest_v1", "upstream_commit": native.CHAINFLOW_COMMIT,
               "training_split": "navtrain_85k", "assets": [
                   {"role": "checkpoint", "sha256": native.OFFICIAL_85K_SHA256},
                   {"role": "backbone", "sha256": "b" * 64}]}
    atomic_write(path, release)
    assert native.validate_release_manifest(path, native.OFFICIAL_85K_SHA256, "b" * 64) == file_hash(path)
    with pytest.raises(ValueError, match="checkpoint differs"):
        native.validate_release_manifest(path, "c" * 64, "b" * 64)
    release["training_split"] = "trainval_103k"
    atomic_write(path, release)
    with pytest.raises(ValueError, match="85k"):
        native.validate_release_manifest(path, native.OFFICIAL_85K_SHA256, "b" * 64)


def test_load_agent_requires_all_checkpoint_keys_and_freezes_complete_model(monkeypatch, tmp_path):
    class MinimalAgent(torch.nn.Linear):
        def __init__(self, **kwargs):
            super().__init__(2, 1)
    config = SimpleNamespace(image_backbone=SimpleNamespace(model_weights=None))
    omega = ModuleType("omegaconf")
    omega.OmegaConf = SimpleNamespace(load=lambda path: SimpleNamespace(config=config))
    agent_module = ModuleType("navsim.agents.chainflow_vla.chainflow_vla_agent")
    agent_module.ChainFlowVLAAgent = MinimalAgent
    data_module = ModuleType("navsim.planning.training.dataset")
    data_module.VlmCacheSpec = lambda *args: args
    for module in (omega, agent_module, data_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(native, "activate_devkit", lambda root: tmp_path)
    monkeypatch.setattr(native, "conditioner_identity", lambda *args: {"native": "pinned"})
    weights = {"agent.weight": torch.ones(1, 2)}
    monkeypatch.setattr(native, "load_tensor", lambda path: {"state_dict": weights})
    paths = [tmp_path] * 4
    with pytest.raises(RuntimeError, match="Missing key"):
        native.load_agent(*paths, "cpu")
    weights["agent.bias"] = torch.ones(1)
    agent = native.load_agent(*paths, "cpu")
    assert not agent.training and not any(p.requires_grad for p in agent.parameters())
    assert torch.equal(agent.weight, torch.ones(1, 2))
