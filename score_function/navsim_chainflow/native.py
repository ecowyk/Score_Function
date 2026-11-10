"""Pinned, frozen ChainFlow inference. Native sampling and candidate ranking are preserved.

Heavy NAVSIM imports are deliberately lazy: metadata/status tools need only Python,
NumPy and Torch, and must never accidentally import the old DiffusionDrive devkit.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import os
import subprocess
import sys

import torch

from score_function.utils.train_utils import file_hash, fingerprint, load_tensor, read_json

CHAINFLOW_COMMIT = "0f3565f66546139a0183bc5807d5f7234ae2d4b0"
AGENT_CONFIG = "navsim/planning/script/config/common/agent/chainflow_vla_stage2.yaml"
FILTER_ROOT = "navsim/planning/script/config/common/train_test_split"
CONDITION = "chainflow-dino64-ego256-internvl1536-v1"
CAMERAS = ("cam_f0", "cam_b0", "cam_l0", "cam_r0")
OFFICIAL_85K_SHA256 = "420bb9254d852234fbe59b7c14c1406157cdf5fde7a1b19bd78175f287748db5"


def activate_devkit(devkit_root):
    root = Path(devkit_root).expanduser().resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != CHAINFLOW_COMMIT:
        raise ValueError(f"Require ChainFlow {CHAINFLOW_COMMIT}, got {commit}")
    for command in (["diff", "--quiet", "HEAD", "--", "navsim"],
                    ["diff", "--cached", "--quiet", "HEAD", "--", "navsim"]):
        if subprocess.run(["git", "-C", str(root), *command], check=False).returncode:
            raise ValueError("Tracked ChainFlow NAVSIM source differs from the pinned release")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import navsim
    actual = Path(navsim.__file__).resolve()
    if root not in actual.parents:
        raise RuntimeError(f"Wrong NAVSIM imported: {actual}; launch with ChainFlow first on PYTHONPATH")
    return root


def validate_release_manifest(path, checkpoint_sha256, backbone_sha256):
    release = read_json(path)
    if (release.get("schema") != "chainflow_release_manifest_v1"
            or release.get("upstream_commit") != CHAINFLOW_COMMIT
            or release.get("training_split") != "navtrain_85k"):
        raise ValueError("Need the pinned 85k ChainFlow release_manifest.json")
    for role, actual in (("checkpoint", checkpoint_sha256), ("backbone", backbone_sha256)):
        matching = [a for a in release.get("assets", []) if a.get("role") == role]
        if len(matching) != 1 or matching[0].get("sha256") != actual:
            raise ValueError(f"Native {role} differs from the pinned release")
    if checkpoint_sha256 != OFFICIAL_85K_SHA256:
        raise ValueError("Require the official 85k Stage2 checkpoint, preserving independent validation")
    return file_hash(path)


def conditioner_identity(devkit_root, checkpoint, backbone, vlm_cache):
    root = activate_devkit(devkit_root)
    checkpoint, backbone, vlm_cache = (Path(p).expanduser().resolve() for p in (checkpoint, backbone, vlm_cache))
    for path in (checkpoint, backbone):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not vlm_cache.is_dir():
        raise FileNotFoundError(vlm_cache)
    source_files = [AGENT_CONFIG, "navsim/agents/chainflow_vla/chainflow_vla_model.py",
                    "navsim/agents/chainflow_vla/chainflow_vla_agent.py",
                    "navsim/agents/chainflow_vla/encoders/image/dinov2_lora.py",
                    "navsim/agents/chainflow_vla/builders/feature_builder.py",
                    "navsim/planning/training/dataset.py"]
    checkpoint_sha256, backbone_sha256 = file_hash(checkpoint), file_hash(backbone)
    release_sha256 = validate_release_manifest(vlm_cache / "release_manifest.json", checkpoint_sha256, backbone_sha256)
    return {
        "condition": CONDITION, "chainflow_commit": CHAINFLOW_COMMIT,
        "checkpoint_sha256": checkpoint_sha256, "backbone_sha256": backbone_sha256,
        "native_source_sha256": {p: file_hash(root / p) for p in source_files},
        "vlm_release_sha256": release_sha256, "bridge_sha256": file_hash(Path(__file__)),
        "vlm_feature_name": "internvl_feature", "vlm_hidden_key": "last_hidden_state",
        "native_noise_seed": 27, "native_denoising_steps": 12,
        "scene_shape": [64, 256], "ego_shape": [256], "vlm_dim": 1536,
    }


def apply_memory_limit(device):
    """Optional allocator cap, not a guarantee on total driver/runtime memory."""
    device = torch.device(device)
    value = os.environ.get("CF_CUDA_MEMORY_GB")
    if value and device.type == "cuda":
        amount = float(value)
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = amount * 1024**3 / total
        if not 0 < fraction <= 1:
            raise ValueError("CF_CUDA_MEMORY_GB must be positive and no larger than GPU memory")
        torch.cuda.set_per_process_memory_fraction(fraction, device)


def normalize_checkpoint_state(saved):
    if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
        raise ValueError("Official ChainFlow checkpoint must contain state_dict")
    raw = saved["state_dict"]
    state = {key.removeprefix("agent."): value for key, value in raw.items()}
    if len(state) != len(raw) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Invalid or duplicate native checkpoint parameter names")
    return state


def load_agent(devkit_root, checkpoint, backbone, vlm_cache, device):
    root = activate_devkit(devkit_root)
    identity = conditioner_identity(root, checkpoint, backbone, vlm_cache)
    from omegaconf import OmegaConf
    from navsim.agents.chainflow_vla.chainflow_vla_agent import ChainFlowVLAAgent
    from navsim.planning.training.dataset import VlmCacheSpec
    config = OmegaConf.load(root / AGENT_CONFIG).config
    config.image_backbone.model_weights = str(Path(backbone).resolve())
    config.vlm_feature_cache_path = str(Path(vlm_cache).resolve())
    # Construct directly: Hydra runtime interpolations and training losses are not
    # needed during frozen extraction, and no training metric caches/Ray are loaded.
    agent = ChainFlowVLAAgent(config=config, lr_args={"name": "AdamW", "base_lr": 2e-4},
                             checkpoint_path=None, checkpoint_strict_load=True, enable_ray=False)
    agent.load_state_dict(normalize_checkpoint_state(load_tensor(checkpoint)), strict=True)
    agent._checkpoint_path = str(Path(checkpoint).resolve())
    agent._score_bridge_identity = identity
    agent._score_devkit_root = root
    agent._score_vlm_spec = VlmCacheSpec(Path(vlm_cache).resolve(), "internvl_feature", "last_hidden_state")
    apply_memory_limit(device)
    return agent.to(device).eval().requires_grad_(False)


def normalize_vlm(value):
    value = torch.as_tensor(value).detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2 or value.shape[1] != 1536 or value.shape[0] < 1:
        raise ValueError(f"Expected nonempty full VLM hidden states [L,1536], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite VLM hidden states")
    return value


def build_features(agent, scene, device):
    from navsim.planning.training.dataset import _load_vlm_cache_features
    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(scene.get_agent_input()))
    if features.get("image") is None or features["image"].shape[0] != 4:
        raise ValueError("ChainFlow requires all four current cameras f0,b0,l0,r0")
    token, log_name = scene.scene_metadata.initial_token, scene.scene_metadata.log_name
    vlm = _load_vlm_cache_features(agent._score_vlm_spec, log_name, token)["vlm_hidden_state"]
    features["vlm_hidden_state"] = normalize_vlm(vlm)
    features = {key: value.unsqueeze(0).to(device) for key, value in features.items()}
    # Omitting this silently changes native diffusion to an unseeded fallback.
    features["scenario_token"] = [token]
    return features


def condition_from_encoded(encoded, features):
    condition = {"scene": encoded["scene_features"], "ego": encoded["ego_token"][:, 0],
                 "vlm": features["vlm_hidden_state"]}
    batch = len(condition["scene"])
    if (condition["scene"].shape != (batch, 64, 256) or condition["ego"].shape != (batch, 256)
            or condition["vlm"].ndim != 3 or condition["vlm"].shape[0] != batch
            or condition["vlm"].shape[2] != 1536 or condition["vlm"].shape[1] < 1):
        raise ValueError("Native conditioning shape differs from released four-camera ChainFlow")
    if any(not torch.isfinite(x).all() for x in condition.values()):
        raise ValueError("Nonfinite native ChainFlow condition")
    condition["vlm_mask"] = torch.ones(condition["vlm"].shape[:2], dtype=torch.bool,
                                        device=condition["vlm"].device)
    return condition


@torch.no_grad()
def encode_scene(agent, scene, device):
    if agent.training or any(p.requires_grad for p in agent.parameters()):
        raise ValueError("Condition extraction requires the frozen native agent in eval mode")
    features = build_features(agent, scene, device)
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        encoded = agent._chainflow_vla_model.scene_encoder(features)
    return condition_from_encoded(encoded, features)


@torch.no_grad()
def native_predict(agent, scene, device):
    """Exact official forward and argmax; observe conditions without re-encoding images."""
    if agent.training or any(p.requires_grad for p in agent.parameters()):
        raise ValueError("Prediction requires the frozen native agent in eval mode")
    features = build_features(agent, scene, device)
    model = agent._chainflow_vla_model
    original = model.scene_encoder
    captured = []

    def observe(inputs):
        encoded = original(inputs)
        captured.append(encoded)
        return encoded

    model.scene_encoder = observe
    try:
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            output = agent(features)
    finally:
        model.scene_encoder = original
    if len(captured) != 1:
        raise ValueError("Native forward must encode the scene exactly once")
    raw = output["trajectory"]
    if raw.shape != (1, 8, 3) or not torch.isfinite(raw).all():
        raise ValueError(f"Invalid native trajectory: {tuple(raw.shape)}")
    index = output["pdm_score"].argmax(dim=1)
    selected = output["proposals"][torch.arange(len(index), device=index.device), index]
    if not torch.equal(raw, selected):
        raise ValueError("Native trajectory differs from its own PDM scorer selection")
    return raw, condition_from_encoded(captured[0], features)


predict_scene = native_predict


@lru_cache(maxsize=4)
def _filter_options(root, name):
    import yaml
    with (Path(root) / FILTER_ROOT / "scene_filter" / f"{name}.yaml").open() as f:
        return yaml.safe_load(f)


def make_scene_loader(data_root, devkit_root, split, log_names=None, tokens=None, load_image_path=False):
    """Official split filters. Overrides only narrow membership; never broaden it."""
    root = Path(devkit_root).resolve()
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    if split not in ("train", "val", "navtest", "navtrain"):
        raise ValueError(f"Unknown split {split}")
    name = "navtest" if split == "navtest" else "navtrain"
    options = {k: v for k, v in _filter_options(str(root), name).items() if not k.startswith("_")}
    if split in ("train", "val"):
        import yaml
        p = root / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
        with p.open() as f:
            logs = yaml.safe_load(f)
        allowed = set(logs[f"{split}_logs"])
        options["log_names"] = sorted(set(options["log_names"]) & allowed)
    if log_names is not None:
        requested = set(log_names)
        if requested - set(options["log_names"]):
            raise ValueError("Requested logs lie outside the official split")
        options["log_names"] = sorted(requested)
    if tokens is not None:
        requested = set(tokens)
        official = options.get("tokens")
        if official is not None and requested - set(official):
            raise ValueError("Requested tokens lie outside the official split")
        options["tokens"] = sorted(requested)
    config = replace(SensorConfig.build_no_sensors(), **{camera: [3] for camera in CAMERAS}, lidar_pc=[])
    part = "test" if split == "navtest" else "trainval"
    data_root = Path(data_root)
    return SceneLoader(data_path=data_root / "navsim_logs" / part,
                       sensor_blobs_path=data_root / "sensor_blobs" / part,
                       scene_filter=SceneFilter(**options), sensor_config=config,
                       load_image_path=load_image_path)
