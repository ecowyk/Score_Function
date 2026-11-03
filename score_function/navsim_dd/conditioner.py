"""Frozen conditioning from the NAVSIM v1 DiffusionDrive front end.

Compatible with hustvl/DiffusionDrive commit 9b52ed0. Install that checkout
as the `navsim` package in the environment used to run real scenes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from score_function.utils.train_utils import load_tensor


def load_diffusiondrive(checkpoint, anchor, backbone, device):
    """Construct the native DD agent and load the trusted official checkpoint strictly."""
    paths = [Path(p).expanduser().resolve() for p in (checkpoint, anchor, backbone)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

    checkpoint, anchor, backbone = paths
    config = TransfuserConfig(plan_anchor_path=str(anchor), bkb_path=str(backbone))
    # The native constructor loads the checkpoint non-strictly when given a path.
    # Initialize the architecture, then do a single strict load on CPU so saved
    # CUDA device IDs never override the worker's requested evaluation device.
    agent = TransfuserAgent(config=config, lr=6e-4, checkpoint_path=None)
    saved = load_tensor(checkpoint)["state_dict"]
    state = {key.removeprefix("agent."): value for key, value in saved.items()}
    if len(state) != len(saved):
        raise ValueError("Duplicate parameter names after removing checkpoint prefix")
    anchor_key = "_transfuser_model._trajectory_head.plan_anchor"
    if anchor_key in state and not torch.equal(
        agent._transfuser_model._trajectory_head.plan_anchor.detach(), state[anchor_key]
    ):
        raise ValueError("Anchor file differs from the plan anchors stored in the checkpoint")
    agent.load_state_dict(state, strict=True)
    agent._checkpoint_path = str(checkpoint)
    return agent.to(device).eval().requires_grad_(False)


def features_from_input(agent, agent_input, device):
    """Use the native input builders, adding the single-scene batch dimension."""
    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    return {key: value.unsqueeze(0).to(device) for key, value in features.items()}


class DDConditioner:
    """Read the frozen low-resolution BEV tokens and unpositioned status embedding."""

    def __init__(self, agent):
        # A paired evaluator may share this very agent with the native planner.
        self.model = agent.eval().requires_grad_(False)._transfuser_model

    @torch.no_grad()
    def encode_features(self, features):
        if self.model.training:
            raise RuntimeError("DDConditioner requires the frozen agent in eval mode")
        _, bev, _ = self.model._backbone(features["camera_feature"], features["lidar_feature"])
        bev = self.model._bev_downscale(bev).flatten(-2, -1).permute(0, 2, 1)
        status = self.model._status_encoding(features["status_feature"])
        if bev.shape[1:] != (64, 256) or status.shape != (bev.shape[0], 256):
            raise ValueError(f"Unexpected DD conditioning shapes: {bev.shape}, {status.shape}")
        memory = torch.cat([bev, status[:, None]], dim=1)
        memory += self.model._keyval_embedding.weight[None, ...]
        scene = memory[:, :-1]
        if not torch.isfinite(scene).all() or not torch.isfinite(status).all():
            raise ValueError("Non-finite DiffusionDrive conditioning")
        return scene, status


@torch.no_grad()
def verify_native_conditions(agent, features, *, rtol=1e-5, atol=1e-6):
    """Check C/g against an unmodified native forward, preserving CPU/CUDA RNG.

    The pre-hook observes the memory actually read by the native decoder. A
    separate hook observes g *before* the positional embedding is added.
    Raises AssertionError if the outputs disagree, including on NaN/Inf.
    """
    conditioner = DDConditioner(agent)
    scene, status = conditioner.encode_features(features)
    captured = {}

    def capture_memory(_module, args):
        captured["scene"] = args[1][:, :-1].detach().clone()

    def capture_status(_module, _args, output):
        captured["status"] = output.detach().clone()

    model = conditioner.model
    device = features["camera_feature"].device
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        memory_hook = model._tf_decoder.register_forward_pre_hook(capture_memory)
        status_hook = model._status_encoding.register_forward_hook(capture_status)
        try:
            agent(features)
        finally:
            status_hook.remove()
            memory_hook.remove()
    if captured.keys() != {"scene", "status"}:
        raise RuntimeError("Expected native DiffusionDrive decoder and status encoder to run")
    result = {}
    for name, feature in (("scene", scene), ("status", status)):
        reference = captured[name]
        torch.testing.assert_close(feature, reference, rtol=rtol, atol=atol)
        result[name + "_shape"] = list(feature.shape)
        result[name + "_max_abs_error"] = (feature - reference).abs().max().item()
    return result
