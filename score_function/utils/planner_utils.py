"""Small checkpoint/configuration helpers around the official Diffusion_Planner.

The network, encoder, route encoder and normalizers are imported from the
official project. This module does not implement a separate planner class.
"""

import torch

from score_function.utils.train_utils import (
    file_hash,
    load_tensor,
    planner_source_hashes,
    resolve_path,
)


def planner_identity(config):
    return {
        "checkpoint_sha256": file_hash(resolve_path(config, "planner_checkpoint")),
        "args_sha256": file_hash(resolve_path(config, "planner_args")),
        "official_sources": planner_source_hashes(config),
    }


def load_planner_config(config, device):
    from diffusion_planner.utils.config import Config

    args = Config(str(resolve_path(config, "planner_args")), None)
    args.device = str(device)
    if (args.future_len, args.predicted_neighbor_num, args.hidden_dim) != (80, 10, 192):
        raise ValueError("Require official ego80 / 10-neighbor / width192 checkpoint")
    return args


def load_frozen_planner(config, device):
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner

    args = load_planner_config(config, device)
    model = Diffusion_Planner(args)
    state = load_tensor(resolve_path(config, "planner_checkpoint"))["ema_state_dict"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model.to(device).eval().requires_grad_(False), args


def ego_normalizer_metadata(args):
    normalizer = args.state_normalizer
    return {
        "mean": normalizer.mean[0, 0].tolist(),
        "std": normalizer.std[0, 0].tolist(),
    }


@torch.no_grad()
def encode_conditions(model, inputs):
    """Call the official scene and route encoders without running its sampler."""
    context = model.encoder(inputs)["encoding"]
    route = model.decoder.decoder.dit.route_encoder(inputs["route_lanes"])
    if context.shape[1:] != (107, 192) or route.shape[1:] != (192,):
        raise ValueError(f"Unexpected official conditioning shapes: {context.shape}, {route.shape}")
    return context, route
