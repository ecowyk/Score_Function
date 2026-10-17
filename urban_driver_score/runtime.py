"""Read-only checkpoint loading compatible with the UrbanDriver torch 1.12 env."""

import inspect
from pathlib import Path

import torch

from score_function.model.score_branch import build_model
from score_function.utils.config import METHOD, load_config, model_config
from score_function.utils.planner_utils import (
    ego_normalizer_metadata, load_planner_config, planner_identity,
)
from score_function.utils.train_utils import resolve_path


def load_cpu(path):
    # These are the user's trusted local research checkpoints. No monkeypatching
    # of torch.load or the frozen score source is needed for old PyTorch.
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(path, **options)


def load_score(config_path, checkpoint_path, device):
    cfg = load_config(config_path)
    state = load_cpu(checkpoint_path)
    if (state.get("schema_version") != 1 or state.get("method") != METHOD
            or state.get("weight_kind") != "ema"):
        raise ValueError("Require the frozen S10-G final EMA checkpoint")
    if model_config(state["config"]["model"]) != model_config(cfg["model"]):
        raise ValueError("Score checkpoint/config architecture mismatch")
    if state["sigma_score"] != .1 or cfg["training"]["sigma"] != .1:
        raise ValueError("This run is fixed to S10-G sigma=0.1")
    if (cfg["model"].get("parameterization", "score") != "score"
            or cfg["model"].get("neighbor_future", False)):
        raise ValueError("Require direct-score G without predicted-neighbor conditioning")
    if not cfg["model"].get("temporal_attention", False):
        raise ValueError("Require the G (global temporal attention) architecture")
    if state["planner"] != planner_identity(cfg):
        raise ValueError("Frozen DP conditioning provenance differs from score training")
    branch = build_model(cfg, device)
    branch.load_state_dict(state["score_branch"], strict=True)
    branch.eval().requires_grad_(False)
    args = load_planner_config(cfg, device)
    from diffusion_planner.utils.config import Config
    imported_root = Path(inspect.getfile(Config)).resolve().parents[2]
    if imported_root != resolve_path(cfg, "planner_dir").resolve():
        raise ValueError("Imported DP source is different from the frozen conditioning source")
    if ego_normalizer_metadata(args) != state["ego_normalizer"]:
        raise ValueError("Ego normalizer differs from score training")
    return cfg, state, branch, args


def load_conditions(cfg, args, device):
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner_Encoder
    from diffusion_planner.model.module.decoder import RouteEncoder

    # Load only the two conditioning encoders. No DP trajectory generator is
    # constructed or sampled, and no fake neighbor predictions are supplied.
    encoder = Diffusion_Planner_Encoder(args)
    route = RouteEncoder(args.route_num, args.lane_len,
                         drop_path_rate=args.encoder_drop_path_rate,
                         hidden_dim=args.hidden_dim)
    saved = load_cpu(resolve_path(cfg, "planner_checkpoint"))["ema_state_dict"]
    saved = {k.removeprefix("module."): v for k, v in saved.items()}
    for model, prefix in ((encoder, "encoder."),
                          (route, "decoder.decoder.dit.route_encoder.")):
        model.load_state_dict({k[len(prefix):]: v for k, v in saved.items()
                               if k.startswith(prefix)}, strict=True)
        model.to(device).eval().requires_grad_(False)
    return encoder, route
