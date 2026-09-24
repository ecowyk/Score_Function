"""Read official preprocessed NPZ features for the ego-only score objective."""

from contextlib import ExitStack

import numpy as np
import torch

from score_function.utils.normalizer import normalize_ego_future

INPUT_KEYS = (
    "ego_current_state",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
)


def read_feature_batch(records, args, device):
    """Reuse official NPZ IO and observation normalization; pack ego DSM targets.

    Unlike the official joint-generation dataset, score learning consumes only
    ego future targets and accepts the per-record paths in our split manifest.
    """
    from diffusion_planner.utils.train_utils import opendata

    with ExitStack() as stack:
        files = [stack.enter_context(opendata(row["feature"])) for row in records]
        raw = {
            key: torch.as_tensor(np.stack([item[key] for item in files]), device=device)
            for key in INPUT_KEYS
        }
        future = torch.as_tensor(
            np.stack([item["ego_agent_future"] for item in files]), device=device
        ).float()
    raw = {key: value.float() if value.is_floating_point() else value for key, value in raw.items()}
    inputs = args.observation_normalizer(raw)
    target = torch.cat((future[..., :2], future[..., 2:3].cos(), future[..., 2:3].sin()), -1)
    return inputs, normalize_ego_future(target, args.state_normalizer)
