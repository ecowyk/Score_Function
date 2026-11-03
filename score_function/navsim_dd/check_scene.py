"""Check native conditioning and trajectory conventions on one real NAVSIM scene."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from score_function.navsim_dd.cache import build_navtrain_filter, verified_devkit_root
from score_function.navsim_dd.conditioner import features_from_input, load_diffusiondrive, verify_native_conditions
from score_function.navsim_dd.evaluate import native_top1, sampling_rng, token_seed
from score_function.navsim_dd.runtime import poses_to_channels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=os.getenv("OPENSCENE_DATA_ROOT"))
    parser.add_argument("--devkit-root", type=Path, default=os.getenv("NAVSIM_DEVKIT_ROOT"))
    parser.add_argument("--split", choices=("navtrain", "mini"), default="navtrain")
    parser.add_argument("--log", help="optionally select a log from the chosen split")
    parser.add_argument("--token", help="optionally select an official navtrain scene")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set OPENSCENE_DATA_ROOT or --data-root")
    from navsim.common.dataloader import SceneLoader
    split_dir = "trainval" if args.split == "navtrain" else "mini"
    logs = args.data_root / "navsim_logs" / split_dir
    sensors = args.data_root / "sensor_blobs" / split_dir
    if args.split == "navtrain":
        scene_filter = build_navtrain_filter("train", args.devkit_root)
    else:
        from navsim.common.dataclasses import SceneFilter
        verified_devkit_root(args.devkit_root)
        # With an explicit token, inspect all keyframes rather than only non-overlapping clips.
        scene_filter = SceneFilter(log_names=sorted(p.stem for p in logs.glob("*.pkl")),
                                  frame_interval=1 if args.token else None)
    if args.token and scene_filter.tokens and args.token not in scene_filter.tokens:
        parser.error("--token is outside the official navtrain filter")
    if args.log and args.log not in scene_filter.log_names:
        parser.error("--log is outside the logs for this split")
    agent = load_diffusiondrive(args.checkpoint, args.anchor, args.backbone, args.device)
    for log in [args.log] if args.log else scene_filter.log_names:
        if not (logs / (log + ".pkl")).is_file():
            continue
        subset = replace(scene_filter, log_names=[log], max_scenes=1,
                         tokens=[args.token] if args.token else scene_filter.tokens)
        loader = SceneLoader(logs, sensors, subset, agent.get_sensor_config())
        if loader.tokens:
            break
    else:
        raise ValueError("No matching real scene under --data-root")
    token = loader.tokens[0]
    features = features_from_input(agent, loader.get_agent_input_from_token(token), args.device)
    checks = verify_native_conditions(agent, features)
    seed = token_seed(args.seed, token)
    with sampling_rng(seed, args.device):
        raw, mode = native_top1(agent, features)
    with torch.no_grad(), sampling_rng(seed, args.device):
        reference = agent(features)["trajectory"]
    torch.testing.assert_close(raw, reference, atol=0, rtol=0)
    expert = torch.as_tensor(loader.get_scene_from_token(token).get_future_trajectory(8).poses)
    encoded = poses_to_channels(expert)
    print(json.dumps({
        "token": token, "log": log, **checks, "top1_index": mode,
        "native_poses_match": True, "raw_shape": list(raw.shape),
        "expert_channels_shape": list(encoded.shape), "native_first_pose": raw[0, 0].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
