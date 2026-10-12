#!/usr/bin/env python3
"""Replay selected trusted nuPlan logs; compare final G at a common DP input.

No simulation is launched. Frozen suite files and original logs are read-only.
The default CPU run captures three temporal probes in each of six selected cases.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import lzma
import os
import pickle
import sys
from pathlib import Path

CASES = [
    ("I01", "cf87d58064425ee7", "following_lane_with_lead"),
    ("W01", "5d392fa38ff65c3f", "changing_lane"),
    ("K01", "6d0875dbcec45b3a", "high_lateral_acceleration"),
    ("K02", "c2c649fcd5325b3d", "starting_right_turn"),
    ("K04", "c6277e74a4f054a7", "following_lane_with_lead"),
    ("W04", "54da00bc66c7575d", "waiting_for_pedestrian_to_cross"),
    ("I02", "99e63d494deb5808", "traversing_pickup_dropoff"),
    ("I03", "99ef05a2e6a45454", "starting_right_turn"),
    ("I04", "3f9e713baeec55a9", "high_magnitude_speed"),
    ("W02", "914f752db44d5143", "starting_left_turn"),
    ("W03", "59199b4d340558b4", "changing_lane"),
    ("K03", "63352660f02f5e3d", "high_lateral_acceleration"),
    ("C01", "521a9f9977d351da", "starting_straight_traffic_light_intersection_traversal"),
    ("C02", "4836b2dd09895237", "behind_long_vehicle"),
    ("C03", "be2029b5dc2c5b78", "low_magnitude_speed"),
    ("C04", "80ababf8dcfb5914", "stopping_with_lead"),
]
KS = (0, 5, 20, 50, 100, 200, 500)
DEFAULT_SUITE = Path("/home/data/wyk/dataset/nuplan/exp/score_full_k_1009_v1")
DEFAULT_OUTPUT = Path("/home/data/wyk/dataset/nuplan/exp/score_G_static_context_1012_v1")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_cases(cohort):
    if cohort == "priority":
        return CASES[:6]
    if cohort == "all":
        return CASES
    names = cohort.split(",")
    by_name = {case[0]: case for case in CASES}
    if any(name not in by_name for name in names):
        raise ValueError("--cohort must be priority, all, or comma-separated case IDs")
    return [by_name[name] for name in dict.fromkeys(names)]


def probe_frames(count, explicit=None):
    frames = sorted(set(explicit if explicit is not None else (0, count // 3, 2 * count // 3)))
    if not frames or any(i < 0 or i >= count for i in frames):
        raise ValueError(f"Frame indices outside available history (0..{count - 1})")
    return frames


def to_local(points, anchor):
    import numpy as np

    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    delta = xy - [anchor.x, anchor.y]
    c, s = np.cos(anchor.heading), np.sin(anchor.heading)
    return delta @ np.array([[c, -s], [s, c]])


def configure_runtime(suite, plan, device, gpu, threads):
    root = Path(plan["root"])
    for key in list(os.environ):
        if key in {
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "RAY_ADDRESS",
        } or key.startswith("TORCHELASTIC_"):
            os.environ.pop(key)
    os.environ.update(
        DP_DIR=str(root / "Diffusion-Planner"),
        NUPLAN_DEVKIT_ROOT=str(root / "nuplan-devkit"),
        NUPLAN_DATA_ROOT=str(root / "dataset/nuplan"),
        NUPLAN_MAPS_ROOT=str(root / "dataset/nuplan/maps"),
        NUPLAN_EXP_ROOT=str(root / "dataset/nuplan/exp"),
        CUDA_VISIBLE_DEVICES=str(gpu) if device == "cuda" else "",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        MPLBACKEND="Agg",
    )
    paths = [suite / "code/lg", root / "Diffusion-Planner", root / "nuplan-devkit"]
    sys.path[:0] = [str(p) for p in paths]
    import torch

    import score_function

    torch.set_num_threads(threads)
    origin = Path(score_function.__file__).resolve()
    if (suite / "code/lg").resolve() not in origin.parents:
        raise ValueError(f"Wrong score implementation imported: {origin}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable")


def cpu_torch_bytes(blob):
    import inspect

    import torch

    kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False
    return torch.load(io.BytesIO(blob), **kw)


class CPUUnpickler(pickle.Unpickler):
    # Only use on the user's own trusted simulation logs: pickle executes code.
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return cpu_torch_bytes
        return super().find_class(module, name)


def load_log(path):
    with lzma.open(path, "rb") as f:
        if path.name.endswith(".msgpack.xz"):
            import msgpack

            blob = msgpack.unpackb(f.read())
            log = CPUUnpickler(io.BytesIO(blob)).load()
        else:
            log = CPUUnpickler(f).load()
    log.planner = None
    gc.collect()
    return log


def find_log(run, token):
    matches = sorted((run / "simulation").rglob(f"{token}.msgpack.xz"))
    matches += sorted((run / "simulation").rglob(f"{token}.pkl.xz"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one saved simulation for {token}; found {len(matches)} in {run}"
        )
    return matches[0]


def make_history(scenario, samples, index, duration):
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

    size = int((duration + scenario.database_interval) / scenario.database_interval) + 1
    history = SimulationHistoryBuffer.initialize_from_scenario(size, scenario, DetectionsTracks)
    for sample in samples[: index + 1]:
        history.append(sample.ego_state, sample.observation)
    return history


def extract_environment(scenario, sample, trajectories):
    import numpy as np
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    anchor = sample.ego_state.rear_axle
    radius = max(
        60.0,
        max(np.linalg.norm(np.asarray(t)[:, :2], axis=1).max() for t in trajectories.values())
        + 20.0,
    )
    api = scenario.map_api
    supported = set(api.get_available_map_objects())
    names = (
        "LANE",
        "LANE_CONNECTOR",
        "INTERSECTION",
        "STOP_LINE",
        "CROSSWALK",
        "WALKWAYS",
        "CARPARK_AREA",
    )
    layers = [getattr(SemanticMapLayer, name) for name in names if hasattr(SemanticMapLayer, name)]
    layers = [layer for layer in layers if layer in supported]
    objects = api.get_proximal_map_objects(Point2D(anchor.x, anchor.y), radius, layers)
    polygons, lines, route = {}, {}, []
    route_ids = set(map(str, scenario.get_route_roadblock_ids()))
    lights = {
        str(light.lane_connector_id): light.status.name.lower()
        for light in sample.traffic_light_status
    }
    for layer, items in objects.items():
        for item in items:
            geometry = item.polygon
            pieces = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
            for polygon in pieces:
                polygons.setdefault(layer.name, []).append(
                    to_local(polygon.exterior.coords, anchor).tolist()
                )
                for hole in polygon.interiors:
                    lines.setdefault("polygon_holes", []).append(
                        to_local(hole.coords, anchor).tolist()
                    )
            if layer.name in ("LANE", "LANE_CONNECTOR"):
                center = to_local(
                    [(p.x, p.y) for p in item.baseline_path.discrete_path], anchor
                ).tolist()
                lines.setdefault("lane_center", []).append(center)
                if str(item.get_roadblock_id()) in route_ids:
                    route.append(center)
                if layer.name == "LANE_CONNECTOR" and str(item.id) in lights:
                    lines.setdefault(lights[str(item.id)], []).append(center)
    agents = []
    for obj in sample.observation.tracked_objects.tracked_objects:
        agents.append(
            dict(
                polygon=to_local(obj.box.geometry.exterior.coords, anchor).tolist(),
                kind=obj.tracked_object_type.name,
                track_token=getattr(obj, "track_token", None),
            )
        )
    return dict(
        map_polygons=polygons,
        map_lines=lines,
        route_lines=route,
        agents=agents,
        ego_polygon=to_local(
            sample.ego_state.car_footprint.geometry.exterior.coords, anchor
        ).tolist(),
        map_query_radius_m=float(radius),
        world_anchor=dict(x=anchor.x, y=anchor.y, heading=anchor.heading),
    )


def replay_error(prediction, sample):
    import numpy as np

    timestamp = sample.ego_state.time_point.time_us
    states = sample.trajectory.get_sampled_trajectory()
    if len(states) != len(prediction) + 1 or states[0].time_point.time_us != timestamp:
        raise ValueError("Saved baseline trajectory has an unexpected horizon or start time")
    states = states[1:]
    offsets = np.array([state.time_point.time_us - timestamp for state in states])
    # Official conversion goes through float seconds and can truncate by 1 us.
    # Comparing stored samples avoids interpolating beyond the saved endpoint.
    if np.max(np.abs(offsets - np.arange(1, len(states) + 1) * 100000)) > 2:
        raise ValueError("Saved baseline future timesteps differ from official 0.1s sampling")
    expected = to_local(
        [(state.rear_axle.x, state.rear_axle.y) for state in states], sample.ego_state.rear_axle
    )
    error = np.linalg.norm(prediction[:, :2] - expected, axis=1)
    return dict(max_xy_m=float(error.max()), rms_xy_m=float(np.sqrt(np.mean(error**2))))


def capture_frame(planner, current_input, sample, case, index, scenario, provenance, tolerance):
    import numpy as np
    import torch

    from score_function.model.refinement import refine_ego
    from score_function.utils.normalizer import denormalize_ego_future, normalize_ego_future

    captured = {}

    def hook(module, args, result):
        captured.update(inputs=args[0], encoding=result[0], output=result[1])

    wrapped = planner._planner
    handle = wrapped.base_model.register_forward_hook(hook)
    try:
        with torch.no_grad():
            planner.compute_planner_trajectory(current_input)
    finally:
        handle.remove()
    with torch.no_grad():
        initial = captured["output"]["prediction"][:, 0].detach().clone()
        physical = initial[0].cpu().numpy()
        error = replay_error(physical, sample)
        if error["max_xy_m"] > tolerance:
            raise ValueError(
                f"DP replay mismatch {error}; tolerance={tolerance}m. No comparison exported. "
                "Check source/history/seed; CPU versus original CUDA can differ. "
                "Try --device cuda --gpu <an idle GPU> in a NEW output directory."
            )
        if getattr(wrapped.score_branch, "uses_neighbor_future", False):
            raise ValueError(
                "This G capture expects scene/route conditioning without neighbor-future input"
            )
        scene = captured["encoding"]["encoding"]
        route = wrapped.base_model.decoder.decoder.dit.route_encoder(
            captured["inputs"]["route_lanes"]
        )
        normalized = normalize_ego_future(initial, wrapped.normalizer)
        arrays = dict(
            dp_physical=physical,
            dp_normalized=normalized.cpu().numpy(),
            scene=scene.cpu().numpy(),
            route=route.cpu().numpy(),
        )
        for key, tensor in captured["inputs"].items():
            if torch.is_tensor(tensor):
                arrays["input_" + key] = tensor.cpu().numpy()
        trajectories = {"0": physical.tolist()}
        current, previous = normalized, 0
        for k in KS[1:]:
            current, _ = refine_ego(
                wrapped.score_branch,
                current,
                scene,
                route,
                wrapped.normalizer,
                sigma=wrapped.sigma,
                gamma=wrapped.gamma,
                steps=k - previous,
                heading_projection=wrapped.heading_projection,
                record_trace=False,
            )
            trajectories[str(k)] = (
                denormalize_ego_future(current, wrapped.normalizer)[0].cpu().tolist()
            )
            previous = k
            print(f"  {case[0]} frame={index} K={k} captured", flush=True)
    metadata = dict(
        **provenance,
        anchor="DP_K0",
        model="S05-G final EMA",
        frame_selection="Fixed temporal probe, not a detected failure event",
        comparison="Common DP closed-loop history/input; counterfactual K endpoints, not separate closed-loop rollouts",
        context_label="Fixed temporal probe; context from DP_K0 closed-loop run",
        replay_error=error,
        gamma=wrapped.gamma,
        sigma=wrapped.sigma,
        heading_projection=wrapped.heading_projection,
        future_dt_s=0.1,
    )
    payload = dict(
        case_id=case[0],
        token=case[1],
        scenario_type=case[2],
        frame_index=index,
        timestamp_us=int(sample.ego_state.time_point.time_us),
        trajectories=trajectories,
        future_time_s=(np.arange(1, len(physical) + 1) * 0.1).tolist(),
        metadata=metadata,
        **extract_environment(scenario, sample, trajectories),
    )
    return payload, arrays


def capture(args):
    suite, output = args.suite.resolve(), args.output.resolve()
    plan = read(suite / "plan.json")
    print("Verifying frozen suite inputs (read-only)...", flush=True)
    for file, sha in plan["files"].items():
        if digest(file) != sha:
            raise ValueError(f"Frozen input changed: {file}")
    configure_runtime(suite, plan, args.device, args.gpu, args.threads)
    import numpy as np
    from nuplan.planning.simulation.planner.abstract_planner import (
        PlannerInitialization,
        PlannerInput,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from omegaconf import OmegaConf

    from score_function.planner.planner import ScoreFunctionPlanner

    job = read(suite / "jobs/DP_K0.json")
    if job.get("state") not in ("complete", "reused"):
        raise ValueError("DP_K0 reference run must be complete")
    run = Path(job["result_dir"])
    cfg = OmegaConf.load(run / "code/hydra/config.yaml")
    if int(cfg.planner.score_function_planner.steps) != 0:
        raise ValueError("Reference run is not DP K=0")
    config_path, checkpoint = (
        suite / "models/S05-G/config.json",
        suite / "models/S05-G/final_ema.pt",
    )
    model_config = read(config_path)
    if model_config["reference_seed"] != plan["reference_seed"]:
        raise ValueError("Reference seed mismatch")
    provenance = dict(
        suite=str(suite),
        plan_sha256=digest(suite / "plan.json"),
        config_sha256=digest(config_path),
        checkpoint_sha256=digest(checkpoint),
        capture_script_sha256=digest(__file__),
        device=args.device,
        reference_seed=plan["reference_seed"],
        K=list(KS),
        replay_tolerance_m=args.replay_tolerance_m,
    )
    manifest_path = output / "capture_manifest.json"
    if manifest_path.exists() and read(manifest_path) != provenance:
        raise ValueError("Output contains a different capture protocol; choose a new --output")
    write(manifest_path, provenance)
    planner = ScoreFunctionPlanner(
        score_config=str(config_path),
        score_checkpoint=str(checkpoint),
        root=str(plan["root"]),
        device=args.device,
        steps=0,
        gamma=plan["gamma"],
        heading_projection=plan["heading_projection"],
        trace_dir=None,
        past_trajectory_sampling=TrajectorySampling(num_poses=20, time_horizon=2.0),
        future_trajectory_sampling=TrajectorySampling(num_poses=80, time_horizon=8.0),
    )
    for case in selected_cases(args.cohort):
        log_path = find_log(run, case[1])
        log_sha = digest(log_path)
        print(f"Loading {case[0]} {case[1]}: {log_path}", flush=True)
        log = load_log(log_path)
        scenario, samples = log.scenario, log.simulation_history.data
        if str(scenario.token) != case[1] or scenario.scenario_type != case[2]:
            raise ValueError("Scenario identity/type differs from selected case")
        frame_indices = probe_frames(len(samples), args.frames)
        planner.initialize(
            PlannerInitialization(
                route_roadblock_ids=scenario.get_route_roadblock_ids(),
                mission_goal=scenario.get_mission_goal(),
                map_api=scenario.map_api,
            )
        )
        for index in frame_indices:
            folder = output / "captures" / case[0] / f"frame_{index:04d}"
            if (folder / "payload.json").exists() and (folder / "inputs.npz").exists():
                existing = read(folder / "payload.json")
                if (
                    all(existing["metadata"].get(k) == v for k, v in provenance.items())
                    and existing["metadata"].get("source_log_sha256") == log_sha
                ):
                    print(f"Reusing {case[0]} frame={index}", flush=True)
                    continue
                raise ValueError(f"Stale capture: {folder}")
            sample = samples[index]
            history = make_history(
                scenario, samples, index, float(cfg.simulation_history_buffer_duration)
            )
            current = PlannerInput(
                iteration=sample.iteration,
                history=history,
                traffic_light_data=sample.traffic_light_status,
            )
            payload, arrays = capture_frame(
                planner, current, sample, case, index, scenario, provenance, args.replay_tolerance_m
            )
            payload["metadata"].update(
                source_log=str(log_path),
                source_log_sha256=log_sha,
                source_frame_count=len(samples),
            )
            if args.frames is not None:
                payload["metadata"].update(
                    frame_selection="User-selected history index; not a detected failure event",
                    context_label="Selected temporal probe; context from DP_K0 closed-loop run",
                )
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "inputs.npz", **arrays)
            write(folder / "payload.json", payload)
            print(
                f"SAVED {case[0]} frame={index}, replay={payload['metadata']['replay_error']}",
                flush=True,
            )
        del log, samples, scenario
        gc.collect()
    print(f"Capture complete: {output}", flush=True)


def render(args):
    os.environ["MPLBACKEND"] = "Agg"
    from matplotlib.backends.backend_pdf import PdfPages
    from static_context_plots import render_case

    paths = []
    for case in selected_cases(args.cohort):
        files = sorted((args.output / "captures" / case[0]).glob("frame_*/payload.json"))
        paths.extend(
            p for p in files if args.frames is None or read(p)["frame_index"] in args.frames
        )
    if not paths:
        raise ValueError("No captured frames found; run capture or run first")
    report = args.output / "static_trajectories.pdf"
    tmp = report.with_name("static_trajectories.tmp.pdf")
    with PdfPages(tmp) as pdf:
        for path in paths:
            payload = read(path)
            folder = (
                args.output / "figures" / payload["case_id"] / f"frame_{payload['frame_index']:04d}"
            )
            render_case(payload, folder, shared_pdf=pdf)
            print(f"RENDERED {folder}", flush=True)
    tmp.replace(report)
    print(f"PDF: {report}\nPNGs: {args.output / 'figures'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "capture", "render"))
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cohort", default="priority", help="priority (6), all (16), or I01,W01,...")
    p.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="Explicit zero-based history indices; default start/one-third/two-thirds",
    )
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--gpu", type=int, help="A single idle physical GPU; required with --device cuda"
    )
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--replay-tolerance-m", type=float, default=0.02)
    args = p.parse_args()
    if args.device == "cuda" and args.gpu is None:
        p.error("--device cuda requires --gpu N; choose an idle GPU")
    if args.threads < 1 or args.replay_tolerance_m <= 0:
        p.error("threads and replay tolerance must be positive")
    if args.command in ("run", "capture"):
        capture(args)
    if args.command in ("run", "render"):
        render(args)


if __name__ == "__main__":
    main()
