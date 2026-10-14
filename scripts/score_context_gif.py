#!/usr/bin/env python3
"""Render refinement snapshots and timestamped closed-loop agents as GIFs.

The existing capture/evaluation files remain unchanged. Refinement animation
uses only saved K endpoints (no invented intermediate score updates). Closed-loop
animation uses each run's own observations, footprints and timestamps.
"""

import argparse
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import expanded_score_context as selection
import static_score_context as core
from static_context_fonts import configure_chinese_font
from static_context_plots import (
    DP_COLOR, REFINED_COLOR, SCENARIO_ZH, _environment, _fit_limits, _limits,
    _score_band, _trajectories,
)

LIGHT_STATES = {"red", "yellow", "green", "unknown"}
REPORT_NAME = "gif_context_v1"
LABELS = (
    "场景动态回放 同一场景输入 周车／行人保持同一时刻 已记录步数 "
    "原始轨迹 当前轨迹 未来时间 相对DP最大位移 米 "
    "自车 周车 行人 自行车 静态物 推动方向 实际行驶 "
    "回放时间 记录时间 已结束 闭环得分 全程场景得分 "
    "前向 左向 秒 路线 当前信号灯 道路／车道边界"
)


def finite_xy(points):
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError("Expected a nonempty finite N x 2 array")
    return value


def validate_clock(frames):
    if not frames:
        raise ValueError("Empty simulation history")
    times = np.asarray([f["timestamp_us"] for f in frames], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Simulation timestamps must be strictly increasing")
    for frame in frames:
        finite_xy([frame["ego_xy"]])
        finite_xy(frame["ego_polygon"])
        if "agents" not in frame:
            raise ValueError("Missing observed agents; cannot use an ego-only overview cache")
        for agent in frame["agents"]:
            finite_xy(agent["polygon"])
    return times


def frame_at(times, timestamp_us):
    """Causal sample-and-hold, never extrapolate before/after a recorded run."""
    if timestamp_us < times[0] or timestamp_us > times[-1]:
        return None
    return int(np.searchsorted(times, timestamp_us, side="right") - 1)


def playback_times(tracks, fps=10, speed=1.0):
    if not np.isfinite([fps, speed]).all() or not 1 <= fps <= 50 or speed <= 0:
        raise ValueError("fps must be in [1, 50] and speed must be positive")
    start = min(int(t[0]) for t in tracks)
    stop = max(int(t[-1]) for t in tracks)
    # Integer microseconds avoid precision loss for epoch-sized timestamps.
    step = max(1, round(speed * 1_000_000 / fps))
    times = list(range(start, stop + 1, step))
    if times[-1] != stop:
        times.append(stop)
    return times


def playback_durations(timeline, fps, speed):
    """Quantize cumulative wall time to GIF centiseconds without FPS drift."""
    ticks = [round((t - timeline[0]) / (10_000 * speed)) for t in timeline]
    ticks.append(round((timeline[-1] - timeline[0]) / (10_000 * speed) + 100 / fps))
    return [max(10, (end - start) * 10) for start, end in zip(ticks, ticks[1:])]


class GifStream:
    """Write complete frames incrementally, without holding an entire run in RAM."""

    def __init__(self, path):
        self.path = Path(path)
        self.temp = self.path.with_name(self.path.name + f".tmp.{os.getpid()}")
        self.count = 0
        self.size = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temp.open("wb")
        return self

    def add(self, image, duration_ms):
        from PIL import GifImagePlugin

        if not np.isfinite(duration_ms) or not 10 <= duration_ms <= 655350:
            raise ValueError("GIF frame duration must be 10..655350 ms")
        frame = image.convert("RGB").quantize(colors=128)
        if self.size is not None and frame.size != self.size:
            raise ValueError("GIF frame size changed")
        self.size = frame.size
        if self.count == 0:
            header, _ = GifImagePlugin.getheader(frame, info={"loop": 0})
            for block in header:
                self.handle.write(block)
        for block in GifImagePlugin.getdata(
            frame, duration=int(round(duration_ms / 10) * 10),
            disposal=2, include_color_table=self.count > 0,
        ):
            self.handle.write(block)
        self.count += 1

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None and self.count:
                self.handle.write(b";")
        finally:
            self.handle.close()
        if exc_type is None and self.count:
            self.temp.replace(self.path)
        else:
            self.temp.unlink(missing_ok=True)
            if exc_type is None:
                raise ValueError("No frames written")


def figure_image(fig):
    from PIL import Image

    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def actor_style(kind):
    kind = str(kind).upper()
    if "PEDESTRIAN" in kind:
        return "#be6687", "#85405b"
    if "BICYCLE" in kind or "CYCLIST" in kind:
        return "#cb973c", "#895f14"
    if "VEHICLE" in kind:
        return "#859f97", "#3f6458"
    return "#a7aab1", "#5b6069"


def draw_actors(ax, frame, ego_color, ids=False):
    from matplotlib.patches import Polygon

    for agent in frame.get("agents", []):
        points = finite_xy(agent["polygon"])
        fill, edge = actor_style(agent.get("kind", ""))
        ax.add_patch(Polygon(points, closed=True, fc=fill, ec=edge, alpha=0.8, lw=0.8, zorder=5))
        vertices = points[:-1] if np.array_equal(points[0], points[-1]) else points
        center = np.asarray(agent.get("center", vertices.mean(axis=0)), dtype=float)
        if "heading" in agent:
            heading = float(agent["heading"])
            direction = np.array([np.cos(heading), np.sin(heading)])
            ax.plot(*np.stack([center, center + direction]).T, color=edge, lw=1, zorder=6)
        if ids and agent.get("track_token"):
            ax.annotate(str(agent["track_token"])[-6:], center, xytext=(3, 4),
                        textcoords="offset points", fontsize=7, clip_on=True, zorder=8)
    polygon = frame.get("ego_polygon")
    if polygon is not None and len(polygon):
        ax.add_patch(Polygon(finite_xy(polygon), closed=True, fc=ego_color, ec=ego_color,
                             alpha=0.65, lw=1, zorder=7))


def static_map(payload):
    return {
        **payload, "agents": [], "ego_polygon": [], "show_gt": False,
        "map_lines": {k: v for k, v in payload.get("map_lines", {}).items()
                      if k.lower() not in LIGHT_STATES},
    }


def legends(fig):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color=DP_COLOR, ls="--", label="DP"),
        Line2D([], [], color=REFINED_COLOR, label="refine"),
        Patch(fc=actor_style("VEHICLE")[0], label="周车"),
        Patch(fc=actor_style("PEDESTRIAN")[0], label="行人"),
        Patch(fc=actor_style("BICYCLE")[0], label="自行车"),
        Patch(fc=actor_style("STATIC")[0], label="静态物"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=6, frameon=False, fontsize=11)


def configure_plot(font=None):
    import matplotlib

    matplotlib.use("Agg")
    family = configure_chinese_font(font, required_text=LABELS)
    return {"font.family": family, "font.size": 11, "axes.unicode_minus": False}


def render_refinement(payload, output, ks=None, hold_s=0.8, dpi=100, font=None, agent_ids=False):
    settings = configure_plot(font)
    import matplotlib.pyplot as plt

    trajectories = _trajectories(payload)
    selected = sorted(set([0] + (list(ks) if ks is not None else list(trajectories))))
    if any(k not in trajectories for k in selected) or len(selected) < 2:
        raise ValueError("Every K must exist in the saved capture, with at least one K > 0")
    if "agents" not in payload:
        raise ValueError("Capture is missing observed agents; recapture this frame")
    if not np.isfinite(hold_s) or not 0.01 <= hold_s <= 600:
        raise ValueError("hold_s must be 0.01..600 seconds")
    dp = trajectories[0]
    bounds = _fit_limits(_limits([np.zeros((1, 2)), *[trajectories[k] for k in selected]],
                                 padding=10, min_span=30), 1.8)
    max_shift = np.max([np.linalg.norm(trajectories[k] - dp, axis=1) for k in selected], axis=0)
    peak = int(np.argmax(max_shift))
    window = slice(max(0, peak - 5), min(len(dp), peak + 6))
    zoom = _fit_limits(_limits([trajectories[k][window] for k in selected], padding=3,
                               min_span=10), 1.8)
    times = np.asarray(payload["future_time_s"], dtype=float)
    if times.shape != (len(dp),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid future_time_s")
    point_ids = sorted(set(int(np.argmin(abs(times - second))) for second in (1, 2, 4, 6, 8)
                           if second <= times[-1] + 1e-6))
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), dpi=dpi)
        fig.subplots_adjust(left=0.065, right=0.98, bottom=0.16, top=0.73, wspace=0.18)
        try:
            legends(fig)
            title = fig.suptitle("", fontsize=16, y=0.97)
            state = fig.text(0.5, 0.90, "", ha="center", fontsize=12)
            fig.text(0.5, 0.84, "同一场景输入；周车／行人保持同一时刻；箭头：前一已记录 K → 当前 K", ha="center")
            with GifStream(output) as writer:
                previous = dp
                previous_k = 0
                for k in selected:
                    current = trajectories[k]
                    shift = np.linalg.norm(current - dp, axis=1)
                    step_shift = np.linalg.norm(current - previous, axis=1)
                    title.set_text(f"{payload['case_id']} · {SCENARIO_ZH.get(payload['scenario_type'], payload['scenario_type'])} · K={k}")
                    state.set_text(f"frame={payload['frame_index']}  |  相对 DP 最大位移 {shift.max():.3f} 米  |  K={previous_k} → {k} 最大位移 {step_shift.max():.3f} 米")
                    for ax, limits, caption in zip(axes, (bounds, zoom), ("完整轨迹", "局部放大")):
                        ax.clear()
                        _environment(ax, {**payload, "agents": [], "ego_polygon": [], "show_gt": False}, limits)
                        draw_actors(ax, payload, DP_COLOR, agent_ids)
                        ax.plot(*dp.T, color=DP_COLOR, ls="--", lw=1.4, zorder=8)
                        ax.plot(*current.T, color=REFINED_COLOR, lw=1.5, marker=".", ms=2, zorder=9)
                        if k > 0:
                            indices = np.arange(0, len(dp), max(1, len(dp) // 16))
                            diff = current[indices] - previous[indices]
                            ax.quiver(previous[indices, 0], previous[indices, 1], diff[:, 0], diff[:, 1],
                                      angles="xy", scale_units="xy", scale=1, color="#9b6a36", width=0.003,
                                      zorder=10)
                        for i in point_ids:
                            ax.plot(*current[i], "o", color=REFINED_COLOR, ms=4, zorder=11)
                            ax.annotate(f"{times[i]:g}s", current[i], xytext=(3, 6),
                                        textcoords="offset points", fontsize=9, clip_on=True, zorder=11)
                        ax.set_title(caption, loc="left")
                    writer.add(figure_image(fig), hold_s * 1000)
                    previous, previous_k = current, k
        finally:
            plt.close(fig)
    return {"gif": str(output), "K": selected, "interpolated_K": False,
            "frame_count": len(selected), "hold_s": hold_s, "dpi": dpi,
            "agent_ids": agent_ids,
            "actor_time": "fixed_capture_timestamp", "arrows": "previous_saved_K_to_current_K_metres"}


def extract_frames(samples, anchor):
    frames = []
    for sample in samples:
        ego = sample.ego_state
        agents = []
        for obj in sample.observation.tracked_objects.tracked_objects:
            center = obj.box.center
            agents.append({
                "polygon": core.to_local(obj.box.geometry.exterior.coords, anchor).tolist(),
                "kind": obj.tracked_object_type.name, "track_token": getattr(obj, "track_token", None),
                "center": core.to_local([(center.x, center.y)], anchor)[0].tolist(),
                "heading": float(center.heading - anchor.heading),
            })
        frames.append({
            "timestamp_us": int(ego.time_point.time_us),
            "ego_xy": core.to_local([(ego.rear_axle.x, ego.rear_axle.y)], anchor)[0].tolist(),
            "ego_polygon": core.to_local(ego.car_footprint.geometry.exterior.coords, anchor).tolist(),
            "agents": agents,
            "traffic_lights": {str(light.lane_connector_id): light.status.name.lower()
                               for light in sample.traffic_light_status},
        })
    validate_clock(frames)
    return frames


def extract_closed_loop(case, scores, ks, radius=35):
    from closed_loop_context_overview import _check_identity, _log_path
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    sources = {"0": scores["baseline"], **{str(k): scores["by_k"][str(k)] for k in ks}}
    runs, anchor, scenario, first_sample, map_api = {}, None, None, None, None
    connectors = {}
    for k, source in sources.items():
        path = _log_path(source, case[1])
        log = core.load_log(path)
        samples = None
        try:
            _check_identity(log, case, source["log_name"])
            samples = log.simulation_history.data
            if not samples:
                raise ValueError(f"Empty history: {path}")
            if anchor is None:
                pose = samples[0].ego_state.rear_axle
                anchor = SimpleNamespace(x=pose.x, y=pose.y, heading=pose.heading)
            frames = extract_frames(samples, anchor)
            if scenario is None:
                scenario, first_sample = log.scenario, samples[0]
                map_api = log.scenario.map_api
            runs[k] = {"frames": frames, "source_log": str(path),
                       "source_log_sha256": core.digest(path), "result_dir": source["result_dir"]}
            for frame in frames:
                for connector_id in frame["traffic_lights"]:
                    if connector_id not in connectors:
                        obj = map_api.get_map_object(connector_id, SemanticMapLayer.LANE_CONNECTOR)
                        connectors[connector_id] = (core.to_local(
                            [(p.x, p.y) for p in obj.baseline_path.discrete_path], anchor).tolist()
                            if obj is not None else [])
        finally:
            del log, samples
            gc.collect()
    # Cover all runs and both viewports, including a refined run that leaves
    # the DP corridor. The extra corners only size the map query, never a path.
    all_positions = [np.asarray([f["ego_xy"] for f in run["frames"]]) for run in runs.values()]
    left, right, bottom, top = _fit_limits(
        _limits(all_positions, padding=radius, min_span=2 * radius), 1.18)
    corners = [[left, bottom], [left, top], [right, bottom], [right, top]]
    environment = static_map(core.extract_environment(scenario, first_sample, {"view_bounds": corners}))
    return {**environment, "case_id": case[0], "token": case[1], "scenario_type": case[2],
            "runs": runs, "signal_connectors": connectors, "closed_loop_scores": scores,
            "metadata": {"kind": "timestamped_closed_loop_agents", "protocol_version": 1,
                         "anchor": vars(anchor), "map_camera_radius_m": radius,
                         "frame_alignment": "timestamp_us; sample-and-hold"}}


def render_closed_loop(payload, k, output, fps=10, speed=1, radius=35, dpi=100, font=None, agent_ids=False):
    settings = configure_plot(font)
    import matplotlib.pyplot as plt

    if payload.get("metadata", {}).get("kind") != "timestamped_closed_loop_agents":
        raise ValueError("Need timestamped agent history, not a static overview")
    if not np.isfinite(radius) or radius < 5:
        raise ValueError("radius must be finite and >= 5 metres")
    if radius > payload["metadata"].get("map_camera_radius_m", radius):
        raise ValueError("Wider camera needs map re-extraction; omit --render-only")
    keys = ("0", str(k))
    frames = [payload["runs"][key]["frames"] for key in keys]
    clocks = [validate_clock(f) for f in frames]
    timeline = playback_times(clocks, fps, speed)
    durations = playback_durations(timeline, fps, speed)
    tracks = [np.asarray([f["ego_xy"] for f in run]) for run in frames]
    band = _score_band(payload, k)
    if band["verdict"] == "missing":
        raise ValueError("Scenario closed-loop scores are required")
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), dpi=dpi)
        fig.subplots_adjust(left=0.065, right=0.98, bottom=0.14, top=0.77, wspace=0.18)
        try:
            legends(fig)
            fig.suptitle(f"{payload['case_id']} · {SCENARIO_ZH.get(payload['scenario_type'], payload['scenario_type'])} · 场景动态回放", fontsize=16, y=0.97)
            fig.text(0.5, 0.91, band["text"], ha="center", fontsize=13)
            state = fig.text(0.5, 0.855, "", ha="center", fontsize=12)
            camera = None
            with GifStream(output) as writer:
                for number, timestamp in enumerate(timeline):
                    indices = [frame_at(clock, timestamp) for clock in clocks]
                    positions = [tracks[i][j] for i, j in enumerate(indices) if j is not None]
                    if positions:
                        # Shared moving camera: both panels retain the same physical XY bounds.
                        camera = _fit_limits(_limits([np.asarray(positions)], padding=radius,
                                                      min_span=2 * radius), 1.18)
                    if camera is None:
                        camera = _fit_limits(_limits([tracks[0][:1]], padding=radius), 1.18)
                    state.set_text(f"回放时间 {(timestamp-timeline[0])/1e6:.2f} 秒  |  speed x{speed:g}  |  全程场景得分")
                    for i, (ax, index, color) in enumerate(zip(axes, indices, (DP_COLOR, REFINED_COLOR))):
                        ax.clear()
                        _environment(ax, static_map(payload), camera)
                        end = int(np.searchsorted(clocks[i], timestamp, side="right"))
                        if end:
                            ax.plot(*tracks[i][:end].T, color=color, lw=1.3,
                                    ls="--" if i == 0 else "-", zorder=3)
                        name = "DP K=0" if i == 0 else f"S05-G K={k}"
                        if index is None:
                            status = "已结束" if timestamp > clocks[i][-1] else "before start"
                            ax.set_title(f"{name} · {status}", loc="left")
                        else:
                            frame = frames[i][index]
                            for connector, light in frame.get("traffic_lights", {}).items():
                                points = payload.get("signal_connectors", {}).get(connector, [])
                                if points:
                                    color_light = {"red": "#c55252", "yellow": "#c69627", "green": "#498864"}.get(light, "#96999e")
                                    ax.plot(*np.asarray(points).T, color=color_light, lw=2, zorder=4)
                            draw_actors(ax, frame, color, agent_ids)
                            actual_time = (int(clocks[i][index]) - timeline[0]) / 1e6
                            ax.set_title(f"{name} · 记录时间 {actual_time:.2f}s · agent={len(frame['agents'])}", loc="left")
                        ax.set(xlabel="DP 起点前向 x（米）", ylabel="DP 起点左向 y（米）")
                    writer.add(figure_image(fig), durations[number])
                    if number % 100 == 0:
                        print(f"  {payload['case_id']} K={k}: GIF {number+1}/{len(timeline)}", flush=True)
        finally:
            plt.close(fig)
    return {"gif": str(output), "frame_count": len(timeline), "fps": fps, "speed": speed,
            "radius_m": radius, "dpi": dpi, "agent_ids": agent_ids,
            "start_timestamp_us": timeline[0], "end_timestamp_us": timeline[-1],
            "actor_source": "each_run_observed_tracks", "alignment": "timestamp_us", "K": k}


def run_refinement(args):
    paths = []
    for case in selection.selected_cases(args.cohort):
        found = sorted((args.source / "captures" / case[0]).glob("frame_*/payload.json"))
        if args.frames is not None:
            found = [p for p in found if core.read(p)["frame_index"] in args.frames]
        if not found:
            raise FileNotFoundError(f"No captures for {case[0]}; run expanded_score_context.py capture first")
        paths.extend(found)
    for path in paths:
        payload = core.read(path)
        target = args.output / "refinement" / payload["case_id"] / f"frame_{payload['frame_index']:04d}.gif"
        info = render_refinement(payload, target, args.refine_ks, args.hold, args.dpi, args.font, args.agent_ids)
        info.update(source=str(path), source_sha256=core.digest(path))
        core.write(target.with_suffix(".json"), info)
        print(f"GIF: {target}", flush=True)


def run_closed_loop(args):
    from static_context_scores import ScenarioScores

    scores = ScenarioScores(args.suite)
    if not args.render_only:
        core.configure_runtime(args.suite.resolve(), scores.plan, "cpu", None, args.threads)
    for case in selection.selected_cases(args.cohort):
        bundle = scores.for_payload({"token": case[1], "scenario_type": case[2]}, args.ks)
        cache = args.output / "closed_loop_data" / f"{case[0]}.json"
        if args.render_only:
            payload = core.read(cache)
            if (payload["case_id"], payload["token"], payload["scenario_type"]) != case:
                raise ValueError(f"Wrong cached scenario: {cache}")
            if payload["closed_loop_scores"] != bundle:
                raise ValueError(f"Changed score/run provenance; extract again: {case[0]}")
            for run in payload["runs"].values():
                if core.digest(run["source_log"]) != run["source_log_sha256"]:
                    raise ValueError(f"Changed source log: {run['source_log']}")
        else:
            print(f"Reading observed agents and ego states: {case[0]}", flush=True)
            payload = extract_closed_loop(case, bundle, args.ks, args.radius)
            core.write(cache, payload)
        for k in args.ks:
            target = args.output / "closed_loop" / case[0] / f"DP_vs_G_K{k:04d}.gif"
            info = render_closed_loop(payload, k, target, args.fps, args.speed, args.radius,
                                      args.dpi, args.font, args.agent_ids)
            info.update(source=str(cache), source_sha256=core.digest(cache))
            core.write(target.with_suffix(".json"), info)
            print(f"GIF: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("refinement", "closed-loop", "all"))
    parser.add_argument("--source", type=Path, default=core.DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, help="Default: SOURCE/gif_context_v1")
    parser.add_argument("--suite", type=Path, default=core.DEFAULT_SUITE)
    parser.add_argument("--cohort", default="I01", help="all (22 cases), or I01,W01,...")
    parser.add_argument("--frames", type=int, nargs="+", help="Refinement capture frame indices")
    parser.add_argument("--refine-ks", type=int, nargs="+", help="Saved K snapshots; default all saved K")
    parser.add_argument("--ks", type=int, nargs="+", default=[20, 50, 200], help="Closed-loop K comparisons")
    parser.add_argument("--hold", type=float, default=0.8, help="Seconds per saved K snapshot")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0, help="Closed-loop playback speed")
    parser.add_argument("--radius", type=float, default=35.0, help="Shared-camera margin in metres")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--agent-ids", action="store_true", help="Show short track tokens")
    parser.add_argument("--render-only", action="store_true", help="Reuse extracted closed-loop actor JSON")
    args = parser.parse_args()
    selection.selected_cases(args.cohort)
    args.ks = sorted(set(args.ks))
    if not args.ks or any(k <= 0 for k in args.ks):
        parser.error("--ks must contain positive K values")
    if args.threads < 1 or not 50 <= args.dpi <= 200:
        parser.error("threads must be positive and dpi must be 50..200")
    playback_times([np.array([0, 100000])], args.fps, args.speed)
    if not np.isfinite([args.hold, args.radius]).all() or not 0.01 <= args.hold <= 600 or args.radius < 5:
        parser.error("hold must be 0.01..600 seconds and radius must be >= 5 metres")
    args.output = (args.output or args.source / REPORT_NAME).resolve()
    if args.mode in ("refinement", "all"):
        run_refinement(args)
    if args.mode in ("closed-loop", "all"):
        run_closed_loop(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
