#!/usr/bin/env python3
"""Read complete saved closed-loop ego histories; never run planner inference.

Each run retains its own timestamps and sample count. All geometry is displayed
in the first DP history frame, so unequal run lengths are never index-aligned.
"""

import argparse
import gc
from pathlib import Path

import expanded_score_context as selection
import numpy as np
import static_score_context as capture_core
from static_context_fonts import configure_chinese_font
from static_context_plots import (
    DP_COLOR,
    REFINED_COLOR,
    SCENARIO_ZH,
    _environment,
    _fit_limits,
    _limits,
    _score_band,
    _trajectory,
    case_category,
)

KS = (20, 50, 200)
REPORT_NAME = "closed_loop_overview_v1"
CHINESE_TEXT = "".join(c for c in Path(__file__).read_text(encoding="utf-8") if ord(c) > 127)


def history_track(samples, anchor):
    """Extract every actual saved rear-axle state, not predicted trajectories."""
    if not samples:
        raise ValueError("The saved simulation has no history samples")
    timestamps = np.array([int(s.ego_state.time_point.time_us) for s in samples], dtype=np.int64)
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("Saved actual-state timestamps must be strictly increasing")
    world = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y] for s in samples])
    if not np.isfinite(world).all():
        raise ValueError("Saved actual ego positions contain nonfinite coordinates")
    xy = capture_core.to_local(world, anchor)
    return dict(
        xy=xy.tolist(),
        timestamps_us=timestamps.tolist(),
        frame_count=len(samples),
        duration_s=float((timestamps[-1] - timestamps[0]) / 1_000_000),
        coordinates="DP first history frame; rear-axle XY in metres",
    )


def validate_track(track):
    xy = np.asarray(track["xy"], dtype=float)
    timestamps = np.asarray(track["timestamps_us"], dtype=np.int64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Actual trajectory must be a nonempty finite N x 2 array")
    if len(xy) != len(timestamps) or len(xy) != track["frame_count"]:
        raise ValueError("Actual trajectory frame count and timestamps do not match")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("Actual trajectory timestamps must increase within each run")
    duration = float((timestamps[-1] - timestamps[0]) / 1_000_000)
    if abs(duration - float(track["duration_s"])) > 1e-6:
        raise ValueError("Actual trajectory duration differs from saved timestamps")
    return xy


def _log_path(source, token):
    from omegaconf import OmegaConf

    run = Path(source["result_dir"])
    cfg = OmegaConf.load(run / "code/hydra/config.yaml")
    callback = cfg.get("callback", {}).get("simulation_log_callback", {})
    root = run / str(callback.get("simulation_log_dir", "simulation_log"))
    return capture_core.find_log(run, token, log_root=root)


def _check_identity(log, case, expected_log):
    scenario = log.scenario
    if str(scenario.token) != case[1] or scenario.scenario_type != case[2]:
        raise ValueError(f"Simulation identity differs from selected case {case[0]}")
    if str(scenario.log_name) != expected_log:
        raise ValueError(f"Simulation recording differs from score row for {case[0]}")


def extract_case(case, score_bundle):
    """Load four trusted logs sequentially and query one shared static map."""
    sources = {"0": score_bundle["baseline"], **score_bundle["by_k"]}
    log_paths = {k: _log_path(row, case[1]) for k, row in sources.items()}
    dp_log = capture_core.load_log(log_paths["0"])
    try:
        _check_identity(dp_log, case, sources["0"]["log_name"])
        scenario, samples = dp_log.scenario, dp_log.simulation_history.data
        if not samples:
            raise ValueError(f"Empty DP actual-state history: {case[0]}")
        anchor = samples[0].ego_state.rear_axle
        tracks = {"0": history_track(samples, anchor)}
        for k in map(str, KS):
            log = capture_core.load_log(log_paths[k])
            try:
                _check_identity(log, case, sources[k]["log_name"])
                tracks[k] = history_track(log.simulation_history.data, anchor)
            finally:
                del log
                gc.collect()
        environment = capture_core.extract_environment(
            scenario, samples[0], {k: value["xy"] for k, value in tracks.items()}
        )
        # A full-run path cannot be compared with actors frozen at the initial
        # instant. Retain static map/route and the initial ego footprint only.
        environment.pop("agents", None)
        # Signals also vary during the run; colored initial signal states must
        # not be misread as a traffic-light history for the complete scenario.
        environment["map_lines"] = {
            layer: lines
            for layer, lines in environment.get("map_lines", {}).items()
            if layer.lower() not in ("red", "yellow", "green", "unknown")
        }
        for k, track in tracks.items():
            track.update(
                source_log=str(log_paths[k]),
                source_log_sha256=capture_core.digest(log_paths[k]),
                result_dir=sources[k]["result_dir"],
            )
        return dict(
            case_id=case[0],
            token=case[1],
            scenario_type=case[2],
            tracks=tracks,
            closed_loop_scores=score_bundle,
            **environment,
            metadata=dict(
                kind="actual_closed_loop_full_history",
                protocol_version=1,
                anchor="DP_K0 first saved ego state",
                common_index_alignment=False,
                description="Every saved ego state, with each run's own timestamps; no predicted futures",
            ),
        )
    finally:
        del dp_log
        gc.collect()


def render_overview(payload, output_dir, shared_pdf=None, font_path=None, zoom_bounds=None):
    """Three pages of whole-scenario actual paths; optional explicit spatial zoom."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MaxNLocator

    if payload.get("metadata", {}).get("kind") != "actual_closed_loop_full_history":
        raise ValueError("This renderer requires complete actual closed-loop histories")
    font = configure_chinese_font(font_path, required_text=CHINESE_TEXT + str(payload["case_id"]))
    tracks = {str(k): validate_track(payload["tracks"][str(k)]) for k in (0, *KS)}
    score_bands = {k: _score_band(payload, k) for k in KS}
    if any(band["verdict"] == "missing" for band in score_bands.values()):
        raise ValueError("Full-run overview requires all four exact scenario scores")
    # Ignore dynamic actors even if an externally prepared payload includes them.
    environment = {**payload, "agents": [], "show_gt": False}
    width, height = 15.5, 10.2
    bounds = _limits([np.zeros((1, 2)), *tracks.values()], padding=5.0, min_span=14.0)
    if zoom_bounds is None:
        positions = [(0.065, 0.185, 0.90, 0.55)]
    else:
        zoom_bounds = tuple(float(x) for x in zoom_bounds)
        if (
            len(zoom_bounds) != 4
            or not np.isfinite(zoom_bounds).all()
            or zoom_bounds[0] >= zoom_bounds[1]
            or zoom_bounds[2] >= zoom_bounds[3]
        ):
            raise ValueError("--zoom-bounds requires xmin xmax ymin ymax in the initial DP frame")
        positions = [(0.065, 0.215, 0.41, 0.53), (0.56, 0.215, 0.41, 0.53)]
    bounds = _fit_limits(bounds, width * positions[0][2] / (height * positions[0][3]))
    windows = [bounds]
    if zoom_bounds is not None:
        windows.append(
            _fit_limits(zoom_bounds, width * positions[1][2] / (height * positions[1][3]))
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = str(payload["case_id"])
    pdf_path = output / f"{stem}_全程实驶.pdf"
    png_paths = [output / f"{stem}_全程实驶_K{k:04d}.png" for k in KS]
    handles = [
        Line2D([], [], color=DP_COLOR, lw=0.7, ls="--", label="DP 实际行驶"),
        Line2D([], [], color=REFINED_COLOR, lw=0.8, label="G 实际行驶"),
        Line2D([], [], color="#aab4bc", lw=0.7, label="道路／车道边界"),
        Line2D([], [], color="#8fb09d", lw=1.2, label="路线"),
        Patch(facecolor="none", edgecolor="#354458", lw=0.7, label="自车初始位置"),
    ]
    with plt.rc_context(
        {
            "font.family": font,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        }
    ):
        with PdfPages(pdf_path) as pdf:
            for page, k in enumerate(KS):
                fig = plt.figure(figsize=(width, height))
                try:
                    title = f"{stem}（{case_category(stem)}）· {SCENARIO_ZH.get(payload['scenario_type'], '驾驶场景')}"
                    fig.suptitle(title + f"  |  全程实际行驶 · G K={k}", y=0.968, fontsize=19)
                    band = score_bands[k]
                    fig.text(
                        0.5,
                        0.909,
                        band["text"],
                        ha="center",
                        fontsize=15,
                        color=band["color"],
                        bbox={
                            "boxstyle": "round,pad=0.5",
                            "facecolor": band["background"],
                            "edgecolor": "none",
                        },
                    )
                    fig.text(
                        0.5,
                        0.863,
                        "这里画的是整个场景中车辆实际走过的路径，不是某一帧预测的未来 8 秒轨迹。",
                        ha="center",
                        fontsize=12,
                        color="#344b5e",
                    )
                    dp_meta, g_meta = payload["tracks"]["0"], payload["tracks"][str(k)]
                    counts = (
                        f"DP：{dp_meta['frame_count']} 个记录点 / {dp_meta['duration_s']:.2f} 秒    "
                        f"G K={k}：{g_meta['frame_count']} 个记录点 / {g_meta['duration_s']:.2f} 秒"
                    )
                    fig.text(0.5, 0.823, counts, ha="center", fontsize=12, color="#53616c")
                    for index, (position, view) in enumerate(zip(positions, windows)):
                        ax = fig.add_axes(position)
                        _environment(ax, environment, view)
                        _trajectory(ax, tracks["0"], 0)
                        _trajectory(ax, tracks[str(k)], k)
                        ax.set(xlabel="DP 起点前向 x（米）", ylabel="DP 起点左向 y（米）")
                        ax.set_title(
                            "完整实际路径（全部记录点）"
                            if index == 0
                            else "指定区域放大（位置比例不变）",
                            loc="left",
                            pad=12,
                        )
                        ax.xaxis.set_major_locator(MaxNLocator(7))
                        ax.yaxis.set_major_locator(MaxNLocator(5))
                        if zoom_bounds is not None and index == 0:
                            region = windows[1]
                            ax.add_patch(
                                Rectangle(
                                    (region[0], region[2]),
                                    region[1] - region[0],
                                    region[3] - region[2],
                                    fill=False,
                                    edgecolor="#8c7a9e",
                                    ls=":",
                                    lw=0.65,
                                )
                            )
                    fig.legend(
                        handles=handles,
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0.107),
                        ncol=5,
                        frameon=False,
                        fontsize=11,
                        columnspacing=1.8,
                    )
                    fig.text(
                        0.5,
                        0.085,
                        "小点按各组自身时间顺序连接；点数和结束时间可以不同，不按下标配对。",
                        ha="center",
                        fontsize=11,
                        color="#53616c",
                    )
                    fig.text(
                        0.5,
                        0.055,
                        "图中仅保留静态地图和初始自车；周车、行人及信号灯动态需另看回放。路径交叉不等于碰撞。",
                        ha="center",
                        fontsize=10.5,
                        color="#53616c",
                    )
                    fig.text(
                        0.05,
                        0.020,
                        f"场景 {stem} · token={payload['token']} · 分组为选样标签，编号不是排名",
                        fontsize=9,
                        color="#75828d",
                    )
                    fig.text(
                        0.97, 0.020, f"{page + 1} / 3", ha="right", fontsize=9, color="#75828d"
                    )
                    fig.savefig(png_paths[page], dpi=210)
                    pdf.savefig(fig)
                    if shared_pdf is not None:
                        shared_pdf.savefig(fig)
                finally:
                    plt.close(fig)
    return dict(
        pdf=str(pdf_path),
        png=list(map(str, png_paths)),
        limits=list(bounds),
        frame_counts={k: value["frame_count"] for k, value in payload["tracks"].items()},
    )


def extract(args):
    from static_context_scores import ScenarioScores

    scores = ScenarioScores(args.suite)
    cases = selection.selected_cases(args.cohort)
    bundles = {
        case[0]: scores.for_payload({"token": case[1], "scenario_type": case[2]}, KS)
        for case in cases
    }
    capture_core.configure_runtime(args.suite.resolve(), scores.plan, "cpu", None, args.threads)
    destination = args.output.resolve() / REPORT_NAME / "data"
    for case in cases:
        print(f"读取全程实际行驶记录：{case[0]} {case[1]}", flush=True)
        payload = extract_case(case, bundles[case[0]])
        capture_core.write(destination / f"{case[0]}.json", payload)
        print(
            "  "
            + " / ".join(
                f"K={k}: {v['frame_count']}点, {v['duration_s']:.2f}s"
                for k, v in payload["tracks"].items()
            ),
            flush=True,
        )


def render(args):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    from matplotlib.backends.backend_pdf import PdfPages
    from static_context_scores import ScenarioScores

    scores = ScenarioScores(args.suite)
    report = args.output.resolve() / REPORT_NAME
    payloads = []
    for case in selection.selected_cases(args.cohort):
        payload = capture_core.read(report / "data" / f"{case[0]}.json")
        if (payload["case_id"], payload["token"], payload["scenario_type"]) != case:
            raise ValueError(f"Wrong cached actual trajectory identity: {case[0]}")
        current_scores = scores.for_payload(payload, KS)
        if current_scores != payload["closed_loop_scores"]:
            raise ValueError(f"Score/run provenance changed for {case[0]}; extract again")
        for k, track in payload["tracks"].items():
            validate_track(track)
            if capture_core.digest(track["source_log"]) != track["source_log_sha256"]:
                raise ValueError(f"Saved source log changed for {case[0]}/K{k}; extract again")
        payloads.append(payload)
    for group in selection.GROUPS:
        items = [p for p in payloads if p["case_id"].startswith(group)]
        if not items:
            continue
        target = report / f"{group}_全程实际行驶.pdf"
        temporary = target.with_suffix(".tmp.pdf")
        with PdfPages(temporary) as combined:
            for payload in items:
                render_overview(
                    payload,
                    report / "cases" / payload["case_id"],
                    shared_pdf=combined,
                    font_path=args.font,
                    zoom_bounds=args.zoom_bounds,
                )
                print(f"已绘制全程路径：{payload['case_id']}", flush=True)
        temporary.replace(target)
        print(f"分组 PDF：{target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "extract", "render"), default="run")
    parser.add_argument("--suite", type=Path, default=capture_core.DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=capture_core.DEFAULT_OUTPUT)
    parser.add_argument("--cohort", default="all", help="all (22 cases), or comma-separated IDs")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--font", type=Path)
    parser.add_argument(
        "--zoom-bounds", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX")
    )
    args = parser.parse_args()
    selection.selected_cases(args.cohort)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command in ("run", "extract"):
        extract(args)
    if args.command in ("run", "render"):
        render(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
