#!/usr/bin/env python3
"""Chinese, one-K-per-page physical trajectory views from saved captures.

The renderer never runs inference. Maps, actors and all trajectories must share
the captured ego rear-axle frame, in metres. GT and diagnostic curves are hidden
by default. Full views and local details preserve equal physical XY scales.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

COLORS = {
    5: "#087e8b",
    20: "#0072b2",
    50: "#d55e00",
    100: "#8b59a5",
    200: "#009e73",
    500: "#bc6c25",
}
OVERVIEW_KS = (5, 20, 50, 100, 200, 500)
DETAIL_KS = (20, 50, 200)
DP_COLOR = "#535c68"
REFINED_COLOR = "#086db7"
SCENARIO_ZH = {
    "behind_long_vehicle": "长车后方",
    "changing_lane": "变道",
    "following_lane_with_lead": "跟车",
    "high_lateral_acceleration": "较大横向加速度",
    "high_magnitude_speed": "高速行驶",
    "low_magnitude_speed": "低速行驶",
    "near_multiple_vehicles": "多车附近",
    "starting_left_turn": "左转起步",
    "starting_right_turn": "右转起步",
    "starting_straight_traffic_light_intersection_traversal": "信号灯路口直行",
    "stationary_in_traffic": "车流中静止",
    "stopping_with_lead": "随前车停车",
    "traversing_pickup_dropoff": "通过接送区域",
    "waiting_for_pedestrian_to_cross": "等待行人通过",
}


def _xy(values):
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError("Geometry must be a finite N x (2 or more) array")
    return points[:, :2]


def _trajectories(payload):
    result = {int(str(k).removeprefix("K")): _xy(v) for k, v in payload["trajectories"].items()}
    if 0 not in result or len(result[0]) < 2:
        raise ValueError("A DP trajectory at K=0 with at least two points is required")
    if any(v.shape != result[0].shape for v in result.values()):
        raise ValueError("All K snapshots must have the same future timestep count")
    if not any(k > 0 for k in result):
        raise ValueError("At least one refined trajectory is required")
    return result


def _limits(arrays, padding=4.0, min_span=12.0):
    points = np.concatenate(arrays, axis=0)
    lo, hi = points.min(axis=0), points.max(axis=0)
    mid = (lo + hi) / 2
    span = np.maximum(hi - lo + 2 * padding, min_span)
    return (
        float(mid[0] - span[0] / 2),
        float(mid[0] + span[0] / 2),
        float(mid[1] - span[1] / 2),
        float(mid[1] + span[1] / 2),
    )


def _intersects(points, limits):
    return not (
        points[:, 0].max() < limits[0]
        or points[:, 0].min() > limits[1]
        or points[:, 1].max() < limits[2]
        or points[:, 1].min() > limits[3]
    )


def _environment(ax, payload, limits):
    from matplotlib.patches import Polygon

    for layer, polygons in payload.get("map_polygons", {}).items():
        name = layer.lower()
        for polygon in polygons:
            points = _xy(polygon)
            if not len(points) or not _intersects(points, limits):
                continue
            crossing = "crosswalk" in name
            drivable = "drivable" in name
            ax.add_patch(
                Polygon(
                    points,
                    closed=True,
                    facecolor="#edf0f3" if crossing else "#f6f7f7",
                    edgecolor="#abb7c0" if crossing else "#b2bac1",
                    hatch="///" if crossing else None,
                    linewidth=0.60 if not drivable else 0.40,
                    alpha=0.8,
                    zorder=0,
                )
            )
    for layer, lines in payload.get("map_lines", {}).items():
        name = layer.lower()
        color, width, style, alpha = "#adb7bf", 0.60, "-", 0.9
        if "center" in name or "baseline" in name:
            color, width, style, alpha = "#c9cfd3", 0.40, "--", 0.8
        if "red" in name:
            color, width, alpha = "#c55252", 0.9, 0.85
        elif "yellow" in name:
            color, width, alpha = "#c69627", 0.9, 0.85
        elif "green" in name:
            color, width, alpha = "#6e9b79", 0.9, 0.85
        for line in lines:
            points = _xy(line)
            if len(points) and _intersects(points, limits):
                ax.plot(*points.T, color=color, lw=width, ls=style, alpha=alpha, zorder=1)
    for line in payload.get("route_lines", []):
        points = _xy(line)
        if len(points) and _intersects(points, limits):
            ax.plot(*points.T, color="#8fb09d", lw=1.2, alpha=0.42, zorder=1)
    for agent in payload.get("agents", []):
        points = _xy(agent["polygon"])
        if not len(points) or not _intersects(points, limits):
            continue
        pedestrian = "pedestrian" in str(agent.get("kind", "")).lower()
        ax.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor=(0.78, 0.65, 0.73, 0.22) if pedestrian else (0.64, 0.69, 0.73, 0.16),
                edgecolor="#937384" if pedestrian else "#7f8b96",
                linewidth=0.60,
                zorder=2,
            )
        )
    if payload.get("ego_polygon") is not None and len(payload["ego_polygon"]):
        ax.add_patch(
            Polygon(
                _xy(payload["ego_polygon"]),
                closed=True,
                facecolor="none",
                edgecolor="#354458",
                lw=0.75,
                zorder=3,
            )
        )
    ax.plot(0, 0, "+", color="#354458", ms=3, mew=0.7, zorder=4)
    if payload.get("show_gt", False) and payload.get("gt_future") is not None:
        points = _xy(payload["gt_future"])
        ax.plot(*points.T, color="#887349", lw=0.55, ls=":", alpha=0.65, zorder=3)
    ax.set(xlim=limits[:2], ylim=limits[2:], xlabel="自车前向 x（米）", ylabel="自车左向 y（米）")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=10, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("#d2d7dc")
        spine.set_linewidth(0.5)
    ax.grid(False)


def _trajectory(ax, points, k, detail=False):
    ax.plot(
        *points.T,
        color=DP_COLOR if k == 0 else REFINED_COLOR,
        lw=0.65 if k == 0 else 0.80,
        ls="--" if k == 0 else "-",
        marker="o",
        markersize=0.85 if k == 0 else 1.05,
        markeredgewidth=0,
        alpha=0.9 if k == 0 else 0.96,
        zorder=4 if k == 0 else 5,
    )


def displacement_components(baseline, refined):
    """Signed XY shift along/left of the local DP tangent, in physical metres."""
    baseline, refined = _xy(baseline), _xy(refined)
    tangent = np.gradient(baseline, axis=0)
    lengths = np.linalg.norm(tangent, axis=1)
    good = np.flatnonzero(lengths > 1e-7)
    if len(good):
        # Fill stationary segments from the closest moving DP point.
        bad = np.flatnonzero(lengths <= 1e-7)
        for i in bad:
            tangent[i] = tangent[good[np.argmin(np.abs(good - i))]]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    else:
        tangent[:] = (1.0, 0.0)
    offset = refined - baseline
    along = np.sum(offset * tangent, axis=1)
    left = offset[:, 1] * tangent[:, 0] - offset[:, 0] * tangent[:, 1]
    return along, left


def _fit_limits(limits, aspect):
    """Expand bounds to fill a panel while retaining equal metres on both axes."""
    xmin, xmax, ymin, ymax = limits
    width, height = xmax - xmin, ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    return (cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2)


def _font_family(font_path, required_text=""):
    # Direct CLI execution and importlib-based tools/tests both work without
    # modifying global sys.path or requiring a package installation.
    try:
        from static_context_fonts import configure_chinese_font
    except ModuleNotFoundError:
        import importlib.util

        font_module = Path(__file__).with_name("static_context_fonts.py")
        spec = importlib.util.spec_from_file_location("static_context_fonts", font_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        configure_chinese_font = module.configure_chinese_font
    return configure_chinese_font(font_path, required_text=required_text)


def _time_labels(ax, refined, times, limits):
    # At most four labels, disabled by default because lane/vehicle context
    # often matters more than time annotations on a tightly cropped view.
    used = set()
    for second in (2, 4, 6, 8):
        if second > times[-1] + 1e-5:
            continue
        index = int(np.argmin(abs(times - second)))
        if index in used:
            continue
        used.add(index)
        x, y = refined[index]
        if limits[0] < x < limits[1] and limits[2] < y < limits[3]:
            ax.annotate(
                f"{times[index]:g}秒",
                (x, y),
                xytext=(3, 6),
                textcoords="offset points",
                fontsize=9,
                color=REFINED_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.6},
                clip_on=True,
                zorder=7,
            )


def _appendix(payload, dp, trajectories, requested, times, font_family):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.2))
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.16, top=0.82, hspace=0.40)
    fig.suptitle("补充曲线：每个未来时刻相对 DP 的位置变化", fontsize=19, y=0.965)
    fig.text(
        0.5,
        0.9,
        "横轴是预测的未来时间，不是推动次数 K；这些曲线不是闭环得分。",
        ha="center",
        fontsize=13,
        color="#435364",
    )
    for k in requested:
        along, left = displacement_components(dp, trajectories[k])
        for ax, values in zip(axes, (along, left)):
            ax.plot(
                times,
                values,
                color=COLORS.get(k, REFINED_COLOR),
                lw=0.8,
                marker="o",
                markersize=1.0,
                markeredgewidth=0,
                label=f"推动 {k} 次",
            )
    for ax, title in zip(
        axes, ("沿 DP 局部方向的位置变化（正值：前方）", "横向位置变化（正值：左侧）")
    ):
        ax.axhline(0, color="#9da6ae", lw=0.6, ls="--")
        ax.set(xlabel="预测的未来时间（秒）", ylabel="相对 DP 的位移（米）", title=title)
        ax.legend(frameon=False, ncol=len(requested), fontsize=11)
        ax.grid(axis="y", alpha=0.15, lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.text(
        0.5,
        0.07,
        "DP 完全静止时，方向取当前自车前向。此页为可选补充，不用于判断碰撞。",
        ha="center",
        fontsize=11,
        color="#65727c",
    )
    return fig


def render_case(
    payload, output_dir, shared_pdf=None, ks=(20, 50, 200), include_curves=False, font_path=None
):
    """Render one large Chinese comparison page per K, with a physical zoom.

    All pages reuse the same DP input and coordinate frame. Selected-K pages
    share overview/zoom bounds; optional curves form a separate appendix page.
    ``shared_pdf`` is an open PdfPages-compatible object owned by the caller.
    Maps/agents/trajectory payload fields are unchanged from capture v1.
    ``show_time_labels`` is optional and off by default. No inference is run.
    """
    import matplotlib

    matplotlib.use("Agg")
    # A caller-owned PdfPages is finalized after this function's rc_context
    # exits. Keep Type 42 active through that later font-embedding step; a
    # Type 3 fallback can corrupt CJK glyphs in the combined report.
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MaxNLocator

    font_family = _font_family(font_path, required_text=str(payload.get("case_id", "")))
    trajectories = _trajectories(payload)
    requested = list(dict.fromkeys(int(k) for k in ks))
    if not requested or any(k <= 0 or k not in trajectories for k in requested):
        raise ValueError("Each requested K must be positive and present in captured trajectories")
    dp = trajectories[0]
    times = np.asarray(
        payload.get(
            "future_time_s",
            np.arange(1, len(dp) + 1) * float(payload.get("metadata", {}).get("future_dt_s", 0.1)),
        ),
        dtype=float,
    )
    if times.shape != (len(dp),) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("future_time_s must be finite and strictly increasing")
    if "limits" in payload:
        limits = tuple(float(v) for v in payload["limits"])
        if (
            len(limits) != 4
            or not np.isfinite(limits).all()
            or limits[0] >= limits[1]
            or limits[2] >= limits[3]
        ):
            raise ValueError("limits must be finite (xmin, xmax, ymin, ymax)")
    else:
        context_arrays = [np.zeros((1, 2)), dp] + [trajectories[k] for k in requested]
        if payload.get("ego_polygon") is not None:
            context_arrays.append(_xy(payload["ego_polygon"]))
        limits = _limits(context_arrays, padding=3.0, min_span=12.0)
    movement = np.max([np.linalg.norm(trajectories[k] - dp, axis=1) for k in requested], axis=0)
    peak = int(payload.get("zoom_index", np.argmax(movement)))
    if not 0 <= peak < len(dp):
        raise ValueError("zoom_index is outside the future trajectory")
    neighborhood = slice(max(0, peak - 5), min(len(dp), peak + 6))
    zoom = _limits(
        [dp[neighborhood]] + [trajectories[k][neighborhood] for k in requested],
        padding=1.3,
        min_span=6.0,
    )

    # A wide straight road gets two full-width horizontal strips; turning and
    # stationary scenes get two large, nearly square panels. Bounds expand to
    # the actual panel dimensions instead of distorting the x/y scale.
    width, height = 15.5, 9.2
    stacked = (limits[1] - limits[0]) / (limits[3] - limits[2]) > 2.3
    if stacked:
        positions = [(0.08, 0.535, 0.88, 0.29), (0.08, 0.19, 0.88, 0.24)]
    else:
        positions = [(0.065, 0.225, 0.41, 0.575), (0.56, 0.225, 0.41, 0.575)]
    limits = _fit_limits(limits, positions[0][2] * width / (positions[0][3] * height))
    zoom = _fit_limits(zoom, positions[1][2] * width / (positions[1][3] * height))
    # Keep the entire enlarged window visible as a rectangle in the overview,
    # including at the endpoint of a long trajectory.
    limits = _fit_limits(
        (
            min(limits[0], zoom[0]),
            max(limits[1], zoom[1]),
            min(limits[2], zoom[2]),
            max(limits[3], zoom[3]),
        ),
        positions[0][2] * width / (positions[0][3] * height),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label = re.sub(
        r"[^a-zA-Z0-9_.-]+", "_", str(payload.get("case_id", payload.get("token", "case")))
    )
    stem = "{}_frame{:04d}".format(label, int(payload.get("frame_index", 0)))
    pdf_path = output / (stem + ".pdf")
    png_paths = [output / (stem + "_K{:04d}.png".format(k)) for k in requested]
    if include_curves:
        png_paths.append(output / (stem + "_补充曲线.png"))
    scenario = SCENARIO_ZH.get(payload.get("scenario_type"), "驾驶场景")
    meta = payload.get("metadata", {})
    anchor = meta.get("anchor", "DP_K0")
    has_map = bool(payload.get("map_polygons") or payload.get("map_lines"))
    regular_dt = float(np.median(np.diff(times)))
    cadence = (
        f"每点间隔 {regular_dt:g} 秒"
        if np.allclose(np.diff(times), regular_dt)
        else "每个小点对应一个预测时刻"
    )
    footer = "场景 {}  ·  token={}  ·  第 {} 帧  ·  背景取自 {}".format(
        payload.get("case_id", ""),
        payload.get("token", "未知"),
        payload.get("frame_index", 0),
        anchor,
    )
    environment_handles = [
        Line2D([], [], color="#abb5be", lw=0.75, label="道路／车道边界"),
        Line2D([], [], color="#8fb09d", lw=1.4, label="路线"),
        Patch(facecolor="none", edgecolor="#354458", lw=0.75, label="当前自车"),
        Patch(facecolor="#e0e5e8", edgecolor="#7f8b96", lw=0.6, label="周车"),
        Patch(facecolor="#e6d8e0", edgecolor="#937384", lw=0.6, label="行人"),
    ]
    if payload.get("show_gt", False) and payload.get("gt_future") is not None:
        environment_handles.append(
            Line2D([], [], color="#887349", lw=0.6, ls=":", label="专家轨迹")
        )
    rc = {
        "font.family": font_family,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(rc):
        with PdfPages(
            pdf_path,
            metadata={
                "Title": f"{label}：同一输入下的轨迹推动",
                "Subject": "地图背景、原始DP轨迹、推动后的轨迹；真实物理比例",
            },
        ) as pdf:
            for page, k in enumerate(requested):
                fig = plt.figure(figsize=(width, height))
                try:
                    fig.suptitle(
                        f"{payload.get('case_id', '')} · {scenario}  |  推动 {k} 次（K={k}）",
                        y=0.968,
                        fontsize=20,
                        fontweight="normal",
                    )
                    fig.text(
                        0.5,
                        0.915,
                        "灰虚线：原始 DP    蓝实线：推动后",
                        ha="center",
                        fontsize=14,
                        color="#24394c",
                    )
                    fig.text(
                        0.5,
                        0.878,
                        f"{cadence}；K 为推动次数。同一帧各 K 页面使用同一输入、同一比例。",
                        ha="center",
                        fontsize=11,
                        color="#64717c",
                    )
                    for index, (position, bounds) in enumerate(zip(positions, (limits, zoom))):
                        ax = fig.add_axes(position)
                        _environment(ax, payload, bounds)
                        _trajectory(ax, dp, 0, detail=index == 1)
                        _trajectory(ax, trajectories[k], k, detail=index == 1)
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                        ax.set_title(
                            "全景：道路与完整轨迹"
                            if index == 0
                            else "局部放大：查看虚框内的轨迹变化",
                            pad=10,
                            loc="left",
                        )
                        if index == 0:
                            ax.add_patch(
                                Rectangle(
                                    (zoom[0], zoom[2]),
                                    zoom[1] - zoom[0],
                                    zoom[3] - zoom[2],
                                    fill=False,
                                    edgecolor="#8c7a9e",
                                    lw=0.7,
                                    ls=":",
                                    zorder=3,
                                )
                            )
                        else:
                            indices = np.unique(
                                np.linspace(neighborhood.start, neighborhood.stop - 1, 4).astype(
                                    int
                                )
                            )
                            for point in indices:
                                delta = trajectories[k][point] - dp[point]
                                if np.linalg.norm(delta) > 0.01:
                                    ax.annotate(
                                        "",
                                        xy=trajectories[k][point],
                                        xytext=dp[point],
                                        arrowprops={
                                            "arrowstyle": "->",
                                            "lw": 0.55,
                                            "color": REFINED_COLOR,
                                            "alpha": 0.70,
                                            "mutation_scale": 6,
                                        },
                                        zorder=6,
                                    )
                        if payload.get("show_time_labels", False):
                            _time_labels(ax, trajectories[k], times, bounds)
                    fig.legend(
                        handles=environment_handles,
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0.107),
                        ncol=len(environment_handles),
                        frameon=False,
                        fontsize=11,
                        handlelength=2.0,
                        columnspacing=2.0,
                    )
                    if has_map:
                        note = "周车、行人为当前帧位置。小箭头连接同一未来时刻的 DP 点与推动后位置；未放大位移。"
                    else:
                        note = "此帧未提供地图，不能据此判断道路边界。小箭头表示同一预测时刻的位置变化。"
                    fig.text(0.5, 0.083, note, ha="center", fontsize=10.5, color="#53616c")
                    fig.text(
                        0.5,
                        0.053,
                        "同一输入下的轨迹推动；非各组闭环实驶轨迹。局部窗口按位置变化选择，不代表事故时刻。",
                        ha="center",
                        fontsize=10.5,
                        color="#53616c",
                    )
                    fig.text(0.05, 0.019, footer, fontsize=8.5, color="#7c8790")
                    fig.text(
                        0.97,
                        0.019,
                        f"{page + 1} / {len(png_paths)}",
                        ha="right",
                        fontsize=9,
                        color="#7c8790",
                    )
                    fig.savefig(png_paths[page], dpi=210)
                    pdf.savefig(fig)
                    if shared_pdf is not None:
                        shared_pdf.savefig(fig)
                finally:
                    plt.close(fig)
            if include_curves:
                fig = _appendix(payload, dp, trajectories, requested, times, font_family)
                try:
                    fig.text(0.05, 0.019, footer, fontsize=8.5, color="#7c8790")
                    fig.text(
                        0.97,
                        0.019,
                        f"{len(png_paths)} / {len(png_paths)}",
                        ha="right",
                        fontsize=9,
                        color="#7c8790",
                    )
                    fig.savefig(png_paths[-1], dpi=210)
                    pdf.savefig(fig)
                    if shared_pdf is not None:
                        shared_pdf.savefig(fig)
                finally:
                    plt.close(fig)
    return {
        "pdf": str(pdf_path),
        "png": [str(p) for p in png_paths],
        "ks": requested,
        "pages": len(png_paths),
        "layout": "stacked" if stacked else "side_by_side",
        "limits": list(limits),
        "zoom_limits": list(zoom),
        "zoom_index": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="Captured same-input scene JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DETAIL_KS))
    parser.add_argument(
        "--include-curves", action="store_true", help="Add a separate optional diagnostics page"
    )
    parser.add_argument("--font", type=Path, help="Chinese font file; see static_context_fonts.py")
    args = parser.parse_args()
    print(
        json.dumps(
            render_case(
                json.loads(args.payload.read_text()),
                args.output,
                ks=args.ks,
                include_curves=args.include_curves,
                font_path=args.font,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
