#!/usr/bin/env python3
"""Thin, per-timestep trajectory plots in a fixed physical nuPlan context.

This module only renders already captured data; it never runs a planner. All
geometry must be in metres, in the same ego rear-axle frame. A case produces a
two-page vector PDF and matching high-resolution PNGs. GT is hidden by default.
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
                    facecolor="#e8f0f5" if crossing else "#f2f3f3",
                    edgecolor="#bdc6cd" if crossing else "#d5d9dc",
                    hatch="///" if crossing else None,
                    linewidth=0.38 if not drivable else 0.25,
                    alpha=0.75 if crossing else 0.60,
                    zorder=0,
                )
            )
    for layer, lines in payload.get("map_lines", {}).items():
        name = layer.lower()
        color, width, style, alpha = "#bbc2c8", 0.45, "-", 0.9
        if "center" in name or "baseline" in name:
            color, width, style, alpha = "#ccd1d5", 0.40, "--", 0.8
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
            ax.plot(*points.T, color="#91bba8", lw=1.15, alpha=0.32, zorder=1)
    for agent in payload.get("agents", []):
        points = _xy(agent["polygon"])
        if not len(points) or not _intersects(points, limits):
            continue
        pedestrian = "pedestrian" in str(agent.get("kind", "")).lower()
        ax.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor="#c7adb8" if pedestrian else "#b8c1c9",
                edgecolor="#937384" if pedestrian else "#7f8b96",
                linewidth=0.55,
                alpha=0.38,
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
    ax.set(xlim=limits[:2], ylim=limits[2:], xlabel="Forward x [m]", ylabel="Left y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7, length=2.5, width=0.5)
    for spine in ax.spines.values():
        spine.set_color("#d2d7dc")
        spine.set_linewidth(0.5)
    ax.grid(False)


def _trajectory(ax, points, k, detail=False):
    ax.plot(
        *points.T,
        color=DP_COLOR if k == 0 else COLORS.get(k, "#a33a76"),
        lw=0.58 if k == 0 else 0.78,
        ls="--" if k == 0 else "-",
        marker="o",
        markersize=0.9 if k == 0 else (1.25 if detail else 1.1),
        markeredgewidth=0,
        alpha=0.8 if k == 0 else 0.96,
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


def render_case(payload, output_dir, shared_pdf=None):
    """Return paths and bounds after writing two PNGs and a two-page PDF.

    Required: trajectories mapping K -> N x 2/4 physical values, including K=0.
    Optional context: map_polygons/map_lines by layer, route_lines, agents with
    polygon/kind, ego_polygon, gt_future (hidden unless show_gt=true). All XY
    uses the same captured rear-axle frame. ``future_time_s`` or metadata's
    ``future_dt_s`` (default 0.1) sets the trajectory-time axis. ``zoom_index``
    overrides the automatically chosen largest-shift region. An optional open
    PdfPages-compatible ``shared_pdf`` also receives both pages for a combined
    report; the caller owns its lifetime. ``metadata.context_label`` adds a
    short provenance label without implying that the chosen frame is an event.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    trajectories = _trajectories(payload)
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
        if len(limits) != 4 or limits[0] >= limits[1] or limits[2] >= limits[3]:
            raise ValueError("limits must be (xmin, xmax, ymin, ymax)")
    else:
        limits = _limits([np.zeros((1, 2)), *trajectories.values()])
    detail_ks = [k for k in DETAIL_KS if k in trajectories]
    if not detail_ks:
        detail_ks = sorted(k for k in trajectories if k > 0)[:3]
    movement = np.max([np.linalg.norm(trajectories[k] - dp, axis=1) for k in detail_ks], axis=0)
    peak = int(payload.get("zoom_index", np.argmax(movement)))
    if not 0 <= peak < len(dp):
        raise ValueError("zoom_index is outside the future trajectory")
    neighborhood = slice(max(0, peak - 5), min(len(dp), peak + 6))
    zoom = _limits(
        [dp[neighborhood]] + [trajectories[k][neighborhood] for k in detail_ks],
        padding=1.5,
        min_span=5.0,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label = re.sub(
        r"[^a-zA-Z0-9_.-]+", "_", str(payload.get("case_id", payload.get("token", "case")))
    )
    stem = "{}_frame{:04d}".format(label, int(payload.get("frame_index", 0)))
    pdf_path = output / (stem + ".pdf")
    png_paths = [output / (stem + "_context.png"), output / (stem + "_detail.png")]
    heading = "{} | {} | frame {}".format(
        payload.get("case_id", "Case"),
        payload.get("scenario_type", ""),
        payload.get("frame_index", 0),
    )
    subtitle = "token={} | same scene, same conditions and DP start | physical XY, no displacement magnification".format(
        payload.get("token", "unknown")
    )
    context_label = str(payload.get("metadata", {}).get("context_label", ""))
    no_map = not (payload.get("map_polygons") or payload.get("map_lines"))
    environment_note = "Roads: gray   Route: pale green   Vehicles: gray boxes   Pedestrians: mauve boxes   Actors shown at capture time"
    if no_map:
        environment_note = (
            "MAP GEOMETRY NOT PROVIDED - do not infer road boundaries from this figure"
        )
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figures = []
        try:
            # Retain equal physical axes without wasting half the page on the
            # common long, narrow road view. A second page keeps full detail.
            overview_height = float(
                np.clip(3.1 + 9.0 * (limits[3] - limits[2]) / (limits[1] - limits[0]), 7.2, 11.0)
            )
            fig, axes = plt.subplots(2, 3, figsize=(15.5, overview_height))
            figures.append(fig)
            fig.subplots_adjust(
                left=0.055,
                right=0.975,
                bottom=0.115,
                top=1.0 - 1.1 / overview_height,
                wspace=0.27,
                hspace=0.35,
            )
            fig.suptitle(
                heading + " | refinement snapshots", y=0.978, fontsize=14, fontweight="medium"
            )
            fig.text(
                0.5,
                1.0 - 0.56 / overview_height,
                subtitle,
                ha="center",
                fontsize=8,
                color="#56616c",
            )
            fig.text(
                0.5,
                1.0 - 0.82 / overview_height,
                "Each panel: dashed DP + one refined trajectory. Every future timestep has a small point. All panels share limits.",
                ha="center",
                fontsize=9,
            )
            for ax, k in zip(axes.flat, OVERVIEW_KS):
                _environment(ax, payload, limits)
                _trajectory(ax, dp, 0)
                if k in trajectories:
                    _trajectory(ax, trajectories[k], k)
                    ax.set_title("K = {}".format(k), color=COLORS[k], pad=7)
                else:
                    ax.set_title("K = {} (not captured)".format(k), color="#899198", pad=7)
                ax.add_patch(
                    Rectangle(
                        (zoom[0], zoom[2]),
                        zoom[1] - zoom[0],
                        zoom[3] - zoom[2],
                        fill=False,
                        edgecolor="#9c88b1",
                        lw=0.55,
                        ls=":",
                        zorder=3,
                    )
                )
            handles = [
                Line2D(
                    [], [], color=DP_COLOR, lw=0.7, ls="--", marker="o", ms=1.3, label="DP / K=0"
                ),
                Line2D(
                    [],
                    [],
                    color="#0072b2",
                    lw=0.8,
                    marker="o",
                    ms=1.3,
                    label="Current K (panel color)",
                ),
                Patch(facecolor="none", edgecolor="#354458", lw=0.7, label="Current ego footprint"),
                Line2D([], [], color="#9c88b1", lw=0.7, ls=":", label="Detail region on page 2"),
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.052),
                ncol=4,
                frameon=False,
                fontsize=8,
            )
            fig.text(0.5, 0.031, environment_note, ha="center", fontsize=8, color="#58636d")
            if context_label:
                fig.text(0.055, 0.012, context_label, fontsize=7, color="#7f8891")
            fig.text(0.975, 0.012, "1 / 2", ha="right", fontsize=8, color="#7f8891")

            detail = plt.figure(figsize=(15.5, 10.6))
            figures.append(detail)
            grid = detail.add_gridspec(
                2,
                6,
                left=0.065,
                right=0.975,
                bottom=0.115,
                top=0.865,
                height_ratios=(1.4, 1),
                hspace=0.40,
                wspace=0.60,
            )
            detail.suptitle(
                heading + " | local movement", y=0.978, fontsize=15, fontweight="medium"
            )
            detail.text(0.5, 0.940, subtitle, ha="center", fontsize=8, color="#56616c")
            detail.text(
                0.5,
                0.910,
                "Shared zoom near future t={:.1f}s | arrows join the same future timestep: DP to refined; lengths are not amplified".format(
                    times[peak]
                ),
                ha="center",
                fontsize=9,
            )
            for j, k in enumerate(detail_ks):
                ax = detail.add_subplot(grid[0, j * 2 : j * 2 + 2])
                _environment(ax, payload, zoom)
                _trajectory(ax, dp, 0, detail=True)
                _trajectory(ax, trajectories[k], k, detail=True)
                indices = np.unique(
                    np.linspace(neighborhood.start, neighborhood.stop - 1, 4).astype(int)
                )
                for idx in indices:
                    delta = trajectories[k][idx] - dp[idx]
                    if np.linalg.norm(delta) > 0.01:
                        ax.annotate(
                            "",
                            xy=trajectories[k][idx],
                            xytext=dp[idx],
                            arrowprops={
                                "arrowstyle": "->",
                                "lw": 0.50,
                                "color": COLORS.get(k, "#a33a76"),
                                "alpha": 0.60,
                                "mutation_scale": 5,
                            },
                            zorder=6,
                        )
                ax.set_title(
                    "K = {} | shared detail window".format(k), color=COLORS.get(k, "#a33a76")
                )
            along_ax = detail.add_subplot(grid[1, :3])
            left_ax = detail.add_subplot(grid[1, 3:])
            for k in detail_ks:
                along, left = displacement_components(dp, trajectories[k])
                for ax, values in ((along_ax, along), (left_ax, left)):
                    ax.plot(
                        times,
                        values,
                        color=COLORS.get(k, "#a33a76"),
                        lw=0.80,
                        marker="o",
                        ms=1.05,
                        mew=0,
                        label="K={}".format(k),
                    )
            for ax, title in (
                (along_ax, "Shift along DP tangent (+ forward)"),
                (left_ax, "Shift across DP tangent (+ left)"),
            ):
                ax.axhline(0, color="#9aa2a9", lw=0.6, ls="--")
                ax.axvspan(
                    times[neighborhood.start],
                    times[neighborhood.stop - 1],
                    color="#a28bbd",
                    alpha=0.07,
                )
                ax.set(
                    xlabel="Future trajectory time [s]", ylabel="Refined minus DP [m]", title=title
                )
                ax.grid(axis="y", alpha=0.15, lw=0.5)
                ax.legend(frameon=False, fontsize=8, ncol=3)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
            detail.text(
                0.5,
                0.069,
                "Displacement curves compare matched future points; they are not GT error or closed-loop scores.",
                ha="center",
                fontsize=8,
            )
            detail.text(
                0.5,
                0.049,
                "Where DP is completely stationary, tangent falls back to the current ego forward direction.",
                ha="center",
                fontsize=8,
                color="#58636d",
            )
            detail.text(0.5, 0.029, environment_note, ha="center", fontsize=8, color="#58636d")
            detail.text(0.975, 0.012, "2 / 2", ha="right", fontsize=8, color="#7f8891")
            if context_label:
                detail.text(0.065, 0.012, context_label, fontsize=7, color="#7f8891")
            with PdfPages(
                pdf_path,
                metadata={
                    "Title": heading,
                    "Subject": "Fixed-input score refinement in physical map context",
                },
            ) as pdf:
                for fig, png in zip(figures, png_paths):
                    fig.savefig(png, dpi=210)
                    pdf.savefig(fig)
                    if shared_pdf is not None:
                        shared_pdf.savefig(fig)
        finally:
            for fig in figures:
                plt.close(fig)
    return {
        "pdf": str(pdf_path),
        "png": [str(p) for p in png_paths],
        "limits": list(limits),
        "zoom_limits": list(zoom),
        "zoom_index": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="Captured same-input scene JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render_case(json.loads(args.payload.read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
