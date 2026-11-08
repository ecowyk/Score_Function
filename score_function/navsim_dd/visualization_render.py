"""Synchronized NAVSIM replays with shared map, scale and metric event markers.

Only the replay backend decides whether an event occurred. This renderer does
not infer collisions or area violations from trajectories or aggregate scores.
It consumes plain JSON-compatible data and does not import NAVSIM or torch.
"""

from __future__ import annotations

import base64
import math
from pathlib import Path
import zlib

import numpy as np


ROAD_FILL = "#d6edda"
ROAD_EDGE = "#257048"
EVENT_RED = "#c5232c"
ARM_COLORS = {"0": "#2563eb", "20": "#d66a26"}
ACTOR_COLORS = {
    "VEHICLE": "#71869c", "PEDESTRIAN": "#c18baf", "BICYCLE": "#8d79ad",
    "STATIC": "#a0a2a5", "TRAFFIC_CONE": "#b49b71", "BARRIER": "#9b9c9e",
}

# Exact outlines of the two required Chinese event labels. Noto Sans CJK is
# licensed under SIL OFL 1.1. These vector outlines avoid missing-font boxes on
# headless company servers; all other text uses Matplotlib's bundled font.
# Each compressed record stores float32 x/y and one uint8 Matplotlib path code.
_EVENT_OUTLINES = {'offroad': 'eJxVlH9oW1UUx1+ft2qDImXBMbGrUJFM7NRl1Mlyz5mb0uDUmU6ZklR0slgtaPJHRVNwYzSMVjHoKjTFKpiODaZPnSBVGtCuOJDUH3UuKAq+KcKWKAPXIv6gfs+97Yt9f32458f33HPOfY5zN6cK93JTNLqbB8N3sjq50MtzlS5Wc5W9HKp2sqqWMux1d7FbmOnnniliVZh5lNvzO1l53Xtw8gArZzmP6yx/u7zux2Bq4aZqKck69BupnqndHPZOkXHO146Qivt3cT1RIWht55ML/5DSoTt4av01IjFgwBXwczdK5m0W/BxDawurrGaezHRb0KE9jPBOA2490QktmOQGoSqc64mNJhzqzkpCh7N6L7mTmW8oqzcRon6h8eI57U6tb0YTriM1mWnjaDRJaEsUMExIuNEU76bTrVwtDVkwpW5I3WwhX7vVXtlxulCqRypfi+Fkv1z5FqhvIzUY7kBXr5TMaxD1rVapwlqYpjWcr4XWixpNaIfE/Rr9uYkXI2+LzxY0yiFcZwdONovEToT/qJXMwURJ5w3E/YyFucqzgJ+1Gi8+z45zgNCxAYx7iNSG1DMY0wip9nw/p9OvEkSfQmG3k4pG+2yF9cTjFvK1pAXHeRAJo4SEuxgTIhX2ejnuT8qU+yBxhlQ6nUHxzaxWNqGxGz1TX+hq6TNqivvzejB8kLBjn+isPqFRxvGYDl1WRsK1OlTtKEvN8HmhDNNhXZiZKCOhDxiYVouRyzGvJ6UtV9Ng+FKZ1z5ajLw+jTJO0GSmt4yaW9mA1COAeWVR/CVl3GIA09m+FbN4DktSimEWfTAdji3PHRD2Wi0Mhn+nZZglP3c6htaVCHfTBqSrrsw8gHriLXLFFMBk5hy50egfDYj7zeyi3gak0zdIo843wOu+zZoCMBs+Xsw2wJgc55gOwISLKQAj4ec26wBMGX7uAAUQ9z+X5X8YzyEku7EVJi3P83pKFQ7K3CNoeFGWVuBde1Itzcrc7UwbU65UjprVahKQpK5APXEFqTOlI2h4h1Zj6wrY+dPT6mLkFTbjjqRGsWPDZfi8hkV6r6zGih7G5MLn2Pssz9MA7k4GRMJd0XL/XdrPRj1ZmGB5Wk1LSx8bcAUWIxfIFVMAxhREjRVnbVTlnrPWJGCcxRTA6ijJ4+fIagkYLXnhRiIAY1qlZaJEwpgEjLOYAlgd1Zb/ynZVIOy9WXYjqe8bUJgZJreUOcsbUk+QasvX8KPeQejhRYygEyfNi/b/8+nCn/j7PYJZdPyN9j5Nat+vNaz6IVKxkR/wiztKao03j6F8iT5H6gZcC/ehsHWj/weYNn3wjoWWagkPDeE/5UYhmif18kweon2kShcO4RYPyckITmKS+SWWHVLJqyasqaXq4e7jEv6RVU8WTtnMQ/0V1PwhqfNfz6MJb8Dnr+/s72ulLY1G/QebdIhE', 'collision': 'eJxVlV1oHFUUx2euV7BTPyhZUJHsKm1hi6VoRjFi5h6pNrtoqEwfGrBBiJgBEelWUHBFq9gxFsRF22g2qH3ICirJUBFkJTtaGzVWNgVB3ZcoHaHxYxd8MFk1UNZ7zt29t9mnH/M///M19862/H9FEPhgO40rwGncB7x01oEguB2451wDjdmdwF23H6LcrcCq6ZvAc+4CfqDaB4vreQzeCpnwIPBWNw+zur+HPGdEBo+CnYpyYFkPAE+KoOxjpUFKyMNmDvLJTmBOYz+0s7JolBuFlr8Xgx+RT3xsQ+UxmVv+oLdrbB/YCEnxDmDL9R2CIAiuFZXCVcDzyXde2IwFr6YvDrnupODt7G755Hl8Enrt7JjglcKU5zQeFDwIPvVa/mMo/eh5zlHBU9Ff8snrClLR9TGLcncLDcv1EY+1/KoB1/1bsEwICvLJARHlrgZ+xHtUWNbNwGfKT1KHcpnnVKsIOAXrjWMGxBKVwvfC9py3CRhCJmx6VF0DSdpVTT8M+BZtBMt6UbCZ8v0GjngpYJXCUwbyyQ6sXhcaSEpF20AD2fPJhtCAJVg7u2KA+kFJQ1K8V5BdQzv7zhBbXK+Jy2Aols0/A5eBlLBVDWTHcTQsrluCJcXHDVAbvdnNNjLhLTSFbVl3qnEQKDNKGrpS1zWfq8m13Ab2gPs5uO4u4J3OF7J6BvjE6hl4LnUD8L5ojoAh4GFj0+X/DJB02LuooD7ym7JPl/+QPWeB/zTbhLApj/qh636XLhfY/FLbwFhpL2ZuGMArwYqp8wZIGh4/bYDsA+5iN093CnapcxRorsPxVyTZfXt+VTETqxfUHZwu/wJ0ROv1FTXgodKSat6vnpNwI7AtjW9VzPzSN8rVS8h1iYnVBXlzK8L+8unzBAzBc14SDCUNJGkXDkiu/rCpJAQKRknDZhfmSUU/q1oIVCtsLqgSGkjaVItcWIIkBApGScNml1/9kMBGmCk/G7PhZMMASuTSUDr7vieldQP4VSJJQz7ZL5g/uGYgyv2DU3xigCScSwPZ+8PTBqgEShqojV7PZorh8SlYrh8EG6Hlb8djExoIgm0IUwacxmeCFU++3H0pV74C+GEkqVI4IbrBq/f0YEEeiZPyyrg1PuCeULD2wRtQTe+rySvzmoK17KTMvL3G//RflXku1eTxOy7P2Asxz64cl9JszIecN3GcmB9rviXfxamYX9goy+qTMe/b867cz3jMt4y+J/93tirANghc95Rgx56Yk66PBet0PpI37gfBUKqmv1ZAA65l5wzQElDSgItivY2ZHf4P0UaTAQ=='}


def _xy(points):
    """Initial ego local coordinates -> forward-up display coordinates."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Replay xy must be finite Nx2 values")
    return np.column_stack((-points[:, 1], points[:, 0]))


def _ring(points, positive):
    points = _xy(points)
    if len(points) < 3:
        raise ValueError("Replay polygon ring needs at least three points")
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    signed_area = np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1])
    if (signed_area > 0) != positive:
        points = points[::-1]
    return points


def polygon_path(polygon):
    """Keep polygon holes unfilled using consistent nonzero winding."""
    from matplotlib.path import Path as MplPath

    rings = []
    for points, positive in [(polygon["exterior"], True)] + [(p, False) for p in polygon.get("holes", [])]:
        vertices = _ring(points, positive)
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
        rings.append(MplPath(vertices, codes))
    return MplPath.make_compound_path(*rings)


def _event_label(ax, kind, xy, offset):
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D

    rows = np.frombuffer(zlib.decompress(base64.b64decode(_EVENT_OUTLINES[kind])),
                         dtype=[("x", "<f4"), ("y", "<f4"), ("code", "u1")])
    path = MplPath(np.column_stack((rows["x"], rows["y"])), rows["code"])
    box = DrawingArea(28, 16, 0, 0)
    box.add_artist(PathPatch(path.transformed(Affine2D().scale(13).translate(1, 3)),
                            facecolor=EVENT_RED, edgecolor="none"))
    artist = AnnotationBbox(box, xy, xybox=offset, boxcoords="offset points",
                           box_alignment=(0, 0), pad=0.15, zorder=20,
                           bboxprops=dict(facecolor="white", edgecolor="none", alpha=.94))
    ax.add_artist(artist)
    return artist


def validate_payload(payload):
    times = np.asarray(payload["times"], dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.isclose(times[0], 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Replay times must start at zero and increase strictly")
    if not payload.get("drivable_polygons"):
        raise ValueError("Refuse to render a replay without drivable-area polygons")
    for polygon in payload["drivable_polygons"]:
        polygon_path(polygon)
    expert = _xy(payload["expert_xy"])
    if len(expert) < 2:
        raise ValueError("Actual expert trajectory is required")
    for arm in ("0", "20"):
        run = payload["runs"][arm]
        frames = run["frames"]
        if len(frames) != len(times) or not np.allclose(
            [frame["time_s"] for frame in frames], times, rtol=0, atol=1e-8
        ):
            raise ValueError("K0/K20 frames must share the exact replay timeline")
        if len(_xy(run["planned_xy"])) < 2:
            raise ValueError("Planned trajectory is required")
        for frame in frames:
            _xy([frame["ego_xy"]])
            _xy(frame["ego_polygon"])
            for actor in frame["actors"]:
                _xy(actor["polygon"])
            events = frame.get("events", {})
            if events.get("offroad"):
                _xy(events["offroad"])
            for event in events.get("collision", []):
                _xy([event["xy"]])
    return times


def poster_frame_index(payload):
    """Prefer the frame exposing most actual violation marks; else final frame."""
    counts = []
    for i in range(len(payload["times"])):
        count = 0
        for arm in ("0", "20"):
            events = payload["runs"][arm]["frames"][i].get("events", {})
            count += len(events.get("offroad", [])) + 4 * len(events.get("collision", []))
        counts.append(count)
    return int(np.argmax(counts)) if max(counts) else len(counts) - 1


def _metric_line(metrics):
    names = [
        ("NC", ("no_at_fault_collisions", "no_ego_at_fault_collisions")),
        ("DAC", ("drivable_area_compliance",)),
        ("TTC", ("time_to_collision_within_bound",)),
        ("EP", ("ego_progress", "ego_progress_along_expert_route")),
        ("C", ("comfort", "ego_is_comfortable")),
    ]
    values = []
    for label, keys in names:
        value = next((metrics[k] for k in keys if k in metrics), None)
        if value is not None:
            values.append(f"{label} {100 * float(value):.1f}")
    return "    ".join(values)


def render_case(payload, gif_path, poster_path, *, radius=35.0, dpi=100, case_id=None):
    """Render faithful 1x playback; return paths and display timing metadata.

    Road polygons, planned/actual/expert trajectories and actors are expressed
    in the same initial-rear-axle local frame. The common camera follows the
    midpoint of the two current ego positions. The first and final frames have
    0.4/0.6 seconds of extra reading time; simulation intervals play at 1x.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, PathPatch, Polygon
    from PIL import Image

    times = validate_payload(payload)
    if not math.isfinite(radius) or radius <= 0 or dpi < 50:
        raise ValueError("Require a positive viewport radius and dpi >= 50")
    gif_path, poster_path = Path(gif_path), Path(poster_path)
    for path in (gif_path, poster_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    metadata = payload["metadata"]
    scores = {arm: float(metadata["scores"][arm]) for arm in ("0", "20")}
    delta = 100 * (scores["20"] - scores["0"])
    category = "Improvement" if delta > 0 else "Degradation" if delta < 0 else "Tie"
    identity = case_id or metadata.get("case_id", metadata["token"])
    if "epoch" in metadata:
        identity = f"{identity} / {metadata['epoch']} epochs"
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 9.0), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(left=.045, right=.985, bottom=.13, top=.82, wspace=.055)
    fig.suptitle(f"{identity}  |  {category}  |  PDMS {100*scores['0']:.2f} -> {100*scores['20']:.2f}  ({delta:+.2f})",
                 x=.5, y=.975, fontsize=18, fontweight="bold", color="#162535")
    fig.text(.5, .94, f"{metadata['log']}  /  {metadata['token']}",
             ha="center", fontsize=10, color="#526173")
    clock = fig.text(.5, .895, "", ha="center", fontsize=12, color="#39495a")
    road_paths = [polygon_path(p) for p in payload["drivable_polygons"]]
    lane_paths = [polygon_path(p) for p in payload.get("lanes", [])]
    expert = _xy(payload["expert_xy"])
    dynamic = {"0": [], "20": []}
    trajectories = {}
    for ax, arm in zip(axes, ("0", "20")):
        ax.set_facecolor("#f1f3f4")
        for path in road_paths:
            ax.add_patch(PathPatch(path, facecolor=ROAD_FILL, edgecolor=ROAD_EDGE, lw=1.2, zorder=0))
        for path in lane_paths:
            ax.add_patch(PathPatch(path, facecolor="none", edgecolor="#83a98c", lw=.6, alpha=.65, zorder=1))
        planned = _xy(payload["runs"][arm]["planned_xy"])
        ax.plot(expert[:, 0], expert[:, 1], color="#26313b", lw=1.7, ls=":", alpha=.9, zorder=3)
        ax.plot(planned[:, 0], planned[:, 1], color=ARM_COLORS[arm], lw=1.6, ls="--", alpha=.65, zorder=4)
        trace, = ax.plot([], [], color=ARM_COLORS[arm], lw=2.6, zorder=5)
        trajectories[arm] = trace
        title = "Original DD | K=0" if arm == "0" else "Score refined | K=20"
        ax.set_title(f"{title}     PDMS {100*scores[arm]:.2f}\n"
                     + _metric_line(payload["runs"][arm]["metrics"]),
                     fontsize=12, linespacing=1.6, color=ARM_COLORS[arm], pad=11)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8, colors="#73808c")
        ax.set_xlabel("Lateral display coordinate (m)", fontsize=9, color="#65727d")
        ax.grid(alpha=.15, color="#6c8275", linewidth=.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd4d9")
    axes[0].set_ylabel("Forward from initial ego (m)", fontsize=9, color="#65727d")
    legend = [
        Line2D([], [], color=ARM_COLORS["0"], lw=2.5, label="DD K0"),
        Line2D([], [], color=ARM_COLORS["20"], lw=2.5, label="Refined K20"),
        Line2D([], [], color="#26313b", ls=":", lw=1.8, label="Expert"),
        Line2D([], [], color="#6b7280", ls="--", lw=1.6, label="Planned trajectory"),
        Line2D([], [], color="#6b7280", lw=2.6, label="Executed trace"),
        Patch(facecolor=ROAD_FILL, edgecolor=ROAD_EDGE, label="Drivable area"),
        Patch(facecolor=ACTOR_COLORS["VEHICLE"], label="Other actors"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .052),
               ncol=len(legend), frameon=False, fontsize=9)
    fig.text(.5, .025, "Red circles: actual simulated off-road corners or contact locations. "
             "Collision marks include contacts not penalized by NC.",
             ha="center", fontsize=9, color="#64707b")
    images, durations = [], []
    poster_index = poster_frame_index(payload)
    try:
        for i, time_s in enumerate(times):
            center = _xy([np.mean([payload["runs"][a]["frames"][i]["ego_xy"]
                                  for a in ("0", "20")], axis=0)])[0]
            clock.set_text(f"Simulation time {time_s:.1f} / {times[-1]:.1f} s  |  synchronized 1x playback")
            for ax, arm in zip(axes, ("0", "20")):
                for artist in dynamic[arm]:
                    artist.remove()
                dynamic[arm] = []
                frames = payload["runs"][arm]["frames"]
                frame = frames[i]
                trace = _xy([f["ego_xy"] for f in frames[:i+1]])
                trajectories[arm].set_data(trace[:, 0], trace[:, 1])
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                for actor in frame["actors"]:
                    kind = str(actor["kind"]).upper().split(".")[-1]
                    patch = Polygon(_xy(actor["polygon"]), closed=True,
                                    facecolor=ACTOR_COLORS.get(kind, ACTOR_COLORS["STATIC"]),
                                    edgecolor="#4e5b66", lw=.65, alpha=.9, zorder=6)
                    ax.add_patch(patch)
                    dynamic[arm].append(patch)
                ego = Polygon(_xy(frame["ego_polygon"]), closed=True, facecolor=ARM_COLORS[arm],
                              edgecolor="#ffffff", lw=1.2, zorder=8)
                ax.add_patch(ego)
                dynamic[arm].append(ego)
                events = frame.get("events", {})
                for kind in ("offroad", "collision"):
                    records = events.get(kind, [])
                    if not records:
                        continue
                    positions = records if kind == "offroad" else [v["xy"] for v in records]
                    positions = _xy(positions)
                    for point in positions:
                        circle = Circle(point, radius=.85 if kind == "offroad" else 1.65,
                                        fill=False, edgecolor=EVENT_RED, lw=2.0, zorder=15)
                        ax.add_patch(circle)
                        dynamic[arm].append(circle)
                    offset = (10, -20) if kind == "offroad" else (10, 10)
                    dynamic[arm].append(_event_label(ax, kind, positions[0], offset))
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
            image = Image.fromarray(rgba).convert("RGB")
            if i == poster_index:
                image.save(poster_path)
            images.append(image)
            duration = float(times[i+1] - time_s) if i+1 < len(times) else .1
            durations.append(max(10, round(duration * 1000)))
        durations[0] += 400
        durations[-1] += 600
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=durations, loop=0, disposal=2, optimize=False)
    finally:
        plt.close(fig)
        for image in images:
            image.close()
    return dict(gif=str(gif_path), poster=str(poster_path), frames=len(times),
                duration_s=sum(durations) / 1000, poster_time_s=float(times[poster_index]))
