"""Static, physical-unit figures for long refinement, without retaining dense traces."""

import numpy as np


def plot_long_refinement(
    stem,
    target,
    snapshots,
    scalars,
    columns,
    normalizer,
    future_dt,
    map_data=None,
    neighbors=None,
    neighbor_valid=None,
    title="",
    failure=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = np.asarray(normalizer["mean"], dtype=np.float64).reshape(4)
    std = np.asarray(normalizer["std"], dtype=np.float64).reshape(4)
    target = np.asarray(target, dtype=np.float64) * std + mean
    physical = {
        key: np.asarray(value, dtype=np.float64) * std + mean for key, value in snapshots.items()
    }
    times = np.arange(1, len(target) + 1) * future_dt
    marker_indices = np.unique(
        np.minimum(
            np.rint(np.array([1.0, 2.0, 4.0, 8.0]) / future_dt).astype(int) - 1, len(target) - 1
        )
    )
    marker_indices = marker_indices[marker_indices >= 0]
    figure = plt.figure(figsize=(13, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.6, 1])
    xy, speed, heading = (
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    )
    for key, color in (("lanes", "#d1d5db"), ("route_lanes", "#b9a8e1")):
        for line in (map_data or {}).get(key, []):
            valid = np.any(line != 0, axis=-1)
            points = np.where(valid[:, None], line[:, :2], np.nan)
            xy.plot(points[:, 0], points[:, 1], color=color, linewidth=0.8, zorder=0)
    for history in (map_data or {}).get("neighbor_agents_past", []):
        valid = np.any(history[:, :4] != 0, axis=-1)
        if np.any(valid):
            points = history[valid, :2]
            xy.plot(points[:, 0], points[:, 1], color="#9ca3af", alpha=0.7, linewidth=0.8)
            xy.scatter(*points[-1], color="#6b7280", s=18, marker="s", zorder=2)
    if neighbors is not None:
        for index, trajectory in enumerate(neighbors):
            if neighbor_valid is not None and not neighbor_valid[index]:
                continue
            xy.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="#6b7280",
                linestyle=":",
                alpha=0.55,
                linewidth=1,
                label="Frozen predicted neighbors" if index == 0 else None,
            )
            xy.scatter(
                trajectory[marker_indices, 0],
                trajectory[marker_indices, 1],
                color="#6b7280",
                alpha=0.5,
                s=9,
            )
    xy.scatter(0, 0, marker="^", color="black", s=50, label="Current ego origin")

    def draw(trajectory, label, color, width, style="-"):
        xy.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            label=label,
            color=color,
            linewidth=width,
            linestyle=style,
        )
        xy.scatter(trajectory[marker_indices, 0], trajectory[marker_indices, 1], color=color, s=20)
        # First interval starts at the physical current ego origin, then fixed 0.1s sampling.
        velocity = np.diff(trajectory[:, :2], axis=0, prepend=np.zeros((1, 2))) / future_dt
        speed.plot(
            times, np.linalg.norm(velocity, axis=-1), color=color, linewidth=width, linestyle=style
        )
        angle = np.unwrap(np.arctan2(trajectory[:, 3], trajectory[:, 2]))
        angle[np.linalg.norm(trajectory[:, 2:], axis=-1) <= 1e-12] = np.nan
        heading.plot(times, np.rad2deg(angle), color=color, linewidth=width, linestyle=style)

    draw(target, "Expert", "black", 2, "--")
    keys = sorted(physical)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.92, max(len(keys), 2)))
    for index, step in enumerate(keys):
        draw(
            physical[step],
            f"K={step}",
            "#d97706" if step == 0 else colors[index],
            2 if step in (0, keys[-1]) else 1,
        )
    for point, second in zip(physical[keys[-1]][marker_indices, :2], times[marker_indices]):
        xy.annotate(f"{second:g}s", point, xytext=(3, 3), textcoords="offset points", fontsize=8)
    suffix = f" | failed at step {failure['step']}; last finite shown" if failure else ""
    xy.set(title=title + suffix, xlabel="Ego-relative x (m)", ylabel="Ego-relative y (m)")
    xy.set_aspect("equal", adjustable="datalim")
    xy.legend(fontsize=8, ncol=4)
    speed.set(xlabel="Trajectory time (s)", ylabel="Finite-difference speed (m/s)")
    heading.set(xlabel="Trajectory time (s)", ylabel="Unwrapped represented heading (degrees)")
    for axis in (xy, speed, heading):
        axis.grid(alpha=0.2)
    figure.savefig(str(stem) + "_trajectories.png", dpi=150)
    plt.close(figure)

    column = {name: scalars[:, index] for index, name in enumerate(columns)}
    steps = column["step"]
    figure, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    panels = [
        (("step_xy_mean_m", "step_xy_max_m"), "Per-update XY movement (m)"),
        (
            ("net_xy_mean_m", "net_xy_max_m", "cumulative_xy_path_mean_m"),
            "Net displacement and traveled path (m)",
        ),
        (("score_l2_normalized",), "Score norm (normalized coordinates)"),
        (
            ("step_l2_normalized", "unprojected_step_l2_normalized", "projection_l2_normalized"),
            "Update and projection norms (normalized)",
        ),
        (
            ("heading_deviation_pre_max", "heading_deviation_post_max"),
            "Maximum heading unit-circle deviation",
        ),
        (("ade_m", "fde_m"), "Expert imitation errors (m)"),
        (
            ("energy_before", "energy_unprojected", "energy_after"),
            "Learned energy (not normalized density)",
        ),
        (("heading_mae_rad",), "Expert heading error (radians)"),
        (("net_l2_normalized",), "Net trajectory displacement (normalized)"),
    ]
    for axis, (names, label) in zip(axes.flat, panels):
        plotted = False
        for name in names:
            if np.any(np.isfinite(column[name])):
                axis.plot(steps, column[name], label=name, linewidth=1.3)
                plotted = True
        axis.set(title=label, xlabel="Refinement iteration K")
        axis.set_xscale("symlog", linthresh=20)
        axis.set_yscale("symlog", linthresh=1e-5)
        axis.grid(alpha=0.2)
        if plotted:
            axis.legend(fontsize=6)
        else:
            axis.text(
                0.5,
                0.5,
                "Not defined for this parameterization",
                ha="center",
                transform=axis.transAxes,
                fontsize=9,
            )
    figure.suptitle(title + " — fixed condition; all finite steps")
    figure.savefig(str(stem) + "_diagnostics.png", dpi=150)
    plt.close(figure)
