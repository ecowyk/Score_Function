"""Trajectory overlays and score-diagnostic plots."""

import numpy as np
import torch

from score_function.evaluation.metrics import normalizer_statistics, to_physical


def plot_trajectory(
    path, target, initial, refined, score, normalizer, title, map_data=None, iterations=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean, noisy, final = [
        to_physical(item.detach().cpu(), normalizer).numpy() for item in (target, initial, refined)
    ]
    # Display physical displacement for one natural sigma^2 update, not the
    # physical-coordinate density gradient (which transforms differently).
    _, std = normalizer_statistics(normalizer, score.device, score.dtype)
    vectors = (score * std).detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if map_data:
        for name, color in (("lanes", "#d1d5db"), ("route_lanes", "#c4b5fd")):
            if name not in map_data:
                continue
            for line in map_data[name]:
                valid = np.any(line != 0, axis=-1)
                points = np.where(valid[:, None], line[:, :2], np.nan)
                axis.plot(points[:, 0], points[:, 1], color=color, linewidth=0.8, zorder=0)
    if iterations is not None:
        for item in iterations:
            points = to_physical(torch.as_tensor(item), normalizer).numpy()
            axis.plot(points[:, 0], points[:, 1], color="#93c5fd", linewidth=0.7, alpha=0.7)
    axis.plot(clean[:, 0], clean[:, 1], color="black", label="Expert")
    axis.plot(noisy[:, 0], noisy[:, 1], color="#d97706", alpha=0.7, label="Initial")
    axis.plot(final[:, 0], final[:, 1], color="#2563eb", label="Refined")
    selected = np.arange(0, len(noisy), max(1, len(noisy) // 12))
    axis.quiver(
        noisy[selected, 0],
        noisy[selected, 1],
        vectors[selected, 0],
        vectors[selected, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#16a34a",
        label="Update displacement",
    )
    axis.set(title=title, xlabel="Ego-relative x (m)", ylabel="Ego-relative y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_cosine(path, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.asarray(summary["histogram_edges"])
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.bar(edges[:-1], summary["histogram_counts"], width=np.diff(edges), align="edge")
    axis.set(
        xlabel="Cosine with sampled corruption target",
        ylabel="Trajectory/noise pairs",
        title="Conditional denoising direction; not a true marginal-score comparison",
        xlim=(-1, 1),
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)
