"""Offline trajectory errors and sampled-corruption score diagnostics."""

import math

import numpy as np
import torch


def normalizer_statistics(normalizer, device, dtype):
    mean = torch.as_tensor(normalizer["mean"], device=device, dtype=dtype).reshape(-1)
    std = torch.as_tensor(normalizer["std"], device=device, dtype=dtype).reshape(-1)
    if mean.numel() != 4 or std.numel() != 4:
        raise ValueError("Evaluation needs ego-only mean/std with four coordinates")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid ego normalizer")
    return mean, std


def to_physical(trajectory, normalizer):
    mean, std = normalizer_statistics(normalizer, trajectory.device, trajectory.dtype)
    return trajectory * std + mean


def trajectory_metrics(prediction, target, normalizer):
    """Return one value per trajectory; angular error uses radians, ADE/FDE metres.

    Heading vectors with exactly zero length have no defined angle. Their count
    is explicit; their angular errors are excluded rather than assigned zero.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 4:
        raise ValueError("Expected matching [batch, future, 4] trajectories")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite trajectory in evaluation")
    physical = to_physical(prediction, normalizer)
    physical_target = to_physical(target, normalizer)
    distance = (physical[..., :2] - physical_target[..., :2]).norm(dim=-1)
    heading = torch.atan2(physical[..., 3], physical[..., 2])
    target_heading = torch.atan2(physical_target[..., 3], physical_target[..., 2])
    delta = heading - target_heading
    angular = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
    valid = (physical[..., 2:].norm(dim=-1) > 1e-12) & (
        physical_target[..., 2:].norm(dim=-1) > 1e-12
    )
    heading_count = valid.sum(dim=-1)
    heading_mean = (angular * valid).sum(dim=-1) / heading_count.clamp_min(1)
    heading_mean = heading_mean.masked_fill(heading_count == 0, float("nan"))
    return {
        "mse_normalized": (prediction - target).square().mean(dim=(-1, -2)),
        "ade_m": distance.mean(dim=-1),
        "fde_m": distance[:, -1],
        "heading_mae_rad": heading_mean,
        "heading_undefined_points": (~valid).sum(dim=-1),
    }


def score_diagnostics(score, noise, sigma):
    """Flatten an entire future trajectory for cosine; do not average point cosines."""
    if score.shape != noise.shape or score.ndim != 3 or score.shape[-1] != 4:
        raise ValueError("Expected matching score/noise [batch, future, 4]")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    if not torch.isfinite(score).all() or not torch.isfinite(noise).all():
        raise FloatingPointError("Non-finite score/noise in evaluation")
    flat, target = score.flatten(1), (-noise / sigma).flatten(1)
    magnitude, target_magnitude = flat.norm(dim=-1), target.norm(dim=-1)
    valid = (magnitude > 1e-12) & (target_magnitude > 1e-12)
    denominator = (magnitude * target_magnitude).clamp_min(torch.finfo(score.dtype).tiny)
    cosine = ((flat * target).sum(dim=-1) / denominator).clamp(-1, 1)
    cosine = cosine.masked_fill(~valid, float("nan"))
    first = torch.diff(score, dim=1)
    second = torch.diff(score, n=2, dim=1)
    return {
        "dsm": (sigma * score + noise).square().mean(dim=(-1, -2)),
        "cosine": cosine,
        "score_l2": magnitude,
        "score_point_l2_mean": score.norm(dim=-1).mean(dim=-1),
        "score_d1": first.norm(dim=-1).mean(dim=-1),
        "score_d2": second.norm(dim=-1).mean(dim=-1),
    }


def one_step_recovery(noisy, target, score, sigma, gamma, normalizer):
    """Unprojected update. In particular gamma=1 is the Tweedie-style estimate."""
    if not math.isfinite(gamma) or gamma < 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("gamma must be nonnegative and sigma positive")
    refined = noisy + gamma * sigma**2 * score
    return refined, trajectory_metrics(refined, target, normalizer)


class MetricAccumulator:
    """Finite scalar means with explicit missing counts and stable float64 sums."""

    def __init__(self):
        self.sums, self.counts, self.missing = {}, {}, {}

    def add(self, values):
        for key, tensor in values.items():
            array = tensor.detach().double().cpu().numpy().reshape(-1)
            valid = np.isfinite(array)
            self.sums[key] = self.sums.get(key, 0.0) + float(array[valid].sum())
            self.counts[key] = self.counts.get(key, 0) + int(valid.sum())
            self.missing[key] = self.missing.get(key, 0) + int((~valid).sum())

    def result(self):
        return {
            key: {
                "mean": self.sums[key] / self.counts[key] if self.counts[key] else None,
                "count": self.counts[key],
                "undefined_count": self.missing[key],
            }
            for key in self.sums
        }


def finite_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def summarize_cosines(path, histogram, undefined):
    count = int(histogram.sum())
    result = {
        "count": count,
        "undefined_count": undefined,
        "mean": None,
        "median": None,
        "fraction_positive": None,
        "histogram_counts": histogram.tolist(),
        "histogram_edges": np.linspace(-1, 1, len(histogram) + 1).tolist(),
    }
    if count:
        # Disk-backed exact median avoids accumulating all per-frame dictionaries.
        values = np.memmap(path, dtype=np.float64, mode="r+", shape=(count,))
        result["mean"] = float(values.mean())
        result["fraction_positive"] = float(np.count_nonzero(values > 0) / count)
        middle = count // 2
        values.partition((middle - 1, middle) if count % 2 == 0 else middle)
        result["median"] = float(
            (values[middle - 1] + values[middle]) / 2 if count % 2 == 0 else values[middle]
        )
        values.flush()
        del values
    return result
