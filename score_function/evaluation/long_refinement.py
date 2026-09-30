"""Fixed-condition, long-horizon refinement diagnostics with bounded trajectory storage.

This is an offline inspection tool. It never trains a model, executes a simulator,
recomputes conditioning inside a path, or interprets convergence as planning quality.
"""

from __future__ import annotations

import csv
import hashlib
import html
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from score_function.evaluation.common import create_output, load_evaluation
from score_function.evaluation.metrics import normalizer_statistics, to_physical, trajectory_metrics
from score_function.utils.normalizer import project_heading
from score_function.utils.train_utils import (
    atomic_write,
    file_hash,
    read_json,
    resolve_path,
    sample_noise,
    stable_seed,
)

DEFAULT_SNAPSHOTS = (0, 5, 20, 100, 500, 1000, 2000, 5000)
SCALAR_COLUMNS = (
    "step",
    "score_l2_normalized",
    "step_l2_normalized",
    "net_l2_normalized",
    "step_xy_mean_m",
    "step_xy_max_m",
    "net_xy_mean_m",
    "net_xy_max_m",
    "cumulative_xy_path_mean_m",
    "unprojected_step_l2_normalized",
    "projection_l2_normalized",
    "heading_deviation_pre_mean",
    "heading_deviation_pre_max",
    "heading_deviation_post_mean",
    "heading_deviation_post_max",
    "ade_m",
    "fde_m",
    "heading_mae_rad",
    "heading_undefined_points",
    "energy_before",
    "energy_unprojected",
    "energy_after",
)


def representative_indices(records, count, seed):
    """Reproducible round-robin across scenario types, or recordings if unlabelled.

    Selection precedes every model evaluation and uses no outcome/metric values.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("max_samples must be a positive integer")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        group = record.get("scenario_type") or record["recording"]
        groups[group].append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: stable_seed(seed, "selection", records[index]["token"]))
    keys = sorted(groups, key=lambda key: stable_seed(seed, "group", key))
    selected, depth = [], 0
    while len(selected) < min(count, len(records)):
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                if len(selected) == min(count, len(records)):
                    break
        depth += 1
    return selected


def _tensor_hash(tensor):
    array = tensor.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _norm(value):
    return float(value.double().norm().item())


def _checked(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {label}")
    return value


@torch.no_grad()
def trace_refinement(
    model,
    initial,
    target,
    scene,
    route,
    normalizer,
    sigma,
    gamma,
    steps,
    heading_projection=True,
    snapshot_steps=DEFAULT_SNAPSHOTS,
    neighbor_future=None,
    neighbor_valid=None,
    convergence_window=50,
    convergence_step_m=1e-4,
    scalar_path=None,
    progress=None,
):
    """Trace one fixed condition, returning O(K) scalars and sparse trajectories.

    Uses exactly the legacy update x += gamma*sigma^2*s(x,c), then optional
    heading projection. No tolerance-based early stop. Numerical failure preserves
    the last finite state and lets the caller continue with other scenes.
    """
    if initial.ndim != 3 or initial.shape[0] != 1 or initial.shape[-1] != 4:
        raise ValueError("Long diagnostics process one [1,T,4] trajectory at a time")
    if target.shape != initial.shape:
        raise ValueError("Target and initial shapes differ")
    if getattr(model, "training", False):
        raise ValueError("Model must be in eval mode")
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("Require finite sigma > 0 and gamma >= 0")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if convergence_window < 1 or not math.isfinite(convergence_step_m) or convergence_step_m < 0:
        raise ValueError("Invalid descriptive convergence settings")
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 0 for k in snapshot_steps):
        raise ValueError("Snapshot steps must be nonnegative integers")
    requested = set(snapshot_steps) | {0, steps}
    mean, std = normalizer_statistics(normalizer, initial.device, initial.dtype)
    projection_normalizer = SimpleNamespace(mean=mean[None, None], std=std[None, None])
    for name, tensor in (
        ("initial", initial),
        ("target", target),
        ("scene", scene),
        ("route", route),
    ):
        _checked(tensor, name)
    condition = {}
    if neighbor_future is not None:
        _checked(neighbor_future, "neighbor condition")
        condition = {"neighbor_future": neighbor_future, "neighbor_valid": neighbor_valid}
    energy_model = getattr(model, "parameterization", "score") == "energy"
    current, physical_initial = initial.detach().clone(), to_physical(initial, normalizer)
    _checked(physical_initial, "physical initial")
    snapshots = {0: current[0].cpu().numpy().copy()}
    # Float64 scalar diagnostics cost O(K), while trajectories cost O(snapshot count*T).
    scalars = np.full((steps + 1, len(SCALAR_COLUMNS)), np.nan, dtype=np.float64)
    field_index = {name: index for index, name in enumerate(SCALAR_COLUMNS)}
    stream = Path(scalar_path).open("w", encoding="utf-8", newline="") if scalar_path else None
    writer = csv.DictWriter(stream, fieldnames=SCALAR_COLUMNS) if stream else None
    if writer:
        writer.writeheader()
    completed, failure, cumulative = 0, None, 0.0

    def energy(trajectory):
        if not energy_model:
            return None
        value = model.energy_value(trajectory, scene, route, **condition)
        _checked(value, "energy")
        if value.numel() != 1:
            raise ValueError("Expected one energy per trajectory")
        return float(value.item())

    def record(step, previous, updated, unprojected, score=None, energies=(None, None, None)):
        nonlocal cumulative
        physical = _checked(to_physical(updated, normalizer), "physical trajectory")
        physical_pre = _checked(
            to_physical(unprojected, normalizer), "unprojected physical trajectory"
        )
        previous_physical = to_physical(previous, normalizer)
        xy_step = (physical[..., :2].double() - previous_physical[..., :2].double()).norm(dim=-1)
        xy_net = (physical[..., :2].double() - physical_initial[..., :2].double()).norm(dim=-1)
        cumulative += float(xy_step.mean())
        pre_heading = (physical_pre[..., 2:].double().norm(dim=-1) - 1).abs()
        post_heading = (physical[..., 2:].double().norm(dim=-1) - 1).abs()
        metrics = trajectory_metrics(updated.double(), target.double(), normalizer)
        row = {
            "step": step,
            "score_l2_normalized": _norm(score) if score is not None else None,
            "step_l2_normalized": _norm(updated.double() - previous.double()),
            "net_l2_normalized": _norm(updated.double() - initial.double()),
            "step_xy_mean_m": float(xy_step.mean()),
            "step_xy_max_m": float(xy_step.max()),
            "net_xy_mean_m": float(xy_net.mean()),
            "net_xy_max_m": float(xy_net.max()),
            "cumulative_xy_path_mean_m": cumulative,
            "unprojected_step_l2_normalized": _norm(unprojected.double() - previous.double()),
            "projection_l2_normalized": _norm(updated.double() - unprojected.double()),
            "heading_deviation_pre_mean": float(pre_heading.mean()),
            "heading_deviation_pre_max": float(pre_heading.max()),
            "heading_deviation_post_mean": float(post_heading.mean()),
            "heading_deviation_post_max": float(post_heading.max()),
            **{
                name: _scalar(metrics[name].item())
                for name in ("ade_m", "fde_m", "heading_mae_rad", "heading_undefined_points")
            },
            **dict(zip(("energy_before", "energy_unprojected", "energy_after"), energies)),
        }
        for key, value in row.items():
            scalars[step, field_index[key]] = np.nan if value is None else value
        if writer:
            writer.writerow(row)
        if step in requested:
            snapshots[step] = updated[0].cpu().numpy().copy()

    started = time.perf_counter()
    try:
        initial_energy = None
        try:
            initial_energy = energy(current)
        except FloatingPointError as error:
            failure = {"step": 0, "error": f"{type(error).__name__}: {error}"}
        record(0, current, current, current, energies=(initial_energy,) * 3)
        if progress is not None:
            progress(0, steps)
        # A zero gamma follows the exact disabled path: no projection or score calls.
        previous_energy = initial_energy
        if gamma > 0 and failure is None:
            for step in range(1, steps + 1):
                try:
                    score = _checked(model(current, scene, route, **condition), "score")
                    if score.shape != current.shape:
                        raise ValueError("Score and trajectory shapes differ")
                    proposed = _checked(current + gamma * sigma**2 * score, "update")
                    updated = (
                        project_heading(proposed, projection_normalizer, previous_norm=current)
                        if heading_projection
                        else proposed
                    )
                    _checked(updated, "projected update")
                    unprojected_energy = energy(proposed)
                    updated_energy = unprojected_energy if updated is proposed else energy(updated)
                    energies = (previous_energy, unprojected_energy, updated_energy)
                    record(step, current, updated, proposed, score, energies)
                except (FloatingPointError, ValueError) as error:
                    if (
                        not isinstance(error, FloatingPointError)
                        and "nonfinite" not in str(error).lower()
                    ):
                        raise
                    failure = {"step": step, "error": f"{type(error).__name__}: {error}"}
                    break
                current, completed = updated.detach(), step
                previous_energy = updated_energy
                if progress is not None:
                    progress(completed, steps)
    except FloatingPointError as error:
        failure = {"step": 0, "error": f"{type(error).__name__}: {error}"}
    finally:
        if stream:
            stream.close()
    snapshots[completed] = current[0].cpu().numpy().copy()
    scalars = scalars[: completed + 1]
    tail = scalars[-convergence_window:, field_index["step_xy_max_m"]]
    small = bool(completed >= convergence_window and np.all(tail <= convergence_step_m))
    result = {
        "requested_steps": steps,
        "completed_steps": completed,
        "disabled": gamma == 0 or steps == 0,
        "failure": failure,
        "snapshot_steps": sorted(snapshots),
        "parameterization": getattr(model, "parameterization", "score"),
        "sigma": sigma,
        "gamma": gamma,
        "heading_projection": heading_projection,
        "small_last_window_xy_updates": small,
        "convergence_window": convergence_window,
        "convergence_step_m": convergence_step_m,
        "interpretation": "Small XY updates describe this iteration only; not a score zero, density maximum, or planning-quality certificate.",
        "energy_note": "Learned energy has arbitrary additive offset; it is not normalized density. Projection may increase energy."
        if energy_model
        else None,
        "elapsed_seconds": time.perf_counter() - started,
        "initial_metrics": {
            key: _scalar(value.item())
            for key, value in trajectory_metrics(
                initial.double(), target.double(), normalizer
            ).items()
        },
        "final_metrics": {
            key: _scalar(value.item())
            for key, value in trajectory_metrics(
                current.double(), target.double(), normalizer
            ).items()
        },
    }
    return result, scalars, snapshots


def _load_map(record, required=False):
    path = Path(record["feature"]) if record.get("feature") else None
    if path is None or not path.is_file():
        if required:
            raise FileNotFoundError(f"Planner initialization requires original NPZ: {path}")
        return {}, "Raw feature unavailable; map and observed neighbors omitted."
    if record.get("feature_sha256") and file_hash(path) != record["feature_sha256"]:
        raise ValueError(f"Changed input feature: {path}")
    with np.load(path, allow_pickle=False) as raw:
        data = {
            key: raw[key].copy()
            for key in ("lanes", "route_lanes", "neighbor_agents_past")
            if key in raw
        }
    return data, None


@torch.no_grad()
def diagnose_refinement(
    config,
    checkpoint,
    split="val",
    output=None,
    max_samples=None,
    steps=None,
    snapshot_steps=None,
    initializations=None,
    seed=None,
    gamma=None,
):
    """Inspect both DP candidates and fixed-noise expert recoveries on <=8 frames by default."""
    from score_function.evaluation.long_refinement_plots import plot_long_refinement
    from score_function.utils.feature_dataset import read_feature_batch
    from score_function.utils.neighbor import neighbor_kwargs, prediction_neighbors
    from score_function.utils.normalizer import normalize_ego_future
    from score_function.utils.planner_utils import (
        ego_normalizer_metadata,
        load_frozen_planner,
        planner_identity,
    )
    from score_function.utils.progress import Progress
    from score_function.utils.sampling import predict_candidates

    cfg = config.get("long_refinement", {})
    max_samples = max_samples if max_samples is not None else cfg.get("max_samples", 8)
    steps = steps if steps is not None else cfg.get("steps", 5000)
    snapshot_steps = (
        snapshot_steps
        if snapshot_steps is not None
        else cfg.get("snapshot_steps", DEFAULT_SNAPSHOTS)
    )
    initializations = (
        initializations
        if initializations is not None
        else cfg.get("initializations", ["planner", "expert_noise"])
    )
    seed = seed if seed is not None else cfg.get("seed", 2026)
    gamma = float(gamma if gamma is not None else config["refinement"]["gamma"])
    if (
        not initializations
        or len(set(initializations)) != len(initializations)
        or any(name not in ("planner", "expert_noise") for name in initializations)
    ):
        raise ValueError(
            "initializations must be a unique nonempty subset of planner, expert_noise"
        )
    future_dt = float(cfg.get("future_dt", 0.1))
    if not math.isfinite(future_dt) or future_dt <= 0:
        raise ValueError("future_dt must be finite and positive")
    directory = create_output(config, output, f"long_refinement_{split}")
    dataset, results = None, []
    try:
        device, model, state, dataset, _ = load_evaluation(config, checkpoint, split, max_samples)
        indices = representative_indices(dataset.records, max_samples, seed)
        records = [dataset.records[index] for index in indices]
        if any("feature" not in record for record in records) and "manifest" in config.get(
            "paths", {}
        ):
            path = resolve_path(config, "manifest")
            if path.is_file():
                manifest = {row["token"]: row for row in read_json(path)}
                records = [{**manifest.get(row["token"], {}), **row} for row in records]
        planner = planner_args = None
        if "planner" in initializations:
            if planner_identity(config) != state["planner"]:
                raise ValueError(
                    "Planner initialization requires checkpoint's exact frozen planner"
                )
            planner, planner_args = load_frozen_planner(config, device)
            if ego_normalizer_metadata(planner_args) != dataset.metadata["ego_normalizer"]:
                raise ValueError("Planner/cache normalizers differ")
        sigma = float(state["config"]["training"]["sigma"])
        normalizer = dataset.metadata["ego_normalizer"]
        provenance = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": file_hash(checkpoint),
            "cache_sha256": dataset.sha256,
            "planner": state["planner"],
            "split": split,
            "selection": "Seeded round-robin by scenario_type, fallback recording; no metric-based selection.",
            "selection_seed": seed,
            "noise_seed": seed,
            "noise_namespace": "long_refinement_expert_noise",
            "reference_seed": config["reference_seed"],
            "initializations": list(initializations),
            "requested_steps": steps,
            "gamma": gamma,
            "sigma": sigma,
            "future_dt": future_dt,
            "snapshot_steps": list(snapshot_steps),
            "records": records,
        }
        atomic_write(directory / "provenance.json", provenance)
        for number, (index, record) in enumerate(zip(indices, records)):
            batch = {
                key: value.unsqueeze(0).to(device)
                for key, value in dataset[index].items()
                if torch.is_tensor(value)
            }
            target, scene, route = batch["target"], batch["context"], batch["route"]
            condition = neighbor_kwargs(batch)
            map_data, map_note = _load_map(record, required=planner is not None)
            neighbors = neighbor_valid = None
            initial = {}
            if planner is not None:
                inputs, raw_target = read_feature_batch([record], planner_args, device)
                # Float32 sin/cos and normalization may differ by a few ULPs
                # between the GPU that built the cache and CPU diagnosis. Raw
                # NPZ provenance is verified separately by _load_map above.
                torch.testing.assert_close(
                    raw_target,
                    target,
                    rtol=1e-5,
                    atol=1e-6,
                    msg="Raw and cached ego targets differ beyond cross-device float32 tolerance",
                )
                encoding, prediction = predict_candidates(
                    planner,
                    planner_args,
                    inputs,
                    [record["start_time_us"]],
                    config["reference_seed"],
                )
                scene = encoding["encoding"]
                route = planner.decoder.decoder.dit.route_encoder(inputs["route_lanes"])
                initial["planner"] = normalize_ego_future(
                    prediction["prediction"][:, 0], planner_args.state_normalizer
                )
                neighbors = prediction["prediction"][0, 1:].cpu().numpy()
                if "neighbor_agents_past" in inputs:
                    neighbor_valid = (
                        inputs["neighbor_agents_past"][0, : len(neighbors), -1, :4]
                        .ne(0)
                        .any(-1)
                        .cpu()
                        .numpy()
                    )
                if getattr(model, "uses_neighbor_future", False):
                    condition = prediction_neighbors(
                        prediction["prediction"], inputs, planner_args.state_normalizer
                    )
            if "expert_noise" in initializations:
                noise = sample_noise(
                    tuple(target.shape[1:]),
                    [record["token"]],
                    seed,
                    "long_refinement_expert_noise",
                    dtype=target.dtype,
                ).to(device)
                initial["expert_noise"] = target + sigma * noise
            # Store the actual fixed tensors; both starts share them when DP is loaded.
            sample_dir = directory / f"sample_{number:03d}"
            sample_dir.mkdir()
            fixed = {"context": scene, "route": route, **condition}
            np.savez_compressed(
                sample_dir / "condition.npz",
                **{key: value.detach().cpu().numpy() for key, value in fixed.items()},
            )
            sample_provenance = {
                "record": record,
                "conditioning_sha256": {key: _tensor_hash(value) for key, value in fixed.items()},
                "map_note": map_note,
                "conditioning_source": "fresh frozen DP evaluation; shared by both initializations"
                if planner is not None
                else "saved cache",
            }
            atomic_write(sample_dir / "provenance.json", sample_provenance)
            for name in initializations:
                stem = sample_dir / name
                with Progress(
                    f"Refine {number + 1}/{len(indices)} {name}", total=steps, unit="step"
                ) as display:
                    result, scalars, snapshots = trace_refinement(
                        model,
                        initial[name],
                        target,
                        scene,
                        route,
                        normalizer,
                        sigma,
                        gamma,
                        steps,
                        heading_projection=bool(config["refinement"]["heading_projection"]),
                        snapshot_steps=snapshot_steps,
                        convergence_window=int(cfg.get("convergence_window", 50)),
                        convergence_step_m=float(cfg.get("convergence_step_m", 1e-4)),
                        scalar_path=stem.with_suffix(".csv"),
                        progress=display,
                        **condition,
                    )
                keys = sorted(snapshots)
                np.savez_compressed(
                    stem.with_suffix(".npz"),
                    snapshot_steps=np.array(keys),
                    trajectories_normalized=np.stack([snapshots[key] for key in keys]),
                    trajectories_physical=np.stack(
                        [snapshots[key].astype(np.float64) for key in keys]
                    )
                    * np.asarray(normalizer["std"])
                    + np.asarray(normalizer["mean"]),
                    target_normalized=target[0].cpu().numpy(),
                    target_physical=to_physical(target[0].double(), normalizer).cpu().numpy(),
                    normalizer_mean=np.asarray(normalizer["mean"]),
                    normalizer_std=np.asarray(normalizer["std"]),
                    future_dt=np.asarray(future_dt),
                )
                result.update(
                    token=record["token"],
                    initialization=name,
                    initial_sha256=_tensor_hash(initial[name]),
                    artifacts=str(stem.relative_to(directory)),
                    map_note=map_note,
                )
                atomic_write(stem.with_suffix(".json"), result)
                plot_long_refinement(
                    stem,
                    target[0].cpu().numpy(),
                    snapshots,
                    scalars,
                    SCALAR_COLUMNS,
                    normalizer,
                    future_dt,
                    map_data,
                    neighbors,
                    neighbor_valid,
                    title=f"{record['token']} | {name}",
                    failure=result["failure"],
                )
                results.append(result)
                atomic_write(
                    directory / "status.json",
                    {
                        "state": "running",
                        "completed_paths": len(results),
                        "total_paths": len(indices) * len(initializations),
                    },
                )
        summary = {
            "kind": "long_fixed_condition_offline",
            "samples": len(indices),
            "paths": len(results),
            "numerical_failures": sum(row["failure"] is not None for row in results),
            "results": results,
            "provenance": provenance,
            "interpretation": [
                "Each path fixes scene, route and predicted neighbors; only ego trajectory changes.",
                "No nuPlan simulation was run. ADE/FDE are imitation diagnostics, not closed-loop scores.",
                "The full step budget is attempted; only numerical failure stops a path.",
                "Only sparse trajectories are stored; scalar CSV diagnostics cover every completed step.",
                "Energy is learned unnormalized energy, not a density/probability estimate.",
            ],
        }
        atomic_write(directory / "summary.json", summary)
        cards = []
        for row in results:
            stem = html.escape(row["artifacts"])
            label = html.escape(
                f"{row['token']} / {row['initialization']} / {row['completed_steps']} steps"
            )
            cards.append(
                f'<h2>{label}</h2><p><a href="{stem}.json">Summary</a> · <a href="{stem}.csv">Per-step CSV</a> · <a href="{stem}.npz">Sparse snapshots</a></p><img src="{stem}_trajectories.png"><img src="{stem}_diagnostics.png">'
            )
        (directory / "report.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Long refinement diagnostics</title><style>body{font:16px sans-serif;margin:2rem;max-width:1300px}img{width:100%}</style><h1>Fixed-condition refinement diagnostics</h1><p>Offline trajectory behavior; convergence does not certify planning quality. Each path attempts its full iteration budget. See JSON for failures and last finite step.</p>'
            + "".join(cards),
            encoding="utf-8",
        )
        atomic_write(
            directory / "status.json",
            {
                "state": "complete_with_failures" if summary["numerical_failures"] else "complete",
                "paths": len(results),
                "numerical_failures": summary["numerical_failures"],
            },
        )
        return summary
    except Exception as error:
        atomic_write(
            directory / "status.json",
            {
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
                "completed_paths": len(results),
            },
        )
        raise
    finally:
        if dataset is not None:
            dataset.close()
