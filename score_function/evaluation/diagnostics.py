"""Fixed-noise expert-trajectory diagnostics; not official nuPlan evaluation."""

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from score_function.evaluation.common import create_output, load_evaluation
from score_function.evaluation.metrics import (
    MetricAccumulator,
    finite_scalar,
    normalizer_statistics,
    one_step_recovery,
    score_diagnostics,
    summarize_cosines,
    trajectory_metrics,
)
from score_function.evaluation.reporting import write_diagnostic_report
from score_function.evaluation.visualization import plot_cosine, plot_trajectory
from score_function.utils.train_utils import (
    atomic_write,
    file_hash,
    sample_noise,
)


@torch.no_grad()
def evaluate(config, checkpoint, split="val", output=None, max_samples=None):
    """Run the Section 8 corruption suite; never launch planner simulation."""
    from score_function.utils.dataset import build_data_loader, device_batch

    directory = create_output(config, output, f"offline_{split}")
    dataset = None
    started = time.perf_counter()
    try:
        device, model, state, dataset, indices = load_evaluation(
            config, checkpoint, split, max_samples
        )
        normalizer = dataset.metadata["ego_normalizer"]
        normalizer_statistics(normalizer, device, torch.float32)
        settings = config.get("evaluation", {})
        sigma = float(state["config"]["training"]["sigma"])
        repeats = int(settings.get("noise_repeats", config["training"]["validation_repeats"]))
        seed = int(settings.get("seed", config["training"]["validation_seed"]))
        gammas = [float(value) for value in settings.get("gammas", [0.1, 0.25, 0.5, 1.0])]
        if 1.0 not in gammas:
            gammas.append(1.0)
        if repeats < 1 or len(set(gammas)) != len(gammas):
            raise ValueError("Positive repeats and unique recovery gammas required")
        if any(not math.isfinite(value) or value < 0 for value in gammas):
            raise ValueError("Recovery gammas must be finite and nonnegative")
        loader_settings = {
            **config["training"],
            "microbatch_size": int(settings.get("batch_size", 64)),
        }
        if loader_settings["microbatch_size"] < 1:
            raise ValueError("Evaluation batch size must be positive")
        diagnostic, baseline = MetricAccumulator(), MetricAccumulator()
        recovered = {value: MetricAccumulator() for value in gammas}
        histogram = np.zeros(40, dtype=np.int64)
        undefined, completed = 0, 0
        cosine_path = directory / "cosine.float64"
        plotted = 0
        plot_limit = int(settings.get("visualize_samples", 8))
        record_by_token = {dataset.records[i]["token"]: dataset.records[i] for i in indices}
        fields = ["token", "recording", "start_time_us", "split", "repeat", "sigma", "gamma"]
        fields += ["dsm", "cosine", "score_l2", "score_point_l2_mean", "score_d1", "score_d2"]
        metric_names = [
            "mse_normalized",
            "ade_m",
            "fde_m",
            "heading_mae_rad",
            "heading_undefined_points",
        ]
        fields += [f"{prefix}_{key}" for prefix in ("before", "after") for key in metric_names]
        fields += [f"improved_{key}" for key in metric_names[:-1]]
        with (
            cosine_path.open("wb") as cosine_stream,
            (directory / "per_sample.csv").open("w", encoding="utf-8", newline="") as csv_stream,
            (directory / "per_sample.jsonl").open("w", encoding="utf-8") as json_stream,
        ):
            writer = csv.DictWriter(csv_stream, fieldnames=fields)
            writer.writeheader()
            for cpu_batch in build_data_loader(dataset, loader_settings, indices=indices):
                batch = device_batch(cpu_batch, device)
                for repeat in range(repeats):
                    noise = sample_noise(
                        (80, 4), batch["tokens"], seed, f"clean_validation_{repeat}"
                    ).to(device)
                    noisy = batch["target"] + sigma * noise
                    score = model(noisy, batch["context"], batch["route"])
                    diagnostics = score_diagnostics(score, noise, sigma)
                    before = trajectory_metrics(noisy, batch["target"], normalizer)
                    diagnostics_cpu = {
                        key: value.cpu().tolist() for key, value in diagnostics.items()
                    }
                    before_cpu = {key: value.cpu().tolist() for key, value in before.items()}
                    diagnostic.add(diagnostics)
                    baseline.add(before)
                    cosine = diagnostics["cosine"].double().cpu().numpy()
                    valid = np.isfinite(cosine)
                    cosine[valid].tofile(cosine_stream)
                    histogram += np.histogram(cosine[valid], bins=40, range=(-1, 1))[0]
                    undefined += int((~valid).sum())
                    for gamma in gammas:
                        refined, after = one_step_recovery(
                            noisy, batch["target"], score, sigma, gamma, normalizer
                        )
                        improvements = {
                            f"improved_{name}": (after[name] < before[name])
                            .float()
                            .masked_fill(
                                ~torch.isfinite(after[name]) | ~torch.isfinite(before[name]),
                                float("nan"),
                            )
                            for name in metric_names[:-1]
                        }
                        recovered[gamma].add({**after, **improvements})
                        after_cpu = {key: value.cpu().tolist() for key, value in after.items()}
                        improvements_cpu = {
                            key: value.cpu().tolist() for key, value in improvements.items()
                        }
                        for index, token in enumerate(batch["tokens"]):
                            record = record_by_token[token]
                            row = {
                                "token": token,
                                "recording": record["recording"],
                                "start_time_us": record["start_time_us"],
                                "split": split,
                                "repeat": repeat,
                                "sigma": sigma,
                                "gamma": gamma,
                            }
                            row.update(
                                {
                                    key: finite_scalar(value[index])
                                    for key, value in diagnostics_cpu.items()
                                }
                            )
                            for prefix, values in (("before", before_cpu), ("after", after_cpu)):
                                row.update(
                                    {
                                        f"{prefix}_{key}": finite_scalar(value[index])
                                        for key, value in values.items()
                                    }
                                )
                            row.update(
                                {
                                    key: finite_scalar(value[index])
                                    for key, value in improvements_cpu.items()
                                }
                            )
                            writer.writerow(row)
                            json_stream.write(json.dumps(row, allow_nan=False) + "\n")
                            if gamma == 1.0 and repeat == 0 and plotted < plot_limit:
                                stem = f"trajectory_{plotted:03d}"
                                plot_trajectory(
                                    directory / f"{stem}.png",
                                    batch["target"][index],
                                    noisy[index],
                                    refined[index],
                                    sigma**2 * score[index],
                                    normalizer,
                                    f"{token}: unprojected gamma=1 recovery",
                                )
                                np.savez_compressed(
                                    directory / f"{stem}.npz",
                                    target=batch["target"][index].cpu().numpy(),
                                    noisy=noisy[index].cpu().numpy(),
                                    refined=refined[index].cpu().numpy(),
                                    score=score[index].cpu().numpy(),
                                    token=np.asarray(token),
                                )
                                plotted += 1
                    completed += len(batch["tokens"])
                atomic_write(
                    directory / "status.json",
                    {
                        "state": "running",
                        "completed_frame_noise_pairs": completed,
                        "total_frame_noise_pairs": len(indices) * repeats,
                    },
                )
        cosine = summarize_cosines(cosine_path, histogram, undefined)
        plot_cosine(directory / "cosine_histogram.png", cosine)
        summary = {
            "schema_version": 1,
            "kind": "synthetic_corruption_offline",
            "split": split,
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": file_hash(checkpoint),
            "cache_sha256": dataset.sha256,
            "samples": len(indices),
            "available_samples": len(dataset),
            "selection": "first cache frames; no metric-based sample selection",
            "noise_repeats": repeats,
            "noise_seed": seed,
            "noise_namespace": "clean_validation_{repeat}",
            "frame_noise_pairs": completed,
            "sigma": sigma,
            "denormalized_noise_std_xy_cos_sin": [sigma * float(x) for x in normalizer["std"]],
            "heading_projection": False,
            "diagnostics": diagnostic.result(),
            "cosine": cosine,
            "corrupted_baseline": baseline.result(),
            "recovery": {str(gamma): recovered[gamma].result() for gamma in gammas},
            "tweedie": {
                "gamma": 1.0,
                "heading_projection": False,
                "metrics": recovered[1.0].result(),
            },
            "numerical_failures": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "interpretation": [
                "Cosine target is sampled -epsilon/sigma, not the true marginal driving score.",
                "Gamma=1 is an unprojected Tweedie-style denoised estimate, not a local-peak test.",
                "These are offline diagnostics, not nuPlan closed-loop planning scores.",
                "Repeated corruptions are not independent driving scenes; samples counts unique frames.",
            ],
        }
        atomic_write(directory / "summary.json", summary)
        write_diagnostic_report(directory / "report.md", summary)
        atomic_write(
            directory / "status.json",
            {
                "state": "complete",
                "samples": len(indices),
                "frame_noise_pairs": completed,
                "elapsed_seconds": summary["elapsed_seconds"],
            },
        )
        return summary
    except Exception as error:
        atomic_write(
            directory / "status.json",
            {
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
                "numerical_failure": isinstance(error, FloatingPointError),
            },
        )
        raise
    finally:
        if dataset is not None:
            dataset.close()
