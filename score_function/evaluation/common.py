"""Evaluation setup, timing, and per-frame trace extraction."""

from pathlib import Path

import torch

from score_function.utils.train_utils import atomic_write, source_hashes


def create_output(config, output, name):
    directory = Path(output) if output is not None else Path(config["output"]) / name
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write(directory / "config.json", config)
    atomic_write(directory / "source_hashes.json", source_hashes())
    atomic_write(directory / "status.json", {"state": "running", "stage": name})
    return directory


def load_evaluation(config, checkpoint, split, max_samples):
    from score_function.utils.checkpoint import load_selected
    from score_function.utils.config import configure_runtime
    from score_function.utils.dataset import ShardedDataset

    device = configure_runtime(
        config, require_cuda=str(config["runtime"]["device"]).startswith("cuda")
    )
    model, state = load_selected(config, checkpoint, device)
    model.eval()
    dataset = ShardedDataset(config["cache"], split)
    if dataset.metadata["planner"] != state["planner"]:
        dataset.close()
        raise ValueError("Evaluation cache and checkpoint use different frozen planners")
    if state.get("ego_normalizer") != dataset.metadata["ego_normalizer"]:
        dataset.close()
        raise ValueError("Evaluation cache and checkpoint use different ego normalizers")
    if max_samples is not None and (int(max_samples) != max_samples or max_samples < 1):
        dataset.close()
        raise ValueError("max_samples must be a positive integer")
    count = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    return device, model, state, dataset, list(range(count))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_frame_trace(trace, index):
    batched = {
        "trajectories",
        "scores",
        "score_norm",
        "displacement_norm",
        "step_displacement_norm",
        "heading_norm_deviation_pre_projection",
        "heading_norm_max_deviation_pre_projection",
    }
    return {
        key: [step[index] for step in value] if key in batched else value
        for key, value in trace.items()
    }
