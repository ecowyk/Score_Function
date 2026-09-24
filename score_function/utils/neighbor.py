"""Shared neighbor normalization and argument packing for train and inference."""

import torch


def neighbor_kwargs(batch):
    if "neighbor_future" not in batch:
        return {}
    return {key: batch[key] for key in ("neighbor_future", "neighbor_valid")}


def prediction_neighbors(prediction, inputs, normalizer):
    """Use official per-agent normalization; ignore padded agent predictions."""
    if prediction.shape[1:] != (11, 80, 4):
        raise ValueError("Expected the official joint prediction [B,11,80,4]")
    current = inputs["neighbor_agents_past"][:, :10, -1, :4]
    valid = current.ne(0).any(dim=-1)
    future = (prediction[:, 1:] - normalizer.mean[1:].to(prediction)) / normalizer.std[1:].to(
        prediction
    )
    future = future.masked_fill(~valid[:, :, None, None], 0)
    if not torch.isfinite(future).all():
        raise FloatingPointError("Nonfinite predicted neighbor condition")
    return {"neighbor_future": future, "neighbor_valid": valid}
