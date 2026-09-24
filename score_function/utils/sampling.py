"""Paired evaluation seeding; trajectory generation remains the official forward."""

import hashlib

import torch


def planner_seed(timestamp, seed):
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{int(timestamp)}".encode()).digest()[:8], "big"
    ) % (2**63 - 1)


@torch.no_grad()
def predict_candidates(model, args, inputs, timestamps, seed):
    """Use one seed per frame, independent of the offline evaluation batch size."""
    encodings, predictions = [], []
    device = inputs["ego_current_state"].device
    devices = [device.index] if device.type == "cuda" else []
    for index, timestamp in enumerate(timestamps):
        one = {key: value[index : index + 1] for key, value in inputs.items()}
        # Official observation_adapter uses this local-frame current-state placeholder.
        # It is not an ascent variable and is not consumed by the scene encoder.
        current = one["ego_current_state"].new_tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        one["ego_current_state"] = args.observation_normalizer({"ego_current_state": current})[
            "ego_current_state"
        ]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(planner_seed(timestamp, seed))
            encoded, output = model(one)
        encodings.append(encoded["encoding"])
        predictions.append(output["prediction"])
    return {"encoding": torch.cat(encodings)}, {"prediction": torch.cat(predictions)}
