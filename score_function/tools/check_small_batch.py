"""Small, fixed-corruption fitting check. Weights are discarded before actual training."""

from pathlib import Path

import torch

from score_function.loss import fixed_sigma_dsm_loss
from score_function.model.score_branch import build_model
from score_function.utils.dataset import ShardedDataset, collate_cpu, device_batch
from score_function.utils.neighbor import neighbor_kwargs
from score_function.utils.train_utils import atomic_write, file_hash, resolve_path, source_hashes


def run_smoke(config, device_override=None):
    device = torch.device(device_override or config["runtime"]["device"])
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.manual_seed(config["training"]["seed"])
    dataset = ShardedDataset(
        config["cache"],
        "train",
        neighbor_index=(
            resolve_path(config, "neighbor_cache")
            if config["model"].get("neighbor_future")
            else None
        ),
    )
    settings = config["smoke"]
    count = min(settings["frames"], len(dataset))
    batch = device_batch(collate_cpu([dataset[i] for i in range(count)]), device)
    dataset.close()
    model = build_model(config, device)
    sigma, noise = config["training"]["sigma"], torch.randn_like(batch["target"])
    noisy = batch["target"] + sigma * noise
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=0)

    def objective():
        score = model(noisy, batch["context"], batch["route"], **neighbor_kwargs(batch))
        return fixed_sigma_dsm_loss(score, noise, sigma)

    model.eval()
    with torch.no_grad():
        initial = objective().item()
    history = []
    for step in range(settings["updates"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite smoke DSM")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        if (step + 1) % 25 == 0:
            history.append({"step": step + 1, "fixed_corruption_train_dsm": loss.item()})
            print(history[-1], flush=True)
    model.eval()
    with torch.no_grad():
        final = objective().item()
    result = {
        "state": "passed" if final < initial else "failed",
        "frames": count,
        "updates": settings["updates"],
        "initial_dsm": initial,
        "final_dsm": final,
        "history": history,
        "cache_sha256": file_hash(config["cache"]),
        "sources": source_hashes(),
        "meaning": "Fixed-noise optimization smoke check only; not generalization or score accuracy",
    }
    atomic_write(Path(config["output"]) / "smoke.json", result)
    if not final < initial:
        raise RuntimeError("Smoke objective did not decrease; full training must not start")
    return result
