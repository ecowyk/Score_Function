"""Fixed-sigma DSM learning and fixed-noise validation for one epoch."""

from contextlib import nullcontext

import torch

from score_function.loss import fixed_sigma_dsm_loss
from score_function.utils import ddp
from score_function.utils.dataset import build_data_loader, device_batch
from score_function.utils.ema import update_ema
from score_function.utils.train_utils import sample_noise


@torch.no_grad()
def validate_epoch(models, dataset, cfg, device):
    """No padding/duplicate validation frames; all ranks reduce sums and counts."""
    for model in models.values():
        model.eval()
    totals = torch.zeros(len(models) + 1, dtype=torch.float64, device=device)
    batches = build_data_loader(
        dataset, cfg, indices=list(range(ddp.rank(), len(dataset), ddp.world_size()))
    )
    for cpu_batch in batches:
        batch = device_batch(cpu_batch, device)
        for repeat in range(cfg["validation_repeats"]):
            noise = sample_noise(
                (80, 4), batch["tokens"], cfg["validation_seed"], f"clean_validation_{repeat}"
            ).to(device)
            noisy = batch["target"] + cfg["sigma"] * noise
            for index, model in enumerate(models.values()):
                score = model(noisy, batch["context"], batch["route"])
                error = (cfg["sigma"] * score + noise).square()
                totals[index] += error.double().sum()
            totals[-1] += noise.numel()
    ddp.sum_tensor(totals)
    if not torch.isfinite(totals).all() or totals[-1] <= 0:
        raise FloatingPointError("Invalid/empty validation")
    return {name: (totals[i] / totals[-1]).item() for i, name in enumerate(models)}


def train_epoch(
    data_loader,
    model,
    optimizer,
    cfg,
    ema,
    device,
    *,
    start_step,
    start_update,
    updates_per_epoch,
    accumulation,
    next_lr,
    distributed_model=None,
):
    """Yield each completed optimizer update so the driver can save exact progress.

    The scientific loop is explicit: perturb the expert trajectory, predict its
    score, minimize DSM, clip gradients, update AdamW, and update EMA.
    """
    wrapped = distributed_model if distributed_model is not None else model
    averaged = ema
    world = ddp.world_size()
    batches = iter(data_loader)
    model.train()
    step = start_step
    for update in range(start_update, updates_per_epoch):
        warmup_updates = cfg["warmup_epochs"] * updates_per_epoch
        fraction = min(1.0, step / max(1, warmup_updates - 1))
        lr = (
            cfg["warmup_learning_rate"]
            + fraction * (cfg["learning_rate"] - cfg["warmup_learning_rate"])
            if step < warmup_updates
            else next_lr
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for micro in range(accumulation):
            batch = device_batch(next(batches), device)
            noise = torch.randn_like(batch["target"])
            sync = wrapped.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
            with sync:
                score = wrapped(
                    batch["target"] + cfg["sigma"] * noise, batch["context"], batch["route"]
                )
                loss = fixed_sigma_dsm_loss(score, noise, cfg["sigma"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training DSM")
                (loss / accumulation).backward()
            update_loss += loss.item() / accumulation
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        step += 1
        decay = (
            min(cfg["ema_decay"], (1 + step) / (10 + step))
            if cfg["ema_warmup"]
            else cfg["ema_decay"]
        )
        update_ema(averaged, model, decay)
        yield {"step": step, "cursor": update + 1, "train_dsm": update_loss, "lr": lr}
