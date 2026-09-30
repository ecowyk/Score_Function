"""Explicit update budgets and learning-rate semantics for comparable training."""

import math


def training_budget(cfg, train_samples):
    """Round the update floor up to whole epochs using the actual training split.

    Incomplete global batches are dropped by EpochBatchSampler. A configured
    maximum is extended when necessary to meet the explicit update floor.
    Legacy configurations without the new floor keep their original maximum.
    """
    updates = train_samples // cfg["batch_size"]
    if updates < 1:
        raise ValueError("Training requires at least one complete global batch")
    minimum_updates = cfg.get("minimum_updates", 0)
    planned_epochs = max(cfg["max_epochs"], math.ceil(minimum_updates / updates))
    planned_updates = planned_epochs * updates
    return {
        "configured_max_epochs": cfg["max_epochs"],
        "minimum_epochs": cfg["minimum_epochs"],
        "minimum_optimizer_updates": minimum_updates,
        "planned_epochs": planned_epochs,
        "updates_per_epoch": updates,
        "planned_optimizer_updates": planned_updates,
        "planned_sample_presentations": planned_updates * cfg["batch_size"],
        "samples_per_epoch": updates * cfg["batch_size"],
        "dropped_samples_per_epoch": train_samples % cfg["batch_size"],
    }


def early_stop_allowed(cfg, epoch, step):
    """An optional plateau stop can never bypass either minimum budget."""
    return (
        cfg.get("early_stopping", True)
        and epoch >= cfg["minimum_epochs"]
        and step >= cfg.get("minimum_updates", 0)
    )


def learning_rate_at_step(cfg, step, updates_per_epoch, next_lr):
    """Official epoch-wise linear warmup, or the unchanged legacy update warmup.

    DP's LinearLR starts at 0.1 * base LR and reaches base LR at the beginning
    of warmup epoch 5 (zero-based epoch 4), then stays constant. We calculate
    directly from the completed update count so mid-epoch resume is exact.
    """
    if cfg.get("lr_schedule", "validation_plateau") == "constant_after_warmup":
        epoch = step // updates_per_epoch
        warmup = cfg["warmup_epochs"]
        if warmup == 0 or epoch >= warmup:
            return cfg["learning_rate"]
        fraction = min(1.0, epoch / max(1, warmup - 1))
        return cfg["warmup_learning_rate"] + fraction * (
            cfg["learning_rate"] - cfg["warmup_learning_rate"]
        )
    warmup_updates = cfg["warmup_epochs"] * updates_per_epoch
    fraction = min(1.0, step / max(1, warmup_updates - 1))
    return (
        cfg["warmup_learning_rate"]
        + fraction * (cfg["learning_rate"] - cfg["warmup_learning_rate"])
        if step < warmup_updates
        else next_lr
    )
