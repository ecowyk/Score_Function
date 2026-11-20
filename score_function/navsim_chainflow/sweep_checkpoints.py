"""Select immutable best and last-trained EMA checkpoints for a paired DD evaluation."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math
import os
from pathlib import Path
import shutil
import tempfile

from .assets import sha256


def _expected_config(base, group):
    cfg = copy.deepcopy(base)
    fields = group.split("_")
    if "lr" in fields:
        cfg["learning_rate"] = 1e-4
    if "wd" in fields:
        cfg["weight_decay"] = .05
    if "dropout" in fields:
        cfg["model"]["dropout"] = .2
    return cfg


def _same_state(left, right):
    import torch
    return (left.keys() == right.keys()
            and all(left[k].dtype == right[k].dtype and torch.equal(left[k], right[k]) for k in left))


def _snapshot(directory, label, source, checkpoint):
    """Use a stream so torch gives the zip archive a stable name regardless of temp filename."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            if source is None:
                torch.save(checkpoint, stream)
            else:
                source.seek(0)
                shutil.copyfileobj(source, stream)
        checksum = sha256(temporary)
        destination = directory / f"{label}-e{checkpoint['epoch']:03d}-{checksum}.pt"
        os.replace(temporary, destination)
        return {"path": str(destination.resolve()), "sha256": checksum, "epoch": checkpoint["epoch"],
                "selection": label, "val_dsm": checkpoint["val_dsm"],
                "training_signature": checkpoint["training_signature"]}
    finally:
        temporary.unlink(missing_ok=True)


def prepare_checkpoints(plan, sweep_dir: Path, output: Path):
    """Preserve the original file hash for portable exports and materialize interrupted EMAs."""
    from score_function.utils.train_utils import fingerprint, load_tensor, read_json
    from .runtime import CHECKPOINT_FORMAT
    from .train import TRAIN_FORMAT

    root = Path(plan["paths"]["training"])
    base = plan["config"]
    if (base["global_batch_size"], base["epochs"], base["learning_rate"], base["weight_decay"],
            base["model"]["dropout"], base["sigma"], base["gamma"]) != (64, 40, 2e-4, .01, .1, .1, .1):
        raise ValueError("Expected the existing B64/E40 reference and its original sweep factors")
    ref_run = read_json(root / "run.json")
    ref_marker = read_json(root / "TRAINING_COMPLETE.json")
    ref_final = root / "final_ema.pt"
    if (ref_marker["signature"] != ref_run["signature"] or ref_marker["epochs"] != base["epochs"]
            or sha256(ref_final) != ref_marker["final_ema_sha256"]):
        raise ValueError("Reference final EMA differs from its completion record")
    ref_export = load_tensor(ref_final)
    cache_id = ref_run["protocol"]["cache_fingerprint"]
    if (ref_export["training_signature"] != ref_run["signature"] or ref_export["cache_fingerprint"] != cache_id
            or ref_export["format"] != CHECKPOINT_FORMAT or ref_export["config"] != base):
        raise ValueError("Reference final EMA has unexpected training provenance")

    outputs = {}
    for group in ("reference", "lr", "wd", "dropout", "lr_wd_dropout"):
        training = root if group == "reference" else Path(sweep_dir) / group / "training"
        run = read_json(training / "run.json")
        config = run["config"]
        expected_config = _expected_config(base, group)
        expected_config["num_workers"] = config["num_workers"]
        protocol = run["protocol"]
        signature = run["signature"]
        if (signature != fingerprint(protocol) or config != expected_config
                or protocol["config"] != {k: v for k, v in config.items() if k != "num_workers"}
                or protocol["cache_fingerprint"] != cache_id):
            raise ValueError(f"{group}: config, training signature or cache differs from expected sweep")
        last = load_tensor(training / "last.pt")
        if last["format"] != TRAIN_FORMAT or last["signature"] != signature:
            raise ValueError(f"{group}: last.pt belongs to a different training run")
        history = last["history"]
        epoch = len(history)
        if (not 0 < epoch <= config["epochs"] or [r["epoch"] for r in history] != list(range(1, epoch + 1))
                or last["step"] != epoch * run["updates_per_epoch"]
                or not all(math.isfinite(r["val_dsm_ema"]) for r in history)):
            raise ValueError(f"{group}: inconsistent training history in last.pt")
        best_row = min(history, key=lambda row: row["val_dsm_ema"])
        best = last["best"]["ema"]
        if (best["epoch"], best["val_dsm"]) != (best_row["epoch"], best_row["val_dsm_ema"]):
            raise ValueError(f"{group}: best EMA metadata disagrees with training history")
        marker_path = training / "TRAINING_COMPLETE.json"
        final_path = training / "final_ema.pt"
        marker = read_json(marker_path) if marker_path.is_file() else None
        if marker is not None:
            if (marker["signature"] != signature or marker["epochs"] != epoch or epoch != config["epochs"]
                    or sha256(final_path) != marker["final_ema_sha256"]):
                raise ValueError(f"{group}: final EMA differs from its completion record")
        if epoch < config["epochs"] and final_path.exists():
            raise ValueError(f"{group}: final_ema.pt exists before the last planned epoch")
        common = dict(format=CHECKPOINT_FORMAT, model=config["model"], normalizer=ref_export["normalizer"],
                      conditioner_identity=ref_export["conditioner_identity"], sigma=config["sigma"], gamma=config["gamma"],
                      training_precision=config["precision"], config=config, cache_fingerprint=cache_id,
                      training_signature=signature, weight_kind="ema")
        for kind, row, state in (("best", best_row, best["state_dict"]), ("final", history[-1], last["ema"])):
            label = "last_available" if kind == "final" and epoch < config["epochs"] else kind
            path = training / f"{kind}_ema.pt"
            expected = dict(common, epoch=row["epoch"], val_dsm=row["val_dsm_ema"])
            with path.open("rb") if path.exists() else nullcontext(None) as source:
                if source is not None:
                    portable = load_tensor(source)
                    if any(portable.get(k) != v for k, v in common.items()):
                        raise ValueError(f"{group}: {path.name} belongs to a different run")
                    if (portable.get("epoch"), portable.get("val_dsm")) == (expected["epoch"], expected["val_dsm"]):
                        if not _same_state(portable["state_dict"], state):
                            raise ValueError(f"{group}: {path.name} weights disagree with last.pt")
                    elif kind == "best" and marker is None and portable["epoch"] < best_row["epoch"]:
                        # The trainer commits last.pt before the export. An interrupt can leave an older best.
                        source = None
                    else:
                        raise ValueError(f"{group}: {path.name} epoch or validation DSM disagrees with last.pt")
                record = _snapshot(Path(output) / "checkpoints" / group, label, source, dict(expected, state_dict=state))
            if kind == "final" and marker is not None and record["sha256"] != marker["final_ema_sha256"]:
                raise ValueError(f"{group}: final EMA changed while preparing its snapshot")
            outputs[f"{group}/{kind}"] = record
    return outputs
