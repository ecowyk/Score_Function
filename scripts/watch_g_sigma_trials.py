#!/usr/bin/env python3
"""Read-only terminal progress for the three fixed-sigma G trials; no CUDA imports."""

import argparse
import json
import math
import time
from pathlib import Path

from tqdm import tqdm


def read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def hours(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "?"
    return f"{max(0, seconds) / 3600:.1f}h"


class ProgressWatch:
    """Track observed progress using the reader's monotonic clock, never NFS mtime.

    Before the first observed change, watch=... reports time watching this value;
    it makes no claim about how fresh that value was before this viewer started.
    """

    def __init__(self):
        self.previous = {}

    def text(self, name, marker, now=None):
        now = time.monotonic() if now is None else now
        if name not in self.previous:
            self.previous[name] = (marker, now, False)
        old_marker, changed_at, observed = self.previous[name]
        if marker != old_marker:
            changed_at, observed = now, True
            self.previous[name] = (marker, changed_at, observed)
        label = "quiet" if observed else "watch"
        return f"{label}={max(0, now - changed_at):.0f}s"


def snapshot(root, name):
    directory = root / name / "score"
    path = directory / "status.json"
    status = read(path, {})
    if not status:
        return {
            "name": name,
            "total": 1,
            "n": 0,
            "label": f"{name} waiting",
            "detail": "no status.json",
            "marker": None,
        }
    budget = status.get("training_budget") or read(directory / "model_info.json", {}).get(
        "training_budget", {}
    )
    updates, epochs = budget.get("updates_per_epoch", 0), budget.get("planned_epochs", 0)
    step = status.get("step", 0)
    phase = status.get("phase", status.get("state", "?"))
    state = status.get("state", "?")
    if updates:
        finished_epoch = phase == "validation" or state == "complete"
        n = updates if finished_epoch else step % updates
        epoch = step // updates + (0 if finished_epoch else 1)
    else:
        n, epoch = 0, status.get("epoch", 0)
    history = read(directory / "history.json", [])
    # Use completed-epoch elapsed differences, including validation/checkpoint gaps.
    rate = None
    if len(history) >= 2:
        before, after = history[max(0, len(history) - 4)], history[-1]
        delta_t = after.get("elapsed_s", 0) - before.get("elapsed_s", 0)
        delta_n = after.get("step", 0) - before.get("step", 0)
        if delta_t > 0 and delta_n > 0:
            rate = delta_n / delta_t
    planned = budget.get("planned_optimizer_updates", 0)
    remaining = max(0, planned - step) / rate if rate else None
    if state == "complete":
        remaining = 0
    loss = status.get("train_dsm")
    if loss is None and history:
        loss = history[-1].get("train_dsm")
    loss_text = f"{loss:.5f}" if isinstance(loss, (int, float)) else "?"
    val = history[-1].get("val_dsm_ema") if history else None
    val_text = f"{val:.5f}" if isinstance(val, (int, float)) else "?"
    lr = status.get("lr", history[-1].get("lr") if history else None)
    lr_text = f"{lr:.2g}" if isinstance(lr, (int, float)) else "?"
    speed = f"{1 / rate:.2f}s/upd" if rate else "speed=?"
    if state == "failed":
        detail = str(status.get("error", "See launcher log")).replace("\n", " ")[:160]
    else:
        detail = (
            f"{phase} loss={loss_text} valEMA={val_text} lr={lr_text} "
            f"step={step} {speed} totalETA={hours(remaining)}"
        )
    return dict(
        name=name,
        total=max(1, updates),
        n=n,
        label=f"{name} epoch {epoch}/{epochs or '?'} [{state}]",
        detail=detail,
        marker=(state, phase, status.get("epoch"), status.get("step"), status.get("elapsed_s")),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/data/wyk/outputs/score_aligned_1004/G_sigma_1012_8gpu_v1/runs"),
    )
    parser.add_argument("--names", nargs="+", default=["S05-G_plus100", "S10-G", "S025-G"])
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--once", action="store_true", help="Print a snapshot without refreshing")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0.5:
        parser.error("--interval must be at least 0.5 seconds")
    bars = []
    watcher = ProgressWatch()
    print(
        "Read-only viewer. Ctrl+C closes this viewer only. "
        "Progress follows saved status (normally every 25 updates).",
        flush=True,
    )
    print(
        "watch=time watching before the first change; quiet=time since an observed progress change. "
        "ETA uses past completed epochs, not a liveness test.",
        flush=True,
    )
    try:
        while True:
            rows = [snapshot(args.root, name) for name in args.names]
            if args.once:
                for row in rows:
                    print(f"{row['label']} {row['n']}/{row['total']} | {row['detail']}")
                return
            if not bars:
                bars = [
                    tqdm(
                        total=row["total"],
                        position=i,
                        dynamic_ncols=True,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
                    )
                    for i, row in enumerate(rows)
                ]
            for bar, row in zip(bars, rows):
                bar.total, bar.n = row["total"], row["n"]
                bar.set_description_str(row["label"], refresh=False)
                observed = watcher.text(row["name"], row["marker"])
                bar.set_postfix_str(row["detail"] + " " + observed, refresh=False)
                bar.refresh()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        for bar in reversed(bars):
            bar.close()


if __name__ == "__main__":
    main()

