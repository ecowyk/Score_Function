"""Rank-zero console progress; display settings never enter the training protocol."""

import os
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout

from tqdm import tqdm

from score_function.utils import ddp


def report(message):
    if ddp.rank() == 0:
        tqdm.write(str(message), file=sys.stdout)
        sys.stdout.flush()


class Progress:
    """TTY bars, or periodic plain lines suitable for tee and batch-job logs.

    Rates/ETA describe this phase on rank zero, excluding work before a resume.
    Updating this display never reads a CUDA tensor or consumes random numbers.
    """

    def __init__(self, description, total=None, initial=0, unit="update"):
        self.description, self.total, self.initial = description, total, initial
        self.current, self.unit = initial, unit
        self.started = self.last_report = time.monotonic()
        self.metrics = {}
        mode = os.environ.get("SCORE_FUNCTION_PROGRESS", "auto")
        self.enabled = ddp.rank() == 0
        interactive = mode == "on" or (mode == "auto" and sys.stderr.isatty())
        self.bar = tqdm(
            total=total,
            initial=initial,
            desc=description,
            unit=unit,
            disable=not (self.enabled and interactive),
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stderr,
        )
        if self.enabled:
            report(f"[{description}] starting ({initial}/{total if total is not None else '?'})")

    def __enter__(self):
        return self

    def __call__(self, current, total=None, **metrics):
        if not self.enabled:
            return
        if total is not None:
            self.total = self.bar.total = total
        self.metrics.update(metrics)
        self.bar.set_postfix(self.metrics, refresh=False)
        self.bar.update(current - self.current)
        self.current = current
        if self.bar.disable and time.monotonic() - self.last_report >= 30:
            self._line("progress")

    def _line(self, state):
        now = time.monotonic()
        elapsed = now - self.started
        completed = self.current - self.initial
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if self.total is not None and rate > 0 else None
        details = " ".join(f"{key}={value}" for key, value in self.metrics.items())
        report(
            f"[{self.description}] {state}: {self.current}/{self.total or '?'} "
            f"{self.unit}, elapsed={elapsed:.1f}s, "
            f"eta={eta:.1f}s, {rate:.2f} {self.unit}/s {details}"
            if eta is not None
            else f"[{self.description}] {state}: {self.current}/{self.total or '?'} "
            f"{self.unit}, elapsed={elapsed:.1f}s, eta=unknown {details}"
        )
        self.last_report = now

    def __exit__(self, exc_type, exc, tb):
        self.bar.close()
        if self.enabled:
            state = "failed" if exc_type else "done" if self.current == self.total else "stopped"
            self._line(state)


@contextmanager
def console_log(path):
    """Persist plain stdout and tracebacks without saving terminal redraws."""
    if ddp.rank() != 0:
        yield
        return
    original = sys.stdout

    class Tee:
        def write(self, text):
            original.write(text)
            return stream.write(text)

        def flush(self):
            original.flush()
            stream.flush()

        def isatty(self):
            return original.isatty()

    with path.open("a", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(Tee()):
            try:
                yield
            except BaseException:
                traceback.print_exc(file=stream)
                raise
