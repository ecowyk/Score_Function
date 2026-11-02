"""Terminate only Linux processes carrying one failed baseline job's exact markers."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import time


def _stat(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rpartition(')')[2].split()
    return int(fields[19]), int(fields[1]), fields[0]  # start ticks, parent, state


def _protected():
    protected, pid = set(), os.getpid()
    while pid > 0 and pid not in protected:
        protected.add(pid)
        try:
            pid = _stat(pid)[1]
        except (OSError, ValueError, IndexError):
            break
    return protected


def _owned(pid, queue, token):
    try:
        before = _stat(pid)
        values = (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
        if (b'SCORE_BASELINE_QUEUE=' + queue.encode() not in values
                or b'SCORE_BASELINE_JOB_TOKEN=' + token.encode() not in values):
            return None
        after = _stat(pid)
        return after[0] if before[0] == after[0] and after[2] != 'Z' else None
    except (OSError, ValueError, IndexError):
        return None


def _matching(queue, token, protected):
    found = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit() and int(path.name) not in protected:
            pid = int(path.name)
            identity = _owned(pid, queue, token)
            if identity is not None:
                found[pid] = identity
    return found


def cleanup_job(queue_path, token, process_group=None, grace_seconds=3):
    """TERM then KILL exact tagged processes, including detached Ray children.

    Never signal by process-group number alone: detached children can have other
    groups, while a stale group number can belong to unrelated work. The optional
    process_group argument is accepted for callers' bookkeeping only.
    """
    queue, token = str(Path(queue_path).resolve()), str(token)
    if not token or '\0' in token:
        raise ValueError('A nonempty per-job process token is required')
    protected, sent, errors = _protected(), {signal.SIGTERM: set(), signal.SIGKILL: set()}, []

    def send(pid, identity, sig):
        if pid in protected or _owned(pid, queue, token) != identity:
            return
        try:
            os.kill(pid, sig)
            sent[sig].add((pid, identity))
        except ProcessLookupError:
            pass  # Normal exit between identity check and signal.
        except OSError as error:
            message = f'{pid}: {type(error).__name__}: {error}'
            if message not in errors:
                errors.append(message)

    deadline = time.monotonic() + max(0, float(grace_seconds))
    while True:
        remaining = _matching(queue, token, protected)
        for pid, identity in remaining.items():
            if (pid, identity) not in sent[signal.SIGTERM]:
                send(pid, identity, signal.SIGTERM)
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(min(.1, max(0, deadline - time.monotonic())))
    for pid, identity in _matching(queue, token, protected).items():
        send(pid, identity, signal.SIGKILL)
    remaining = _matching(queue, token, protected)
    return dict(terminated=len(sent[signal.SIGTERM]), killed=len(sent[signal.SIGKILL]),
                remaining=sorted(remaining), errors=errors)
