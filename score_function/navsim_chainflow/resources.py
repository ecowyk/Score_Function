"""Cooperative GPU leases for the ChainFlow experiment.

This is a scheduler, not GPU resource isolation.  It never stops another process,
changes device settings, starts MPS, or claims that free VRAM means spare compute.
``wait-idle`` waits for low utilization AND negligible existing VRAM use.  The
explicit ``shared`` policy only checks memory headroom and can slow other jobs.

Selectors are physical nvidia-smi indices or GPU UUIDs.  Every child receives full
UUIDs, so a selected physical GPU cannot silently change after CUDA renumbering.
Leases coordinate users of this module; unrelated jobs can start at any time.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time


QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]
DEFAULT_LOCK_DIR = Path("/tmp/score-function-gpu-leases")


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    total_mib: int
    used_mib: int
    utilization: int

    @property
    def free_mib(self):
        return self.total_mib - self.used_mib


def parse_gpu_query(output):
    """Reject unknown telemetry instead of treating it as an idle device."""
    result = []
    for row in csv.reader(io.StringIO(output)):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi telemetry row: {row!r}")
        try:
            index, uuid, total, used, util = (value.strip() for value in row)
            item = GPU(int(index), uuid, int(total), int(used), int(util))
        except ValueError as exc:
            raise RuntimeError("GPU telemetry is unavailable or not numeric") from exc
        if (item.index < 0 or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", item.uuid)
                or item.total_mib <= 0 or not 0 <= item.used_mib <= item.total_mib
                or not 0 <= item.utilization <= 100):
            raise RuntimeError(f"Invalid GPU telemetry: {item!r}")
        result.append(item)
    if len({gpu.index for gpu in result}) != len(result) or len({gpu.uuid for gpu in result}) != len(result):
        raise RuntimeError("Duplicate GPU index or UUID in telemetry")
    return sorted(result, key=lambda gpu: gpu.index)


def query_gpus():
    """Read telemetry only; nvidia-smi is given a bounded execution timeout."""
    try:
        result = subprocess.run(QUERY, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot read GPU telemetry with nvidia-smi") from exc
    return parse_gpu_query(result.stdout)


def _select(gpus, selector, *, allow_auto=False):
    if selector is None or (isinstance(selector, str) and selector.strip() == "auto"):
        if allow_auto:
            return list(gpus)
        raise ValueError("Inherited CUDA_VISIBLE_DEVICES must list GPU indices or UUIDs")
    values = selector.split(",") if isinstance(selector, str) else list(selector)
    values = [str(value).strip() for value in values]
    if values in ([], [""], ["-1"]):
        return []
    selected = []
    for value in values:
        if value.startswith("MIG-"):
            raise ValueError("MIG selectors are not supported by the full-GPU scheduler")
        matches = [gpu for gpu in gpus if (
            str(gpu.index) == value if value.isdigit()
            else value.startswith("GPU-") and gpu.uuid.startswith(value)
        )]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous GPU selector: {value!r}")
        if matches[0] in selected:
            raise ValueError(f"Duplicate GPU selector: {value!r}")
        selected.append(matches[0])
    return selected


def select_pool(gpus, pool="auto", *, environ=None):
    """Intersect the requested physical pool with the parent's CUDA visibility."""
    environ = os.environ if environ is None else environ
    permitted = list(gpus)
    if "CUDA_VISIBLE_DEVICES" in environ:
        permitted = _select(gpus, environ["CUDA_VISIBLE_DEVICES"])
    requested = _select(gpus, pool, allow_auto=True)
    permitted_uuids = {gpu.uuid for gpu in permitted}
    # Explicit requests outside the inherited allocation are errors, not a cue
    # to silently use devices the launcher did not allocate to this process.
    if pool not in (None, "auto") and any(gpu.uuid not in permitted_uuids for gpu in requested):
        raise ValueError("Requested GPU pool exceeds inherited CUDA_VISIBLE_DEVICES")
    return [gpu for gpu in requested if gpu.uuid in permitted_uuids]


def eligible(gpu, *, budget_gb, reserve_gb, policy, idle_utilization=10, idle_memory_mib=1024):
    memory_ok = gpu.free_mib >= math.ceil((budget_gb + reserve_gb) * 1024)
    return memory_ok and (policy == "shared" or (
        gpu.utilization <= idle_utilization and gpu.used_mib <= idle_memory_mib
    ))


def gpu_environment(selected, budget_gb):
    """The budget limits PyTorch's allocator, not every CUDA allocation."""
    if not selected or not math.isfinite(float(budget_gb)) or budget_gb <= 0:
        raise ValueError("Require selected GPUs and a positive finite memory budget")
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(gpu.uuid for gpu in selected),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CF_CUDA_MEMORY_GB": str(float(budget_gb)),
    }


def _release(handles):
    error = None
    for handle in handles:
        if handle.closed:
            continue
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError as exc:
            error = exc if error is None else error
        finally:
            handle.close()
    if error is not None:
        raise error


def _print_status(state):
    devices = ", ".join(
        f"{gpu['index']}: {gpu['free_mib'] / 1024:.1f}GiB free, {gpu['utilization']}% util"
        for gpu in state["pool"]
    )
    print(f"GPU {state['state']} ({state['policy']}, workers={state['requested_workers']}, "
          f"wait={state['wait_seconds']:.1f}s): {devices}. {state['notice']}", flush=True)


@contextmanager
def acquire_gpus(pool="auto", workers=4, budget_gb=12, reserve_gb=4,
                 policy="wait-idle", wait=True, poll=15, lock_dir=DEFAULT_LOCK_DIR,
                 on_status=None, *, timeout=None, idle_utilization=10,
                 idle_memory_mib=1024, environ=None):
    """Yield selected GPUs while retaining all advisory locks in this process.

    ``workers='auto'`` chooses the largest available power-of-two group (up to
    eight).  Use an explicit worker count for resume, whose world size is fixed.
    ``wait=False`` makes one attempt; ``timeout`` bounds the total waiting time.
    The callback receives JSON-serializable ``waiting``/``acquired`` snapshots.
    KeyboardInterrupt, callback exceptions and body exceptions release all locks.
    """
    if policy not in ("wait-idle", "shared"):
        raise ValueError("GPU policy must be wait-idle or shared")
    if workers != "auto" and (type(workers) is not int or workers not in (1, 2, 4, 8)):
        raise ValueError("GPU workers must be 1, 2, 4, 8, or auto")
    if (not all(math.isfinite(float(x)) for x in (budget_gb, reserve_gb, poll))
            or budget_gb <= 0 or reserve_gb < 0 or not 0 < poll <= 60):
        raise ValueError("Require positive budget, nonnegative reserve, and polling in (0,60] seconds")
    if timeout is not None and (not math.isfinite(float(timeout)) or timeout < 0):
        raise ValueError("GPU wait timeout must be nonnegative and finite")
    if not 0 <= idle_utilization <= 100 or idle_memory_mib < 0:
        raise ValueError("Invalid idle thresholds")
    inherited = dict(os.environ if environ is None else environ)
    directory = Path(lock_dir)
    directory.mkdir(parents=True, exist_ok=True)
    acquired = []
    start = time.monotonic()
    try:
        while True:
            gpus = select_pool(query_gpus(), pool, environ=inherited)
            if not gpus:
                raise RuntimeError("No GPUs permitted by the selected pool and inherited visibility")
            if workers != "auto" and len(gpus) < workers:
                raise ValueError(f"Requested {workers} workers but the permitted pool has {len(gpus)} GPUs")
            options = dict(budget_gb=budget_gb, reserve_gb=reserve_gb, policy=policy,
                           idle_utilization=idle_utilization, idle_memory_mib=idle_memory_mib)
            possible = [gpu for gpu in gpus if eligible(gpu, **options)]
            selected, locked = [], []
            for gpu in possible:
                handle = (directory / f"{gpu.uuid}.lock").open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    locked.append(gpu.uuid)
                    continue
                except BaseException:
                    handle.close()
                    raise
                acquired.append(handle)
                selected.append(gpu)
                if workers != "auto" and len(selected) == workers:
                    break
            requested = (max((n for n in (1, 2, 4, 8) if n <= len(selected)), default=0)
                         if workers == "auto" else workers)
            if requested and len(selected) >= requested:
                _release(acquired[requested:])
                acquired = acquired[:requested]
                selected = selected[:requested]
                # Recheck after acquiring the locks: they do not prevent jobs
                # outside this scheduler from changing free memory/utilization.
                current = {gpu.uuid: gpu for gpu in query_gpus()}
                gpus = [current.get(gpu.uuid, gpu) for gpu in gpus]
                ready = all(gpu.uuid in current and eligible(current[gpu.uuid], **options)
                            for gpu in selected)
                if ready:
                    selected = [current[gpu.uuid] for gpu in selected]
            else:
                ready = False
            state = {
                "state": "acquired" if ready else "waiting", "policy": policy,
                "host": socket.gethostname(), "pid": os.getpid(),
                "updated": datetime.now(timezone.utc).isoformat(),
                "requested_workers": workers, "budget_gb_per_worker": budget_gb,
                "reserve_gb": reserve_gb, "wait_seconds": round(time.monotonic() - start, 1),
                "pool": [{**asdict(gpu), "free_mib": gpu.free_mib} for gpu in gpus],
                "locked_by_other_pipeline": locked,
                "selected_uuids": [gpu.uuid for gpu in selected] if ready else [],
                "notice": ("Shared GPU execution may slow existing jobs; memory headroom is not compute isolation."
                           if policy == "shared" else
                           "Idle launch requires low utilization and low existing VRAM use; unrelated jobs can still start later."),
                "memory_budget_scope": "PyTorch allocator only; CUDA and other libraries may allocate additional memory",
            }
            if ready:
                for handle, gpu in zip(acquired, selected):
                    handle.seek(0)
                    handle.truncate()
                    json.dump({**state, "gpu_uuid": gpu.uuid}, handle)
                    handle.write("\n")
                    handle.flush()
            (on_status or _print_status)(state)
            if ready:
                yield selected
                return
            _release(acquired)
            acquired = []
            elapsed = time.monotonic() - start
            if not wait or (timeout is not None and elapsed >= timeout):
                raise TimeoutError("No eligible unlocked GPU group is available under this resource policy")
            time.sleep(min(poll, max(0, timeout - elapsed)) if timeout is not None else poll)
    finally:
        _release(acquired)
