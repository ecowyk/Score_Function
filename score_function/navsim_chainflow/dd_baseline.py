"""Read a completed DiffusionDrive paired run as the source of frozen K0 poses.

DD saved float32 top-1 poses as JSON numbers. Only K0 is imported, together
with the original per-token seed, mode and scores. No DD or ChainFlow model
is needed. The per-file hashes bind a new cross-planner evaluation to these
specific source files, including on resume.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from score_function.navsim_chainflow.handoff import DIGEST, METRICS, NAVTEST_TOKEN_DIGEST, TOKEN, TOTAL, _json
from score_function.navsim_dd.k_sweep import DD_COMMIT, _token_digest, _token_seed

DD_SEED = 20260922
DD_CHECKPOINT_SHA256 = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
DD_BACKBONE_SHA256 = "f6c1d7785b5d0e796a83f9643fe2cb952ac95740daa9fb2d5c8c6514fdadaeb3"
METRIC_SET = frozenset(METRICS)


def resolve_dd_source(source):
    """Accept a DD run directory or the absolute run path in a *_latest.txt file."""
    source = Path(source).expanduser().resolve()
    if source.is_file():
        target = source.read_text(encoding="utf-8").strip()
        if not target or "\n" in target or not Path(target).is_absolute():
            raise ValueError(f"Expected a DD run directory or absolute path in pointer: {source}")
        source = Path(target).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Missing completed DiffusionDrive run directory: {source}")
    return source


def _read_metadata(root, path, hashes):
    content = path.read_bytes()
    hashes[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
    return _json(content)


def _validate_protocol(run, expected_count, expected_token_digest):
    identity = run["identity"]
    if (identity.get("dd_commit") != DD_COMMIT or identity.get("condition") != "dd-keyval64-status-v1"
            or identity.get("checkpoint_sha256") != DD_CHECKPOINT_SHA256
            or identity.get("backbone_sha256") != DD_BACKBONE_SHA256
            or not DIGEST.fullmatch(identity.get("anchor_sha256", ""))):
        raise ValueError("Expected pinned 88.1-PDMS DiffusionDrive planner assets in DD source")
    if (run.get("seed") != DD_SEED or run.get("split") != "navtest"
            or run.get("count") != expected_count or type(run.get("num_shards")) is not int
            or not 1 <= run["num_shards"] <= expected_count or 0 not in run["ks"]
            or run.get("max_scenes") not in (None, expected_count)
            or run.get("token_digest") != expected_token_digest):
        raise ValueError("Unexpected DD source seed, split, count, token digest or K0 protocol")
    for key in ("data_root", "metric_cache"):
        if not isinstance(run.get(key), str) or not Path(run[key]).is_absolute():
            raise ValueError(f"DD source protocol lacks absolute {key}")


def _validate_k0(row):
    token, log = row["token"], row["log"]
    if (not isinstance(token, str) or not TOKEN.fullmatch(token) or not isinstance(log, str) or not log
            or row.get("valid", True) is not True or row.get("seed") != _token_seed(DD_SEED, token)
            or type(row.get("top1_index")) is not int or not 0 <= row["top1_index"] < 20):
        raise ValueError(f"Invalid DD token, log, validity, native seed or top-1 mode: {token}")
    poses, metrics = row["trajectories"]["0"], row["metrics"]["0"]
    values = np.asarray(poses)
    if (values.shape != (8, 3) or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
            or not np.array_equal(values, values.astype(np.float32))):
        raise ValueError(f"Expected finite, losslessly representable float32 DD [8,3] K0 poses: {token}")
    deltas = row["delta"]["0"]
    if (set(metrics) != METRIC_SET or set(deltas) != METRIC_SET or any(
            type(metrics[m]) not in (int, float) or not 0 <= metrics[m] <= 1 or deltas[m] != 0
            for m in METRICS)):
        raise ValueError(f"Invalid original DD K0 metrics or nonzero K0 deltas: {token}")


def load_dd_baseline(source, *, expected_count=TOTAL, expected_token_digest=NAVTEST_TOKEN_DIGEST, needed=None):
    """Stream all source shards once, retaining only requested K0s.

    Passing an empty iterable as needed still validates every shard, original
    subscore mean and canonical token. Legacy DD records omit a validity flag
    and publish a completion manifest only after every token scores successfully.
    """
    root = resolve_dd_source(source)
    hashes, records, seen = {}, {}, set()
    totals = dict.fromkeys(METRICS, 0.)
    required = None if needed is None else set(needed)
    summary = _read_metadata(root, root / "summary.json", hashes)
    if (expected_count < 1 or summary.get("count") != expected_count
            or summary.get("valid_count", expected_count) != expected_count or summary.get("failed_count", 0) != 0
            or summary.get("score_scale") != "0..1"):
        raise ValueError(f"DD source summary requires {expected_count} valid navtest scenes on the 0..1 scale")
    protocol = None
    shards = {}
    for part in sorted(root.glob("part-*-of-*")):
        manifest = _read_metadata(root, part / "complete.json", hashes)
        if _read_metadata(root, part / "run.json", hashes) != manifest:
            raise ValueError(f"DD run/completion manifest mismatch: {part}")
        run, index = manifest["run"], manifest["shard_index"]
        if protocol is None:
            _validate_protocol(run, expected_count, expected_token_digest)
            protocol = run
        if (run != protocol or type(index) is not int or not 0 <= index < run["num_shards"]
                or index in shards or part.name != f"part-{index:04d}-of-{run['num_shards']:04d}"):
            raise ValueError(f"Invalid or mixed DD source shard: {part}")
        digest = hashlib.sha256()
        tokens = []
        path = part / "scenes.jsonl"
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = _json(line)
                _validate_k0(row)
                token = row["token"]
                if token in seen:
                    raise ValueError(f"Duplicate DD source token: {token}")
                seen.add(token)
                tokens.append(token)
                metrics = row["metrics"]["0"]
                for name in METRICS:
                    totals[name] += metrics[name]
                if required is None or token in required:
                    poses = row["trajectories"]["0"]
                    encoded = json.dumps(poses, separators=(",", ":"), allow_nan=False).encode("ascii")
                    records[token] = {
                        "token": token, "log": row["log"], "native_seed": row["seed"],
                        "top1_index": row["top1_index"], "native_origin": "diffusiondrive",
                        "trajectories": {"0": poses}, "metrics": {"0": metrics},
                        "trajectory_dtype": "float32", "trajectory_sha256": hashlib.sha256(encoded).hexdigest(),
                    }
        if tokens != manifest["shard_tokens"]:
            raise ValueError(f"DD scene order/coverage differs from completion manifest: {part}")
        shards[index] = tokens
        hashes[str(path.relative_to(root))] = digest.hexdigest()
    if protocol is None or len(seen) != expected_count or _token_digest(seen) != expected_token_digest:
        raise ValueError("DD baseline does not cover the pinned navtest token set")
    ordered = sorted(seen)
    count = protocol["num_shards"]
    if set(shards) != set(range(count)) or any(shards[i] != ordered[i::count] for i in range(count)):
        raise ValueError("Incomplete or mixed DD source shard membership")
    if required is not None and set(records) != required:
        raise ValueError("Requested tokens missing from the DD baseline")
    means = summary["by_k"]["0"]["mean"]
    if set(means) != METRIC_SET or any(not math.isclose(totals[m] / expected_count, means[m],
                                                       rel_tol=0., abs_tol=1e-10) for m in METRICS):
        raise ValueError("DD K0 summary differs from recorded per-token metrics")
    provenance = {
        "source": str(root), "source_planner": "diffusiondrive", "source_seed": DD_SEED,
        "count": expected_count, "mean_k0_score": means["score"], "protocol": protocol, "files_sha256": hashes,
        "trajectory_hash_encoding": "sha256(compact JSON of original DD K0 poses)",
    }
    return records, provenance
