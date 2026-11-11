"""Read-only adapter for the independently reproduced CSV + NPZ native K0.

This accepts the real ``chainflow-native-reproduction-v1`` handoff, not a
synthetic copy of our sharded evaluator format.  Auditing files now cannot
recover condition hashes, metric-cache hashes, or timings from the native run;
that distinction is kept explicit in the returned provenance.

Only NumPy and the standard library are needed.  In particular, importing this
module does not initialize Torch, CUDA, NAVSIM, or the native model.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = "chainflow-native-reproduction-v1"
ADAPTER_SCHEMA = "chainflow-external-baseline-v1"
TOTAL = 12146
NAVTEST_TOKEN_DIGEST = "8d42fef68542f095d0e03894fdaf38babfdc89f3fd7cdc26080c3a6009cb4108"
UPSTREAM_REPO = "https://github.com/AFARI-Research/ChainFlow-VLA"
UPSTREAM_COMMIT = "0f3565f66546139a0183bc5807d5f7234ae2d4b0"
CHECKPOINT_SHA256 = "420bb9254d852234fbe59b7c14c1406157cdf5fde7a1b19bd78175f287748db5"
CHECKPOINT_REVISION = "d00244bd0e2b12575251f4f47370034831b0d6c1"
BACKBONE_SHA256 = "dca70548ecd7b03ffba6172c4db403014511b5ee6073f9fca72ba9e6e602a25d"
VLM_ARCHIVE_SHA256 = "5c045c8842685651f5ec95cca166d1785a36f14011cab1c279e609571d00877b"
METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score",
)
TOKEN = re.compile(r"^[0-9a-f]{16}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


def handoff_path(path):
    path = Path(path).expanduser()
    return path / "reproduction_ready.json" if path.is_dir() else path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Nonfinite JSON value: {value}")))


def is_handoff(path):
    """Recognize a completed-format manifest without loading its large assets."""
    source = handoff_path(path)
    if not source.is_file():
        return False
    return _json(source.read_bytes()).get("schema") == SCHEMA


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value, label):
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise ValueError(f"Invalid SHA-256: {label}")
    return value


def _read_verified(path, expected, label):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != _digest(expected, label):
        raise ValueError(f"SHA-256 mismatch: {label}: {path}")
    return data


def _path(value, source, label, *, directory=False):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing handoff path: {label}")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = source.parent / result
    result = result.resolve()
    if not (result.is_dir() if directory else result.is_file()):
        raise ValueError(f"Missing handoff {'directory' if directory else 'file'}: {label}: {result}")
    return result


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Invalid numeric value: {label}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value: {label}") from None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"Metric outside 0..1: {label}")
    return result


def _metrics_csv(data, *, summary=False):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    required = {"token", "valid", *METRICS}
    if not summary:
        required.add("log_name")
    if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or not required.issubset(reader.fieldnames)):
        raise ValueError("Invalid native metrics CSV columns")
    rows, average = {}, None
    for line, record in enumerate(reader, 2):
        if None in record or any(record.get(key) is None for key in required):
            raise ValueError(f"Malformed native CSV record at line {line}")
        token = record["token"]
        if record["valid"].strip().lower() != "true":
            raise ValueError(f"Invalid native scene: {token}")
        metrics = {name: _number(record[name], f"{token}/{name}") for name in METRICS}
        if summary and token == "average":
            if average is not None:
                raise ValueError("Duplicate average row in native summary CSV")
            average = metrics
            continue
        if not TOKEN.fullmatch(token) or token in rows:
            raise ValueError(f"Invalid/duplicate native token: {token}")
        log = record.get("log_name")
        if not summary and (not log or log.strip() != log or "/" in log or "\\" in log):
            raise ValueError(f"Invalid native log_name: {token}")
        rows[token] = {"log": log, "metrics": metrics}
    return rows, average


def _close(left, right, label, tolerance=1e-12):
    if not math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"Native handoff values differ: {label}: {left} != {right}")


def load_handoff(path, *, expected_count=TOTAL, expected_token_digest=NAVTEST_TOKEN_DIGEST,
                 verify_payloads=True):
    """Validate the independent reproduction and return exact saved K0 records.

    ``path`` can name its directory or ``reproduction_ready.json`` directly.
    No source files are changed.  The default audits every referenced NPZ plus
    the checkpoint/backbone contents against their published hashes.  Setting
    ``verify_payloads=False`` skips only these costly payload hashes; the return
    value explicitly marks the result as unaudited.  This option is for display,
    never an acceptance gate.  CSV/index/config hashes and all trajectory values
    are checked in either mode.

    Returned ``rows`` maps each token to an evaluator-like K0 record.  Native
    arrays retain their dtype in ``trajectory_dtype`` and their full values in
    JSON-compatible lists; callers must avoid an implicit float32 cast for K0.
    """
    source = handoff_path(path).resolve()
    manifest_bytes = source.read_bytes()
    manifest = _json(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Native handoff manifest must be an object")
    fixed = {
        "schema": SCHEMA, "state": "complete", "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT, "training_split": "navtrain",
        "evaluation_split": "navtest", "score_scale": "0..1",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_hf_revision": CHECKPOINT_REVISION, "backbone_sha256": BACKBONE_SHA256,
        "vlm_cache_archive_sha256": VLM_ARCHIVE_SHA256,
        "native_noise_seed": 27, "native_sampling": "ddim_12step_64proposal",
        "trajectory_shape": [8, 3], "trajectory_columns": ["x", "y", "heading"],
        "trajectory_frame": "ego_local", "trajectory_interval_s": 0.5,
        "trajectory_horizon_s": 4.0,
    }
    for name, expected in fixed.items():
        if manifest.get(name) != expected:
            raise ValueError(f"Native handoff identity mismatch: {name}")
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    for name, expected in (("count", expected_count), ("valid_count", expected_count),
                           ("failed_count", 0)):
        if type(manifest.get(name)) is not int or manifest[name] != expected:
            raise ValueError(f"Native handoff count mismatch: {name}")
    paths = {}
    for name in ("checkpoint", "backbone", "resolved_config", "metrics", "summary_csv", "token_index"):
        paths[name] = _path(manifest.get(name), source, name)
    for name in ("trajectories", "vlm_cache", "metric_cache"):
        paths[name] = _path(manifest.get(name), source, name, directory=True)
    if manifest.get("vlm_provenance"):
        paths["vlm_provenance"] = _path(manifest["vlm_provenance"], source, "vlm_provenance")
    # Canonical absolute paths avoid guesses about the token-index location.
    manifest = {**manifest, **{key: str(value) for key, value in paths.items()}}
    audited = {str(source): hashlib.sha256(manifest_bytes).hexdigest()}
    payload_verified = bool(verify_payloads)
    for name in ("checkpoint", "backbone"):
        if payload_verified:
            actual = _sha256(paths[name])
            if actual != manifest[f"{name}_sha256"]:
                raise ValueError(f"SHA-256 mismatch: {name}: {paths[name]}")
            audited[str(paths[name])] = actual
    contents = {}
    for name in ("resolved_config", "metrics", "summary_csv", "token_index"):
        contents[name] = _read_verified(paths[name], manifest.get(f"{name}_sha256"), name)
        audited[str(paths[name])] = manifest[f"{name}_sha256"]
    index = _json(contents["token_index"])
    csv_rows, _ = _metrics_csv(contents["metrics"])
    if not isinstance(index, dict) or len(index) != expected_count or set(index) != set(csv_rows):
        raise ValueError("Native token-index/CSV coverage mismatch")
    tokens = sorted(csv_rows)
    token_digest = hashlib.sha256("\n".join(tokens).encode()).hexdigest()
    if expected_token_digest is not None and token_digest != expected_token_digest:
        raise ValueError("Native token membership differs from the canonical navtest split")
    original_rows, average = _metrics_csv(contents["summary_csv"], summary=True)
    if set(original_rows) != set(csv_rows):
        raise ValueError("Original summary CSV and per-token metrics have different tokens")
    for token in tokens:
        for name in METRICS:
            _close(original_rows[token]["metrics"][name], csv_rows[token]["metrics"][name],
                   f"original summary CSV/{token}/{name}")
    means = {name: math.fsum(csv_rows[token]["metrics"][name] for token in tokens) / expected_count
             for name in METRICS}
    subscores = manifest.get("subscores")
    if not isinstance(subscores, dict) or not set(METRICS).issubset(subscores):
        raise ValueError("Missing native handoff metric means")
    for name in METRICS:
        _close(means[name], _number(subscores[name], f"subscores/{name}"), name)
        if average is not None:
            _close(means[name], average[name], f"summary average/{name}")
    _close(means["score"], _number(manifest.get("mean_pdms"), "mean_pdms"), "mean_pdms")
    # The public handoff rounds its display value to four decimal places.
    display = manifest.get("mean_pdms_x100")
    if isinstance(display, bool) or not isinstance(display, (int, float)) or not math.isfinite(display):
        raise ValueError("Invalid mean_pdms_x100")
    _close(100 * means["score"], display, "mean_pdms_x100", tolerance=5.000001e-5)
    rows, trajectory_hashes, dtypes = {}, {}, set()
    for token in tokens:
        entry = index[token]
        if not isinstance(entry, dict) or entry.get("shape") != [8, 3]:
            raise ValueError(f"Invalid trajectory index entry: {token}")
        npz_path = _path(entry.get("npz"), source, f"trajectory/{token}")
        if (not npz_path.is_relative_to(paths["trajectories"])
                or npz_path.name != f"{token}.npz"):
            raise ValueError(f"Indexed trajectory does not match its token/directory: {token}")
        expected_sha = _digest(entry.get("sha256"), token)
        payload = npz_path.read_bytes()
        if payload_verified and hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"SHA-256 mismatch: trajectory/{token}: {npz_path}")
        # Load exactly the bytes that were hashed, so no second-open race can
        # replace the payload between validation and array conversion.
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"trajectory"}:
                raise ValueError(f"Unexpected NPZ fields: {token}")
            trajectory = archive["trajectory"]
        if (trajectory.shape != (8, 3) or trajectory.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(trajectory).all()):
            raise ValueError(f"Invalid native trajectory shape/dtype/values: {token}")
        dtype = str(trajectory.dtype)
        dtypes.add(dtype)
        trajectory_hashes[token] = expected_sha
        values = csv_rows[token]["metrics"]
        rows[token] = {
            "token": token, "log": csv_rows[token]["log"], "valid": True,
            "native_seed": int(hashlib.md5(f"{token}:27".encode()).hexdigest()[:16], 16) % (2**63 - 1),
            "native_sampling": manifest["native_sampling"], "trajectory_dtype": dtype,
            "trajectory_sha256": expected_sha, "native_origin": "external-handoff",
            "trajectories": {"0": trajectory.tolist()}, "metrics": {"0": values},
            "delta": {"0": {name: 0.0 for name in METRICS}},
            "origin": ADAPTER_SCHEMA,
        }
    summary = {
        "count": expected_count, "valid_count": expected_count, "failed_count": 0,
        "score_scale": "0..1", "by_k": {"0": {"mean": means,
            "mean_delta": {name: 0.0 for name in METRICS}, "improved": 0,
            "worsened": 0, "tied": expected_count}},
    }
    identity = {
        "schema": ADAPTER_SCHEMA, "upstream_commit": UPSTREAM_COMMIT,
        "chainflow_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256, "backbone_sha256": BACKBONE_SHA256,
        "vlm_cache_archive_sha256": VLM_ARCHIVE_SHA256,
        "resolved_config_sha256": manifest["resolved_config_sha256"],
        "handoff_sha256": audited[str(source)], "token_digest": token_digest,
        "count": expected_count, "native_noise_seed": 27,
        "native_sampling": manifest["native_sampling"], "trajectory_dtypes": sorted(dtypes),
    }
    provenance = {
        "schema": ADAPTER_SCHEMA, "source": str(source), "payloads_verified": payload_verified,
        "files_sha256": audited, "trajectory_index_sha256": manifest["token_index_sha256"],
        "trajectory_sha256": trajectory_hashes,
        "checks_performed": "current files versus recorded hashes; exact CSV/NPZ/token coverage",
        "native_condition_hashes": None, "native_metric_cache_hashes": None,
        "native_timings": None, "native_execution_identity": None,
        "native_token_seed_source": "derived from declared seed 27 using the pinned upstream sampler rule",
        "vlm_archive_hash_source": "native handoff declaration; extracted cache/archive not rehashed here",
        "resolved_config_validation": "recorded SHA-256; evaluation settings also checked against declared metadata",
    }
    if "vlm_provenance" in paths:
        provenance["current_vlm_provenance_sha256"] = _sha256(paths["vlm_provenance"])
    return {"schema": ADAPTER_SCHEMA, "source": str(source), "manifest_ready_path": str(source), "manifest": manifest,
            "identity": identity, "rows": rows, "summary": summary, "provenance": provenance}
