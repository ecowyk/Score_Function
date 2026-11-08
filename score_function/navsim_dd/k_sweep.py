"""Strict paired comparison of new refinement steps with saved K0/K20 arms.

This module only reads saved results. It never loads a neural network, changes
an evaluation summary, or recomputes a PDM score.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path


PROTOCOL_KEYS = (
    "split", "seed", "gamma", "identity", "score_checkpoint_sha256",
    "token_digest", "count", "data_root", "metric_cache", "max_scenes",
)
DD_COMMIT = "9b52ed0ec06b073d82d6f392ab084c7b301c8681"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, contents):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(contents, encoding="utf-8")
    tmp.replace(path)


def _token_seed(seed, token):
    return int.from_bytes(hashlib.sha256(f"{seed}\0{token}".encode()).digest()[:8], "big") % (2**63 - 1)


def _token_digest(tokens):
    return hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()


def _load_completed(directory, expected_count):
    """Validate every arm, row, shard and summary before permitting reuse."""
    directory = Path(directory).resolve()
    summary = _read(directory / "summary.json")
    if summary.get("count") != expected_count or expected_count < 1:
        raise ValueError(f"Expected {expected_count} completed scenes: {directory}")
    records, shard_ids, hashes, run, arms, metric_names = {}, set(), {}, None, None, None
    for part in sorted(directory.glob("part-*-of-*")):
        complete = _read(part / "complete.json")
        if _read(part / "run.json") != complete:
            raise ValueError(f"Run/completion metadata mismatch: {part}")
        if run is None:
            run = complete["run"]
            if any(key not in run for key in PROTOCOL_KEYS):
                raise ValueError(f"Missing evaluation protocol fields: {directory}")
            ks = run.get("ks", [])
            if (not ks or any(type(k) is not int or k < 0 for k in ks)
                    or len(set(ks)) != len(ks) or 0 not in ks):
                raise ValueError("Evaluation requires unique nonnegative K values including K0")
            arms = [str(k) for k in sorted(ks)]
            if set(summary.get("by_k", {})) != set(arms):
                raise ValueError("Summary arms differ from the evaluation protocol")
            metric_names = set(summary["by_k"]["0"].get("mean", {}))
            if "score" not in metric_names or any(
                set(summary["by_k"][k].get("mean", {})) != metric_names for k in arms
            ):
                raise ValueError("Summary arms need identical metric sets including score")
        index = complete["shard_index"]
        if complete["run"] != run or type(index) is not int or index in shard_ids:
            raise ValueError(f"Mixed or duplicate shard protocol: {part}")
        shard_ids.add(index)
        found, digest = [], hashlib.sha256()
        with (part / "scenes.jsonl").open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                token = row["token"]
                if not isinstance(token, str) or not token or token in records:
                    raise ValueError(f"Invalid or duplicate scene token: {token}")
                if not isinstance(row.get("log"), str) or not row["log"]:
                    raise ValueError(f"Missing scene log: {token}")
                if row.get("seed") != _token_seed(run["seed"], token):
                    raise ValueError(f"Scene seed does not match protocol: {token}")
                if type(row.get("top1_index")) is not int or row["top1_index"] < 0:
                    raise ValueError(f"Invalid native mode index: {token}")
                if any(set(row.get(field, {})) != set(arms) for field in ("trajectories", "metrics", "delta")):
                    raise ValueError(f"Scene arms differ from protocol: {token}")
                for k in arms:
                    poses = row["trajectories"][k]
                    if (len(poses) != 8 or any(len(p) != 3 for p in poses)
                            or not all(math.isfinite(float(v)) for p in poses for v in p)):
                        raise ValueError(f"Invalid K{k} trajectory: {token}")
                    values, deltas = row["metrics"][k], row["delta"][k]
                    if (set(values) != metric_names or set(deltas) != metric_names
                            or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in values.values())):
                        raise ValueError(f"Invalid K{k} metric set: {token}")
                    for metric in metric_names:
                        actual = float(values[metric]) - float(row["metrics"]["0"][metric])
                        if not math.isclose(actual, float(deltas[metric]), rel_tol=0, abs_tol=1e-12):
                            raise ValueError(f"Incorrect saved delta: {token} K{k} {metric}")
                records[token] = row
                found.append(token)
        if sorted(found) != complete["shard_tokens"]:
            raise ValueError(f"Incomplete scene records: {part}")
        hashes[str((part / "scenes.jsonl").relative_to(directory))] = digest.hexdigest()
        for filename in ("run.json", "complete.json"):
            hashes[str((part / filename).relative_to(directory))] = _hash(part / filename)
    if (run is None or run.get("split") != "navtest" or run.get("count") != expected_count
            or type(run.get("num_shards")) is not int or run["num_shards"] < 1
            or shard_ids != set(range(run["num_shards"])) or len(records) != expected_count
            or _token_digest(records) != run["token_digest"]):
        raise ValueError(f"Incomplete navtest coverage: {directory}")
    for k in arms:
        for metric in metric_names:
            actual = math.fsum(float(r["metrics"][k][metric]) for r in records.values()) / expected_count
            if not math.isclose(actual, float(summary["by_k"][k]["mean"][metric]), rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Summary differs from scene results: K{k} {metric}")
    hashes["summary.json"] = _hash(directory / "summary.json")
    return records, run, {"directory": str(directory), "files_sha256": hashes}


def _require_500_reference(reference):
    launcher = _read(Path(reference) / "launcher.json")
    if launcher.get("epochs") != 500 or launcher.get("status") != "complete":
        raise ValueError("Reference must be the completed aligned 500-epoch evaluation")
    return launcher


def _require_reference_protocol(run):
    if run["seed"] != 20260922 or run["gamma"] != 0.1:
        raise ValueError("Reference requires seed=20260922 and gamma=0.1")
    if "dd_commit" in run["identity"] and run["identity"]["dd_commit"] != DD_COMMIT:
        raise ValueError("Reference uses a different DiffusionDrive source commit")


def validate_reference(reference, checkpoint_path, *, expected_count=12146):
    """Preflight the source and bind it to the exact final checkpoint bytes."""
    reference = Path(reference).resolve()
    launcher = _require_500_reference(reference)
    _, run, provenance = _load_completed(reference, expected_count)
    _require_reference_protocol(run)
    if set(run["ks"]) != {0, 20}:
        raise ValueError("Reference must contain exactly K0 and K20")
    if run["score_checkpoint_sha256"] != _hash(checkpoint_path):
        raise ValueError("Reference K20 uses a different score checkpoint")
    provenance["files_sha256"]["launcher.json"] = _hash(reference / "launcher.json")
    provenance["launcher"] = launcher
    return {"run": run, "provenance": provenance}


def merge_comparison(output, reference, *, expected_count=12146):
    """Join saved K20 to newly computed arms, requiring exact per-scene K0."""
    output, reference = Path(output).resolve(), Path(reference).resolve()
    if output == reference:
        raise ValueError("K sweep needs an independent output directory")
    launcher = _require_500_reference(reference)
    original, ref_run, ref_provenance = _load_completed(reference, expected_count)
    _require_reference_protocol(ref_run)
    new, new_run, new_provenance = _load_completed(output, expected_count)
    if set(ref_run["ks"]) != {0, 20}:
        raise ValueError("Reference must contain exactly K0 and K20")
    if 20 in new_run["ks"] or len(new_run["ks"]) < 2:
        raise ValueError("New sweep must compute positive K values excluding reused K20")
    for key in PROTOCOL_KEYS:
        if ref_run[key] != new_run[key]:
            raise ValueError(f"K sweep/reference protocol mismatch: {key}")
    if not ref_run["score_checkpoint_sha256"]:
        raise ValueError("Missing score checkpoint identity")
    if set(new) != set(original):
        raise ValueError("K sweep/reference token sets differ")
    for token, row in new.items():
        old = original[token]
        for key in ("log", "seed", "top1_index"):
            if row[key] != old[key]:
                raise ValueError(f"K sweep/reference {key} mismatch: {token}")
        if row["trajectories"]["0"] != old["trajectories"]["0"]:
            raise ValueError(f"K sweep/reference K0 trajectory mismatch: {token}")
        if row["metrics"]["0"] != old["metrics"]["0"]:
            raise ValueError(f"K sweep/reference K0 metric mismatch: {token}")
    tokens = sorted(new)
    arms = sorted(set(new_run["ks"]) | {20})
    metric_names = sorted(original[tokens[0]]["metrics"]["0"])
    statistics = {}
    for k in arms:
        key = str(k)
        source = original if k in (0, 20) else new
        deltas = [source[t]["metrics"][key]["score"] - original[t]["metrics"]["0"]["score"] for t in tokens]
        statistics[key] = {
            "origin": "reused" if k in (0, 20) else "computed",
            "mean": {m: math.fsum(source[t]["metrics"][key][m] for t in tokens) / expected_count for m in metric_names},
            "mean_delta": {m: math.fsum(source[t]["metrics"][key][m] - original[t]["metrics"]["0"][m] for t in tokens) / expected_count for m in metric_names},
            "mean_delta_vs_k20": {m: math.fsum(source[t]["metrics"][key][m] - original[t]["metrics"]["20"][m] for t in tokens) / expected_count for m in metric_names},
            "improved": sum(d > 0 for d in deltas), "worsened": sum(d < 0 for d in deltas), "tied": sum(d == 0 for d in deltas),
        }
    ref_provenance["files_sha256"]["launcher.json"] = _hash(reference / "launcher.json")
    report = {"format": "navsim-dd-paired-k-sweep-v1", "count": expected_count, "score_scale": "0..1",
              "epochs": 500, "seed": ref_run["seed"], "gamma": ref_run["gamma"],
              "score_checkpoint_sha256": ref_run["score_checkpoint_sha256"], "ks": arms,
              "by_k": statistics, "pairing": "exact same saved K0 trajectory and metrics for every token",
              "provenance": {"reference": ref_provenance, "computed": new_provenance,
                             "reference_launcher": launcher, "reference_protocol": ref_run, "computed_protocol": new_run}}
    buffer = io.StringIO(newline="")
    fields = ["K", "origin", "count", "PDMS_100", "delta_K0_points", "delta_K20_points",
              "improved", "worsened", "tied"] + [f"mean_{m}" for m in metric_names if m != "score"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for k in arms:
        stats = statistics[str(k)]
        writer.writerow({"K": k, "origin": stats["origin"], "count": expected_count,
                         "PDMS_100": stats["mean"]["score"] * 100,
                         "delta_K0_points": stats["mean_delta"]["score"] * 100,
                         "delta_K20_points": stats["mean_delta_vs_k20"]["score"] * 100,
                         **{m: stats[m] for m in ("improved", "worsened", "tied")},
                         **{f"mean_{m}": stats["mean"][m] for m in metric_names if m != "score"}})
    # comparison.json is published last, after every source and generated row passes.
    _atomic_text(output / "comparison.csv", buffer.getvalue())
    _atomic_text(output / "comparison.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def print_comparison(output):
    report = _read(Path(output) / "comparison.json")
    print(f"Aligned {report['epochs']} epoch RAW; {report['count']} paired scenes; "
          f"seed={report['seed']}; gamma={report['gamma']}", flush=True)
    print("K    origin       PDMS     delta K0   delta K20   improved / worsened / tied", flush=True)
    for k in report["ks"]:
        stats = report["by_k"][str(k)]
        print(f"{k:<4} {stats['origin']:<9} {100*stats['mean']['score']:9.4f} "
              f"{100*stats['mean_delta']['score']:+10.4f} {100*stats['mean_delta_vs_k20']['score']:+11.4f} "
              f"  {stats['improved']} / {stats['worsened']} / {stats['tied']}", flush=True)
    return report
