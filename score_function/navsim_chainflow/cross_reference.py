"""Read the scene identities and responses saved by the original ChainFlow-start experiment."""
import hashlib
import json
import math
from pathlib import Path

from .handoff import METRICS, NAVTEST_TOKEN_DIGEST, TOTAL


def load_reference(path, *, needed=None, expected_count=TOTAL):
    """Check the completed reference and return selected scene metadata, its summary and file hashes."""
    path = Path(path).resolve()
    needed = set(needed) if needed is not None else None
    summary_bytes = (path / "summary.json").read_bytes()
    hashes = {"summary.json": hashlib.sha256(summary_bytes).hexdigest()}
    summary = json.loads(summary_bytes)
    protocol = summary["protocol"]
    ks = (0, 5, 10, 15, 20, 30)
    arms = set(map(str, ks))
    metric_names = set(METRICS)
    if (summary["count"] != expected_count or summary["valid_count"] != expected_count
            or summary["failed_count"] != 0 or protocol["count"] != expected_count
            or protocol["mode"] != "refine" or protocol["split"] != "navtest"
            or protocol["score_epoch"] != 40 or protocol["score_weight_kind"] != "ema"
            or protocol["gamma"] != .1 or protocol["ks"] != list(ks)
            or protocol["seed"] != 20260922 or protocol["native_seed"] != 27
            or set(summary["by_k"]) != arms):
        raise ValueError(f"Expected the completed epoch-40 ChainFlow-start EMA comparison: {path}")

    # Keep just the requested conditions and scores. An eight-GPU run need not
    # hold eight copies of all six trajectories of all reference scenes in RAM.
    records, partitions = {}, {}
    scores = {k: [] for k in arms}
    for directory in sorted(path.glob("part-*-of-*")):
        run_bytes = (directory / "run.json").read_bytes()
        completed_bytes = (directory / "complete.json").read_bytes()
        manifest, completed = json.loads(run_bytes), json.loads(completed_bytes)
        index = manifest["shard_index"]
        if (manifest["run"] != protocol or completed["manifest"] != manifest
                or index in partitions or directory.name != f"part-{index:04d}-of-{protocol['num_shards']:04d}"):
            raise ValueError(f"Mixed reference shards: {directory}")
        tokens = []
        digest = hashlib.sha256()
        with (directory / "scenes.jsonl").open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                token, metrics = row["token"], row["metrics"]
                if (row.get("valid") is not True or not row["log"] or set(metrics) != arms
                        or any(set(m) != metric_names for m in metrics.values())):
                    raise ValueError(f"Incomplete reference metrics: {token}")
                for key in ("condition_sha256", "metric_cache_sha256"):
                    value = row[key]
                    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                        raise ValueError(f"Invalid reference {key}: {token}")
                tokens.append(token)
                for k in arms:
                    score = metrics[k]["score"]
                    if not 0 <= score <= 1:
                        raise ValueError(f"Invalid reference PDMS: {token} K={k}")
                    scores[k].append(score)
                if needed is None or token in needed:
                    records[token] = {key: row[key] for key in
                                      ("log", "condition_sha256", "metric_cache_sha256", "metrics")}
        if (tokens != manifest["shard_tokens"] or completed["count"] != len(tokens)
                or digest.hexdigest() != completed["scenes_sha256"]):
            raise ValueError(f"Reference records differ from completion marker: {directory}")
        partitions[index] = tokens
        for name, data in (("run.json", run_bytes), ("complete.json", completed_bytes)):
            hashes[f"{directory.name}/{name}"] = hashlib.sha256(data).hexdigest()
        hashes[f"{directory.name}/scenes.jsonl"] = digest.hexdigest()

    all_tokens = sorted(t for part in partitions.values() for t in part)
    token_digest = hashlib.sha256("\n".join(all_tokens).encode()).hexdigest()
    n = protocol["num_shards"]
    if (set(partitions) != set(range(n)) or len(all_tokens) != expected_count
            or len(set(all_tokens)) != expected_count or token_digest != protocol["token_digest"]
            or (expected_count == TOTAL and token_digest != NAVTEST_TOKEN_DIGEST)
            or any(partitions[i] != all_tokens[i::n] for i in partitions)
            or (needed is not None and records.keys() != needed)):
        raise ValueError("Incomplete or incorrect reference token coverage")
    base = math.fsum(scores["0"]) / expected_count
    for k, values in scores.items():
        mean = math.fsum(values) / expected_count
        arm = summary["by_k"][k]
        if (not math.isclose(mean, arm["mean"]["score"], rel_tol=0, abs_tol=1e-10)
                or not math.isclose(mean - base, arm["mean_delta"]["score"], rel_tol=0, abs_tol=1e-10)):
            raise ValueError(f"Reference PDMS summary does not match saved rows: K={k}")
    provenance = {"source": str(path), "files_sha256": hashes,
                  "summary_sha256": hashes["summary.json"], "token_digest": token_digest}
    return records, summary, provenance
