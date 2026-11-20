"""Evaluate EMA checkpoints at K=20, sharing each DD start, PDM replay and frozen scene encoding."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from . import cross_dd
from .assets import atomic_json, lock, sha256
from .handoff import METRICS, NAVTEST_TOKEN_DIGEST, TOTAL

SCHEMA = "chainflow-dd-k20-checkpoint-matrix-shard-v1"


def computed_arms(request):
    names = list(request["checkpoints"])
    if request["reuse_reference_final"] is not None:
        names.remove("reference/final")
    return names


def matching_math(current, previous):
    # Reproduction manifests use float32_matmul_precision; some older tools
    # called the same torch.get_float32_matmul_precision() value matmul_precision.
    def canonical(runtime):
        values = {runtime[k] for k in ("float32_matmul_precision", "matmul_precision") if k in runtime}
        if len(values) != 1 or None in values:
            raise ValueError("Missing or conflicting float32 matmul precision in runtime")
        return {**runtime, "matmul_precision": values.pop()}
    cross_dd.require_matching_math(canonical(current), canonical(previous))


def check_metrics(values):
    if (set(values) != set(METRICS)
            or any(type(v) not in (float, int) or not 0 <= v <= 1 for v in values.values())):
        raise ValueError("Invalid NAVSIM metrics")


def validate_row(row, token, source, control, names):
    if (row["token"] != token or row["log"] != source["log"] or row["log"] != control["log"]
            or row["dd_k0_sha256"] != source["trajectory_sha256"]
            or row["k0"] != source["metrics"]["0"]
            or row["condition_sha256"] != control["condition_sha256"]
            or row["metric_cache_sha256"] != control["metric_cache_sha256"]
            or set(row["k20"]) != set(names)):
        raise ValueError(f"DD start, features, metric cache or checkpoint set changed for {token}")
    check_metrics(row["k0_replay"])
    if any(abs(row["k0_replay"][m] - row["k0"][m]) > cross_dd.K0_TOLERANCE for m in METRICS):
        raise ValueError(f"DD K0 replay mismatch for {token}")
    for arm in row["k20"].values():
        check_metrics(arm["metrics"])
        poses = arm["trajectory"]
        if len(poses) != 8 or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in poses):
            raise ValueError(f"Invalid K20 trajectory for {token}")


def read_records(directory, manifest, dd, reference, names, consume=None):
    """Validate the existing prefix and return its length; keep only complete rows for resume."""
    marker = directory / "complete.json"
    final, partial = directory / "scenes.jsonl", directory / "scenes.jsonl.tmp"
    completed = marker.exists()
    if completed:
        proof = cross_dd.read_json(marker)
        if (proof["manifest"] != manifest or proof["count"] != len(manifest["shard_tokens"])
                or not final.is_file() or proof["scenes_sha256"] != sha256(final)):
            raise ValueError(f"Invalid published matrix shard: {directory}")
    if final.exists() and partial.exists():
        raise ValueError(f"Conflicting matrix result files: {directory}")
    if final.exists() and not completed:
        # Interrupted between publishing all rows and publishing their completion marker.
        final.replace(partial)
    path = final if completed else partial
    if not path.exists():
        return 0
    count = offset = 0
    tokens = manifest["shard_tokens"]
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n") and not completed:
                break
            if count >= len(tokens) or not line.endswith(b"\n"):
                raise ValueError(f"Unexpected matrix row in {path}")
            row = json.loads(line)
            token = tokens[count]
            validate_row(row, token, dd[token], reference[token], names)
            if consume is not None:
                consume(row)
            count += 1
            offset += len(line)
    if completed and count != len(tokens):
        raise ValueError(f"Incomplete matrix shard: {directory}")
    if not completed and offset < path.stat().st_size:
        with path.open("r+b") as stream:
            stream.truncate(offset)
    return count


def create_refiners(request, conditioner_identity, device):
    from .runtime import ChainFlowRefiner
    by_hash, refiners = {}, {}
    for name in computed_arms(request):
        record = request["checkpoints"][name]
        digest, path = record["sha256"], record["path"]
        if sha256(path) != digest:
            raise ValueError(f"Score checkpoint changed: {name}")
        if digest not in by_hash:
            refiner = ChainFlowRefiner.from_checkpoint(path, device)
            refiner.require_conditioner(conditioner_identity)
            if (refiner.sigma, refiner.gamma) != (.1, .1):
                raise ValueError(f"Unexpected Score checkpoint sigma/gamma: {name}")
            by_hash[digest] = refiner
        refiners[name] = by_hash[digest]
    return refiners


def evaluate_scene(token, source, control, metric_hash, score, get_condition, refiners, device):
    import torch
    from . import evaluate as cf

    if metric_hash != control["metric_cache_sha256"]:
        raise ValueError(f"NAVSIM metric cache changed: {token}")
    raw = torch.tensor(source["trajectories"]["0"], dtype=torch.float32, device=device)[None]
    k0 = source["metrics"]["0"]
    replay = score(raw[0].cpu().numpy())
    cf.verify_k0_metrics(replay, k0)
    condition = get_condition()
    digest = cf.condition_digest(condition)
    if digest != control["condition_sha256"]:
        raise ValueError(f"Frozen scene, ego or VLM condition changed: {token}")

    predictions, arms = {}, {}
    for name, refiner in refiners.items():
        if id(refiner) not in predictions:
            trajectories, metrics, _, _ = cf.score_arms(raw, condition, [0, 20], refiner, score, device, .1, k0)
            predictions[id(refiner)] = {"trajectory": trajectories["20"], "metrics": metrics["20"]}
        arms[name] = predictions[id(refiner)]
    row = dict(token=token, log=source["log"], dd_k0_sha256=source["trajectory_sha256"],
               condition_sha256=digest, metric_cache_sha256=metric_hash,
               k0=k0, k0_replay=replay, k20=arms)
    validate_row(row, token, source, control, refiners)
    return row


def run(request_path, index, device):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from . import evaluate as cf
    from .native import activate_devkit, encode_scene, load_agent, make_scene_loader

    request_path = Path(request_path).resolve()
    request = cross_dd.read_json(request_path)
    base = request["base"]
    root = activate_devkit(base["devkit_root"])
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score

    cf.apply_cuda_budget(device)
    torch.set_num_threads(1)
    torch.manual_seed(base["seed"])
    workers = request["workers"]
    loader = make_scene_loader(base["data_root"], root, "navtest")
    tokens, shard = cf.select_tokens(loader.tokens, None, index, workers)
    if len(tokens) != TOTAL or cf.digest_tokens(tokens) != NAVTEST_TOKEN_DIGEST:
        raise ValueError("Scene loader differs from canonical navtest")
    dd, reference, previous = cross_dd.sources(base, needed=shard)
    env = cf.execution_identity(device)
    matching_math(env, previous["protocol"]["execution_identity"])
    if cf.scoring_identity(root) != base["scoring_identity"]:
        raise ValueError("PDM scorer identity changed after launch")
    agent = load_agent(root, base["checkpoint"], base["backbone"], base["vlm_cache"], device)
    if agent._score_bridge_identity != base["native_identity"]:
        raise ValueError("Frozen ChainFlow assets changed after launch")
    refiners = create_refiners(request, agent._score_bridge_identity, device)
    metric_root = Path(base["metric_cache"])
    paths = cf.metric_index(metric_root)
    if set(shard) - paths.keys():
        raise ValueError("Missing NAVSIM metrics for DD scenes")

    class AllNodeMetricCache(MetricCacheLoader):
        _load_metric_cache_paths = staticmethod(cf.metric_index)

    cache = AllNodeMetricCache(metric_root)
    cfg = OmegaConf.load(root / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Inconsistent PDM scoring horizons")
    directory = cross_dd.shard_directory(request_path.parent, index, workers)
    manifest = dict(schema=SCHEMA, request_sha256=cross_dd.request_digest(request), runtime=env,
                    shard_index=index, shard_tokens=shard)
    with lock(directory / ".lock"):
        manifest_path = directory / "run.json"
        if manifest_path.exists():
            if cross_dd.read_json(manifest_path) != manifest:
                raise ValueError(f"Matrix shard belongs to another run: {directory}")
        elif any((directory / f).exists() for f in ("complete.json", "scenes.jsonl", "scenes.jsonl.tmp")):
            raise ValueError(f"Matrix shard records found without a manifest: {directory}")
        else:
            atomic_json(manifest_path, manifest)
        start = read_records(directory, manifest, dd, reference, refiners)
        if (directory / "complete.json").exists():
            return
        tmp = directory / "scenes.jsonl.tmp"
        print(f"Matrix shard {index}: resuming at {start}/{len(shard)}", flush=True)
        with tmp.open("a") as stream:
            for i, token in enumerate(shard[start:], start):
                try:
                    metric = cache.get_from_token(token)

                    def score(poses):
                        return asdict(pdm_score(metric, Trajectory(poses), simulator.proposal_sampling, simulator, scorer))

                    def get_condition():
                        scene = loader.get_scene_from_token(token)
                        metadata = scene.scene_metadata
                        if (metadata.initial_token, metadata.log_name) != (token, dd[token]["log"]):
                            raise ValueError(f"Wrong navtest scene for DD trajectory: {token}")
                        return encode_scene(agent, scene, device)

                    row = evaluate_scene(token, dd[token], reference[token], sha256(paths[token]),
                                         score, get_condition, refiners, device)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    if (i+1) % 25 == 0 or i+1 == len(shard):
                        print(f"Matrix shard {index}: {i+1}/{len(shard)} scenes", flush=True)
                except Exception as exc:
                    atomic_json(directory / "failure.json", {"token": token, "completed": i, "error": str(exc)})
                    raise
        final = directory / "scenes.jsonl"
        tmp.replace(final)
        atomic_json(directory / "complete.json", dict(manifest=manifest, count=len(shard), scenes_sha256=sha256(final)))
        (directory / "failure.json").unlink(missing_ok=True)


def merge(output, request):
    output = Path(output)
    if cross_dd.read_json(output / "request.json") != request:
        raise ValueError("Matrix output belongs to a different request")
    dd, reference, previous = cross_dd.sources(request["base"])
    tokens = sorted(dd)
    if len(tokens) != TOTAL or hashlib.sha256("\n".join(tokens).encode()).hexdigest() != NAVTEST_TOKEN_DIGEST:
        raise ValueError("Expected complete canonical DD navtest")
    workers = request["workers"]
    directories = [cross_dd.shard_directory(output, i, workers) for i in range(workers)]
    if set(output.glob("part-*-of-*")) != set(directories):
        raise ValueError("Missing or unexpected matrix shards")
    names = computed_arms(request)
    results = {name: dict(mean=dict.fromkeys(METRICS, 0.), mean_delta=dict.fromkeys(METRICS, 0.),
                          improved=0, worsened=0, tied=0) for name in names}

    def accumulate(row):
        baseline = row["k0"]
        for name, arm in row["k20"].items():
            stats = results[name]
            for m, value in arm["metrics"].items():
                stats["mean"][m] += value
                stats["mean_delta"][m] += value - baseline[m]
            delta = arm["metrics"]["score"] - baseline["score"]
            outcome = "improved" if delta > 0 else "worsened" if delta < 0 else "tied"
            stats[outcome] += 1

    environment = None
    for i, directory in enumerate(directories):
        manifest = cross_dd.read_json(directory / "run.json")
        if (not (directory / "complete.json").is_file() or manifest["schema"] != SCHEMA
                or manifest["request_sha256"] != cross_dd.request_digest(request) or manifest["shard_index"] != i
                or manifest["shard_tokens"] != tokens[i::workers]):
            raise ValueError(f"Invalid matrix shard: {directory}")
        if environment is not None and environment != manifest["runtime"]:
            raise ValueError("Mixed matrix runtime environments")
        environment = manifest["runtime"]
        matching_math(environment, previous["protocol"]["execution_identity"])
        read_records(directory, manifest, dd, reference, names, accumulate)
    count = len(tokens)
    for stats in results.values():
        for key in ("mean", "mean_delta"):
            stats[key] = {m: value / count for m, value in stats[key].items()}
    k0 = math.fsum(dd[token]["metrics"]["0"]["score"] for token in tokens) / count
    reused = request["reuse_reference_final"]
    if reused is not None:
        stats = reused["by_k20"]
        check_metrics(stats["mean"])
        if (set(stats["mean_delta"]) != set(METRICS)
                or any(not -1 <= v <= 1 for v in stats["mean_delta"].values())
                or any(type(stats[k]) != int or stats[k] < 0 for k in ("improved", "worsened", "tied"))
                or sum(stats[k] for k in ("improved", "worsened", "tied")) != count
                or abs(stats["mean"]["score"] - stats["mean_delta"]["score"] - k0) > 1e-9):
            raise ValueError("Historical reference final K20 differs from common DD baseline")
        results["reference/final"] = {**stats, "per_scene_directory": reused["output"]}
    summary = dict(schema=SCHEMA, count=count, valid_count=count, k0=k0, by_checkpoint=results,
                   ks=[0, 20], score_scale="0..1", protocol=request, runtime=environment)
    atomic_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.request, args.shard_index, args.device)
