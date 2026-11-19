"""Apply an already trained ChainFlow Score to saved DiffusionDrive trajectories."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .assets import atomic_json, lock, sha256
from .handoff import METRICS, NAVTEST_TOKEN_DIGEST, TOTAL

KS = [0, 5, 10, 15, 20, 30]
SCHEMA = "chainflow-score-on-diffusiondrive-v1"
K0_TOLERANCE = 1e-9


def read_json(path):
    return json.loads(Path(path).read_text())


def request_digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sources(request, needed=None):
    from .cross_reference import load_reference
    from .dd_baseline import load_dd_baseline

    if request["ks"] != KS or request["seed"] != 20260922 or request["gamma"] != .1:
        raise ValueError("Unexpected cross-planner K, seed or gamma")
    dd, provenance = load_dd_baseline(request["dd_source"], needed=needed)
    if provenance != request["dd_baseline"]:
        raise ValueError("DiffusionDrive baseline changed after launch")
    for key in ("data_root", "metric_cache"):
        if Path(provenance["protocol"][key]).resolve() != Path(request[key]).resolve():
            raise ValueError(f"DiffusionDrive source differs from selected {key}")
    reference, summary, reference_provenance = load_reference(request["reference_evaluation"], needed=needed)
    if reference_provenance != request["reference_provenance"] or reference_provenance["summary_sha256"] != request["reference_summary_sha256"]:
        raise ValueError("Original ChainFlow evaluation changed after launch")
    protocol = summary["protocol"]
    expected = {"score_epoch": 40, "score_weight_kind": "ema", "gamma": .1, "ks": KS,
                "score_checkpoint_sha256": request["score_checkpoint_sha256"],
                "identity": request["native_identity"], "scoring_identity": request["scoring_identity"]}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Original ChainFlow reference differs from this experiment: {key}")
    if set(dd) != set(reference) or (needed is not None and set(dd) != set(needed)):
        raise ValueError("DD and ChainFlow evaluations have different scene membership")
    for token in dd:
        if dd[token]["log"] != reference[token]["log"]:
            raise ValueError(f"Log mismatch between planners: {token}")
    return dd, reference, summary


def validate_row(row, token, dd, reference):
    """Bind a new/resumed record to both its independently verified source records."""
    if (row["token"] != token or row["valid"] is not True or row["log"] != dd["log"]
            or row["log"] != reference["log"] or row["native_origin"] != "diffusiondrive"
            or row["native_seed"] != dd["native_seed"] or row["top1_index"] != dd["top1_index"]
            or row["native_forward_executed"] is not False or row["native_ms"] != 0
            or row["condition_sha256"] != reference["condition_sha256"]
            or row["metric_cache_sha256"] != reference["metric_cache_sha256"]):
        raise ValueError(f"Cross-planner scene, condition or metric-cache mismatch: {token}")
    if row["trajectories"]["0"] != dd["trajectories"]["0"] or row["metrics"]["0"] != dd["metrics"]["0"]:
        raise ValueError(f"Saved DD K0 changed: {token}")
    audit = row["k0_metric_audit"]
    if (audit["trajectory_sha256"] != dd["trajectory_sha256"] or audit["tolerance"] != K0_TOLERANCE
            or set(audit["metrics"]) != set(METRICS)):
        raise ValueError(f"Missing DD K0 replay audit: {token}")
    errors = [abs(audit["metrics"][m] - dd["metrics"]["0"][m]) for m in METRICS]
    if any(not math.isfinite(v) or v > K0_TOLERANCE for v in errors) or audit["max_abs_error"] != max(errors):
        raise ValueError(f"DD K0 replay differs: {token}")
    arms = {str(k) for k in KS}
    if any(set(row[name]) != arms for name in ("trajectories", "metrics", "delta")):
        raise ValueError(f"Missing/unexpected K: {token}")
    for k in arms:
        poses, metrics = row["trajectories"][k], row["metrics"][k]
        if (len(poses) != 8 or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in poses)
                or set(metrics) != set(METRICS) or not all(math.isfinite(v) and 0 <= v <= 1 for v in metrics.values())
                or row["delta"][k] != {m: v - row["metrics"]["0"][m] for m, v in metrics.items()}):
            raise ValueError(f"Invalid trajectory, metrics or paired delta: {token}, K={k}")
    if any(not math.isfinite(row[key]) or row[key] < 0 for key in ("conditioner_ms", "refine_ms")):
        raise ValueError(f"Invalid timing: {token}")
    return {k: row[k] for k in ("token", "metrics", "delta", "refine_ms")}


def read_records(directory, manifest, dd, reference, *, completed):
    """Validate before truncating only an unfinished final line of an interrupted shard."""
    final, partial = directory / "scenes.jsonl", directory / "scenes.jsonl.tmp"
    if final.exists() and partial.exists():
        raise ValueError(f"Both final and temporary scene records exist: {directory}")
    if completed:
        marker = read_json(directory / "complete.json")
        if (marker.get("manifest") != manifest or marker.get("count") != len(manifest["shard_tokens"])
                or marker.get("scenes_sha256") != sha256(final)):
            raise ValueError(f"Invalid shard completion: {directory}")
    path = final if final.exists() else partial
    if not path.exists():
        return []
    tokens = manifest["shard_tokens"]
    rows, end = [], 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                if completed:
                    raise ValueError(f"Unfinished row in published results: {directory}")
                break
            if len(rows) >= len(tokens):
                raise ValueError(f"Too many rows in {directory}")
            token = tokens[len(rows)]
            rows.append(validate_row(json.loads(line), token, dd[token], reference[token]))
            end += len(line)
    if completed:
        if len(rows) != len(tokens):
            raise ValueError(f"Incomplete published records: {directory}")
    else:
        if path == final:
            final.replace(partial)
        with partial.open("r+b") as stream:
            stream.truncate(end)
    return rows


def shard_directory(output, index, count):
    return output / f"part-{index:04d}-of-{count:04d}"


def merge(output, request):
    from score_function.navsim_dd.evaluate import summarize
    output = Path(output).resolve()
    if read_json(output / "request.json") != request:
        raise ValueError("Cross-planner output is bound to a different request")
    dd, reference, original = sources(request)
    tokens = sorted(dd)
    if len(tokens) != TOTAL or hashlib.sha256("\n".join(tokens).encode()).hexdigest() != NAVTEST_TOKEN_DIGEST:
        raise ValueError("Cross-planner results require canonical navtest")
    workers = request["workers"]
    directories = [shard_directory(output, i, workers) for i in range(workers)]
    if set(output.glob("part-*-of-*")) != set(directories):
        raise ValueError("Missing, extra or stale cross-planner shards")
    rows, environment = [], None
    for i, directory in enumerate(directories):
        manifest = read_json(directory / "run.json")
        if (manifest["schema"] != SCHEMA or manifest["request_sha256"] != request_digest(request)
                or manifest["shard_index"] != i or manifest["shard_tokens"] != tokens[i::workers]):
            raise ValueError(f"Wrong cross-planner shard: {directory}")
        if environment is not None and environment != manifest["runtime"]:
            raise ValueError("Mixed cross-planner worker environments")
        environment = manifest["runtime"]
        require_matching_math(environment, original["protocol"]["execution_identity"])
        rows.extend(read_records(directory, manifest, dd, reference, completed=True))
    rows.sort(key=lambda r: r["token"])
    report = summarize(rows)
    report.update(schema=SCHEMA, mode="chainflow_score_on_dd", valid_count=len(rows), failed_count=0,
                  primary_k=20, protocol=request, runtime=environment,
                  chainflow_reference_by_k=original["by_k"], k0_replayed=len(rows))
    atomic_json(output / "summary.json", report)
    return report


def evaluate_pair(token, source, control, *, metric_hash, score, get_condition, refiner, device, gamma):
    """Replay the saved DD start, then refine only that trajectory using matching CF features."""
    import torch
    from . import evaluate as cf

    if metric_hash != control["metric_cache_sha256"]:
        raise ValueError("Metric cache differs from same-token ChainFlow reference")
    raw = torch.tensor(source["trajectories"]["0"], dtype=torch.float32, device=device)[None]
    replay = score(raw[0].cpu().numpy())
    error = cf.verify_k0_metrics(replay, source["metrics"]["0"])
    condition, encode_ms = cf.timed(get_condition, device)
    digest = cf.condition_digest(condition)
    if digest != control["condition_sha256"]:
        raise ValueError("Frozen scene/ego/VLM features differ from same-token ChainFlow reference")
    trajectories, metrics, delta, refine_ms = cf.score_arms(
        raw, condition, KS, refiner, score, device, gamma, source["metrics"]["0"])
    row = dict(token=token, log=source["log"], valid=True, native_origin="diffusiondrive",
               native_seed=source["native_seed"], top1_index=source["top1_index"],
               native_forward_executed=False, native_ms=0., condition_sha256=digest,
               metric_cache_sha256=metric_hash,
               k0_metric_audit=dict(metrics=replay, max_abs_error=error, tolerance=K0_TOLERANCE,
                                    trajectory_sha256=source["trajectory_sha256"]),
               trajectories=trajectories, metrics=metrics, delta=delta,
               conditioner_ms=encode_ms, refine_ms=refine_ms)
    validate_row(row, token, source, control)
    return row


def require_matching_math(current, reference):
    # Driver inventory and device names are recorded separately; compare the
    # math policies that control the actual tensor operations in both arms.
    for key in ("device_type", "matmul_allow_tf32", "cudnn_allow_tf32", "matmul_precision",
                "cudnn_deterministic", "cudnn_benchmark", "deterministic_algorithms", "cublas_workspace_config"):
        if key not in reference or current[key] != reference[key]:
            raise ValueError(f"Cross-planner math policy differs from original ChainFlow evaluation: {key}")


def run(request_path, index, device):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from . import evaluate as cf
    from .native import activate_devkit, encode_scene, load_agent, make_scene_loader
    from .runtime import ChainFlowRefiner

    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    root = activate_devkit(request["devkit_root"])
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score

    cf.apply_cuda_budget(device)
    torch.set_num_threads(1)
    torch.manual_seed(request["seed"])
    workers = request["workers"]
    loader = make_scene_loader(request["data_root"], root, "navtest")
    tokens, shard = cf.select_tokens(loader.tokens, None, index, workers)
    if len(tokens) != TOTAL or cf.digest_tokens(tokens) != NAVTEST_TOKEN_DIGEST:
        raise ValueError("Scene loader has the wrong navtest membership")
    dd, reference, predecessor = sources(request, needed=shard)
    metric_root = Path(request["metric_cache"])
    paths = cf.metric_index(metric_root)
    if set(shard) - set(paths):
        raise ValueError("Missing NAVSIM metric cache for this shard")
    if cf.scoring_identity(root) != request["scoring_identity"]:
        raise ValueError("NAVSIM scorer changed after launch")
    if sha256(request["score_checkpoint"]) != request["score_checkpoint_sha256"]:
        raise ValueError("Score checkpoint changed after launch")
    agent = load_agent(root, request["checkpoint"], request["backbone"], request["vlm_cache"], device)
    if agent._score_bridge_identity != request["native_identity"]:
        raise ValueError("Frozen ChainFlow native assets changed after launch")
    refiner = ChainFlowRefiner.from_checkpoint(request["score_checkpoint"], device)
    refiner.require_conditioner(agent._score_bridge_identity)
    if (refiner.sigma, refiner.gamma) != (.1, .1):
        raise ValueError("Wrong sigma/gamma in the ChainFlow Score checkpoint")
    class AllNodeMetricCache(MetricCacheLoader):
        _load_metric_cache_paths = staticmethod(cf.metric_index)
    cache = AllNodeMetricCache(metric_root)
    config = OmegaConf.load(root / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator, scorer = instantiate(config.simulator), instantiate(config.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Inconsistent NAVSIM scoring horizons")
    environment = cf.execution_identity(device)
    require_matching_math(environment, predecessor["protocol"]["execution_identity"])
    directory = shard_directory(request_path.parent, index, workers)
    manifest = dict(schema=SCHEMA, request_sha256=request_digest(request), runtime=environment,
                    shard_index=index, shard_tokens=shard)
    with lock(directory / ".lock"):
        manifest_path = directory / "run.json"
        if manifest_path.is_file():
            if read_json(manifest_path) != manifest:
                raise ValueError(f"Shard is bound to another protocol: {directory}")
        elif any((directory / name).exists() for name in ("scenes.jsonl.tmp", "scenes.jsonl", "complete.json")):
            raise ValueError(f"Cannot resume scene records without a shard manifest: {directory}")
        else:
            atomic_json(manifest_path, manifest)
        completed = (directory / "complete.json").exists()
        prefix = read_records(directory, manifest, dd, reference, completed=completed)
        if completed:
            return
        start = len(prefix)
        del prefix
        print(f"DD × ChainFlow shard {index}: resuming {start}/{len(shard)}", flush=True)
        path = directory / "scenes.jsonl.tmp"
        with path.open("a") as stream:
            for i in range(start, len(shard)):
                token = shard[i]
                source, control = dd[token], reference[token]
                try:
                    cache_hash = sha256(paths[token])
                    metric = cache.get_from_token(token)

                    def score(poses):
                        return asdict(pdm_score(metric, Trajectory(poses), simulator.proposal_sampling, simulator, scorer))

                    def get_condition():
                        scene = loader.get_scene_from_token(token)
                        metadata = scene.scene_metadata
                        if (metadata.initial_token, metadata.log_name) != (token, source["log"]):
                            raise ValueError("Scene loader token/log differs from DD trajectory")
                        return encode_scene(agent, scene, device)

                    row = evaluate_pair(token, source, control, metric_hash=cache_hash, score=score,
                                        get_condition=get_condition, refiner=refiner, device=device,
                                        gamma=request["gamma"])
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    if (i+1) % 25 == 0 or i+1 == len(shard):
                        print(f"shard {index}: {i+1}/{len(shard)}", flush=True)
                except Exception as exc:
                    atomic_json(directory / "failure.json", {"token": token, "completed": i, "error": str(exc)})
                    raise RuntimeError(f"Cross-planner evaluation failed at {token} ({i}/{len(shard)})") from exc
        final = directory / "scenes.jsonl"
        path.replace(final)
        atomic_json(directory / "complete.json", dict(manifest=manifest, count=len(shard), scenes_sha256=sha256(final)))
        (directory / "failure.json").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.request, args.shard_index, args.device)
