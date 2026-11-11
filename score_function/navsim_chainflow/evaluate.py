"""Native ChainFlow reproduction and strictly paired conditional-score evaluation.

The native sampler keeps its released token-dependent seed (27). A refined run
reads the completed native trajectories; it never invokes the native planner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path

import torch

from score_function.navsim_dd.evaluate import metric_index, select_tokens, summarize, timed
from score_function.utils.train_utils import atomic_write, file_hash, read_json

SCHEMA = "navsim-chainflow-evaluation-v1"
TOTAL = 12146
# SHA-256 of sorted newline-separated tokens in the pinned upstream navtest.yaml.
NAVTEST_TOKEN_DIGEST = "8d42fef68542f095d0e03894fdaf38babfdc89f3fd7cdc26080c3a6009cb4108"
KS = (5, 10, 15, 20, 30)
NATIVE_SEED = 27
METRICS = {"no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
           "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"}
K0_AUDIT_TOLERANCE = 1e-9
COMPATIBLE = ("schema", "split", "seed", "native_seed", "identity", "scoring_identity",
              "execution_identity", "token_digest", "count", "data_root", "metric_cache", "max_scenes")


def digest_tokens(tokens):
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def native_token_seed(token):
    """Mirror the released sampler only for recording/verification, not sampling."""
    return int(hashlib.md5(f"{token}:{NATIVE_SEED}".encode()).hexdigest()[:16], 16) % (2**63 - 1)


def condition_digest(condition):
    """Bind K0 to the same FP16-quantized frozen features used by score training."""
    digest = hashlib.sha256()
    for key in ("scene", "ego", "vlm"):
        value = condition[key].detach().to(device="cpu", dtype=torch.float16).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite condition: {key}")
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    mask = condition.get("vlm_mask")
    if mask is None:
        mask = torch.ones(condition["vlm"].shape[:2], dtype=torch.bool)
    mask = mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
    digest.update(mask.numpy().tobytes())
    return digest.hexdigest()


def scoring_identity(devkit_root):
    root = Path(devkit_root)
    names = [
        "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml",
        "navsim/evaluate/pdm_score.py",
        "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
    ]
    return {name: file_hash(root / name) for name in names}


def execution_identity(device):
    """Record official runtime math policy without changing it for reproduction."""
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "shapely", "timm", "diffusers",
                 "hydra-core", "omegaconf"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = {"packages": packages, "device_type": torch.device(device).type,
              "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if torch.device(device).type == "cuda":
        result.update(device_name=torch.cuda.get_device_name(device),
                      device_capability=list(torch.cuda.get_device_capability(device)))
    return result


def apply_cuda_budget(device):
    """The outer queue decides which devices are available; workers respect its cap."""
    device = torch.device(device)
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    limit = os.environ.get("CF_CUDA_MEMORY_GB")
    if limit:
        gib = float(limit)
        if not math.isfinite(gib) or gib <= 0:
            raise ValueError("CF_CUDA_MEMORY_GB must be positive")
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., gib * 1024**3 / total), device)


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_row(row, token, log_name, steps, baseline=None):
    arms = {str(k) for k in steps}
    if (row.get("token") != token or row.get("log") != log_name or row.get("valid") is not True
            or row.get("native_seed") != native_token_seed(token)
            or set(row.get("trajectories", {})) != arms
            or set(row.get("metrics", {})) != arms or set(row.get("delta", {})) != arms
            or not _valid_digest(row.get("condition_sha256"))
            or not _valid_digest(row.get("metric_cache_sha256"))):
        raise ValueError(f"Invalid evaluation record: {token}")
    for k in arms:
        poses, metrics = row["trajectories"][k], row["metrics"][k]
        if (len(poses) != 8 or any(len(pose) != 3 for pose in poses)
                or not all(math.isfinite(float(v)) for pose in poses for v in pose)
                or set(metrics) != METRICS
                or not all(math.isfinite(float(v)) and -1e-6 <= float(v) <= 1 + 1e-6
                           for v in metrics.values())
                or row["delta"][k] != {m: v - row["metrics"]["0"][m] for m, v in metrics.items()}):
            raise ValueError(f"Invalid trajectory/metric/delta: {token}, K={k}")
    for key in ("native_ms", "conditioner_ms", "refine_ms"):
        if not isinstance(row.get(key), (float, int)) or not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError(f"Invalid timing: {token}/{key}")
    if row.get("native_origin") == "external-handoff":
        audit = row.get("k0_metric_audit", {})
        if (row.get("native_forward_executed") is not False
                or row["native_ms"] != 0. or audit.get("tolerance") != K0_AUDIT_TOLERANCE
                or not _valid_digest(audit.get("trajectory_sha256"))):
            raise ValueError(f"Missing external K0 replay audit: {token}")
        verified = verify_k0_metrics(audit.get("metrics", {}), row["metrics"]["0"])
        if audit.get("max_abs_error") != verified:
            raise ValueError(f"External K0 replay audit changed: {token}")
    if baseline is not None:
        external = baseline.get("native_origin") == "external-handoff"
        keys = ("log", "native_seed") if external else ("log", "native_seed", "condition_sha256", "metric_cache_sha256")
        for key in keys:
            if row[key] != baseline[key]:
                raise ValueError(f"Reused native record differs: {token}/{key}")
        if external and (row.get("native_origin") != "external-handoff"
                         or row["k0_metric_audit"]["trajectory_sha256"] != baseline["trajectory_sha256"]):
            raise ValueError(f"External native trajectory provenance changed: {token}")
        if (row["trajectories"]["0"] != baseline["trajectories"]["0"]
                or row["metrics"]["0"] != baseline["metrics"]["0"]):
            raise ValueError(f"K0 changed: {token}")
    return row


def verify_k0_metrics(current, original):
    """Replay the saved trajectory, never the native sampler, before K0 reuse.

    External CSV/NPZ handoffs lack native-time per-cache hashes. Matching every
    metric under the current scorer is an explicit compatibility check, not a
    claim that missing historical hashes or conditions were recorded.
    """
    if set(current) != METRICS or set(original) != METRICS:
        raise ValueError("K0 metric replay requires all seven original metrics")
    errors = {key: abs(float(current[key]) - float(original[key])) for key in METRICS}
    if any(not math.isfinite(value) or value > K0_AUDIT_TOLERANCE for value in errors.values()):
        raise ValueError(f"Saved K0 metric replay differs from the handoff: {errors}")
    return max(errors.values())


def read_completed_shard(directory):
    directory = Path(directory)
    completed = read_json(directory / "complete.json")
    manifest = read_json(directory / "run.json")
    if completed.get("manifest") != manifest:
        raise ValueError(f"Completion manifest changed: {directory}")
    source = directory / "scenes.jsonl"
    if completed.get("scenes_sha256") != file_hash(source):
        raise ValueError(f"Completed scene records changed: {directory}")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if ([r.get("token") for r in rows] != manifest["shard_tokens"]
            or completed.get("count") != len(rows)):
        raise ValueError(f"Incomplete scene records: {directory}")
    for row in rows:
        validate_row(row, row["token"], row["log"], manifest["run"]["ks"])
    return manifest, rows


def collect_completed(output):
    """Verify exact all-shard coverage before any averages or K0 reuse."""
    output = Path(output)
    protocol, rows, ids, hashes = None, [], set(), {}
    directories = sorted(output.glob("part-*-of-*"))
    if not directories:
        raise ValueError(f"No completed evaluation shards: {output}")
    for directory in directories:
        manifest, part = read_completed_shard(directory)
        if protocol is None:
            protocol = manifest["run"]
        index = manifest["shard_index"]
        if manifest["run"] != protocol or index in ids:
            raise ValueError("Mixed/duplicate evaluation shard protocols")
        ids.add(index)
        rows.extend(part)
        hashes[str(directory.relative_to(output) / "complete.json")] = file_hash(directory / "complete.json")
        hashes[str(directory.relative_to(output) / "scenes.jsonl")] = file_hash(directory / "scenes.jsonl")
    tokens = sorted(row["token"] for row in rows)
    if (ids != set(range(protocol["num_shards"])) or len(tokens) != protocol["count"]
            or len(set(tokens)) != len(tokens) or digest_tokens(tokens) != protocol["token_digest"]):
        raise ValueError("Incomplete/duplicate full evaluation coverage")
    # Validate each shard's actual sorted stride, not just the union.
    for directory in directories:
        manifest = read_json(directory / "run.json")
        if manifest["shard_tokens"] != tokens[manifest["shard_index"]::protocol["num_shards"]]:
            raise ValueError("Incorrect deterministic shard assignment")
    # Aggregate in token order so changing the worker count cannot even change
    # the order of floating-point summation for the reused native mean.
    return protocol, sorted(rows, key=lambda row: row["token"]), hashes


def _verified_summary(output, protocol, rows):
    summary = read_json(Path(output) / "summary.json")
    actual = summarize(rows)
    if (summary.get("protocol") != protocol or summary.get("mode") != protocol["mode"]
            or summary.get("valid_count") != len(rows) or summary.get("failed_count") != 0
            or any(summary.get(key) != value for key, value in actual.items())):
        raise ValueError("Evaluation summary does not match completed scene records")
    return summary


def collect_baseline(output, expected_count=TOTAL):
    """Read either our shards or the independently produced immutable CSV/NPZ."""
    from .handoff import is_handoff, load_handoff
    if not is_handoff(output):
        return collect_completed(output)
    imported = load_handoff(output, expected_count=expected_count,
                            expected_token_digest=NAVTEST_TOKEN_DIGEST if expected_count == TOTAL else None)
    rows = sorted(imported["rows"].values(), key=lambda row: row["token"])
    protocol = dict(schema=imported["schema"], mode="baseline", split="navtest", ks=[0],
                    native_seed=NATIVE_SEED, count=len(rows), token_digest=digest_tokens([r["token"] for r in rows]),
                    identity=imported["identity"], external_handoff=True,
                    handoff_manifest=imported["manifest"], handoff_summary=imported["summary"],
                    handoff_provenance=imported["provenance"])
    return protocol, rows, imported["provenance"]["files_sha256"]


def baseline_reference(output, protocol=None, hashes=None):
    if protocol is None or hashes is None:
        protocol, _, hashes = collect_baseline(output)
    if protocol.get("external_handoff"):
        return protocol["handoff_provenance"]
    return {"source": str(Path(output).resolve()), "files_sha256": hashes,
            "summary_sha256": file_hash(Path(output) / "summary.json")}


def audit_external_baseline(output, devkit_root, metric_cache, expected_identity=None, max_scenes=8):
    """CPU compatibility preflight on diverse saved K0s before expensive training.

    All tokens are checked again during paired evaluation. This small check is
    explicitly not certification of full metric-cache or historical feature parity.
    """
    from .native import activate_devkit
    import numpy as np
    activate_devkit(devkit_root)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    protocol, rows, hashes = collect_baseline(output)
    if not protocol.get("external_handoff"):
        raise ValueError("Saved-K0 preflight requires an external handoff")
    _require_external_identity(protocol, expected_identity)
    if Path(protocol["handoff_manifest"]["metric_cache"]).resolve() != Path(metric_cache).resolve():
        raise ValueError("External baseline metric cache path differs")
    paths = metric_index(Path(metric_cache))
    if set(row["token"] for row in rows) - paths.keys():
        raise ValueError("Missing native metric-cache coverage")
    class AllNodeMetricCache(MetricCacheLoader):
        _load_metric_cache_paths = staticmethod(metric_index)
    cache = AllNodeMetricCache(Path(metric_cache))
    cfg = OmegaConf.load(Path(devkit_root) / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("PDM scorer and simulator horizons differ")
    ordered = sorted(rows, key=lambda row: (row["metrics"]["0"]["score"], row["token"]))
    selected = {ordered[round(i * (len(rows)-1) / max(1, min(max_scenes, len(rows))-1))]["token"]
                for i in range(min(max_scenes, len(rows)))}
    checks = []
    for row in rows:
        if row["token"] not in selected:
            continue
        token = row["token"]
        poses = np.asarray(row["trajectories"]["0"], dtype=row["trajectory_dtype"])
        metrics = asdict(pdm_score(cache.get_from_token(token), Trajectory(poses),
                                  simulator.proposal_sampling, simulator, scorer))
        checks.append(dict(token=token, max_abs_error=verify_k0_metrics(metrics, row["metrics"]["0"]),
                           metric_cache_sha256=file_hash(paths[token])))
        print(f"Saved K0 metric preflight: {len(checks)}/{len(selected)} {token}", flush=True)
    return dict(scope="sampled_saved_trajectory_metric_replay", passed=True,
                sample_count=len(checks), total_count=len(rows), checks=checks,
                full_replay_required_during_evaluation=True, tolerance=K0_AUDIT_TOLERANCE,
                scoring_identity=scoring_identity(devkit_root), baseline=baseline_reference(output, protocol, hashes))


def _require_external_identity(protocol, expected_identity):
    if expected_identity is None:
        return
    for key in ("chainflow_commit", "checkpoint_sha256", "backbone_sha256"):
        if not expected_identity.get(key) or protocol["identity"].get(key) != expected_identity[key]:
            raise ValueError(f"External native conditioner identity differs: {key}")


def validate_native_baseline(output, expected_identity=None, expected_scoring=None, expected_count=TOTAL):
    """CPU-only gate for an independently completed native reproduction handoff.

    Callers should obtain expected_identity from native.conditioner_identity and
    expected_scoring from scoring_identity using the assets they will train with.
    No aggregate CSV is accepted in place of saved per-token native trajectories.
    """
    protocol, rows, _ = collect_baseline(output, expected_count=expected_count)
    if protocol.get("external_handoff"):
        _require_external_identity(protocol, expected_identity)
        # The original score config is hash-checked by the adapter; current
        # scorer equivalence is established per token by saved-K0 metric replay.
        return {**protocol["handoff_summary"], "mode": "baseline", "protocol": protocol,
                "current_scoring_identity": expected_scoring,
                "k0_metric_replay_required": True}
    if (protocol.get("schema") != SCHEMA or protocol.get("mode") != "baseline"
            or protocol.get("split") != "navtest" or protocol.get("ks") != [0]
            or protocol.get("native_seed") != NATIVE_SEED or len(rows) != expected_count):
        raise ValueError("Require complete canonical native ChainFlow navtest reproduction")
    if expected_count == TOTAL and protocol.get("token_digest") != NAVTEST_TOKEN_DIGEST:
        raise ValueError("Native baseline token membership differs from the pinned official navtest")
    if expected_identity is not None and protocol.get("identity") != expected_identity:
        raise ValueError("Native baseline conditioner identity differs from the training assets")
    if expected_scoring is not None and protocol.get("scoring_identity") != expected_scoring:
        raise ValueError("Native baseline scoring identity differs from the current evaluator")
    return _verified_summary(output, protocol, rows)


def load_baseline(output, expected, selected, needed=None):
    protocol, rows, hashes = collect_baseline(output, expected_count=len(selected))
    external = protocol.get("external_handoff", False)
    if protocol.get("mode") != "baseline" or protocol["ks"] != [0]:
        raise ValueError("Require a native ChainFlow baseline, not DD or a refined run")
    if external:
        _require_external_identity(protocol, expected.get("identity"))
        if (expected.get("split") != "navtest" or expected.get("native_seed") != NATIVE_SEED
                or expected.get("token_digest") != protocol["token_digest"]):
            raise ValueError("External baseline split/seed/token membership differs")
        original_cache = Path(protocol["handoff_manifest"]["metric_cache"]).resolve()
        if original_cache != Path(expected["metric_cache"]).resolve():
            raise ValueError("External baseline metric cache path differs")
    else:
        for key in COMPATIBLE:
            if protocol.get(key) != expected.get(key):
                raise ValueError(f"Baseline protocol mismatch: {key}")
    if sorted(r["token"] for r in rows) != selected:
        raise ValueError("Baseline token membership changed")
    if not external:
        _verified_summary(output, protocol, rows)
    required = set(selected if needed is None else needed)
    result = {r["token"]: r for r in rows if r["token"] in required}
    if set(result) != required:
        raise ValueError("Missing required native baseline tokens")
    return result, baseline_reference(output, protocol, hashes)


def resume_prefix(directory, manifest, logs, baseline=None):
    """Keep all complete valid rows; only truncate a final partial write."""
    directory = Path(directory)
    path = directory / "scenes.jsonl.tmp"
    if not path.exists():
        if (directory / "scenes.jsonl").exists():
            path = directory / "scenes.jsonl"
        else:
            return []
    rows, offset = [], 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            if len(rows) >= len(manifest["shard_tokens"]):
                raise ValueError("Too many resumable records")
            token = manifest["shard_tokens"][len(rows)]
            row = json.loads(line)
            validate_row(row, token, logs[token], manifest["run"]["ks"],
                         None if baseline is None else baseline[token])
            rows.append(row)
            offset += len(line)
    destination = directory / "scenes.jsonl.tmp"
    if path != destination:
        path.replace(destination)
    with destination.open("r+b") as stream:
        stream.truncate(offset)
    return rows


def score_arms(raw, condition, steps, refiner, score_fn, device, gamma=None, baseline_metrics=None):
    if refiner is None:
        if steps != [0]:
            raise ValueError("Nonzero K requires a score model")
        predictions, elapsed = {0: raw}, 0.
    else:
        predictions, elapsed = timed(lambda: refiner.refine_many(raw.clone(), condition, steps, gamma=gamma), device)
        if set(predictions) != set(steps) or not torch.equal(predictions[0], raw):
            raise ValueError("Refiner changed K0 or returned incorrect K arms")
    trajectories, metrics = {}, {}
    for k in steps:
        array = predictions[k][0].detach().cpu().numpy()
        if array.shape != (8, 3):
            raise ValueError(f"Invalid trajectory shape for K={k}")
        trajectories[str(k)] = array.tolist()
        result = baseline_metrics if k == 0 and baseline_metrics is not None else score_fn(array)
        metrics[str(k)] = {name: float(v) for name, v in result.items()}
    delta = {k: {name: v - metrics["0"][name] for name, v in arm.items()} for k, arm in metrics.items()}
    return trajectories, metrics, delta, elapsed


def merge_shards(output):
    protocol, rows, _ = collect_completed(output)
    report = {**summarize(rows), "valid_count": len(rows), "failed_count": 0,
              "protocol": protocol, "mode": protocol["mode"]}
    if protocol.get("baseline"):
        report["baseline"] = protocol["baseline"]
    atomic_write(Path(output) / "summary.json", report)
    return report


def run(args):
    # Heavy upstream imports happen only in a configured worker process.
    from .native import activate_devkit, load_agent, make_scene_loader, native_predict, encode_scene
    activate_devkit(args.devkit_root)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    apply_cuda_budget(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    agent = load_agent(args.devkit_root, args.checkpoint, args.backbone, args.vlm_cache, args.device)
    identity = agent._score_bridge_identity
    loader = make_scene_loader(args.data_root, args.devkit_root, "navtest")
    selected, shard = select_tokens(loader.tokens, args.max_scenes, args.shard_index, args.num_shards)
    if args.max_scenes is None and len(selected) != TOTAL:
        raise ValueError(f"Expected {TOTAL} canonical navtest tokens, got {len(selected)}")
    if len(selected) == TOTAL and digest_tokens(selected) != NAVTEST_TOKEN_DIGEST:
        raise ValueError("Loaded scene tokens differ from the pinned official navtest")
    logs = {t: log for log, ts in loader.get_tokens_list_per_log().items() for t in ts}
    paths = metric_index(args.metric_cache)
    missing = set(selected) - set(paths)
    if missing:
        raise ValueError(f"Missing metric cache tokens: {len(missing)}; examples {sorted(missing)[:5]}")

    class AllNodeMetricCache(MetricCacheLoader):
        _load_metric_cache_paths = staticmethod(metric_index)

    cache = AllNodeMetricCache(args.metric_cache)
    cfg = OmegaConf.load(args.devkit_root / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("PDM scorer and simulator horizons differ")
    refiner, saved = None, None
    if args.mode == "refine":
        from .runtime import ChainFlowRefiner
        refiner = ChainFlowRefiner.from_checkpoint(args.score_checkpoint, args.device)
        refiner.require_conditioner(identity)
        saved = torch.load(args.score_checkpoint, map_location="cpu", weights_only=True)
    steps = [0] if args.mode == "baseline" else sorted({0, *args.ks})
    protocol = dict(schema=SCHEMA, mode=args.mode, split="navtest", seed=args.seed, native_seed=NATIVE_SEED,
                    identity=identity, scoring_identity=scoring_identity(args.devkit_root),
                    execution_identity=execution_identity(args.device),
                    count=len(selected), token_digest=digest_tokens(selected), num_shards=args.num_shards,
                    data_root=str(args.data_root.resolve()), metric_cache=str(args.metric_cache.resolve()),
                    max_scenes=args.max_scenes, ks=steps, gamma=args.gamma if refiner else None,
                    score_checkpoint_sha256=file_hash(args.score_checkpoint) if refiner else None,
                    score_epoch=None if saved is None else saved.get("epoch"),
                    score_weight_kind=None if saved is None else saved.get("weight_kind"))
    baseline = None
    if args.mode == "refine":
        baseline, provenance = load_baseline(args.baseline, protocol, selected, shard)
        protocol["baseline"] = provenance
    directory = args.output / f"part-{args.shard_index:04d}-of-{args.num_shards:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(run=protocol, shard_index=args.shard_index, shard_tokens=shard)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "run.json").exists() and read_json(directory / "run.json") != manifest:
            raise ValueError(f"Output belongs to another evaluation protocol: {directory}")
        if (directory / "complete.json").exists():
            read_completed_shard(directory)
            print(f"Already complete: {directory}", flush=True)
            return
        atomic_write(directory / "run.json", manifest)
        rows = resume_prefix(directory, manifest, logs, baseline)
        print(f"{args.mode} shard {args.shard_index}: resuming {len(rows)}/{len(shard)}", flush=True)
        with (directory / "scenes.jsonl.tmp").open("a", encoding="utf-8") as stream:
            for index in range(len(rows), len(shard)):
                token = shard[index]
                try:
                    scene = loader.get_scene_from_token(token)
                    cache_hash = file_hash(paths[token])
                    native_ms, conditioner_ms = 0., 0.
                    if baseline is None:
                        (raw, condition), native_ms = timed(lambda: native_predict(agent, scene, args.device), args.device)
                    else:
                        source = baseline[token]
                        external = source.get("native_origin") == "external-handoff"
                        if source["log"] != logs[token] or (not external and source["metric_cache_sha256"] != cache_hash):
                            raise ValueError("Native K0 log or metric cache changed")
                        dtype = getattr(torch, source["trajectory_dtype"]) if external else torch.float32
                        raw = torch.tensor(source["trajectories"]["0"], device=args.device, dtype=dtype)[None]
                        condition, conditioner_ms = timed(lambda: encode_scene(agent, scene, args.device), args.device)
                    condition_hash = condition_digest(condition)
                    if baseline is not None and not external and condition_hash != baseline[token]["condition_sha256"]:
                        raise ValueError("Frozen scene/VLM conditions differ from the native baseline")
                    metric = cache.get_from_token(token)
                    def score_fn(poses):
                        return asdict(pdm_score(metric, Trajectory(poses), simulator.proposal_sampling, simulator, scorer))
                    audit = None
                    if baseline is not None and external:
                        replay = score_fn(raw[0].detach().cpu().numpy())
                        audit = dict(metrics=replay, max_abs_error=verify_k0_metrics(replay, source["metrics"]["0"]),
                                     tolerance=K0_AUDIT_TOLERANCE, trajectory_sha256=source["trajectory_sha256"])
                    trajectories, metrics, delta, elapsed = score_arms(
                        raw, condition, steps, refiner, score_fn, args.device, args.gamma,
                        None if baseline is None else baseline[token]["metrics"]["0"],
                    )
                    row = dict(token=token, log=logs[token], valid=True, native_seed=native_token_seed(token),
                               condition_sha256=condition_hash, metric_cache_sha256=cache_hash,
                               trajectories=trajectories, metrics=metrics, delta=delta,
                               native_ms=native_ms, conditioner_ms=conditioner_ms, refine_ms=elapsed)
                    if audit is not None:
                        row.update(native_origin="external-handoff", native_forward_executed=False,
                                   k0_metric_audit=audit,
                                   condition_hash_scope="first_observed_during_refinement")
                    validate_row(row, token, logs[token], steps, None if baseline is None else baseline[token])
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    if (index + 1) % 10 == 0 or index + 1 == len(shard):
                        print(f"{args.mode} shard {args.shard_index}: {index+1}/{len(shard)}", flush=True)
                except Exception as exc:
                    atomic_write(directory / "failure.json", {"token": token, "completed": len(rows),
                                                             "error": f"{type(exc).__name__}: {exc}"})
                    raise RuntimeError(f"Evaluation failed at {token}; completed records are resumable") from exc
        (directory / "scenes.jsonl.tmp").replace(directory / "scenes.jsonl")
        atomic_write(directory / "complete.json", {"manifest": manifest, "count": len(rows),
                                                   "scenes_sha256": file_hash(directory / "scenes.jsonl")})
        if (directory / "failure.json").exists():
            (directory / "failure.json").unlink()
    if args.num_shards == 1:
        merge_shards(args.output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "refine"), default="baseline")
    for name in ("devkit-root", "checkpoint", "backbone", "vlm-cache", "data-root", "metric-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--score-checkpoint", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--ks", nargs="+", type=int, default=list(KS))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args(argv)
    if not math.isfinite(args.gamma) or args.gamma < 0 or any(k < 1 for k in args.ks):
        parser.error("Require nonnegative finite gamma and positive refinement K values")
    if args.mode == "refine" and (args.baseline is None or args.score_checkpoint is None):
        parser.error("Refinement requires --baseline and --score-checkpoint")
    if args.baseline and args.baseline.resolve() == args.output.resolve():
        parser.error("Baseline and refinement outputs must differ")
    return args


if __name__ == "__main__":
    run(parse_args())
