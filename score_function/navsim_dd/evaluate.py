"""Paired DiffusionDrive + score evaluation against the official NAVSIM v1 scorer."""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from score_function.utils.train_utils import file_hash


def token_seed(seed, token):
    digest = hashlib.sha256(f"{seed}\0{token}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@contextmanager
def sampling_rng(seed, device):
    """Assign a repeatable noise stream independent of token/shard processing order."""
    device = torch.device(device)
    cuda_index = (torch.cuda.current_device() if device.index is None else device.index) if device.type == "cuda" else None
    with torch.random.fork_rng(devices=[] if cuda_index is None else [cuda_index]):
        torch.random.default_generator.manual_seed(seed)
        if cuda_index is not None:
            torch.cuda.default_generators[cuda_index].manual_seed(seed)
        yield


def select_tokens(tokens, max_scenes, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if max_scenes is not None and max_scenes < 1:
        raise ValueError("max-scenes must be positive")
    selected = sorted(tokens)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate scene tokens")
    selected = selected[:max_scenes]
    if len(selected) < num_shards:
        raise ValueError("Fewer selected scenes than evaluation shards")
    return selected, selected[shard_index::num_shards]


def require_coverage(expected, observed, name):
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"Missing {len(missing)} {name}; examples: {missing[:8]}")


def timed(fn, device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    result = fn()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return result, 1000 * (perf_counter() - start)


@torch.no_grad()
def native_top1(agent, features):
    """Observe the last decoder outputs without changing the native DD rollout."""
    last = []

    def capture(_module, _inputs, outputs):
        poses, logits = outputs
        last[:] = [poses[-1], logits[-1]]

    decoder = agent._transfuser_model._trajectory_head.diff_decoder
    handle = decoder.register_forward_hook(capture)
    try:
        raw = agent(features)["trajectory"]
    finally:
        handle.remove()
    if len(last) != 2 or raw.shape != (1, 8, 3) or not torch.isfinite(raw).all():
        raise ValueError("Invalid native DiffusionDrive prediction")
    poses, logits = last
    index = int(logits[0].argmax().item())
    if not torch.equal(raw[0], poses[0, index]):
        raise ValueError("Native returned trajectory differs from its last decoder top-1")
    return raw, index


def score_pose_group(raw, steps, refiner, context, global_context, gamma, scorer, device, *, baseline_metrics=None):
    """Every arm starts from exactly the same raw tensor; K0 performs no conversion."""
    trajectories, scores = {}, {}
    if steps == [0]:
        predictions, refine_ms = {0: raw}, 0.0
    else:
        predictions, refine_ms = timed(
            lambda: refiner.refine_many(raw.clone(), context, global_context, steps, gamma=gamma), device
        )
        if not torch.equal(predictions[0], raw):
            raise ValueError("K0 differs from the shared native initial trajectory")
    for k in steps:
        prediction = predictions[k]
        if prediction.shape != (1, 8, 3) or not torch.isfinite(prediction).all():
            raise ValueError(f"Invalid trajectory for K={k}")
        array = prediction.detach().cpu().numpy()[0]
        values = baseline_metrics if k == 0 and baseline_metrics is not None else scorer(array)
        metrics = {name: float(value) for name, value in values.items()}
        if 'score' not in metrics or not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Invalid PDM subscore for K={k}: {metrics}")
        trajectories[k], scores[k] = array.tolist(), metrics
    deltas = {k: {name: value - scores[0][name] for name, value in metrics.items()} for k, metrics in scores.items()}
    return trajectories, scores, deltas, refine_ms


def summarize(rows):
    if not rows or len({row['token'] for row in rows}) != len(rows):
        raise ValueError("Expected non-empty results with unique tokens")
    arms = list(rows[0]['metrics'])
    stats = {}
    for k in arms:
        metrics = rows[0]['metrics'][k].keys()
        deltas = [row['delta'][k]['score'] for row in rows]
        stats[k] = {
            'mean': {m: sum(row['metrics'][k][m] for row in rows) / len(rows) for m in metrics},
            'mean_delta': {m: sum(row['delta'][k][m] for row in rows) / len(rows) for m in metrics},
            'improved': sum(v > 0 for v in deltas), 'worsened': sum(v < 0 for v in deltas),
            'tied': sum(v == 0 for v in deltas),
        }
    return {'count': len(rows), 'score_scale': '0..1', 'by_k': stats,
            'mean_refinement_ms_through_max_k': sum(row['refine_ms'] for row in rows) / len(rows)}


def dump_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest_tokens(tokens):
    return hashlib.sha256('\n'.join(tokens).encode()).hexdigest()


def load_reusable_k0(source, expected_run, selected, needed=None):
    """Read a completed baseline without modifying it; reject changed protocols."""
    source = Path(source).resolve()
    summary = json.loads((source / 'summary.json').read_text())
    if summary.get('count') != len(selected) or '0' not in summary.get('by_k', {}):
        raise ValueError('Reusable K0 needs a completed summary for the full selected split')
    required = set(selected if needed is None else needed)
    records, observed, ids, hashes = {}, [], set(), {}
    original = None
    score_sum = 0.0
    compatible = ('split', 'seed', 'identity', 'token_digest', 'count',
                  'data_root', 'max_scenes', 'metric_cache')
    for directory in sorted(source.glob('part-*-of-*')):
        manifest = json.loads((directory / 'complete.json').read_text())
        if json.loads((directory / 'run.json').read_text()) != manifest:
            raise ValueError(f'Baseline run/completion manifest mismatch: {directory}')
        run = manifest['run']
        if original is None:
            original = run
            for key in compatible:
                if key not in run or key not in expected_run or run[key] != expected_run[key]:
                    raise ValueError(f'Cannot reuse K0: {key} differs from the completed baseline')
            if 0 not in run['ks']:
                raise ValueError('Baseline has no K0 arm')
        index = manifest['shard_index']
        if run != original or index in ids:
            raise ValueError('Incompatible or duplicate baseline shards')
        ids.add(index)
        path = directory / 'scenes.jsonl'
        digest, shard_tokens = hashlib.sha256(), []
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                token, trajectory, metrics = row['token'], row['trajectories']['0'], row['metrics']['0']
                if row['seed'] != token_seed(run['seed'], token):
                    raise ValueError(f'Baseline token seed mismatch: {token}')
                if (len(trajectory) != 8 or any(len(pose) != 3 for pose in trajectory)
                        or not all(math.isfinite(float(v)) for pose in trajectory for v in pose)
                        or 'score' not in metrics or not all(math.isfinite(float(v)) for v in metrics.values())):
                    raise ValueError(f'Invalid baseline trajectory/metrics: {token}')
                shard_tokens.append(token)
                score_sum += float(metrics['score'])
                if token in required:
                    if token in records:
                        raise ValueError(f'Duplicate baseline token: {token}')
                    records[token] = dict(trajectory=trajectory, metrics=metrics,
                                          top1_index=row['top1_index'], seed=row['seed'])
        if sorted(shard_tokens) != manifest['shard_tokens']:
            raise ValueError(f'Incomplete baseline records: {directory}')
        observed.extend(shard_tokens)
        hashes[str(path.relative_to(source))] = digest.hexdigest()
        hashes[str((directory / 'complete.json').relative_to(source))] = file_hash(directory / 'complete.json')
    if (original is None or ids != set(range(original['num_shards']))
            or sorted(observed) != sorted(selected) or len(set(observed)) != len(observed)
            or digest_tokens(sorted(observed)) != expected_run['token_digest']
            or set(records) != required):
        raise ValueError('Incomplete reusable K0 token/shard coverage')
    if not math.isclose(score_sum / len(observed), summary['by_k']['0']['mean']['score'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Baseline K0 summary differs from its scene records')
    return records, dict(source=str(source), files_sha256=hashes,
                         summary_sha256=file_hash(source / 'summary.json'))


def bounded_loader(scene_filter, available_logs, limit, load):
    """Load just enough official samples from the local logs, in deterministic log order."""
    if limit < 1:
        raise ValueError("max-scenes must be positive")
    result = None
    for name in sorted(set(scene_filter.log_names) & set(available_logs)):
        remaining = limit - (len(result) if result is not None else 0)
        part = load(replace(scene_filter, log_names=[name], max_scenes=remaining))
        if result is None:
            result = part
        else:
            if result.scene_frames_dicts.keys() & part.scene_frames_dicts.keys():
                raise ValueError(f"Duplicate tokens in official log {name}")
            result.scene_frames_dicts.update(part.scene_frames_dicts)
        if len(result) == limit:
            break
    if result is None or len(result) != limit:
        raise ValueError(f"Local official logs supply fewer than {limit} requested scenes")
    return result


def metric_index(cache_root):
    """Combine official node-level CSV metadata, including uneven or empty worker shards."""
    paths = {}
    files = sorted((cache_root / 'metadata').glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No metric-cache CSV manifests in {cache_root}/metadata')
    for file in files:
        with file.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                continue  # nuPlan writes an empty CSV when the worker had no samples.
            if 'file_name' not in reader.fieldnames:
                raise ValueError(f'Invalid metric-cache metadata header in {file}')
            for row in reader:
                path = Path(row['file_name'])
                token = path.parent.name
                if not token or (token in paths and paths[token] != path):
                    raise ValueError(f'Conflicting or invalid token metadata for {token!r} in {file}')
                paths[token] = path
    return paths


def build_loader(args, agent, config_root):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from .cache import build_navtrain_filter

    if args.split == 'val':
        scene_filter = build_navtrain_filter('val', args.devkit_root)
        data_split = 'trainval'
    else:
        scene_filter = instantiate(OmegaConf.load(config_root / 'common/train_test_split/scene_filter/navtest.yaml'))
        data_split = 'test'
    root = args.data_root
    sensor_dir = root / 'sensor_blobs' / data_split
    log_dir = root / 'navsim_logs' / data_split
    if not sensor_dir.is_dir() or not log_dir.is_dir():
        raise FileNotFoundError(f"Expected NAVSIM log and sensor directories: {log_dir}, {sensor_dir}")
    available_logs = {p.stem for p in log_dir.glob('*.pkl')}

    def load(filt):
        return SceneLoader(data_path=log_dir, sensor_blobs_path=sensor_dir, scene_filter=filt,
                           sensor_config=agent.get_sensor_config())

    if args.max_scenes is not None:
        return bounded_loader(scene_filter, available_logs, args.max_scenes, load)
    require_coverage(scene_filter.log_names, available_logs, 'official scene logs')
    loader = load(scene_filter)
    if args.split == 'navtest':
        require_coverage(scene_filter.tokens, loader.tokens, 'official navtest tokens in loaded logs')
    return loader


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='native DiffusionDrive checkpoint')
    parser.add_argument('--anchor', type=Path)
    parser.add_argument('--backbone', type=Path)
    parser.add_argument('--score-checkpoint', type=Path)
    parser.add_argument('--metric-cache', type=Path, help='official metric cache for the selected split')
    parser.add_argument('--data-root', type=Path, default=os.environ.get('OPENSCENE_DATA_ROOT'))
    parser.add_argument('--devkit-root', type=Path, default=os.environ.get('NAVSIM_DEVKIT_ROOT'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=['val', 'navtest'], default='navtest')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--ks', nargs='+', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None, help='override checkpoint default step size')
    parser.add_argument('--max-scenes', type=int)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--native-only', action='store_true', help='only evaluate the unchanged native trajectory')
    parser.add_argument('--verify-native', action='store_true', help='on the first token compare native and extracted conditions')
    parser.add_argument('--reuse-k0-from', type=Path,
                        help='reuse trajectories and K0 scores from a completed compatible paired evaluation')
    parser.add_argument('--merge', action='store_true', help='merge completed shards and check coverage')
    args = parser.parse_args(argv)
    if args.merge:
        return args
    for name in ('checkpoint', 'anchor', 'backbone', 'metric_cache', 'data_root'):
        if getattr(args, name) is None:
            parser.error(f'missing --{name.replace("_", "-")} (or corresponding environment variable)')
    if args.gamma is not None and (not math.isfinite(args.gamma) or args.gamma < 0):
        parser.error('--gamma must be finite and nonnegative')
    if args.native_only and args.score_checkpoint:
        parser.error('use either --native-only or --score-checkpoint')
    if args.native_only and args.reuse_k0_from:
        parser.error('--reuse-k0-from requires a score checkpoint for a refined arm')
    if not args.native_only and not args.score_checkpoint:
        parser.error('supply --score-checkpoint or --native-only')
    args.ks = sorted(set(args.ks or ([0] if args.native_only else [0, 20])) | {0})
    if args.reuse_k0_from and args.ks == [0]:
        parser.error('--reuse-k0-from requires at least one nonzero refinement step')
    if args.ks[0] < 0 or (args.native_only and args.ks != [0]):
        parser.error('invalid steps for the requested evaluation')
    return args


def merge_shards(output):
    dirs = sorted(output.glob('part-*-of-*'))
    if not dirs:
        raise ValueError(f'No evaluation shards under {output}')
    run, rows, ids = None, [], set()
    for directory in dirs:
        manifest = json.loads((directory / 'complete.json').read_text())
        if run is None:
            run = manifest['run']
        if manifest['run'] != run or manifest['shard_index'] in ids:
            raise ValueError(f'Incompatible or duplicate evaluation shard {directory}')
        ids.add(manifest['shard_index'])
        with (directory / 'scenes.jsonl').open() as f:
            shard_rows = [json.loads(line) for line in f]
        if sorted(row['token'] for row in shard_rows) != manifest['shard_tokens']:
            raise ValueError(f'Incomplete records in {directory}')
        rows.extend(shard_rows)
    if ids != set(range(run['num_shards'])) or digest_tokens(sorted(row['token'] for row in rows)) != run['token_digest']:
        raise ValueError('Incomplete merged token coverage')
    report = summarize(rows)
    if run.get('k0_reuse'):
        report['k0_reuse'] = run['k0_reuse']
    dump_json(output / 'summary.json', report)
    print(json.dumps(report, indent=2))
    return report


def run(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import MetricCacheLoader
    from .cache import conditioner_identity, verified_devkit_root
    from .conditioner import DDConditioner, load_diffusiondrive
    from .runtime import NavsimRefiner

    class AllNodeMetricCache(MetricCacheLoader):
        # Upstream get_from_token handles its serialized payload; override only its
        # first-CSV index lookup for caches generated across multiple nodes.
        _load_metric_cache_paths = staticmethod(metric_index)

    config_root = verified_devkit_root(args.devkit_root) / "navsim/planning/script/config"
    agent = load_diffusiondrive(args.checkpoint, args.anchor, args.backbone, args.device)
    identity = conditioner_identity(args.checkpoint, args.anchor, args.backbone)
    refiner = None
    if not args.native_only:
        refiner = NavsimRefiner.from_checkpoint(args.score_checkpoint, args.device)
        refiner.require_conditioner(identity)
    conditioner = DDConditioner(agent)
    loader = build_loader(args, agent, config_root)
    selected, shard = select_tokens(loader.tokens, args.max_scenes, args.shard_index, args.num_shards)
    logs = {token: log for log, tokens in loader.get_tokens_list_per_log().items() for token in tokens}
    cache = AllNodeMetricCache(args.metric_cache)
    require_coverage(shard, cache.tokens, 'metric cache tokens')
    config = OmegaConf.load(config_root / 'pdm_scoring/default_scoring_parameters.yaml')
    simulator, scorer = instantiate(config.simulator), instantiate(config.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError('Inconsistent PDM simulation/scoring horizons')
    run_config = dict(split=args.split, seed=args.seed, ks=args.ks, gamma=args.gamma, identity=identity,
                      num_shards=args.num_shards, token_digest=digest_tokens(selected), count=len(selected),
                      data_root=str(args.data_root.resolve()), max_scenes=args.max_scenes,
                      score_checkpoint_sha256=file_hash(args.score_checkpoint) if refiner else None,
                      metric_cache=str(args.metric_cache.resolve()))
    baseline = None
    if args.reuse_k0_from is not None:
        if args.reuse_k0_from.resolve() == args.output.resolve():
            raise ValueError('Use a separate output directory when reusing K0')
        baseline, provenance = load_reusable_k0(args.reuse_k0_from, run_config, selected, shard)
        run_config['k0_reuse'] = provenance
        print(f'Reusing {len(baseline)} K0 trajectories and scores from {args.reuse_k0_from}', flush=True)
    dest = args.output / f'part-{args.shard_index:04d}-of-{args.num_shards:04d}'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dict(run=run_config, shard_index=args.shard_index, shard_tokens=shard)
    with (dest / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = dest / 'run.json'
        if existing.exists() and json.loads(existing.read_text()) != manifest:
            raise ValueError(f'Output already belongs to another protocol: {dest}')
        if (dest / 'complete.json').exists():
            raise FileExistsError(f'Already complete: {dest}')
        dump_json(existing, manifest)
        _run_shard(args, dest, shard, logs, agent, refiner, conditioner, cache, simulator, scorer, loader, manifest, baseline)


def _run_shard(args, dest, shard, logs, agent, refiner, conditioner, cache, simulator, scorer, loader, manifest, baseline=None):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from .conditioner import features_from_input, verify_native_conditions
    rows = []
    with (dest / 'scenes.jsonl.tmp').open('w', encoding='utf-8') as out:
        for i, token in enumerate(shard):
            try:
                obs = loader.get_agent_input_from_token(token)
                features = features_from_input(agent, obs, args.device)
                seed = token_seed(args.seed, token)
                checks, reference = {}, []
                if args.verify_native and i == 0:
                    def observe_reference(_module, _inputs, result):
                        reference.append(result['trajectory'].detach().clone())

                    handle = agent.register_forward_hook(observe_reference)
                    try:
                        with sampling_rng(seed, args.device):
                            checks = verify_native_conditions(agent, features)
                    finally:
                        handle.remove()
                if baseline is None:
                    with sampling_rng(seed, args.device):
                        (raw, mode), native_ms = timed(lambda: native_top1(agent, features), args.device)
                else:
                    cached = baseline[token]
                    raw = torch.tensor(cached['trajectory'], dtype=torch.float32, device=args.device)[None]
                    mode, native_ms = cached['top1_index'], 0.0
                    checks['k0_reused'] = True
                if args.verify_native and i == 0:
                    if len(reference) != 1 or not torch.equal(reference[0], raw):
                        raise ValueError('K0 differs from the same-seed unmodified native forward')
                    checks['raw_k0_matches_native_forward'] = True
                context, global_context, context_ms = None, None, 0.0
                if refiner:
                    (context, global_context), context_ms = timed(lambda: conditioner.encode_features(features), args.device)
                    if args.verify_native and i == 0:
                        k0 = refiner.refine(raw, context, global_context, steps=0)
                        if not torch.equal(k0, raw):
                            raise ValueError('Score K0 altered native poses')
                        checks['score_k0_matches_native'] = True
                metric = cache.get_from_token(token)

                def score(poses):
                    return asdict(pdm_score(metric, Trajectory(poses), simulator.proposal_sampling, simulator, scorer))

                trajectories, metrics, delta, refine_ms = score_pose_group(
                    raw, args.ks, refiner, context, global_context, args.gamma, score, args.device,
                    baseline_metrics=baseline[token]['metrics'] if baseline is not None else None)
                row = dict(token=token, log=logs[token], seed=seed, top1_index=mode, trajectories=trajectories,
                           metrics=metrics, delta=delta, native_ms=native_ms, conditioner_ms=context_ms,
                           refine_ms=refine_ms, checks=checks)
                out.write(json.dumps(row, allow_nan=False) + '\n')
                out.flush()
                # Summaries only need metrics/timings. Keep trajectories in the streaming file.
                rows.append({k: row[k] for k in ('token', 'metrics', 'delta', 'refine_ms')})
            except Exception as exc:
                raise RuntimeError(f'Paired evaluation failed at token {token}; shard is incomplete') from exc
    report = summarize(rows)
    if manifest['run'].get('k0_reuse'):
        report['k0_reuse'] = manifest['run']['k0_reuse']
    (dest / 'scenes.jsonl.tmp').replace(dest / 'scenes.jsonl')
    dump_json(dest / 'summary.json', report)
    dump_json(dest / 'complete.json', manifest)
    if args.num_shards == 1:
        dump_json(args.output / 'summary.json', report)
    print(json.dumps(report, indent=2))


def main(argv=None):
    args = parse_args(argv)
    if args.merge:
        merge_shards(args.output)
    else:
        run(args)


if __name__ == '__main__':
    main()
