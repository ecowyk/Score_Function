#!/usr/bin/env python3
"""Resource-aware ChainFlow Score queue; native reproduction is a separate job."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
KS = [5, 10, 15, 20, 30]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def paths_for(root):
    root = Path(root).resolve()
    exp = root / 'dataset/navsim/exp'
    return dict(python=root/'envs/chainflow/bin/python', devkit_root=root/'ChainFlow-VLA',
                checkpoint=root/'ChainFlow-VLA/weights/stage_2/stage2_navtrain_9382.ckpt',
                backbone=root/'ChainFlow-VLA/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors',
                vlm_cache=root/'dataset/navsim/chainflow_vlm_cache', data_root=root/'dataset/navsim',
                metric_cache=exp/'metric_cache', source_cache=exp/'dd_score_cache_v1',
                baseline=exp/'chainflow_native_navtest_v1', cache=exp/'chainflow_score_cache_v1',
                training=exp/'chainflow_score_v1', evaluation=exp/'chainflow_score_navtest_v1',
                output=exp/'chainflow_score_pipeline', gpu_lock_dir=exp/'chainflow_gpu_leases')


def environment(args):
    env = os.environ.copy()
    path = [str(REPO), str(args.devkit_root), str(args.root/'nuplan-devkit')]
    if env.get('PYTHONPATH'):
        path.append(env['PYTHONPATH'])
    env.update(PYTHONPATH=os.pathsep.join(path), PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', CHAINFLOW_ENABLE_RAY='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OPENSCENE_DATA_ROOT=str(args.data_root),
               NAVSIM_DEVKIT_ROOT=str(args.devkit_root), NAVSIM_EXP_ROOT=str(args.root/'dataset/navsim/exp'),
               NAVSIM_METRIC_CACHE_ROOT=str(args.metric_cache.parent),
               NUPLAN_MAPS_ROOT=str(args.root/'dataset/nuplan/maps'), NUPLAN_MAP_VERSION='nuplan-maps-v1.0')
    return env


def effective_config(args):
    cfg = read_json(args.config)
    cfg['microbatch_size'] = args.microbatch_size
    cfg['num_workers'] = args.loader_workers
    if cfg.get('epochs') != 500 or cfg.get('checkpoint_every_epochs') != 50:
        raise ValueError('This queue requires 500 epochs and a checkpoint every 50 epochs')
    if cfg.get('global_batch_size') != 512 or cfg.get('sigma') != .1 or cfg.get('gamma') != .1:
        raise ValueError('This queue requires global batch 512, sigma=0.1 and gamma=0.1')
    for workers in ((1, 2, 4, 8) if args.workers == 'auto' else (args.workers,)):
        if 512 % (args.microbatch_size * workers):
            raise ValueError('microbatch * workers must divide global batch 512')
    return cfg


def plan_for(args):
    cfg = effective_config(args)
    source_files = sorted((REPO/'score_function/navsim_chainflow').glob('*.py'))
    for directory in ('model', 'utils'):
        source_files += sorted((REPO/'score_function'/directory).rglob('*.py'))
    source_files += [Path(__file__), REPO/'scripts/eval_navsim_chainflow.py']
    return dict(schema='chainflow-score-pipeline-v1', config=cfg, ks=KS,
                paths={name: str(getattr(args, name)) for name in paths_for(args.root)},
                source_sha256={str(p.relative_to(REPO)): sha256(p) for p in source_files},
                baseline_mode='external_native_only', final_weights='ema')


def bind_plan(path, plan):
    if path.exists() and read_json(path) != plan:
        raise ValueError('Pipeline configuration, code or paths changed; choose a new --output directory')
    atomic_json(path, plan)


@contextmanager
def pipeline_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'pipeline.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This ChainFlow pipeline is already running') from None
        yield


def lock_active(path):
    if not path.exists():
        return False
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def cached_world(args, stage):
    # A resumed training/evaluation must retain its partitioning and RNG protocol.
    path = args.training/'run.json' if stage == 'train' else args.evaluation/'launcher.json'
    if not path.exists():
        return None
    data = read_json(path)
    world = data['protocol']['world_size'] if stage == 'train' else data['request']['workers']
    if args.workers != 'auto' and args.workers != world:
        raise ValueError(f'{stage} was started with {world} GPUs; resume with --workers {world}')
    if stage == 'train' and data['config']['microbatch_size'] != args.microbatch_size:
        raise ValueError('Resume microbatch differs from the saved training config')
    return world


def training_complete(args):
    marker = args.training/'TRAINING_COMPLETE.json'
    if not marker.is_file():
        return False
    data = read_json(marker)
    run = read_json(args.training/'run.json')
    if data.get('epochs') != 500 or data.get('signature') != run.get('signature'):
        raise ValueError('Invalid training completion marker')
    metadata = read_json(args.cache/'manifest.json')
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if run['protocol']['cache_fingerprint'] != fingerprint:
        raise ValueError('Completed training belongs to a different feature cache')
    if run['config'] != effective_config(args):
        raise ValueError('Completed training configuration differs from this queue')
    for kind in ('raw', 'ema'):
        if sha256(args.training/f'final_{kind}.pt') != data[f'final_{kind}_sha256']:
            raise ValueError(f'Final {kind} checkpoint checksum mismatch')
    return True


def cache_command(args, workers, check=False):
    cmd = [str(args.python), '-u', '-m', 'score_function.navsim_chainflow.cache']
    values = dict(source_cache=args.source_cache, cache_dir=args.cache, checkpoint=args.checkpoint,
                  backbone=args.backbone, vlm_cache=args.vlm_cache, data_root=args.data_root, devkit_root=args.devkit_root)
    for key, value in values.items():
        cmd += ['--'+key.replace('_', '-'), str(value)]
    cmd += ['--workers', str(workers)]
    if check:
        cmd.append('--check-data')
    return cmd


def train_command(args, workers):
    cmd = [str(args.python), '-u']
    if workers > 1:
        cmd += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc_per_node={workers}',
                '--module', 'score_function.navsim_chainflow.train']
    else:
        cmd += ['-m', 'score_function.navsim_chainflow.train']
    cmd += ['--cache-dir', str(args.cache), '--output', str(args.training), '--config', str(args.config),
            '--microbatch-size', str(args.microbatch_size), '--workers', str(args.loader_workers)]
    if (args.training/'run.json').exists():
        cmd.append('--resume')
    return cmd


def eval_command(args, workers, baseline=False):
    cmd = [str(args.python), '-u', str(REPO/'scripts/eval_navsim_chainflow.py'), 'run',
           '--root', str(args.root), '--mode', 'baseline' if baseline else 'refine',
           '--gpu-lease-held', '--workers', str(workers), '--output', str(args.baseline if baseline else args.evaluation)]
    for key in ('devkit_root', 'checkpoint', 'backbone', 'vlm_cache', 'data_root', 'metric_cache'):
        cmd += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    if not baseline:
        cmd += ['--baseline', str(args.baseline), '--score-checkpoint', str(args.training/'final_ema.pt'),
                '--ks', *map(str, KS), '--gamma', '.1', '--expected-epoch', '500', '--weight-kind', 'ema']
    return cmd


def helper_command(args, mode):
    cmd = [str(args.python), '-u', str(Path(__file__).resolve()), mode, '--root', str(args.root),
           '--config', str(args.config), '--microbatch-size', str(args.microbatch_size),
           '--loader-workers', str(args.loader_workers)]
    for key in paths_for(args.root):
        cmd += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    return cmd


class Queue:
    def __init__(self, args):
        self.args = args
        self.state = dict(status='running', stage='starting', pid=os.getpid(), host=socket.gethostname(),
                          started=now(), training=str(args.training), evaluation=str(args.evaluation),
                          baseline=str(args.baseline), cache=str(args.cache), epochs=500, ks=KS)

    def update(self, **values):
        self.state.update(values, updated=now())
        atomic_json(self.args.output/'state.json', self.state)

    def note(self, message):
        print(f'[{now()}] {message}', flush=True)

    def execute(self, stage, command, env=None):
        logfile = self.args.output/f'{stage}.log'
        self.update(stage=stage, status='running', log=str(logfile))
        self.note(f'{stage}: starting; log={logfile}')
        with logfile.open('a', buffering=1) as log:
            log.write(f'\n=== {now()} {stage} ===\n')
            child = subprocess.Popen(command, cwd=REPO, env=env or environment(self.args), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            self.update(child_pid=child.pid)
            started, reported = time.monotonic(), 0.
            try:
                while child.poll() is None:
                    elapsed = time.monotonic()-started
                    if elapsed-reported >= 15:
                        self.note(f'{stage}: running {elapsed:.0f}s; log={logfile}')
                        self.update(elapsed_seconds=round(elapsed))
                        reported = elapsed
                    time.sleep(1)
                if child.returncode:
                    raise RuntimeError(f'{stage} exited {child.returncode}; see {logfile}')
            except BaseException:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                raise
            finally:
                self.update(child_pid=None)
        self.note(f'{stage}: complete')

    def gpu(self, stage, command_builder, workers=None):
        from score_function.navsim_chainflow.resources import acquire_gpus, gpu_environment
        args = self.args
        self.update(stage=stage, status='waiting_gpu', log=None)
        requested = workers if workers is not None else args.workers
        def resource_status(value):
            self.update(status='running' if value['state'] == 'acquired' else 'waiting_gpu', resource=value)
            values = ', '.join(f"{g['index']}:{g['utilization']}%/{g['free_mib']/1024:.1f}GiB free" for g in value['pool'])
            self.note(f"{stage}: GPU {value['state']} ({value['policy']}); {values}")
        with acquire_gpus(pool=args.gpu_pool, workers=requested, budget_gb=args.gpu_budget_gb,
                          reserve_gb=args.gpu_reserve_gb, policy=args.gpu_policy, poll=args.poll,
                          lock_dir=args.gpu_lock_dir, on_status=resource_status) as selected:
            env = environment(args)
            env.update(gpu_environment(selected, args.gpu_budget_gb))
            env.update(CF_GPU_LEASE_OWNER_PID=str(os.getpid()), CF_GPU_LEASE_LOCK_DIR=str(args.gpu_lock_dir))
            self.execute(stage, command_builder(len(selected)), env)

    def wait_baseline(self):
        self.update(stage='wait_native', status='waiting_native', log=None)
        while not (self.args.baseline/'summary.json').is_file():
            self.note(f'Waiting for independent native reproduction: {self.args.baseline}/summary.json')
            self.update(status='waiting_native')
            time.sleep(self.args.poll)
        self.execute('verify_native', helper_command(self.args, '_verify-native'))

    def run(self):
        args = self.args
        if not args.python.is_file():
            raise FileNotFoundError(f'Create the ChainFlow conda environment first: {args.python}')
        self.execute('check_data', cache_command(args, 1, check=True))
        self.wait_baseline()
        if (args.cache/'TRAINING_READY.json').is_file():
            self.execute('verify_cache', helper_command(args, '_verify-cache'))
        else:
            self.gpu('cache', lambda n: cache_command(args, n))
        # A production-model update and fixed-noise validation on real train/val data.
        # The model/optimizer are temporary; full training starts from the configured seed.
        if not training_complete(args):
            world = cached_world(args, 'train')
            self.gpu('preflight', lambda n: helper_command(args, '_preflight'), workers=1)
            self.gpu('train', lambda n: train_command(args, n), workers=world)
            if not training_complete(args):
                raise RuntimeError('Training exited without a verified 500-epoch final checkpoint')
        world = cached_world(args, 'evaluate')
        # The eval launcher validates completed shards and skips them on a restart.
        if (args.evaluation/'summary.json').is_file():
            self.execute('verify_evaluation', helper_command(args, '_verify-evaluation'))
        else:
            self.gpu('evaluate', lambda n: eval_command(args, n), workers=world)
        self.update(status='complete', stage='complete', finished=now())
        self.note(f'Complete: {args.evaluation}/summary.json')


def verify_native(args):
    from score_function.navsim_chainflow.native import conditioner_identity
    from score_function.navsim_chainflow.evaluate import scoring_identity, validate_native_baseline
    identity = conditioner_identity(args.devkit_root, args.checkpoint, args.backbone, args.vlm_cache)
    result = validate_native_baseline(args.baseline, expected_identity=identity,
                                     expected_scoring=scoring_identity(args.devkit_root), expected_count=12146)
    atomic_json(args.output/'native_verified.json', dict(verified=now(), baseline=str(args.baseline),
                summary_sha256=sha256(args.baseline/'summary.json'), count=result['count'],
                native_pdms=100*result['by_k']['0']['mean']['score']))
    print(f"Verified native baseline: {result['count']} scenes, PDMS={100*result['by_k']['0']['mean']['score']:.4f}", flush=True)


def verify_cache(args, full=True):
    from score_function.navsim_chainflow.cache import (validate_training_ready, validate_chunk, source_tasks, make_identity, inside)
    from score_function.navsim_chainflow.native import conditioner_identity
    metadata = validate_training_ready(args.cache)
    if metadata['conditioner_identity'] != conditioner_identity(args.devkit_root, args.checkpoint, args.backbone, args.vlm_cache):
        raise ValueError('Cached conditions do not match the current native assets')
    if metadata['identity']['source_manifest_sha256'] != sha256(args.source_cache/'manifest.json'):
        raise ValueError('Source DD split/target manifest changed since ChainFlow caching')
    if {k: v['num_records'] for k, v in metadata['splits'].items()} != {'train': 85109, 'val': 18179}:
        raise ValueError('Cache split counts do not match the canonical source split')
    if full:
        source, tasks = source_tasks(args.source_cache)
        if make_identity(args.source_cache, source, metadata['conditioner_identity']) != metadata['identity']:
            raise ValueError('Cache source identity changed')
        entries = {entry['path']: entry for split in metadata['splits'].values() for entry in split['shards']}
        if set(entries) != {task['path'] for task in tasks}:
            raise ValueError('Cache shard membership differs from the original split')
        for index, task in enumerate(tasks, 1):
            directory = inside(args.cache, task['path'])
            if sha256(directory/'complete.json') != entries[task['path']]['marker_sha256']:
                raise ValueError(f'Changed cache completion marker: {directory}')
            validate_chunk(directory, metadata['identity'], task, checksums=True)
            print(f'Cache verified {index}/{len(tasks)}: {task["path"]}', flush=True)
    print('Frozen condition cache verified', flush=True)
    return metadata


def verify_evaluation(args):
    from score_function.navsim_chainflow.evaluate import collect_completed, _verified_summary, scoring_identity, validate_row
    from score_function.navsim_chainflow.native import conditioner_identity
    protocol, rows, _ = collect_completed(args.evaluation)
    summary = _verified_summary(args.evaluation, protocol, rows)
    request = read_json(args.evaluation/'launcher.json')['request']
    _, baseline_rows, hashes = collect_completed(args.baseline)
    baseline = {row['token']: row for row in baseline_rows}
    if (protocol['mode'] != 'refine' or len(rows) != 12146 or protocol['ks'] != [0, *KS]
            or request['score_checkpoint_sha256'] != sha256(args.training/'final_ema.pt')
            or request['baseline'] != str(args.baseline)
            or protocol['baseline']['files_sha256'] != hashes
            or protocol['baseline']['summary_sha256'] != sha256(args.baseline/'summary.json')
            or protocol['score_checkpoint_sha256'] != sha256(args.training/'final_ema.pt')
            or protocol['score_epoch'] != 500 or protocol['score_weight_kind'] != 'ema'
            or protocol['gamma'] != .1 or protocol['scoring_identity'] != scoring_identity(args.devkit_root)
            or request['native_identity'] != conditioner_identity(args.devkit_root, args.checkpoint, args.backbone, args.vlm_cache)):
        raise ValueError('Completed paired evaluation differs from this pipeline')
    for row in rows:
        validate_row(row, row['token'], row['log'], [0, *KS], baseline=baseline[row['token']])
    print(f"Paired evaluation verified: {summary['count']} scenes", flush=True)


def preflight(args):
    import torch
    from score_function.navsim_chainflow import train
    cfg = effective_config(args)
    cfg['num_workers'] = 0
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('Real preflight requires one leased CUDA device')
    train.configure_device_memory(device)
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    verify_cache(args, full=False)
    metadata, training, validation = train.load_training_data(args.cache, cfg)
    if len(training) != 85109 or len(validation) != 18179:
        raise ValueError('Expected the original 85,109/18,179 train/val split')
    model = train.build_training_model(cfg, metadata, args.cache, device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    # Include a full requested microbatch to exercise padding and memory, not one toy row.
    batch = train.collate_rows([training[i] for i in range(args.microbatch_size)])
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = train.run_training_epoch([batch], model, optimizer, ema, cfg, device,
                                      epoch=0, step=0, updates_per_epoch=1, accumulation=1)
    if not any(not torch.equal(initial[name], value) for name, value in model.named_parameters()):
        raise RuntimeError('Preflight did not change any trainable parameter')
    scores = train.validate_models({'raw': model, 'ema': ema}, validation, cfg, device,
                                  indices=list(range(min(args.microbatch_size, len(validation)))), description='Preflight val')
    atomic_json(args.output/'preflight.json', dict(status='complete', time=now(), train=result, validation=scores,
                train_count=len(training), val_count=len(validation), microbatch=args.microbatch_size,
                max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device)))
    print('REAL PREFLIGHT OK: production model update and RAW/EMA validation on real cache.', flush=True)


def snapshot(args):
    state_path = args.output/'state.json'
    state = read_json(state_path) if state_path.is_file() else {'status': 'not_started'}
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print('pipeline lock:', 'active' if lock_active(args.output/'pipeline.lock') else 'idle')
    history_path = args.training/'history.json'
    if history_path.is_file():
        history = read_json(history_path)
        if history:
            row = history[-1]
            print(f"Training: {row['epoch']}/500; train DSM={row['train_dsm']:.6f}; "
                  f"val RAW={row['val_dsm_raw']:.6f}; EMA={row['val_dsm_ema']:.6f}")
    summary_path = args.evaluation/'summary.json'
    if summary_path.is_file():
        data = read_json(summary_path)
        print(f"Paired scenes: {data['valid_count']}/{data['count']}")
        for k, result in sorted(data['by_k'].items(), key=lambda item: int(item[0])):
            print(f"K={int(k):2d} PDMS={100*result['mean']['score']:.4f} delta={100*result['mean_delta']['score']:+.4f}")
    sys.stdout.flush()
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status', '_verify-native', '_verify-cache', '_verify-evaluation', '_preflight'])
    parser.add_argument('--root', type=Path, default=Path('/home/data/wyk'))
    for key in paths_for(Path('/home/data/wyk')):
        parser.add_argument('--'+key.replace('_', '-'), type=Path)
    parser.add_argument('--config', type=Path, default=REPO/'configs/navsim_chainflow_s10g.json')
    parser.add_argument('--workers', default='4', choices=['1', '2', '4', '8', 'auto'])
    parser.add_argument('--microbatch-size', type=int, default=16)
    parser.add_argument('--loader-workers', type=int, default=2)
    parser.add_argument('--gpu-policy', choices=['wait-idle', 'shared'], default='wait-idle')
    parser.add_argument('--gpu-pool', default='auto')
    parser.add_argument('--gpu-budget-gb', type=float, default=12)
    parser.add_argument('--gpu-reserve-gb', type=float, default=4)
    parser.add_argument('--poll', type=float, default=15)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    for key, value in paths_for(args.root).items():
        chosen = (getattr(args, key) or value).expanduser()
        # Resolving a venv's python symlink would silently select the base interpreter.
        setattr(args, key, Path(os.path.abspath(chosen)) if key == 'python' else chosen.resolve())
    args.config = args.config.expanduser().resolve()
    args.workers = int(args.workers) if args.workers != 'auto' else args.workers
    if not 0 < args.poll <= 60 or args.microbatch_size < 1 or args.loader_workers < 0:
        parser.error('Require poll in (0,60], positive microbatch and nonnegative loader workers')
    if args.gpu_budget_gb <= 0 or args.gpu_reserve_gb < 0:
        parser.error('Require positive GPU memory budget and nonnegative reserve')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'status':
        try:
            while True:
                snapshot(args)
                if not args.watch:
                    return
                time.sleep(args.poll)
        except KeyboardInterrupt:
            return  # Stops only this monitor.
    if args.command == '_verify-native':
        return verify_native(args)
    if args.command == '_verify-cache':
        return verify_cache(args)
    if args.command == '_verify-evaluation':
        return verify_evaluation(args)
    if args.command == '_preflight':
        return preflight(args)
    with pipeline_lock(args.output):
        bind_plan(args.output/'plan.json', plan_for(args))
        queue = Queue(args)
        queue.update()
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            queue.run()
        except BaseException as exc:
            queue.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                         error=f'{type(exc).__name__}: {exc}', finished=now())
            queue.note(f"{queue.state['status']}: {exc}")
            raise
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    main()
