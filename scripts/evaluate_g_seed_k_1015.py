#!/usr/bin/env python3
"""Paired inference-seed x refinement-K evaluation using frozen S10-G weights.

Default: Test14-Hard NR then Test14 NR, five seeds, K=0/5/10/20/50/100. Only reference_seed
changes in per-seed configs. Every seed gets a fresh K=0 control. No training,
noise rescaling, dynamic gamma, early stopping, or best-of-N trajectory choice.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import full_test14hard_suite as suite
import queue_g_sigma_trials as training

ROOT = Path('/home/data/wyk')
SOURCE = ROOT / 'dataset/nuplan/exp/score_G_transfer_1014_v1'
OUTPUT = ROOT / 'dataset/nuplan/exp/score_G_seed_k_1015_v1'
KS = [0, 5, 10, 20, 50, 100]
SETTINGS = {
    'hard_NR': ('test14-hard', 'closed_loop_nonreactive_agents'),
    'hard_R': ('test14-hard', 'closed_loop_reactive_agents'),
    'random_NR': ('test14-random', 'closed_loop_nonreactive_agents'),
    'random_R': ('test14-random', 'closed_loop_reactive_agents'),
}


def lock(stack, path):
    handle = stack.enter_context(Path(path).open('a'))
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def axes(old, args):
    seeds = args.seeds if args.seeds is not None else [int(old['reference_seed']) + i for i in range(5)]
    ks = args.ks if args.ks is not None else KS
    settings = args.settings if args.settings is not None else ['hard_NR', 'random_NR']
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 - 1 for s in seeds):
        raise ValueError('Require at least two unique nonnegative inference seeds below 2**63-1')
    if not ks or len(set(ks)) != len(ks) or 0 not in ks or max(ks) == 0 or min(ks) < 0:
        raise ValueError('K must contain 0 and at least one positive value, without duplicates')
    if not settings or len(set(settings)) != len(settings) or not set(settings) <= SETTINGS.keys():
        raise ValueError('Invalid or duplicate evaluation settings')
    return list(seeds), sorted(ks), list(settings)


def jobs(seeds, ks, settings):
    # Finish controls for every seed before the first refined K, then fill columns.
    return [dict(id=f'{setting}_seed{seed}_K{k}', setting=setting,
                 split=SETTINGS[setting][0], challenge=SETTINGS[setting][1],
                 model=f'S10-G_seed{seed}', seed=seed, K=k)
            for setting in settings for k in ks for seed in seeds]


def variant_config(original, seed):
    variant = copy.deepcopy(original)
    variant['reference_seed'] = seed
    return variant


def inventory(root):
    dbs = sorted((Path(root) / 'dataset/nuplan/nuplan-v1.1/test').rglob('*.db'))
    if not dbs:
        raise ValueError('No test databases found')
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in dbs}


def verify(out, plan):
    if plan['schema_version'] != 1 or plan['jobs'] != jobs(plan['seeds'], plan['ks'], plan['settings']):
        raise ValueError('Matrix plan axes/order changed')
    source = Path(plan['source_suite'])
    if not (out / 'code').is_symlink() or (out / 'code').resolve() != (source / 'code').resolve():
        raise ValueError('Frozen simulator link changed')
    suite.verify(out, plan)
    original = suite.read(source / 'models/S10-G/config.json')
    for seed in plan['seeds']:
        config, checkpoint = suite.model_paths(out, f'S10-G_seed{seed}')
        if suite.read(config) != variant_config(original, seed):
            raise ValueError('A seed config changed more than reference_seed')
        if not checkpoint.is_symlink() or checkpoint.resolve() != (source / 'models/S10-G/final_ema.pt').resolve():
            raise ValueError('Score checkpoint link changed')


PREFLIGHT = r'''
import hashlib, json, sys, torch
from pathlib import Path
from score_function.utils.checkpoint import load_selected
from score_function.utils.config import load_config
from score_function.utils.planner_utils import planner_identity
from score_function.utils.sampling import planner_seed
p=json.loads(sys.argv[1]); source=Path(p['source']); out=Path(p['output'])
torch.set_num_threads(2)
cfg=load_config(source/'models/S10-G/config.json', p['root'])
branch,state=load_selected(cfg,source/'models/S10-G/final_ema.pt','cpu')
assert (state['weight_kind'],state['epoch'],state['sigma_score']) == ('ema',529,0.1)
assert state['planner']==planner_identity(cfg), 'Frozen DP/normalizer identity changed'
assert all(torch.isfinite(t).all() for t in branch.state_dict().values())
noise={}
for seed in p['seeds']:
    loaded=load_config(out/f'models/S10-G_seed{seed}/config.json',p['root'])
    expected=dict(cfg,reference_seed=seed)
    assert loaded==expected, 'Inference config differs beyond reference_seed'
    probes=[]
    for timestamp in (0,100000,200000):
        derived=planner_seed(timestamp,seed)
        expected_seed=int.from_bytes(hashlib.sha256(f'{seed}:{timestamp}'.encode()).digest()[:8],'big')%(2**63-1)
        assert derived==expected_seed, 'Noise seed rule changed'
        with torch.random.fork_rng():
            torch.manual_seed(derived)
            a=torch.randn(32)
            torch.manual_seed(derived)
            assert torch.equal(a,torch.randn(32)), 'Seed replay failed'
        probes.append(hashlib.sha256(a.numpy().tobytes()).hexdigest())
    noise[str(seed)]=probes
assert len(set(tuple(v) for v in noise.values()))==len(noise), 'Seeds have identical probes'
print(json.dumps(dict(weight_kind=state['weight_kind'],epoch=state['epoch'],sigma=state['sigma_score'],
    noise_probe_sha256=noise,scope='CPU seed-contract probes; not a GPU simulation')))
'''


def prepare(args):
    out, source = args.output.resolve(), args.source.resolve()
    if out == source:
        raise ValueError('Use a separate output directory')
    old = suite.read(source / 'plan.json')
    seeds, ks, settings = axes(old, args)
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock(stack, out / '.run.lock')
        lock(stack, source / '.run.lock')
        if (out / 'plan.json').exists():
            plan = suite.read(out / 'plan.json')
            if (plan['source_suite'], plan['seeds'], plan['ks'], plan['settings']) != (str(source), seeds, ks, settings):
                raise ValueError('Already prepared with different axes/source; use a new --output')
            verify(out, plan)
            print('Already prepared; frozen inputs verified.', flush=True)
            return
        if any(p.name != '.run.lock' for p in out.iterdir()):
            raise ValueError('Partial preparation exists; preserve it and use a new --output')
        suite.verify(source, old)
        if suite.read(source / 'status.json')['state'] != 'complete':
            raise ValueError('Finish the existing transfer evaluation first')
        if len(old['tokens']) != 272 or len(set(old['tokens'])) != 272:
            raise ValueError('Expected frozen full 272-scenario hard split')
        original = suite.read(source / 'models/S10-G/config.json')
        if original['reference_seed'] != old['reference_seed']:
            raise ValueError('Original reference seed mismatch')
        if original['training']['sigma'] != 0.1 or original['refinement']['gamma'] != old['gamma'] or old['gamma'] != 0.1:
            raise ValueError('Expected S10-G sigma=0.1, fixed gamma=0.1')
        if original['refinement']['heading_projection'] != old['heading_projection']:
            raise ValueError('Heading projection mismatch')
        root = Path(old['root'])
        if inventory(root) != old['db_inventory']:
            raise ValueError('Test database inventory differs from transfer suite')
        scenario_keys = {}
        anchor_paths = []
        for split in {SETTINGS[s][0] for s in settings}:
            anchor_id = 'hard_R_DP_K0' if split == 'test14-hard' else 'random_NR_DP_K0'
            anchor_path = source / 'jobs' / f'{anchor_id}.json'
            anchor = suite.read(anchor_path)
            keys = anchor.get('scenario_keys', [])
            if anchor['state'] != 'complete' or not keys or len({tuple(k) for k in keys}) != len(keys):
                raise ValueError(f'Missing verified scenario membership: {anchor_id}')
            if split == 'test14-hard' and (len(keys) != 272 or {k[1] for k in keys} != set(old['tokens'])):
                raise ValueError('Source hard baseline has different scenarios')
            scenario_keys[split] = sorted(keys)
            anchor_paths.append(anchor_path)
        # Preserve the original seeded planner, rather than adding a new RNG path.
        planner_text = (source / 'code/lg/score_function/planner/planner.py').read_text()
        if 'torch.manual_seed(planner_seed(timestamp, self.settings["reference_seed"]))' not in planner_text:
            raise ValueError('Unrecognized planner seeding; inspect before launching a matrix')
        (out / 'code').symlink_to(source / 'code', target_is_directory=True)
        files = copy.deepcopy(old['files'])
        for seed in seeds:
            config, checkpoint = suite.model_paths(out, f'S10-G_seed{seed}')
            suite.write(config, variant_config(original, seed))
            checkpoint.symlink_to(source / 'models/S10-G/final_ema.pt')
            files[str(config)] = suite.digest(config)
        for path in (source / 'plan.json', source / 'models/S10-G/config.json',
                     source / 'models/S10-G/final_ema.pt', Path(__file__).resolve(),
                     Path(suite.__file__).resolve(), Path(training.__file__).resolve(), *anchor_paths):
            files[str(path)] = suite.digest(path)
        payload = dict(source=str(source), output=str(out), root=str(root), seeds=seeds)
        checked = subprocess.run([sys.executable, '-c', PREFLIGHT, json.dumps(payload)],
                                 cwd=out / 'code/lg', env=suite.env_for(root, out / 'code/lg', ''),
                                 capture_output=True, text=True)
        if checked.returncode:
            raise RuntimeError(f'CPU preflight failed:\n{checked.stdout}\n{checked.stderr}')
        preflight = json.loads(checked.stdout.strip().splitlines()[-1])
        suite.write(out / 'preflight.json', preflight)
        files[str(out / 'preflight.json')] = suite.digest(out / 'preflight.json')
        plan = dict(schema_version=1, source_suite=str(source), root=str(root),
                    seeds=seeds, ks=ks, settings=settings, jobs=jobs(seeds, ks, settings),
                    reference_seed=old['reference_seed'], seed_rule=old['seed_rule'],
                    gamma=old['gamma'], sigma=0.1, heading_projection=old['heading_projection'],
                    tokens=old['tokens'], filters=old['filters'], scenario_keys=scenario_keys,
                    db_inventory=old['db_inventory'],
                    training_queue=old['training_queue'], files=files,
                    gpu_fraction=1.0, gpus=list(range(8)), threads=128,
                    comparison='Paired K minus K=0 within each inference seed; fresh controls',
                    created_unix=time.time())
        suite.write(out / 'plan.json', plan)
        suite.write(out / 'status.json', dict(state='prepared'))
        print(f"READY: {len(plan['jobs'])} evaluations; settings={settings}; seeds={seeds}; K={ks}; gamma=0.1, sigma=0.1", flush=True)
        print('All K=0 controls will be rerun with the same GPU allocation as refinement.', flush=True)


def command_for(job, plan, out, label):
    config, checkpoint = suite.model_paths(out, job['model'])
    return ['bash', str(out / 'code/lg/sim_score_planner.sh'), '--python', sys.executable,
            '--root', plan['root'], '--config', str(config), '--checkpoint', str(checkpoint),
            '--name', label, '--split', job['split'], '--challenge', job['challenge'],
            '--steps', str(job['K']), '--gamma', str(plan['gamma']),
            '--gpus', ','.join(map(str, plan['gpus'])), '--gpu-fraction', str(plan['gpu_fraction']),
            '--threads', str(plan['threads'])]


def validate_result(run, job, plan, out):
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf
    run = Path(run)
    cfg = OmegaConf.load(run / 'code/hydra/config.yaml')
    planner = cfg.planner.score_function_planner
    config, checkpoint = suite.model_paths(out, job['model'])
    actual_config = suite.read(planner.score_config)
    if actual_config != suite.read(config) or actual_config['reference_seed'] != job['seed']:
        raise ValueError('Saved config/seed mismatch')
    if suite.digest(planner.score_checkpoint) != suite.digest(checkpoint):
        raise ValueError('Saved checkpoint mismatch')
    if int(planner.steps) != job['K'] or float(planner.gamma) != plan['gamma']:
        raise ValueError('K/gamma mismatch')
    heading = planner.heading_projection
    if (actual_config['refinement']['heading_projection'] if heading is None else heading) != plan['heading_projection']:
        raise ValueError('Heading projection mismatch')
    if planner.trace_dir is not None or str(planner.device) != 'cuda':
        raise ValueError('Unexpected trace/device setting')
    parent = Path(plan['root']) / 'dataset/nuplan/exp/exp/simulation' / job['challenge'] / 'score_function' / job['split']
    if run.parent.resolve() != parent.resolve():
        raise ValueError('Wrong result mode/split path')
    required = 'IDMAgents' if job['challenge'] == 'closed_loop_reactive_agents' else 'TracksObservation'
    if not str(cfg.observation._target_).endswith('.' + required):
        raise ValueError('Wrong traffic observation')
    if OmegaConf.to_container(cfg.scenario_filter, resolve=True) != plan['filters'][job['split']]:
        raise ValueError('Scenario filter changed')
    if Path(str(cfg.scenario_builder.data_root)).resolve() != (Path(plan['root']) / 'dataset/nuplan/nuplan-v1.1/test').resolve():
        raise ValueError('Wrong dataset root')
    if not str(cfg.worker._target_).endswith('RayDistributed') or float(cfg.number_of_gpus_allocated_per_simulation) != plan['gpu_fraction']:
        raise ValueError('Wrong Ray worker/GPU reservation')
    aggregates = list((run / 'aggregator_metric').glob('*.parquet'))
    if len(aggregates) != 1:
        raise ValueError('Expected one aggregate')
    frame = pd.read_parquet(aggregates[0])
    detail = frame.loc[frame.log_name.notna()]
    report = pd.read_parquet(run / str(cfg.runner_report_file))
    keys = sorted(zip(detail.log_name.astype(str), detail.scenario.astype(str)))
    actual = sorted(zip(report.log_name.astype(str), report.scenario_name.astype(str)))
    n = len(keys)
    if not n or len(set(keys)) != n or actual != keys or not report.succeeded.eq(True).all():
        raise ValueError('Missing/duplicate/failed scenario results')
    if [list(k) for k in keys] != plan['scenario_keys'][job['split']]:
        raise ValueError('Scenario membership differs from the previous transfer evaluation')
    if job['split'] == 'test14-hard':
        if n != 272 or set(detail.scenario.astype(str)) != set(plan['tokens']):
            raise ValueError('Require all 272 frozen hard scenarios')
    else:
        counts = detail.groupby('scenario_type').size()
        if set(counts.index) != set(plan['filters'][job['split']]['scenario_types']) or (counts > 20).any():
            raise ValueError('Wrong random split membership')
    anchor_id = f"{job['setting']}_seed{plan['seeds'][0]}_K0"
    if job['id'] != anchor_id:
        anchor = suite.read(out / 'jobs' / f'{anchor_id}.json')
        if anchor['state'] != 'complete' or anchor['scenario_keys'] != [list(k) for k in keys]:
            raise ValueError('Scenario membership differs across matrix cells')
    final = frame.loc[frame.scenario.eq('final_score')]
    if len(final) != 1 or int(final.iloc[0].num_scenarios) != n:
        raise ValueError('Wrong final count')
    metrics = {key: float(final.iloc[0][key]) for key in suite.METRICS}
    if not np.isfinite(detail[list(suite.METRICS)].to_numpy(dtype=float)).all() or not np.isfinite(list(metrics.values())).all():
        raise ValueError('Missing/nonfinite metrics')
    return dict(metrics=metrics, scenario_keys=[list(k) for k in keys], num_scenarios=n)


def write_csv(path, fields, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def summarize(out, plan, save=True):
    records, rows, matrix, stats = {}, [], [], []
    for job in plan['jobs']:
        path = out / 'jobs' / f"{job['id']}.json"
        record = suite.read(path) if path.exists() else dict(state='pending')
        records[(job['setting'], job['seed'], job['K'])] = record
        rows.append(dict(job, state=record['state'], result_dir=record.get('result_dir', ''),
                         **{f'{k}_100': 100*v for k, v in record.get('metrics', {}).items()}))
    for setting in plan['settings']:
        print(f"\n{setting} | scores on 0..100 scale | fixed gamma={plan['gamma']}, sigma={plan['sigma']}", flush=True)
        print('seed'.ljust(14) + ''.join(f'K={k: <12}' for k in plan['ks']), flush=True)
        for seed in plan['seeds']:
            row = dict(setting=setting, seed=seed)
            cells = []
            for k in plan['ks']:
                rec = records[(setting, seed, k)]
                value = 100*rec['metrics']['score'] if rec['state'] == 'complete' else None
                row[f'K{k}'] = value
                cells.append(f'{value:.4f}' if value is not None else rec['state'])
            matrix.append(row)
            print(str(seed).ljust(14) + ''.join(f'{c:<15}' for c in cells), flush=True)
        for k in plan['ks']:
            for metric in suite.METRICS:
                values, deltas = [], []
                for seed in plan['seeds']:
                    rec, base = records[(setting, seed, k)], records[(setting, seed, 0)]
                    if rec['state'] == 'complete' and base['state'] == 'complete':
                        if rec['scenario_keys'] != base['scenario_keys']:
                            raise ValueError('Cannot summarize mismatched paired scenarios')
                        value, control = 100*rec['metrics'][metric], 100*base['metrics'][metric]
                        values.append(value)
                        deltas.append(value-control)
                if not values:
                    continue
                row = dict(setting=setting, K=k, metric=metric, paired_seeds=len(values),
                           planned_seeds=len(plan['seeds']), complete=len(values)==len(plan['seeds']),
                           mean_100=statistics.mean(values),
                           sample_std_100=statistics.stdev(values) if len(values)>1 else None,
                           delta_mean_100=statistics.mean(deltas),
                           delta_sample_std_100=statistics.stdev(deltas) if len(deltas)>1 else None,
                           delta_min_100=min(deltas), delta_max_100=max(deltas),
                           improved_seeds=sum(d>1e-9 for d in deltas),
                           tied_seeds=sum(abs(d)<=1e-9 for d in deltas),
                           worse_seeds=sum(d < -1e-9 for d in deltas))
                stats.append(row)
                if metric == 'score':
                    sd = f"{row['delta_sample_std_100']:.4f}" if len(values)>1 else 'NA'
                    print(f"K={k:<5} pairs={len(values)}/{len(plan['seeds'])} "
                          f"mean={row['mean_100']:.4f} paired_delta={row['delta_mean_100']:+.4f} "
                          f"sample_std={sd} improved={row['improved_seeds']}/{len(values)}", flush=True)
    if save:
        write_csv(out / 'summary.csv', [*plan['jobs'][0], 'state', 'result_dir', *[f'{m}_100' for m in suite.METRICS]], rows)
        write_csv(out / 'matrix.csv', ['setting', 'seed', *[f'K{k}' for k in plan['ks']]], matrix)
        if stats:
            write_csv(out / 'paired_summary.csv', list(stats[0]), stats)
    return stats


def run(args):
    out = args.output.resolve()
    plan = suite.read(out / 'plan.json')
    if plan['source_suite'] != str(args.source.resolve()):
        raise ValueError('Source differs from frozen plan')
    root, code = Path(plan['root']), out / 'code/lg'
    os.environ.update(suite.env_for(root, code))
    with ExitStack() as stack:
        locks = [lock(stack, out / '.run.lock')]
        paths = {Path(plan['training_queue']) / 'queue.lock', Path(plan['source_suite']) / '.run.lock'}
        paths.update(root / 'dataset/nuplan/exp' / name / '.run.lock' for name in (
            'score_full_k_1009_v1', 'score_G_local_k_1012_v1', 'score_G_sigma_closed_loop_1014_v1'))
        for path in sorted(paths):
            if path.parent.exists():
                locks.append(lock(stack, path))
        for job in plan['jobs']:
            verify(out, plan)
            path = out / 'jobs' / f"{job['id']}.json"
            old = suite.read(path) if path.exists() else {}
            if old.get('state') == 'complete':
                checked = validate_result(old['result_dir'], job, plan, out)
                if any(old[key] != value for key, value in checked.items()):
                    raise ValueError('Completed receipt changed')
                print('SKIP verified', job['id'], flush=True)
                continue
            if old.get('state') == 'running' and old.get('pid') and suite.process_identity(old['pid']):
                raise RuntimeError('Previous simulation PID may still be active; inspect before retrying')
            if inventory(root) != plan['db_inventory']:
                raise ValueError('Test database inventory changed')
            busy = training.gpu_busy(plan['gpus'])
            if busy:
                raise RuntimeError(f'GPUs occupied; no processes stopped: {busy}')
            label = f"{job['id']}_matrix_{time.time_ns()}"
            log = out / 'logs' / f'{label}.log'
            log.parent.mkdir(exist_ok=True)
            command = command_for(job, plan, out, label)
            if old:
                suite.write(out / 'retry_history' / f'{label}.json', old)
            record = dict(job, state='running', log=str(log), command=command,
                          plan_sha256=suite.digest(out / 'plan.json'), started_unix=time.time())
            print(f"START {job['id']} | 8 GPUs, one scenario per GPU | log={log}", flush=True)
            try:
                with log.open('w') as handle:
                    child = subprocess.Popen(command, cwd=code, env=suite.env_for(root, code),
                                             stdout=handle, stderr=subprocess.STDOUT,
                                             start_new_session=True, pass_fds=tuple(h.fileno() for h in locks))
                    record['pid'] = child.pid
                    suite.write(path, record)
                    try:
                        while child.poll() is None:
                            suite.write(out / 'status.json', dict(state='running', job=job['id'],
                                        log=str(log), pid=child.pid, updated_unix=time.time()))
                            try:
                                child.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                pass
                    except BaseException:
                        if child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                        raise
                if child.returncode:
                    raise RuntimeError(f'Simulation exited {child.returncode}; see {log}')
                result = next(line.split(': ', 1)[1] for line in log.read_text(errors='replace').splitlines() if line.startswith('Results: '))
                record.update(result_dir=result, **validate_result(result, job, plan, out),
                              state='complete', finished_unix=time.time())
                suite.write(path, record)
                summarize(out, plan)
            except BaseException as exc:
                record.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
                suite.write(path, record)
                suite.write(out / 'status.json', dict(state=record['state'], job=job['id'], error=str(exc)))
                summarize(out, plan)
                raise
        suite.write(out / 'status.json', dict(state='complete', updated_unix=time.time()))
        summarize(out, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'status'))
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seeds', type=int, nargs='+', help='prepare only; default original reference_seed plus 0..4')
    parser.add_argument('--ks', type=int, nargs='+', help='prepare only; default 0 5 10 20 50 100')
    parser.add_argument('--settings', choices=tuple(SETTINGS), nargs='+', help='prepare only; default hard_NR random_NR')
    args = parser.parse_args()
    if args.mode != 'prepare' and any(x is not None for x in (args.seeds, args.ks, args.settings)):
        parser.error('Axes are frozen by prepare; run/status read plan.json')
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'run':
        run(args)
    else:
        print(suite.read(args.output / 'status.json'), flush=True)
        summarize(args.output, suite.read(args.output / 'plan.json'), save=False)


if __name__ == '__main__':
    main()
