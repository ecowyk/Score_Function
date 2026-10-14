#!/usr/bin/env python3
"""Frozen DP/S10-G evaluation: hard R, random NR, random R, in that order.

Uses the existing frozen simulator/model exports. Never starts or stops training.
Random-split membership is established by its complete official NR baseline;
every subsequent random run must contain exactly the same (log, token) pairs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import full_test14hard_suite as suite
import queue_g_sigma_trials as training

ROOT = Path('/home/data/wyk')
SOURCE = ROOT / 'dataset/nuplan/exp/score_G_sigma_closed_loop_1014_v1'
OUTPUT = ROOT / 'dataset/nuplan/exp/score_G_transfer_1014_v1'
SETTINGS = (
    ('hard_R', 'test14-hard', 'closed_loop_reactive_agents'),
    ('random_NR', 'test14-random', 'closed_loop_nonreactive_agents'),
    ('random_R', 'test14-random', 'closed_loop_reactive_agents'),
)


def jobs():
    return [dict(id=f'{label}_{name}', setting=label, split=split, challenge=challenge,
                 model=model, K=k)
            for label, split, challenge in SETTINGS
            for name, model, k in [('DP_K0', 'S05-G', 0), ('S10-G_K20', 'S10-G', 20)]]


def lock(stack, path):
    handle = stack.enter_context(Path(path).open('a'))
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def verify(out, plan):
    if plan['jobs'] != jobs():
        raise ValueError('Frozen evaluation order/settings differ from this runner')
    source = Path(plan['source_suite'])
    for name in ('code', 'models'):
        if not (out / name).is_symlink() or (out / name).resolve() != (source / name).resolve():
            raise ValueError(f'Frozen {name} link changed')
    suite.verify(out, plan)


def prepare(args):
    import yaml
    out, source = args.output.resolve(), args.source.resolve()
    if out == source:
        raise ValueError('Output must differ from the previous evaluation')
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock(stack, out / '.run.lock')
        if (out / 'plan.json').exists():
            plan = suite.read(out / 'plan.json')
            if plan['source_suite'] != str(source):
                raise ValueError('Different frozen source requested')
            verify(out, plan)
            print('Already prepared; frozen inputs verified.', flush=True)
            return
        if any(p.name != '.run.lock' for p in out.iterdir()):
            raise ValueError('Incomplete preparation exists; preserve it and use a new --output')
        old = suite.read(source / 'plan.json')
        suite.verify(source, old)
        if len(old['tokens']) != 272 or len(set(old['tokens'])) != 272:
            raise ValueError('Expected the existing full 272-scenario hard baseline')
        for name in ('DP_K0', 'S10-G_K20'):
            if suite.read(source / 'jobs' / f'{name}.json')['state'] not in ('complete', 'reused'):
                raise ValueError(f'Previous evaluation not complete: {name}')
        common = suite.read(source / 'models/S05-G/config.json')
        target = suite.read(source / 'models/S10-G/config.json')
        for cfg in (common, target):
            if cfg['reference_seed'] != old['reference_seed']:
                raise ValueError('Reference seed mismatch')
            if cfg['refinement']['gamma'] != old['gamma'] or cfg['refinement']['heading_projection'] != old['heading_projection']:
                raise ValueError('Refinement settings mismatch')
        root = Path(old['root'])
        filters = {}
        files = copy.deepcopy(old['files'])
        for split in ('test14-hard', 'test14-random'):
            path = root / 'Diffusion-Planner/diffusion_planner/config/scenario_filter' / f'{split}.yaml'
            filters[split] = yaml.safe_load(path.read_text())
            files[str(path)] = suite.digest(path)
        if set(filters['test14-hard']['scenario_tokens']) != set(old['tokens']):
            raise ValueError('Hard split changed')
        random = filters['test14-random']
        if (random.get('scenario_tokens') is not None or random.get('shuffle') is not False
                or random.get('num_scenarios_per_type') != 20
                or random.get('limit_total_scenarios') is not None
                or len(set(random.get('scenario_types') or [])) != 14):
            raise ValueError('Expected the official deterministic 14-type random filter')
        for path in (source / 'plan.json', Path(__file__).resolve(), Path(suite.__file__).resolve(), Path(training.__file__).resolve()):
            files[str(path)] = suite.digest(path)
        for model in ('S05-G', 'S10-G'):
            for name in ('config.json', 'final_ema.pt'):
                path = source / 'models' / model / name
                files[str(path)] = suite.digest(path)
        # Fail if the test database inventory changes between the paired evaluations.
        dbs = sorted((root / 'dataset/nuplan/nuplan-v1.1/test').rglob('*.db'))
        if not dbs:
            raise ValueError('No nuPlan test database files found')
        inventory = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in dbs}
        plan = {key: copy.deepcopy(old[key]) for key in ('root', 'tokens', 'gamma', 'heading_projection', 'reference_seed', 'seed_rule', 'training_queue')}
        plan.update(source_suite=str(source), jobs=jobs(), filters=filters, files=files,
                    db_inventory=inventory, created_unix=time.time())
        for name in ('code', 'models'):
            (out / name).symlink_to(source / name, target_is_directory=True)
        suite.write(out / 'plan.json', plan)
        suite.write(out / 'status.json', dict(state='prepared'))
        print(f"READY: six evaluations; gamma={plan['gamma']}, seed={plan['reference_seed']}; hard R -> random NR -> random R", flush=True)


def validate_result(run, job, plan, out):
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf
    run = Path(run)
    cfg = OmegaConf.load(run / 'code/hydra/config.yaml')
    planner = cfg.planner.score_function_planner
    config, checkpoint = suite.model_paths(out, job['model'])
    if suite.read(planner.score_config) != suite.read(config) or suite.digest(planner.score_checkpoint) != suite.digest(checkpoint):
        raise ValueError('Saved model/config mismatch')
    if int(planner.steps) != job['K'] or float(planner.gamma) != plan['gamma']:
        raise ValueError('K/gamma mismatch')
    heading = planner.heading_projection
    if (suite.read(config)['refinement']['heading_projection'] if heading is None else heading) != plan['heading_projection']:
        raise ValueError('Heading projection mismatch')
    if planner.trace_dir is not None or str(planner.device) != 'cuda':
        raise ValueError('Unexpected trace/device setting')
    expected_parent = Path(plan['root']) / 'dataset/nuplan/exp/exp/simulation' / job['challenge'] / 'score_function' / job['split']
    if run.parent.resolve() != expected_parent.resolve():
        raise ValueError('Wrong simulation mode or split output path')
    observation = str(cfg.observation._target_)
    required = 'IDMAgents' if job['challenge'] == 'closed_loop_reactive_agents' else 'TracksObservation'
    if not observation.endswith('.' + required):
        raise ValueError(f'Wrong traffic observation: {observation}; expected {required}')
    if OmegaConf.to_container(cfg.scenario_filter, resolve=True) != plan['filters'][job['split']]:
        raise ValueError('Scenario filter differs from the frozen official filter')
    if Path(str(cfg.scenario_builder.data_root)).resolve() != (Path(plan['root']) / 'dataset/nuplan/nuplan-v1.1/test').resolve():
        raise ValueError('Not the nuPlan test data root')
    if not str(cfg.worker._target_).endswith('RayDistributed'):
        raise ValueError('Expected Ray scenario parallelism')
    aggregates = list((run / 'aggregator_metric').glob('*.parquet'))
    if len(aggregates) != 1:
        raise ValueError('Expected exactly one official aggregate')
    frame = pd.read_parquet(aggregates[0])
    detail = frame.loc[frame.log_name.notna()]
    report = pd.read_parquet(run / str(cfg.runner_report_file))
    keys = sorted(zip(detail.log_name.astype(str), detail.scenario.astype(str)))
    actual = sorted(zip(report.log_name.astype(str), report.scenario_name.astype(str)))
    n = len(keys)
    if not n or len(set(keys)) != n or actual != keys or not report.succeeded.eq(True).all():
        raise ValueError('Missing/duplicate/failed scenarios in simulation report or metrics')
    if job['split'] == 'test14-hard':
        if n != 272 or set(detail.scenario.astype(str)) != set(plan['tokens']):
            raise ValueError('Hard run does not contain the full frozen 272 scenarios')
    else:
        counts = detail.groupby('scenario_type').size()
        if set(counts.index) != set(plan['filters'][job['split']]['scenario_types']) or (counts > 20).any():
            raise ValueError('Random run does not cover the official 14 types with <=20 each')
    anchor_id = 'hard_R_DP_K0' if job['split'] == 'test14-hard' else 'random_NR_DP_K0'
    if job['id'] != anchor_id:
        anchor = suite.read(out / 'jobs' / f'{anchor_id}.json')
        if anchor['state'] != 'complete' or anchor['scenario_keys'] != [list(k) for k in keys]:
            raise ValueError('Scenario membership differs from its paired baseline')
    final = frame.loc[frame.scenario.eq('final_score')]
    if len(final) != 1 or int(final.iloc[0].num_scenarios) != n:
        raise ValueError('Wrong aggregate scenario count')
    metrics = {key: float(final.iloc[0][key]) for key in suite.METRICS}
    if not np.isfinite(detail[list(suite.METRICS)].to_numpy(dtype=float)).all() or not np.isfinite(list(metrics.values())).all():
        raise ValueError('Missing/nonfinite metrics')
    return dict(metrics=metrics, scenario_keys=[list(k) for k in keys], num_scenarios=n)


def summarize(out, plan):
    rows = []
    for job in plan['jobs']:
        path = out / 'jobs' / f"{job['id']}.json"
        record = suite.read(path) if path.exists() else {'state': 'pending'}
        rows.append(dict(job, state=record['state'], num_scenarios=record.get('num_scenarios'),
                         result_dir=record.get('result_dir', ''), **record.get('metrics', {})))
    fields = [*jobs()[0], 'state', 'num_scenarios', 'result_dir', *suite.METRICS]
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        score = row.get('score')
        print(f"{row['id']:26s} {row['state']:12s} " + (f'{100 * score:.4f}' if score is not None else ''), flush=True)


def command_for(job, plan, out, label):
    code = out / 'code/lg'
    config, checkpoint = suite.model_paths(out, job['model'])
    return ['bash', str(code / 'sim_score_planner.sh'), '--python', sys.executable,
            '--root', plan['root'], '--config', str(config), '--checkpoint', str(checkpoint),
            '--name', label, '--split', job['split'], '--challenge', job['challenge'],
            '--steps', str(job['K']), '--gamma', str(plan['gamma']), '--gpus', '0,1,2,3,4,5,6,7',
            '--gpu-fraction', '0.15', '--threads', '128']


def run(args):
    out = args.output.resolve()
    plan = suite.read(out / 'plan.json')
    if plan['source_suite'] != str(args.source.resolve()):
        raise ValueError('Requested source differs from frozen plan')
    root, source = Path(plan['root']), Path(plan['source_suite'])
    code = out / 'code/lg'
    os.environ.update(suite.env_for(root, code))
    with ExitStack() as stack:
        locks = [lock(stack, out / '.run.lock')]
        paths = {Path(plan['training_queue']) / 'queue.lock', source / '.run.lock',
                 root / 'dataset/nuplan/exp/score_full_k_1009_v1/.run.lock',
                 root / 'dataset/nuplan/exp/score_G_local_k_1012_v1/.run.lock'}
        for path in sorted(paths):
            if path.parent.exists():
                locks.append(lock(stack, path))
        for job in plan['jobs']:
            record_path = out / 'jobs' / f"{job['id']}.json"
            old = suite.read(record_path) if record_path.exists() else {}
            verify(out, plan)
            if old.get('state') == 'complete':
                checked = validate_result(old['result_dir'], job, plan, out)
                if any(old[key] != value for key, value in checked.items()):
                    raise ValueError('Completed result receipt changed')
                print('SKIP verified', job['id'], flush=True)
                continue
            if old.get('state') == 'running' and old.get('pid') and suite.process_identity(old['pid']):
                raise RuntimeError('Previous simulation PID may still be active; inspect before retrying')
            dbs = sorted((root / 'dataset/nuplan/nuplan-v1.1/test').rglob('*.db'))
            inventory = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in dbs}
            if inventory != plan['db_inventory']:
                raise ValueError('Test database inventory changed')
            busy = training.gpu_busy(list(range(8)))
            if busy:
                raise RuntimeError(f'GPUs occupied; no processes were stopped: {busy}')
            label = f"{job['id']}_transfer_{time.time_ns()}"
            log = out / 'logs' / f'{label}.log'
            log.parent.mkdir(exist_ok=True)
            command = command_for(job, plan, out, label)
            record = dict(job, state='running', log=str(log), command=command, started_unix=time.time())
            print(f"START {job['id']} | GPUs=0..7 | log={log}", flush=True)
            try:
                with log.open('w') as handle:
                    child = subprocess.Popen(command, cwd=code, env=suite.env_for(root, code),
                                             stdout=handle, stderr=subprocess.STDOUT,
                                             start_new_session=True, pass_fds=tuple(h.fileno() for h in locks))
                    record['pid'] = child.pid
                    suite.write(record_path, record)
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
                record.update(result_dir=result, **validate_result(result, job, plan, out), state='complete', finished_unix=time.time())
                suite.write(record_path, record)
                summarize(out, plan)
            except BaseException as exc:
                record.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
                suite.write(record_path, record)
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
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'run':
        run(args)
    else:
        print(suite.read(args.output / 'status.json'), flush=True)
        summarize(args.output, suite.read(args.output / 'plan.json'))


if __name__ == '__main__':
    main()
