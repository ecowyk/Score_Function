#!/usr/bin/env python3
"""Resume the prepared baselines8 queue with four isolated two-GPU lanes.

Default: 4 jobs x 2 GPUs x 16 simulations/GPU = 128 simulations.
No model, seed, cohort, or original plan/script is modified. The current legacy
job drains before takeover. Run with the existing urbandriver Python environment.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

VERSION = 1
SELF = Path(__file__).resolve()
PROC = Path('/proc')
ROOT = Path('/home/data/wyk')
DEFAULT_OUT = ROOT / 'dataset/nuplan/exp/baselines8_flowcfg18_seed20260922_1028_v4'
LEGACY_NAME = 'evaluate_baselines_val_reactive_1028.py'
LANES = ('0,1', '2,3', '4,5', '6,7')
PROFILE = dict(jobs=4, gpus_per_job=2, scenarios_per_gpu=16,
               cpus_per_job=32, cpus_per_simulation=1,
               gpus_per_simulation=0.0625, object_store_gib_per_job=8,
               ray_logical_memory_gib_per_job=128)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def workdir(args):
    return args.output / 'parallel128_1102'


def legacy(args):
    scripts = args.repo / 'scripts'
    require((scripts / LEGACY_NAME).is_file(), f'Missing original script: {scripts / LEGACY_NAME}')
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    module = importlib.import_module(LEGACY_NAME[:-3])
    require(Path(module.__file__).resolve() == scripts / LEGACY_NAME, 'Wrong queue module imported')
    return module


def config_for(plan, job):
    cfg = copy.deepcopy(plan['configs'][job['setting']][job['variant']][str(job['K'])])
    require(str(cfg['worker']['_target_']).endswith('.RayDistributed'), 'Expected RayDistributed')
    cfg['worker'].update(threads_per_node=32, master_node_ip=None,
                         use_distributed=False, debug_mode=False)
    cfg['number_of_gpus_allocated_per_simulation'] = 0.0625
    cfg['number_of_cpus_allocated_per_simulation'] = 1
    cfg['max_callback_workers'] = min(2, int(cfg.get('max_callback_workers', 2)))
    return cfg


def environment(mod, plan, name, devices, token, output):
    env = mod.environment(plan, name)
    env.update(CUDA_VISIBLE_DEVICES=devices, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               PYTHONUNBUFFERED='1', SCORE_BASELINE_QUEUE=str(output),
               SCORE_BASELINE_JOB_TOKEN=token)
    for name in ('RAY_ADDRESS', 'RAY_TMPDIR'):
        env.pop(name, None)
    return env


def proc(pid):
    try:
        directory = PROC / str(int(pid))
        fields = (directory / 'stat').read_text().rpartition(')')[2].split()
        return dict(pid=int(pid), ticks=int(fields[19]), ppid=int(fields[1]),
                    state=fields[0], cmd=(directory / 'cmdline').read_bytes().split(b'\0'))
    except (OSError, ValueError, IndexError):
        return None


def same_process(identity):
    now = proc(identity['pid'])
    return now if now and now['ticks'] == identity['ticks'] and now['state'] != 'Z' else None


def identity(pid):
    p = proc(pid)
    return dict(pid=p['pid'], ticks=p['ticks'], host=socket.gethostname()) if p else None


def owns_token(pid, output, token):
    try:
        env = (PROC / str(pid) / 'environ').read_bytes().split(b'\0')
        return (f'SCORE_BASELINE_QUEUE={output}'.encode() in env and
                f'SCORE_BASELINE_JOB_TOKEN={token}'.encode() in env)
    except OSError:
        return False


def running_old_worker(controller, output):
    """Read children after STOP, including a child launched before status.json."""
    found = []
    for directory in PROC.iterdir():
        if not directory.name.isdigit():
            continue
        info = proc(int(directory.name))
        if not info or info['ppid'] != controller['pid'] or info['state'] == 'Z':
            continue
        try:
            entries = (directory / 'environ').read_bytes().split(b'\0')
            env = dict(e.split(b'=', 1) for e in entries if b'=' in e)
        except OSError:
            continue
        if env.get(b'SCORE_BASELINE_QUEUE') != str(output).encode():
            continue
        cmd = [x.decode(errors='replace') for x in info['cmd'] if x]
        require('worker' in cmd and '--job' in cmd, 'Unknown tagged child of the old controller')
        token = env.get(b'SCORE_BASELINE_JOB_TOKEN', b'').decode()
        require(bool(token), 'Old worker has no attempt marker')
        found.append((identity(info['pid']), cmd[cmd.index('--job') + 1], token))
    require(len(found) <= 1, 'Multiple legacy workers detected; restoring original controller')
    return found[0] if found else (None, None, None)


def signal_identity(who, sig):
    require(who.get('host') == socket.gethostname(), 'Refusing to signal a process from another node')
    if same_process(who):
        try:
            os.kill(who['pid'], sig)
        except ProcessLookupError:
            pass


def holds_file(pid, path):
    try:
        wanted = Path(path).stat()
        for fd in (PROC / str(pid) / 'fd').iterdir():
            try:
                actual = fd.stat()
                if (actual.st_dev, actual.st_ino) == (wanted.st_dev, wanted.st_ino):
                    return True
            except OSError:
                pass
    except OSError:
        pass
    return False


def find_old_controller(args):
    record = read(args.output / 'process.json', {})
    if not record.get('pid') or record.get('host') != socket.gethostname():
        return None
    p = proc(record['pid'])
    if not p or p['state'] == 'Z':
        return None
    command = [b.decode(errors='replace') for b in p['cmd'] if b]
    expected = str(args.repo / 'scripts' / LEGACY_NAME)
    if expected not in command or 'run' not in command:
        return None
    require('--output' in command and
            Path(command[command.index('--output') + 1]).resolve() == args.output,
            'Old controller command does not identify this output directory')
    require(holds_file(p['pid'], args.output / 'queue.lock'),
            'Old controller does not own the expected queue file descriptor')
    require(p['pid'] not in (os.getpid(), os.getppid()), 'Refusing to pause this process or its parent')
    return identity(p['pid'])


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield f
    finally:
        f.close()


def snapshot(args, phase, **extra):
    value = dict(state=phase, host=socket.gethostname(), pid=os.getpid(),
                 updated_unix=time.time(), profile=PROFILE, **extra)
    write(workdir(args) / 'status.json', value)
    return value


def preflight(args, mod, plan):
    require(plan['seed'] == 20260922 and plan.get('flow_cfg_weight') == 1.8,
            'Expected seed20260922 / Flow CFG1.8')
    require(plan['ks'] == [0, 20] and plan['gamma'] == .1 and plan['sigma'] == .1,
            'Unexpected score-refinement protocol')
    require(len({j['id'] for j in plan['jobs']}) == len(plan['jobs']), 'Duplicate job IDs')
    print('Checking original prepared inputs and checkpoints...', flush=True)
    mod.verify(plan, Path(plan['source']))
    for job in plan['jobs']:
        cfg = config_for(plan, job)
        require(args.output in Path(job['output']).resolve().parents, 'Output outside original queue')
        require(Path(cfg['output_dir']).resolve() == Path(job['output']).resolve(), 'Job/output mismatch')
        require(cfg['seed'] == plan['seed'], 'Job seed differs from plan')
    gpu_ids = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True)
    require(set(range(8)) <= {int(x.strip()) for x in gpu_ids.splitlines() if x.strip()}, 'Require GPUs 0-7')
    if hasattr(os, 'sched_getaffinity'):
        require(len(os.sched_getaffinity(0)) >= 128, 'CPU affinity exposes fewer than 128 CPUs')
    limits = []
    for prefix in ('cpu', 'cpu,cpuacct'):
        q = Path('/sys/fs/cgroup') / prefix / 'cpu.cfs_quota_us'
        p = q.with_name('cpu.cfs_period_us')
        if q.exists() and p.exists() and int(q.read_text()) > 0:
            limits.append(int(q.read_text()) / int(p.read_text()))
    q = Path('/sys/fs/cgroup/cpu.max')
    if q.exists():
        quota, period = q.read_text().split()
        if quota != 'max':
            limits.append(int(quota) / int(period))
    require(not limits or min(limits) >= 128, f'CPU quota is below 128: {limits}')
    # Import checks use each original interpreter without starting Ray or a model.
    runtimes = {}
    for job in plan['jobs']:
        runtime = plan['sources'][job['variant']]['runtime']
        runtimes.setdefault(runtime['python'], (runtime, job['variant']))
    for runtime, name in runtimes.values():
        for key in ('python', 'cwd', 'entry'):
            require(Path(runtime[key]).exists(), f'Missing {name} runtime: {runtime[key]}')
        env = environment(mod, plan, name, '', 'preflight', args.output)
        command = [runtime['python'], '-u', str(SELF), '_probe', '--repo', str(args.repo)]
        result = subprocess.run(command, cwd=runtime['cwd'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        require(result.returncode == 0, f'Environment check failed for {name}:\n{result.stdout[-5000:]}')
        print(f'Environment OK: {name}; {result.stdout.strip()}', flush=True)
    print('Preflight OK: 4 lanes, 32 scenarios/lane, 8 GiB object store/lane.', flush=True)


def restore_old(args):
    paused = read(workdir(args) / 'handover.json', {})
    who = paused.get('controller')
    if paused.get('state') == 'paused' and who:
        require(who['host'] == socket.gethostname(), 'Restore on the node that paused the old queue')
        if same_process(who):
            signal_identity(who, signal.SIGCONT)
            print('RESUMED original controller', who['pid'], flush=True)
        paused.update(state='restored', restored_unix=time.time())
        write(workdir(args) / 'handover.json', paused)


def drain_old(args, mod, plan):
    pending = read(workdir(args) / 'handover.json', {})
    who = pending.get('controller') if pending.get('state') == 'paused' else find_old_controller(args)
    if who and not same_process(who):
        who = None
    if not who:
        return
    require(who['host'] == socket.gethostname(), 'Cannot drain a controller on another node')
    handover = dict(state='paused', controller=who, started_unix=time.time())
    write(workdir(args) / 'handover.json', handover)
    # Stop only dispatch; the worker and its Ray processes continue independently.
    signal_identity(who, signal.SIGSTOP)
    try:
        deadline = time.monotonic() + 5
        while same_process(who) and same_process(who)['state'] not in ('T', 't'):
            require(time.monotonic() < deadline, 'Original controller did not pause')
            time.sleep(.05)
        status = read(args.output / 'status.json', {})
        worker, job_id, token = running_old_worker(who, args.output)
        if worker and same_process(worker):
            require(token and job_id and owns_token(worker['pid'], args.output, token),
                    'Cannot establish identity of the currently running worker; restoring old controller')
            snapshot(args, 'draining_old_job', job=job_id, old_controller=who['pid'])
            print(f'DRAIN {job_id}: current evaluation continues; no new legacy job will start.', flush=True)
            last_notice = 0.0
            while same_process(worker):
                if time.monotonic() - last_notice > 30:
                    print(f'WAIT current evaluation {job_id}; log={status.get("log", "")}', flush=True)
                    last_notice = time.monotonic()
                time.sleep(2)
        # Worker exited (including failure). Let the original handler clean only
        # that attempt and release its locks, without dispatching another job.
        signal_identity(who, signal.SIGINT)
        signal_identity(who, signal.SIGCONT)
        deadline = time.monotonic() + 60
        while same_process(who):
            require(time.monotonic() < deadline,
                    'Original controller has not exited; no parallel simulations were launched')
            time.sleep(.2)
        handover.update(state='complete', finished_unix=time.time(), job=job_id)
        write(workdir(args) / 'handover.json', handover)
        print('HANDOVER complete; validated completed receipts retained.', flush=True)
    except BaseException:
        restore_old(args)
        raise


def assert_gpus_free():
    deadline = time.monotonic() + 30
    while True:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
        if not apps:
            return
        require(time.monotonic() < deadline,
                f'GPU processes remain (PIDs below). Nothing else was stopped:\n{apps}')
        time.sleep(1)


def command(args, action, script=None):
    return [sys.executable, '-u', str(script or SELF), action,
            '--repo', str(args.repo), '--output', str(args.output)]


def launch(args):
    require((args.output / 'plan.json').is_file(), 'Original plan.json is required; no new queue will be created')
    mod, plan = legacy(args), read(args.output / 'plan.json')
    with lock(workdir(args) / 'controller.lock') as launch_lock:
        preflight(args, mod, plan)
        runner = workdir(args) / ('runner_' + digest(SELF)[:16] + '.py')
        if not runner.exists():
            shutil.copyfile(SELF, runner)
        require(digest(runner) == digest(SELF), 'Frozen runner differs')
        manifest = dict(version=VERSION, profile=PROFILE, original_plan_sha256=digest(args.output / 'plan.json'),
                        runner_sha256=digest(runner), repo=str(args.repo), output=str(args.output),
                        launched_unix=time.time(), host=socket.gethostname())
        write(workdir(args) / 'manifest.json', manifest)
        log = workdir(args) / 'queue.log'
        with log.open('a') as f:
            child = subprocess.Popen(command(args, '_run', runner), stdout=f, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(launch_lock.fileno(),))
        who = identity(child.pid)
        require(who is not None, f'Controller exited during launch; inspect {log}')
        write(workdir(args) / 'launcher.json', dict(**who, command=command(args, '_run', runner), log=str(log)))
        print(f'LAUNCHED PID {child.pid}; current job drains automatically.\nLog: {log}', flush=True)


def cleanup_attempt(mod, args, attempt):
    result = mod.cleanup_job(args.output, attempt['token'], process_group=attempt['child'].pid)
    write(workdir(args) / 'cleanup' / (attempt['token'] + '.json'), result)
    # A KILL may take a short time to be reflected in /proc. Never reuse a lane
    # while a process with the previous attempt's exact marker remains alive.
    deadline = time.monotonic() + 10
    while True:
        remaining = []
        for p in PROC.iterdir():
            if p.name.isdigit() and owns_token(int(p.name), args.output, attempt['token']):
                info = proc(int(p.name))
                if info and info['state'] != 'Z':
                    remaining.append(int(p.name))
        if not remaining:
            break
        require(time.monotonic() < deadline, f'Owned processes did not exit: {remaining}; lane not reused')
        time.sleep(.2)


def start_job(args, mod, plan, job, lane):
    token = uuid.uuid4().hex
    log = workdir(args) / 'logs' / (job['id'] + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    runtime = plan['sources'][job['variant']]['runtime']
    env = environment(mod, plan, job['variant'], LANES[lane], token, args.output)
    path = mod.receipt_path(args.output, job)
    previous = read(path, {})
    if previous:
        write(workdir(args) / 'history' / (job['id'] + '.' + token + '.json'), previous)
    record = dict(job, state='running', host=socket.gethostname(), log=str(log), token=token,
                  lane=lane, gpus=LANES[lane], profile=PROFILE, started_unix=time.time())
    write(path, record)
    cmd = command(args, '_worker') + ['--job', job['id'], '--lane', str(lane), '--token', token]
    cmd[0] = runtime['python']
    with log.open('a') as stream:
        child = subprocess.Popen(cmd, cwd=runtime['cwd'], env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    print(f'START lane={lane} GPUs={LANES[lane]} scenes<=32 {job["id"]}; log={log}', flush=True)
    return dict(child=child, job=job, lane=lane, token=token, record=record)


def update_status(args, mod, plan, active, phase):
    current = [dict(job=a['job']['id'], lane=lane, gpus=LANES[lane], pid=a['child'].pid,
                    log=a['record']['log']) for lane, a in sorted(active.items())]
    s = snapshot(args, phase, active=current, **mod.counts(args.output, plan))
    write(args.output / 'status.json', s)


def run(args):
    mod, plan = legacy(args), read(args.output / 'plan.json')
    manifest = read(workdir(args) / 'manifest.json')
    require(manifest['original_plan_sha256'] == digest(args.output / 'plan.json'), 'Original plan changed after launch')
    require(manifest['runner_sha256'] == digest(SELF), 'Frozen runner changed')
    snapshot(args, 'starting')
    drain_old(args, mod, plan)
    with contextlib.ExitStack() as stack:
        for path in (args.output / '.launcher.lock', args.output / 'queue.lock',
                     Path('/tmp/score_random_baselines_1025.lock')):
            stack.enter_context(lock(path))
        assert_gpus_free()
        require(manifest['original_plan_sha256'] == digest(args.output / 'plan.json'), 'Original plan changed during drain')
        write(workdir(args) / 'process.json', identity(os.getpid()))
        write(args.output / 'process.json', dict(**identity(os.getpid()), runner=str(SELF)))
        pending = []
        for job in plan['jobs']:
            record = read(mod.receipt_path(args.output, job), {})
            if record.get('state') == 'complete':
                require(record.get('id') == job['id'] and record.get('scenarios') == len(plan['cohorts'][job['setting']]),
                        f'Completed receipt identity differs: {job["id"]}')
                require(Path(record['result_dir']).is_dir(), f'Missing completed output: {record["result_dir"]}')
                print('SKIP complete', job['id'], flush=True)
            else:
                pending.append(job)
        print(f'PARALLEL128: {len(pending)} pending evaluations; groups={LANES}', flush=True)
        active = {}
        try:
            while pending or active:
                for lane, attempt in list(active.items()):
                    code = attempt['child'].poll()
                    if code is None:
                        continue
                    cleanup_attempt(mod, args, attempt)
                    job = attempt['job']
                    receipt = read(mod.receipt_path(args.output, job), {})
                    if receipt.get('state') != 'complete':
                        receipt = dict(attempt['record'], state='failed', finished_unix=time.time(),
                                       error=f'Worker exited {code}; see {attempt["record"]["log"]}')
                        write(mod.receipt_path(args.output, job), receipt)
                        print('FAILED', job['id'], receipt['error'], flush=True)
                    else:
                        print(f'COMPLETE {job["id"]}: {receipt["score_100"]:.4f}', flush=True)
                    del active[lane]
                    mod.try_pair(args.output, plan, job)
                    mod.summarize(args.output, plan)
                stopping = (workdir(args) / 'STOP_AFTER_CURRENT').exists()
                if not stopping:
                    for lane in range(4):
                        if lane not in active and pending:
                            job = pending.pop(0)
                            active[lane] = start_job(args, mod, plan, job, lane)
                if stopping and not active:
                    break
                update_status(args, mod, plan, active, 'draining' if stopping else 'running')
                time.sleep(2)
        except BaseException:
            for attempt in active.values():
                cleanup_attempt(mod, args, attempt)
                path = mod.receipt_path(args.output, attempt['job'])
                receipt = read(path, {})
                if receipt.get('state') != 'complete':
                    write(path, dict(attempt['record'], state='interrupted', finished_unix=time.time()))
            update_status(args, mod, plan, {}, 'interrupted')
            raise
        for job in plan['jobs']:
            mod.try_pair(args.output, plan, job)
        mod.summarize(args.output, plan)
        totals = mod.counts(args.output, plan)
        phase = 'paused' if pending else ('complete_with_errors' if totals['failed'] or totals['pair_failures'] else 'complete')
        update_status(args, mod, plan, {}, phase)
        print(f'{phase.upper()}: {totals["completed"]}/{totals["total"]}; summary={args.output / "summary.csv"}', flush=True)


def worker(args):
    mod, plan = legacy(args), read(args.output / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == args.job)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == LANES[args.lane], 'GPU lane environment mismatch')
    require(owns_token(os.getpid(), args.output, args.token), 'Attempt ownership marker mismatch')
    cfg = mod.order_metric_callbacks(config_for(plan, job))
    base = plan['configs'][job['setting']][job['variant']][str(job['K'])]
    shared = mod.old.shared()
    require(shared.evaluation_signature(cfg) == shared.evaluation_signature(base), 'Evaluation semantics changed')
    cohort = mod.old.keys(plan['cohorts'][job['setting']])
    target = Path(job['output'])
    result = None
    reused = False
    if target.exists():
        try:
            result = mod.validate_stage(target, cfg, cohort, job['variant'], job['K'])
            reused = True
            print('REUSE verified', job['id'], flush=True)
        except (ValueError, FileNotFoundError, KeyError):
            archive = target.with_name(target.name + '.interrupted.' + str(time.time_ns()))
            target.rename(archive)
            print('PRESERVED', archive, flush=True)
    if result is None:
        from omegaconf import OmegaConf
        launch_dir = target / 'launch'
        launch_dir.mkdir(parents=True)
        OmegaConf.save(OmegaConf.create(cfg), launch_dir / 'config.yaml')
        entry = plan['sources'][job['variant']]['runtime']['entry']
        cmd = command(args, '_simulation') + ['--entry', entry, '--config', str(launch_dir / 'config.yaml'),
                                              '--lane', str(args.lane), '--token', args.token]
        env = environment(mod, plan, job['variant'], LANES[args.lane], args.token, args.output)
        with (target / 'console.log').open('w') as log:
            child = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
            require(code == 0, f'Simulation exited {code}: {target / "console.log"}')
        result = mod.validate_stage(target, cfg, cohort, job['variant'], job['K'])
    write(mod.receipt_path(args.output, job), dict(job, state='complete', score_100=result[3],
          scenarios=len(cohort), result_dir=str(target), finished_unix=time.time(),
          host=socket.gethostname(), token=args.token, lane=args.lane, gpus=LANES[args.lane],
          profile=PROFILE, reused_existing_output=reused))
    print(f'VALIDATED {job["id"]}: {result[3]:.4f}; {len(cohort)}/{len(cohort)}', flush=True)


def ray_options(args, kwargs):
    kwargs.update(address='local', num_cpus=32, num_gpus=2,
                  object_store_memory=8 * 1024**3, _memory=128 * 1024**3,
                  include_dashboard=False, dashboard_port=0,
                  _temp_dir=f'/tmp/sbp128_{args.token[:16]}', _node_ip_address='127.0.0.1')
    return kwargs


def simulation(args):
    import ray
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == LANES[args.lane], 'Simulation GPU group changed')
    require(torch.cuda.device_count() == 2, 'Simulation must see exactly two GPUs')
    original_init = ray.init

    def bounded_init(*positional, **kwargs):
        require(not positional, 'Unexpected positional Ray initialization arguments')
        result = original_init(**ray_options(args, kwargs))
        resources = ray.cluster_resources()
        require(resources.get('CPU') == 32 and resources.get('GPU') == 2, f'Wrong Ray resources: {resources}')
        print('PARALLEL128_RAY', json.dumps(dict(lane=args.lane, devices=LANES[args.lane],
              resources=resources, scenarios_per_lane=32, object_store_gib=8)), flush=True)
        return result

    ray.init = bounded_init
    sys.path.insert(0, str(args.entry.parent))
    sys.argv = [str(args.entry), '--config-path', str(args.config.parent), '--config-name', args.config.stem]
    runpy.run_path(str(args.entry), run_name='__main__')


def probe(args):
    import ray
    import torch
    import hydra
    import pandas
    from omegaconf import OmegaConf
    from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
    mod = legacy(args)
    mod.old.shared()
    require(hasattr(ray, 'cluster_resources'), 'Ray resource API missing')
    print(f'Python={sys.version.split()[0]} Ray={ray.__version__} Torch={torch.__version__}')


def status(args):
    s = read(workdir(args) / 'status.json', {})
    print(json.dumps(s, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('launch', 'status', 'pause', 'resume', 'restore-old',
                                      '_run', '_worker', '_simulation', '_probe'))
    p.add_argument('--repo', type=Path, default=ROOT / 'Score_Function_TwoNode_1025')
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--job')
    p.add_argument('--lane', type=int, choices=range(4))
    p.add_argument('--token')
    p.add_argument('--entry', type=Path)
    p.add_argument('--config', type=Path)
    args = p.parse_args()
    args.repo, args.output = args.repo.expanduser().resolve(), args.output.expanduser().resolve()
    if args.action in ('_run', '_worker'):
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.action == 'pause':
            write(workdir(args) / 'STOP_AFTER_CURRENT', dict(requested_unix=time.time()))
            print('Pause requested: running parallel evaluations will finish; no new ones will start.')
        elif args.action == 'resume':
            # Check lock before changing a live controller's requested stop.
            with lock(workdir(args) / 'controller.lock'):
                (workdir(args) / 'STOP_AFTER_CURRENT').unlink(missing_ok=True)
            launch(args)
        elif args.action == 'restore-old':
            with lock(workdir(args) / 'controller.lock'):
                restore_old(args)
        else:
            {'launch': launch, 'status': status,
             '_run': run, '_worker': worker, '_simulation': simulation, '_probe': probe}[args.action](args)
    except BlockingIOError:
        print('LOCKED: another controller owns this queue or node. No lock was deleted.', file=sys.stderr)
        raise SystemExit(2)
    except BaseException as error:
        if args.action == '_run':
            snapshot(args, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        traceback.print_exc()
        raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1)


if __name__ == '__main__':
    main()
