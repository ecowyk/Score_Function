#!/usr/bin/env python3
"""Resume one frozen eight-GPU Energy training run, then five paired evaluations."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import evaluate_g_seed_k_1015 as matrix
import full_test14hard_suite as suite
import queue_g_sigma_trials as training
from prepare_energy_aligned_1025 import DEFAULT_QUEUE, KS, SEED, validate_control


FINAL_PREFLIGHT = r'''
import json,sys,torch
from score_function.utils.config import load_config
from score_function.utils.checkpoint import load_selected
from score_function.utils.planner_utils import planner_identity
torch.set_num_threads(2)
cfg=load_config(sys.argv[1]); model,state=load_selected(cfg,sys.argv[2],'cpu')
if (state['epoch'],state['step'],state['weight_kind'],state['sigma_score'])!=(529,244398,'ema',0.1):
    raise ValueError('Expected full-budget final-epoch Energy EMA')
if model.parameterization!='energy' or cfg['reference_seed']!=20260922:
    raise ValueError('Wrong Energy parameterization or inference seed')
if state['planner']!=planner_identity(cfg):
    raise ValueError('Frozen DP identity changed')
if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
    raise ValueError('Final Energy weights contain nonfinite values')
with torch.inference_mode():
    value=model(torch.randn(1,80,4),torch.randn(1,107,192),torch.randn(1,192))
if value.shape!=(1,80,4) or not torch.isfinite(value).all():
    raise ValueError('Final Energy checkpoint inference preflight failed')
print(json.dumps(dict(weight_kind='ema',epoch=529,step=244398,seed=20260922,finite=True)))
'''


def evaluation_jobs(setting):
    split, challenge = matrix.SETTINGS[setting]
    return [dict(id=f'{setting}_seed{SEED}_K{k}',setting=setting,split=split,
        challenge=challenge,model='DP_K0_reused' if k==0 else f'E10-G_seed{SEED}',seed=SEED,K=k)
        for k in (0,*KS)]


def smoke_identity(queue,plan):
    job=plan['jobs'][0]
    return dict(plan_sha256=suite.digest(queue/'plan.json'),config_sha256=suite.digest(job['config']),
        sources=job['sources'],gpus=job['gpus'],world_size=8,steps=2,microbatch_per_rank=256)


def run_gpu_smoke(queue,plan,poll_seconds,pipeline_lock):
    job=plan['jobs'][0]; identity=smoke_identity(queue,plan)
    receipt_path=queue/'energy_ddp_smoke.json'
    if receipt_path.exists():
        receipt=suite.read(receipt_path)
        result=receipt.get('result',{})
        if (receipt.get('identity')!=identity or not result.get('finite_gradients')
                or not result.get('synchronized') or not result.get('updated')
                or result.get('world_size')!=8 or result.get('steps')!=2):
            raise ValueError('Existing Energy GPU smoke receipt is invalid or belongs to another plan')
        print('SKIP verified eight-GPU Energy double-backprop smoke',flush=True)
        return
    while True:
        occupied=training.gpu_busy(job['gpus'])
        if not occupied: break
        suite.write(queue/'pipeline_status.json',dict(state='waiting_gpu_smoke',busy_gpus=occupied))
        print('WAIT smoke GPUs:',occupied,flush=True); time.sleep(poll_seconds)
    training.verify_plan(plan)
    result_path=queue/'logs'/f'energy_ddp_smoke_{time.time_ns()}.json'
    log=queue/'logs/energy_ddp_smoke.log'
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc_per_node=8',str(Path(job['code'])/'energy_ddp_smoke_1025.py'),
        '--config',job['config'],'--result',str(result_path)]
    print(f'START eight-GPU Energy smoke (two real-width microbatch256 updates): {log}',flush=True)
    with log.open('a') as handle:
        child=subprocess.Popen(command,cwd=job['code'],env=training.clean_env(job['code'],job['gpus']),
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(pipeline_lock.fileno(),))
        try:
            while child.poll() is None:
                suite.write(queue/'pipeline_status.json',dict(state='gpu_smoke',pid=child.pid,log=str(log)))
                try: child.wait(timeout=poll_seconds)
                except subprocess.TimeoutExpired: pass
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait()
            raise
    if child.returncode or not result_path.is_file():
        raise RuntimeError(f'Eight-GPU Energy smoke failed before training (exit {child.returncode}): {log}')
    result=suite.read(result_path)
    if not result.get('finite_gradients') or not result.get('synchronized') or not result.get('updated') or result.get('world_size')!=8 or result.get('steps')!=2:
        raise ValueError('Eight-GPU Energy smoke did not certify two synchronized finite updates')
    suite.write(receipt_path,dict(identity=identity,result=result,log=str(log),completed_unix=time.time()))
    print('PASS eight-GPU Energy double-backprop smoke; starting full-budget training',flush=True)


def prepare_evaluation(queue, plan):
    target=queue/'evaluation'; spec=plan['evaluation']; job=plan['jobs'][0]
    matrix_out=Path(spec['matrix']); old=suite.read(matrix_out/'plan.json')
    anchor_path,anchor=validate_control(matrix_out,old,spec['setting'])
    final=training.inspect_checkpoint(job,suite.digest(queue/'plan.json'),final=True)
    if (target/'plan.json').exists():
        ep=suite.read(target/'plan.json')
        verify_evaluation(queue,plan,target,ep)
        return target,ep
    target.mkdir(exist_ok=True)
    config=copy.deepcopy(suite.read(job['config']))
    config['reference_seed']=SEED
    config_path,checkpoint=suite.model_paths(target,f'E10-G_seed{SEED}')
    allowed={config_path,checkpoint,target/'checkpoint_preflight.json',
             target/'jobs'/f"{spec['setting']}_seed{SEED}_K0.json"}
    if any(path not in allowed for path in target.rglob('*') if path.is_file() or path.is_symlink()):
        raise ValueError(f'Unknown files in partial evaluation preparation: {target}; inspect before retrying')
    if config_path.exists() and suite.read(config_path)!=config:
        raise ValueError('Partial Energy evaluation config has different settings')
    if checkpoint.is_symlink():
        if checkpoint.resolve()!=Path(final['final_ema']).resolve():
            raise ValueError('Partial Energy checkpoint link points elsewhere')
    elif checkpoint.exists():
        raise ValueError('Partial Energy checkpoint must be a link to the final EMA')
    else:
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        checkpoint.symlink_to(final['final_ema'])
    suite.write(config_path,config)
    checked=subprocess.run([sys.executable,'-c',FINAL_PREFLIGHT,str(config_path),str(checkpoint)],
        cwd=spec['code'],env=suite.env_for(Path(plan['root']),Path(spec['code']),''),
        text=True,capture_output=True)
    if checked.returncode:
        raise RuntimeError(f'Final Energy checkpoint preflight failed:\n{checked.stdout}\n{checked.stderr}')
    suite.write(target/'checkpoint_preflight.json',json.loads(checked.stdout.strip().splitlines()[-1]))
    ep=copy.deepcopy(spec)
    ep.update(schema_version=1,root=plan['root'],training_plan_sha256=suite.digest(queue/'plan.json'),
        source_matrix_plan_sha256=suite.digest(matrix_out/'plan.json'),jobs=evaluation_jobs(spec['setting']),
        seeds=[SEED],ks=[0,*KS],settings=[spec['setting']],reference_seed=SEED,
        code=spec['code'],checkpoint_policy='final_epoch_ema_not_validation_selected',
        created_unix=time.time())
    ep['files']={str(config_path.resolve()):suite.digest(config_path),
                 str(checkpoint.resolve()):suite.digest(checkpoint),
                 str((target/'checkpoint_preflight.json').resolve()):suite.digest(target/'checkpoint_preflight.json'),
                 str(anchor_path.resolve()):suite.digest(anchor_path)}
    borrowed=copy.deepcopy(anchor)
    borrowed.update(ep['jobs'][0],state='complete',reused=True,
        reuse_source_receipt=str(anchor_path),reuse_source_sha256=suite.digest(anchor_path),
        reuse_source_matrix=str(matrix_out),note='DP K0 from completed seed-matched matrix; not rerun')
    suite.write(target/'jobs'/f"{borrowed['id']}.json",borrowed)
    (target/'logs').mkdir(exist_ok=True)
    suite.write(target/'plan.json',ep)
    suite.write(target/'status.json',dict(state='prepared'))
    verify_evaluation(queue,plan,target,ep)
    return target,ep


def verify_evaluation(queue,plan,out,ep):
    training.verify_plan(plan)
    spec=plan['evaluation']; matrix_out=Path(spec['matrix'])
    if ep['training_plan_sha256']!=suite.digest(queue/'plan.json') or ep['source_matrix_plan_sha256']!=suite.digest(matrix_out/'plan.json'):
        raise ValueError('Training/source matrix plan changed after evaluation preparation')
    if ep['jobs']!=evaluation_jobs(spec['setting']) or ep['seeds']!=[SEED] or ep['ks']!=[0,*KS]:
        raise ValueError('Energy evaluation seed/K axes changed')
    for key in ('setting','seed','gamma','sigma','filters','tokens','scenario_keys','db_inventory',
                'heading_projection','gpu_fraction','gpus','threads','code','sources'):
        if ep.get(key)!=spec.get(key):
            raise ValueError(f'Frozen Energy evaluation setting changed: {key}')
    if ep['code']!=spec['code'] or training.sources(spec['code'])!=spec['sources']:
        raise ValueError('Frozen matrix inference snapshot changed')
    for path,expected in ep['files'].items():
        if suite.digest(path)!=expected:
            raise ValueError(f'Frozen evaluation input changed: {path}')
    config,checkpoint=suite.model_paths(out,f'E10-G_seed{SEED}')
    expected=copy.deepcopy(suite.read(plan['jobs'][0]['config'])); expected['reference_seed']=SEED
    if suite.read(config)!=expected or not checkpoint.is_symlink() or checkpoint.resolve()!=Path(plan['jobs'][0]['output'])/'score/final_ema.pt':
        raise ValueError('Energy evaluation config/final EMA link changed')
    _,anchor=validate_control(matrix_out,suite.read(matrix_out/'plan.json'),spec['setting'])
    receipt=suite.read(out/'jobs'/f"{spec['setting']}_seed{SEED}_K0.json")
    if receipt.get('state')!='complete' or receipt['metrics']!=anchor['metrics'] or receipt['scenario_keys']!=anchor['scenario_keys'] or receipt['result_dir']!=anchor['result_dir']:
        raise ValueError('Borrowed DP K0 differs from verified original matrix result')
    if matrix.inventory(Path(plan['root']))!=spec['db_inventory']:
        raise ValueError('Test database inventory changed')


def summarize(queue,plan,out,ep,save=True):
    matrix.summarize(out,ep,save=save)
    rows=[]
    control=suite.read(plan['evaluation']['control_receipts']['0'])
    baseline=100*control['metrics']['score']
    for k in KS:
        direct=suite.read(plan['evaluation']['control_receipts'][str(k)])
        path=out/'jobs'/f"{plan['evaluation']['setting']}_seed{SEED}_K{k}.json"
        energy=suite.read(path) if path.exists() else dict(state='pending')
        value=100*energy['metrics']['score'] if energy['state']=='complete' else None
        if value is not None and energy['scenario_keys']!=direct['scenario_keys']:
            raise ValueError('Cannot compare Energy and direct-score results from different scenes')
        direct_score=100*direct['metrics']['score']
        rows.append(dict(setting=plan['evaluation']['setting'],seed=SEED,K=k,
            state=energy['state'],DP_K0_100=baseline,S10_G_100=direct_score,E10_G_100=value,
            energy_minus_DP=None if value is None else value-baseline,
            energy_minus_S10_G=None if value is None else value-direct_score,
            source_direct_result=direct['result_dir'],energy_result=energy.get('result_dir','')))
    if save:
        matrix.write_csv(out/'energy_vs_direct.csv',list(rows[0]),rows)
    return rows


def run_evaluations(queue,plan,poll_seconds):
    out,ep=prepare_evaluation(queue,plan)
    code,root=Path(ep['code']),Path(ep['root'])
    lock=(out/'.run.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in ep['jobs'][1:]:
            verify_evaluation(queue,plan,out,ep)
            path=out/'jobs'/f"{job['id']}.json"
            previous=suite.read(path) if path.exists() else {}
            if previous.get('state')=='complete':
                checked=matrix.validate_result(previous['result_dir'],job,ep,out)
                if any(previous.get(k)!=v for k,v in checked.items()):
                    raise ValueError('Completed Energy evaluation receipt changed')
                print('SKIP verified',job['id'],flush=True)
                continue
            if previous.get('state')=='running' and previous.get('pid') and suite.process_identity(previous['pid']):
                raise RuntimeError('Previous evaluation may still be active; inspect it before retrying')
            while True:
                busy=training.gpu_busy(ep['gpus'])
                if not busy:
                    break
                suite.write(out/'status.json',dict(state='waiting_gpu',job=job['id'],busy_gpus=busy))
                print('WAIT eight GPUs:',busy,flush=True)
                time.sleep(poll_seconds)
            label=f"E10G_{job['id']}_{time.time_ns()}"
            log=out/'logs'/f'{label}.log'
            config,checkpoint=suite.model_paths(out,job['model'])
            command=['bash',str(code/'sim_score_planner.sh'),'--python',sys.executable,
                '--root',str(root),'--config',str(config),'--checkpoint',str(checkpoint),
                '--name',label,'--split',job['split'],'--challenge',job['challenge'],
                '--steps',str(job['K']),'--gamma','0.1','--gpus',','.join(map(str,ep['gpus'])),
                '--gpu-fraction','1.0','--threads',str(ep['threads'])]
            if previous:
                suite.write(out/'retry_history'/f'{label}.json',previous)
            record=dict(job,state='running',log=str(log),command=command,
                plan_sha256=suite.digest(out/'plan.json'),started_unix=time.time())
            print(f"START {job['id']}: eight GPUs, final-epoch EMA; log={log}",flush=True)
            try:
                with log.open('w') as handle:
                    child=subprocess.Popen(command,cwd=code,env=suite.env_for(root,code,','.join(map(str,ep['gpus']))),
                        stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                    record['pid']=child.pid; suite.write(path,record)
                    try:
                        while child.poll() is None:
                            suite.write(out/'status.json',dict(state='running',job=job['id'],pid=child.pid,
                                log=str(log),updated_unix=time.time()))
                            try:
                                child.wait(timeout=poll_seconds)
                            except subprocess.TimeoutExpired:
                                pass
                    except BaseException:
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            try: child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid,signal.SIGKILL); child.wait()
                        raise
                if child.returncode:
                    raise RuntimeError(f'Simulation exited {child.returncode}: {log}')
                results=[line.split(': ',1)[1] for line in log.read_text(errors='replace').splitlines() if line.startswith('Results: ')]
                if len(results)!=1:
                    raise ValueError('Simulation did not report exactly one result directory')
                result=results[0]
                record.update(result_dir=result,**matrix.validate_result(result,job,ep,out),state='complete',finished_unix=time.time())
                suite.write(path,record)
                summarize(queue,plan,out,ep)
            except BaseException as exc:
                record.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
                suite.write(path,record); suite.write(out/'status.json',dict(state=record['state'],job=job['id'],error=str(exc)))
                summarize(queue,plan,out,ep)
                raise
        suite.write(out/'status.json',dict(state='complete',updated_unix=time.time()))
        summarize(queue,plan,out,ep)


def run(args):
    queue=args.queue_dir.resolve(); plan=suite.read(queue/'plan.json')
    with (queue/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        training.verify_plan(plan)
        run_gpu_smoke(queue,plan,args.poll_seconds,lock)
        suite.write(queue/'pipeline_status.json',dict(state='training',updated_unix=time.time()))
        training.run(SimpleNamespace(queue_dir=queue,poll_seconds=args.poll_seconds,
            gpu_memory_mib=256,gpu_util_percent=5))
        suite.write(queue/'pipeline_status.json',dict(state='evaluating',updated_unix=time.time()))
        run_evaluations(queue,plan,args.poll_seconds)
        suite.write(queue/'pipeline_status.json',dict(state='complete',updated_unix=time.time()))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','status'))
    p.add_argument('--queue-dir',type=Path,default=DEFAULT_QUEUE)
    p.add_argument('--poll-seconds',type=int,default=30)
    args=p.parse_args(argv)
    if not 1<=args.poll_seconds<=60:
        p.error('--poll-seconds must be 1..60')
    try:
        if args.action=='run':
            run(args)
        else:
            queue=args.queue_dir.resolve(); plan=suite.read(queue/'plan.json')
            print(training.read(queue/'pipeline_status.json',dict(state='prepared')))
            training.show_status(queue,plan)
            if (queue/'evaluation/plan.json').exists():
                print(training.read(queue/'evaluation/status.json',{}))
                summarize(queue,plan,queue/'evaluation',suite.read(queue/'evaluation/plan.json'),save=False)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Training retains its own queue lock if still running; inspect status before resuming.',flush=True)
        return 130
    except Exception as exc:
        if args.queue_dir.is_dir() and args.action=='run':
            suite.write(args.queue_dir/'pipeline_status.json',dict(state='failed',error=f'{type(exc).__name__}: {exc}',updated_unix=time.time()))
        print(f'ERROR: {type(exc).__name__}: {exc}',flush=True)
        return 1


if __name__=='__main__':
    sys.exit(main())
