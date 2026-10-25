#!/usr/bin/env python3
"""Freeze an eight-GPU E10-G run from the completed, actual S10-G run.

Only the score/energy parameterization and output paths change. Training seed,
data, augmentation, optimizer, schedule, EMA, batch and update budget are copied
from the source run. The existing matrix supplies the evaluation cohort and K0.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import evaluate_g_seed_k_1015 as matrix
import full_test14hard_suite as suite
import prepare_g_sigma_trials as prepare_g
import queue_g_sigma_trials as training

ROOT = Path('/home/data/wyk')
DEFAULT_QUEUE = ROOT / 'outputs/score_aligned_1004/E10_G_aligned_1025_v1'
SEED = 20260922
KS = (5, 10, 20, 50, 100)


def energy_config(original, output):
    result = copy.deepcopy(original)
    result['model']['parameterization'] = 'energy'
    result['output'] = result['paths']['run_dir'] = str(output)
    restored = copy.deepcopy(result)
    restored['model'] = copy.deepcopy(original['model'])
    restored['output'] = original['output']
    restored['paths']['run_dir'] = original['paths']['run_dir']
    if restored != original:
        raise AssertionError('Energy preparation changed an unrelated setting')
    return result


def verify_source_metadata(config, info, status):
    m, t = config['model'], config['training']
    if (m.get('architecture') != 'temporal_score_function'
            or m.get('parameterization', 'score') != 'score'
            or not m.get('temporal_attention') or m.get('neighbor_future', False)
            or m.get('sigma_conditioning', False) or t['sigma'] != 0.1
            or t.get('early_stopping', True)
            or t.get('lr_schedule') != 'constant_after_warmup'
            or not t.get('data_augmentation', {}).get('enabled')):
        raise ValueError('Source must be the completed fixed-sigma aligned S10-G run')
    expected = dict(world_size=8, global_batch=2048, microbatch_per_rank=256,
                    gradient_accumulation=1, train_samples=947912, precision='float32')
    if any(info.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Source training metadata differs from the recorded protocol: {expected}')
    budget = info['training_budget']
    if (t['batch_size'] != 2048 or t['microbatch_size'] != 256
            or budget['planned_epochs'] != 529
            or budget['planned_optimizer_updates'] != 244398
            or status.get('state') != 'complete' or status.get('epoch') != 529
            or status.get('step') != 244398):
        raise ValueError('Require full-budget source completion: 529 epochs / 244398 updates')


def source_job(plan, setting):
    wanted = f'{setting}_seed{SEED}_K0'
    selected = [job for job in plan['jobs'] if job['id'] == wanted]
    if len(selected) != 1:
        raise ValueError(f'Existing matrix does not contain exactly one {wanted}')
    return selected[0]


def validate_control(matrix_out, plan, setting):
    job = source_job(plan, setting)
    receipt_path = matrix_out / 'jobs' / f"{job['id']}.json"
    record = suite.read(receipt_path)
    if record.get('state') != 'complete':
        raise ValueError(f'Existing seed {SEED} K0 is not complete: {receipt_path}')
    checked = matrix.validate_result(record['result_dir'], job, plan, matrix_out)
    if any(record.get(k) != v for k, v in checked.items()):
        raise ValueError('Existing matrix K0 receipt no longer matches its simulation')
    return receipt_path, record


INSPECT = r'''
import json, sys, torch
from pathlib import Path
from types import SimpleNamespace
from score_function.utils.config import load_config, model_config
from score_function.utils.dataset import ShardedDataset
from score_function.utils.training_budget import training_budget
from score_function.utils.train_utils import load_tensor, file_hash, source_hashes
from score_function.utils.planner_utils import planner_identity
from score_function.model.score_branch import build_model
from score_function.model.refinement import refine_ego
from score_function.loss import fixed_sigma_dsm_loss
p=json.loads(sys.argv[1]); torch.set_num_threads(2)
cfg=load_config(p['config']); source=Path(p['source'])
if cfg != json.loads(Path(p['config']).read_text()):
    raise ValueError('Prepared config does not round-trip unchanged')
last=load_tensor(source/'last.pt'); final=load_tensor(source/'final_ema.pt')
provenance=json.loads((source/'provenance.json').read_text())
original=json.loads((source/'config.json').read_text())
saved=dict(last['config']); saved['model']=model_config(saved['model'])
requested=dict(original); requested['model']=model_config(requested['model'])
saved_runtime=dict(saved.get('runtime',{})); requested_runtime=dict(requested.get('runtime',{}))
if str(saved_runtime.get('device','')).startswith('cuda') and str(requested_runtime.get('device','')).startswith('cuda'):
    saved_runtime['device']=requested_runtime['device']='cuda'
saved['runtime']=saved_runtime; requested['runtime']=requested_runtime
if saved != requested:
    raise ValueError('Source config.json differs from actual completed last.pt configuration')
if last['sources'] != source_hashes() or last['sources'] != provenance['sources']:
    raise ValueError('Restored source differs from actual S10-G training sources')
if not last.get('finished') or last.get('termination')!='epoch_budget_exhausted':
    raise ValueError('Source last.pt is not a completed full-budget checkpoint')
if (last['epoch'],last['step'],final['epoch'],final['step'],final['weight_kind']) != (529,244398,529,244398,'ema'):
    raise ValueError('Source is not the final-epoch EMA convention')
if set(last['ema_branch']) != set(final['score_branch']) or any(not torch.equal(v,final['score_branch'][k]) for k,v in last['ema_branch'].items()):
    raise ValueError('Source final_ema.pt differs from last.pt EMA branch')
ref=load_tensor(p['matrix_checkpoint'])
if (ref['epoch'],ref['step'],ref['sigma_score'],ref['weight_kind'])!=(529,244398,0.1,'ema'):
    raise ValueError('Matrix does not use the final S10-G EMA')
if ref['planner'] != final['planner'] or ref['ego_normalizer'] != final['ego_normalizer'] or set(ref['score_branch'])!=set(final['score_branch']) or any(not torch.equal(v,ref['score_branch'][k]) for k,v in final['score_branch'].items()):
    raise ValueError('Matrix and source training refer to different S10-G weights/normalizer')
if planner_identity(cfg) != final['planner']:
    raise ValueError('DP checkpoint, arguments or source identity changed')
dataset=ShardedDataset(cfg['cache'],'train')
if cfg['training'].get('data_augmentation',{}).get('enabled',False):
    from score_function.data_process.online_training import RawTrainingDataset
    dataset=RawTrainingDataset(dataset,cfg)
if dataset.sha256 != provenance['cache_sha256'] or len(dataset)!=947912:
    raise ValueError('Training cache/split differs from original S10-G')
budget=training_budget(cfg['training'],len(dataset))
if budget != last['training_budget']:
    raise ValueError('Training update budget changed')
del dataset
torch.manual_seed(cfg['training']['seed'])
model=build_model(cfg,'cpu').train()
optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'])
x=torch.randn(2,80,4); c=torch.randn(2,107,192); r=torch.randn(2,192)
for _ in range(2):
    optimizer.zero_grad(set_to_none=True); epsilon=torch.randn_like(x)
    prediction=model(x+0.1*epsilon,c,r)
    loss=fixed_sigma_dsm_loss(prediction,epsilon,0.1); loss.backward()
    if any(q.grad is None or not torch.isfinite(q.grad).all() for q in model.parameters()):
        raise ValueError('Energy double-backprop has a missing/nonfinite parameter gradient')
    optimizer.step()
model.eval().requires_grad_(False)
with torch.inference_mode():
    y=model(x.clone(),c.clone(),r.clone())
if y.shape != x.shape or not torch.isfinite(y).all() or y.requires_grad:
    raise ValueError('Energy inference-mode gradient preflight failed')
normalizer=SimpleNamespace(**{key:torch.tensor(final['ego_normalizer'][key]).reshape(1,1,4) for key in ('mean','std')})
expected={str(k):refine_ego(model,x,c,r,normalizer,0.1,0.1,k,
    heading_projection=cfg['refinement']['heading_projection'],record_trace=False)[0] for k in (0,5,20)}
torch.save(dict(weights=model.state_dict(),x=x,scene=c,route=r,score=y,
    normalizer=final['ego_normalizer'],refinements=expected),p['parity_probe'])
print(json.dumps(dict(parameters=sum(q.numel() for q in model.parameters()),budget=budget,
    cache_sha256=provenance['cache_sha256'],source_weight_kind='final_epoch_ema',
    source_epoch=529,source_step=244398,energy_backward_steps=2,inference_mode_ok=True)))
'''


EVAL_PARITY = r'''
import json,sys,torch
from types import SimpleNamespace
from score_function.utils.config import load_config
from score_function.utils.train_utils import load_tensor
from score_function.model.score_branch import build_model
from score_function.model.refinement import refine_ego
torch.set_num_threads(2)
cfg=load_config(sys.argv[1]); probe=load_tensor(sys.argv[2])
model=build_model(cfg,'cpu').eval().requires_grad_(False)
model.load_state_dict(probe['weights'],strict=True)
with torch.inference_mode():
    score=model(probe['x'].clone(),probe['scene'].clone(),probe['route'].clone())
torch.testing.assert_close(score,probe['score'],rtol=1e-6,atol=1e-7)
normalizer=SimpleNamespace(**{key:torch.tensor(probe['normalizer'][key]).reshape(1,1,4) for key in ('mean','std')})
errors={}
for key,expected in probe['refinements'].items():
    actual,_=refine_ego(model,probe['x'],probe['scene'],probe['route'],normalizer,
        0.1,0.1,int(key),heading_projection=cfg['refinement']['heading_projection'],record_trace=False)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    errors[key]=float((actual-expected).abs().max())
print(json.dumps(dict(matrix_inference_snapshot=True,score_max_error=float((score-probe['score']).abs().max()),refinement_max_errors=errors)))
'''


def prepare(args):
    root, out = args.root.resolve(), args.queue_dir.resolve()
    source_queue = (args.source_queue or root / 'outputs/score_aligned_1004/G_sigma_1012_8gpu_v1').resolve()
    source = source_queue / 'runs/S10-G/score'
    matrix_out = (args.matrix or root / 'dataset/nuplan/exp/score_G_seed_k_1015_v1').resolve()
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) != 8 or len(set(gpus)) != 8 or min(gpus) < 0:
        raise ValueError('This protocol requires exactly eight distinct physical GPUs')
    if out.exists():
        if not (out/'plan.json').is_file():
            raise FileExistsError(f'Partial preparation exists: {out}; preserve and inspect it before preparing again.')
        plan=suite.read(out/'plan.json'); spec=plan['evaluation']
        if (plan['root']!=str(root) or plan['source_queue']!=str(source_queue)
                or spec['matrix']!=str(matrix_out) or spec['setting']!=args.setting
                or spec['gpus']!=gpus or spec['seed']!=SEED or spec['ks']!=list(KS)):
            raise ValueError('Existing Energy queue was prepared with different inputs/settings')
        training.verify_plan(plan)
        print(f'Already prepared and verified: {out}. Resume the same run command.',flush=True)
        return plan
    if out == source_queue or source_queue in out.parents or out in source_queue.parents:
        raise ValueError('Energy output must be separate from the source training queue')
    original, info, provenance, status = (suite.read(source / n) for n in
        ('config.json','model_info.json','provenance.json','status.json'))
    verify_source_metadata(original, info, status)
    if Path(original['paths']['root']).resolve() != root:
        raise ValueError('Keep original absolute data/root paths mounted on node B; root remapping is not implicit')
    old = suite.read(matrix_out / 'plan.json')
    if (old['gamma'] != 0.1 or old['sigma'] != 0.1 or old['gpus'] != gpus
            or float(old['gpu_fraction']) != 1.0):
        raise ValueError('Matrix gamma/sigma/GPU allocation does not match the requested protocol')
    if original['refinement']['heading_projection'] != old['heading_projection']:
        raise ValueError('Source training and matrix heading projection differ')
    anchor_path, anchor = validate_control(matrix_out, old, args.setting)
    if matrix.inventory(root) != old['db_inventory']:
        raise ValueError('Test database inventory differs from completed matrix')
    out.mkdir(parents=True)
    try:
        code = out / 'code/energy'
        repo = Path(__file__).resolve().parents[1]
        origins = prepare_g.restore_original_sources(code, provenance['sources'],
            [source_queue / 'code/fixed', repo], [repo, root / 'Score_Function_G_sigma_8gpu_1012'])
        # Reuse precisely the simulation launcher/config from the original matrix.
        sim_source = matrix_out / 'code/lg'
        eval_code=out/'code/evaluation'
        shutil.copytree(sim_source/'score_function',eval_code/'score_function',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        source_bindings={str(Path(name).resolve()):sha for name,sha in old['files'].items()}
        for name in training.sources(eval_code):
            original_path=(sim_source/'score_function'/name).resolve()
            expected=source_bindings.get(str(original_path))
            if expected is None or suite.digest(original_path)!=expected or suite.digest(eval_code/'score_function'/name)!=expected:
                raise ValueError(f'Matrix inference source is not frozen by its original plan: {original_path}')
        for name in ('sim_score_planner.sh', 'configs/nuplan_planner.yaml'):
            original_path=(sim_source/name).resolve()
            if source_bindings.get(str(original_path))!=suite.digest(original_path):
                raise ValueError(f'Matrix inference launcher/config changed: {original_path}')
            dest = eval_code / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_path,dest)
        shutil.copy2(Path(__file__).with_name('energy_ddp_smoke_1025.py'),code/'energy_ddp_smoke_1025.py')
        planner = (eval_code / 'score_function/planner/planner.py').read_text()
        if 'torch.manual_seed(planner_seed(timestamp, self.settings["reference_seed"]))' not in planner:
            raise ValueError('Restored training snapshot does not preserve matrix DP seeding')
        reference = out / 'reference/S10-G'; reference.mkdir(parents=True)
        for name in ('config.json','model_info.json','provenance.json','status.json'):
            shutil.copy2(source / name, reference / name)
        run_dir = out / 'runs/E10-G'
        config = energy_config(original, run_dir)
        config_path = out / 'configs/E10-G.json'; suite.write(config_path,config)
        payload = dict(config=str(config_path),source=str(source),parity_probe=str(out/'energy_parity_probe.pt'),
                       matrix_checkpoint=str(suite.model_paths(matrix_out,source_job(old,args.setting)['model'])[1]))
        check = subprocess.run([sys.executable,'-c',INSPECT,json.dumps(payload)],
            cwd=code,env=prepare_g.clean_env(code,root),text=True,capture_output=True)
        if check.returncode:
            raise RuntimeError(f'Energy CPU/data preflight failed:\n{check.stdout}\n{check.stderr}')
        inspection = json.loads(check.stdout.strip().splitlines()[-1])
        parity=subprocess.run([sys.executable,'-c',EVAL_PARITY,str(config_path),payload['parity_probe']],
            cwd=eval_code,env=prepare_g.clean_env(eval_code,root),text=True,capture_output=True)
        if parity.returncode:
            raise RuntimeError(f'Matrix inference/training Energy parity failed:\n{parity.stdout}\n{parity.stderr}')
        inspection['matrix_inference_parity']=json.loads(parity.stdout.strip().splitlines()[-1])
        suite.write(out/'preflight.json',inspection)
        job = dict(id='E10-G',code=str(code),config=str(config_path),output=str(run_dir),
            resume=False,world_size=8,gradient_accumulation=1,gpus=gpus,target_epochs=529,
            target_updates=244398,sources=training.sources(code),parameters=inspection['parameters'],
            training_budget=inspection['budget'])
        # Freeze the control and all five existing direct-score comparison receipts.
        controls = {}
        for k in (0,*KS):
            path = matrix_out/'jobs'/f'{args.setting}_seed{SEED}_K{k}.json'
            record = suite.read(path)
            if record.get('state') != 'complete':
                raise ValueError(f'Required direct-score comparison is incomplete: {path}')
            srcjob = next(j for j in old['jobs'] if j['id']==record['id'])
            checked = matrix.validate_result(record['result_dir'],srcjob,old,matrix_out)
            if any(record.get(key)!=v for key,v in checked.items()):
                raise ValueError(f'Source result receipt changed: {path}')
            target = out/'reference/matrix'/path.name; suite.write(target,record)
            controls[str(k)] = str(target)
        files = {str(p.resolve()):suite.digest(p) for p in out.rglob('*') if p.is_file()}
        # Bind existing source/checkpoints and metric inputs without copying large artifacts.
        additional = [source/'last.pt',source/'final_ema.pt',Path(payload['matrix_checkpoint']),
                      matrix_out/'plan.json',Path(original['cache'])]
        for path in [Path(__file__),Path(training.__file__),Path(matrix.__file__),Path(suite.__file__),
                     Path(__file__).with_name('energy_ddp_smoke_1025.py'),
                     Path(__file__).with_name('run_energy_aligned_1025.py'),*additional]:
            files[str(path.resolve())] = suite.digest(path)
        # Preserve the DP/devkit/filter bytes that produced the existing matrix.
        # Other old queues' locks and mutable status files are deliberately not used.
        external_roots = (root/'Diffusion-Planner',root/'nuplan-devkit')
        for name,expected in old['files'].items():
            path=Path(name).resolve()
            if any(base.resolve() in path.parents for base in external_roots):
                if suite.digest(path)!=expected:
                    raise ValueError(f'DP/devkit input differs from completed matrix: {path}')
                files[str(path)]=expected
        plan = dict(schema_version=1,root=str(root),queue_dir=str(out),hostname=socket.gethostname(),
            source_run=str(source.parent),source_queue=str(source_queue),source_recovery=origins,
            jobs=[job],files=files,created_unix=time.time(),
            protocol='Actual S10-G configuration and code; only energy head/output changed; final epoch EMA',
            evaluation=dict(setting=args.setting,seed=SEED,ks=list(KS),gamma=0.1,sigma=0.1,
                code=str(eval_code),sources=training.sources(eval_code),
                matrix=str(matrix_out),control_receipts=controls,source_job=source_job(old,args.setting),
                filters=old['filters'],tokens=old['tokens'],scenario_keys=old['scenario_keys'],
                db_inventory=old['db_inventory'],heading_projection=old['heading_projection'],
                gpu_fraction=1.0,gpus=gpus,threads=old['threads']))
        suite.write(out/'plan.json',plan)
        (out/'logs').mkdir(exist_ok=True)
        training.verify_plan(plan)
        print(f'READY: {out}\nE10-G: 8 GPUs, batch 2048, microbatch 256, 529 epochs / 244398 updates.')
        print(f'After training: final-epoch EMA; {args.setting}; seed={SEED}; K={list(KS)}; existing K0 reused.')
        return plan
    except Exception as exc:
        suite.write(out/'prepare_failed.json',dict(error=repr(exc)))
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--queue-dir',type=Path,default=DEFAULT_QUEUE)
    p.add_argument('--source-queue',type=Path)
    p.add_argument('--matrix',type=Path)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--setting',choices=('hard_NR','random_NR'),default='hard_NR')
    return p


if __name__=='__main__':
    prepare(parser().parse_args())
