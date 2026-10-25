#!/usr/bin/env python3
"""Two synthetic updates of the real Energy G model on eight NCCL ranks.

This does not read or write training checkpoints and does not count toward the
scientific training budget. Invoke only through the prepared Energy pipeline.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from score_function.loss import fixed_sigma_dsm_loss
from score_function.model.score_branch import build_model
from score_function.utils.config import load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--result',required=True,type=Path)
    args=p.parse_args()
    cfg=load_config(args.config)
    if (cfg['model']['parameterization']!='energy' or not cfg['model'].get('temporal_attention')
            or cfg['training']['microbatch_size']!=256 or cfg['training']['sigma']!=0.1):
        raise ValueError('Smoke must use the actual aligned G-energy configuration')
    if int(os.environ.get('WORLD_SIZE','0'))!=8 or torch.cuda.device_count()!=8:
        raise ValueError('Energy smoke requires exactly eight visible GPUs and eight DDP ranks')
    rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    torch.set_num_threads(cfg['runtime'].get('cpu_threads',2))
    torch.cuda.set_per_process_memory_fraction(cfg['runtime'].get('memory_fraction',1.0),rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    try:
        device=torch.device('cuda',rank)
        torch.manual_seed(cfg['training']['seed'])
        model=build_model(cfg,device).train()
        wrapped=DistributedDataParallel(model,device_ids=[rank],broadcast_buffers=False)
        optimizer=torch.optim.AdamW(wrapped.parameters(),lr=cfg['training']['learning_rate'],
            weight_decay=cfg['training']['weight_decay'])
        initial=torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()
        # Distinct rank-local inputs make incorrect gradient synchronization observable.
        torch.manual_seed(cfg['training']['seed']+rank+1)
        batch=cfg['training']['microbatch_size']
        x=torch.randn(batch,80,4,device=device)
        scene=torch.randn(batch,107,192,device=device)
        route=torch.randn(batch,192,device=device)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            epsilon=torch.randn_like(x)
            prediction=wrapped(x+0.1*epsilon,scene,route)
            loss=fixed_sigma_dsm_loss(prediction,epsilon,0.1)
            loss.backward()
            good=bool(torch.isfinite(loss)) and all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in model.parameters())
            flag=torch.tensor(int(good),device=device)
            dist.all_reduce(flag,op=dist.ReduceOp.MIN)
            if not bool(flag):
                raise RuntimeError('Energy smoke has missing/nonfinite loss or parameter gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip'])
            optimizer.step()
            if rank==0:
                print(f'Energy eight-GPU smoke update {step+1}/2, loss={loss.item():.8g}',flush=True)
        current=torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        updated=torch.tensor(int(not torch.equal(current,initial)),device=device)
        dist.all_reduce(updated,op=dist.ReduceOp.MIN)
        reference=current.clone(); dist.broadcast(reference,src=0)
        difference=(current-reference).abs().max()
        dist.all_reduce(difference,op=dist.ReduceOp.MAX)
        if not bool(updated) or not bool(torch.isfinite(difference)) or difference.item()>1e-6:
            raise RuntimeError('Energy DDP update was absent or parameters failed to synchronize')
        peak=torch.tensor(torch.cuda.max_memory_allocated(device),device=device,dtype=torch.float64)
        dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if rank==0:
            result=dict(world_size=8,steps=2,microbatch_per_rank=256,global_batch=2048,
                parameterization='energy',updated=True,synchronized=True,finite_gradients=True,
                max_replica_difference=difference.item(),max_rank_peak_allocated_bytes=int(peak.item()))
            args.result.parent.mkdir(parents=True,exist_ok=True)
            temporary=args.result.with_suffix('.tmp')
            temporary.write_text(json.dumps(result,indent=2)+'\n'); temporary.replace(args.result)
            print(json.dumps(result),flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
