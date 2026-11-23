#!/usr/bin/env python3
"""
Communication profiling for CPDP / DDP / LocalSGD. Reports two directly-measured,
clearly-labelled quantities per method, using the SAME code path as train_scaling.py:

  (1) nccl_kernel_s : GPU time in NCCL device kernels (ncclKernel_*), per-GPU mean,
                      from torch.profiler. Kernel occupancy; may overlap compute.
  (2) exposed_comm_s: wall(comm ON) - wall(comm OFF), BOTH measured with the
                      profiler OFF (so profiler overhead cancels). The part of
                      communication exposure estimated by the controlled window difference.
                      comm OFF = all steps under no_sync() + reconciliation disabled.

Scaled estimates use actual boundary counts, including final partial intervals.
SyncBatchNorm and DDP buffer communication are not disabled by no_sync; see docs/profiling.md.
No drift diagnostic is executed. Does NOT run full training.
"""
import os, csv, time, argparse
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, ProfilerActivity

import train_scaling as T
from schedule import step_policy
from synchronization import gradient_context, reconcile_parameters
import math

def is_nccl_device_kernel(name: str) -> bool:
    n = name.lower()
    return n.startswith('ncclkernel') or 'nccldevkernel' in n

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='resnet50')
    p.add_argument('--dataset', type=str, default='cifar100')
    p.add_argument('--method', type=str, default='ddp')
    p.add_argument('--sync-period', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--slowmo-beta', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--backend', type=str, default='nccl')
    p.add_argument('--no-syncbn', action='store_true', default=False)
    p.add_argument('--warmup-iters', type=int, default=10)
    p.add_argument('--measure-cycles', type=int, default=25)
    p.add_argument('--wall-repeats', type=int, default=3)
    p.add_argument('--steps-per-epoch', type=int, default=0)
    p.add_argument('--scale-tag', type=str, required=True)
    p.add_argument('--outfile', type=str, required=True)
    p.add_argument('--dump-kernels', action='store_true', default=False)
    args = p.parse_args()
    step_policy(args.method, args.sync_period, 1, 1)
    if args.measure_cycles < 1 or args.wall_repeats < 1 or args.warmup_iters < 0:
        p.error('Invalid profiling window')
    return args

def main():
    args = parse()
    dist.init_process_group(backend=args.backend)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    T.set_seed(args.seed, rank)

    K = args.sync_period
    measured_steps = (args.measure_cycles * K) if K > 1 else 50
    units = args.measure_cycles if K > 1 else measured_steps

    num_classes = T.get_num_classes(args.dataset)
    train_ds = T.build_dataset(args.dataset, args.data_root, train=True)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=False)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else len(loader)

    model = T.build_model(args.model, num_classes, args.dataset).cuda()
    if not T.is_transformer(args.model) and not args.no_syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])
    if T.is_transformer(args.model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train(); sampler.set_epoch(0)
    data_iter = iter(loader)
    def get_batch():
        nonlocal data_iter
        try: x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); x, y = next(data_iter)
        return x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    slowmo_on  = T.SlowMoController(model, beta=args.slowmo_beta, enabled=(args.method=='cpdp'))

    def step(step_idx, comm_on):
        policy = step_policy(args.method, K, step_idx % steps_per_epoch + 1,
                             steps_per_epoch)
        ctx = gradient_context(model, policy) if comm_on else model.no_sync()
        x, y = get_batch()
        with ctx:
            out = model(x); loss = criterion(out, y); loss.backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if comm_on:
            reconcile_parameters(slowmo_on, policy)

    # ----- PASS A: profiler ON -> NCCL kernel time only (ignore its wall) -----
    for i in range(args.warmup_iters): step(i, True)
    torch.cuda.synchronize(); dist.barrier()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(measured_steps): step(args.warmup_iters+i, True)
        torch.cuda.synchronize()
    dist.barrier()
    nccl_us=0.0; comm_names={}; all_names={}
    for evt in prof.key_averages():
        dev=getattr(evt,"cuda_time_total",0) or getattr(evt,"device_time_total",0)
        if dev:
            all_names[evt.key]=all_names.get(evt.key,0)+dev
            if is_nccl_device_kernel(evt.key):
                nccl_us+=dev; comm_names[evt.key]=comm_names.get(evt.key,0)+dev

    # ----- PASS B/C: profiler OFF, wall-clock comm ON vs OFF (averaged) -----
    def timed_window(comm_on):
        for i in range(args.warmup_iters): step(i, comm_on)
        torch.cuda.synchronize(); dist.barrier()
        t0=time.perf_counter()
        for i in range(measured_steps): step(args.warmup_iters+i, comm_on)
        torch.cuda.synchronize()
        return time.perf_counter()-t0
    on_list=[timed_window(True)  for _ in range(args.wall_repeats)]; dist.barrier()
    off_list=[timed_window(False) for _ in range(args.wall_repeats)]; dist.barrier()
    wall_on=sum(on_list)/len(on_list); wall_off=sum(off_list)/len(off_list)

    # rank-reduce
    tt=torch.tensor([nccl_us, wall_on, wall_off], device='cuda')
    s=tt.clone(); dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx=tt.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    nccl_s_pergpu=(s[0].item()/world)/1e6
    wall_on_s=mx[1].item(); wall_off_s=mx[2].item()
    exposed_s=max(0.0, wall_on_s-wall_off_s)

    units_per_epoch = math.ceil(steps_per_epoch / K) if K > 1 else steps_per_epoch
    units = sum(step_policy(args.method, K, (args.warmup_iters+i) % steps_per_epoch + 1,
                            steps_per_epoch).boundary for i in range(measured_steps))
    if units == 0:
        raise ValueError('Measured window contains no synchronization boundary')
    nccl_per_epoch=(nccl_s_pergpu/units)*units_per_epoch
    exposed_per_epoch=(exposed_s/units)*units_per_epoch

    if rank==0:
        print(f"\n[{args.scale_tag}] method={args.method} K={K} world={world} seed={args.seed}")
        print(f"  measured_steps={measured_steps} units={units} steps/epoch={steps_per_epoch}")
        print(f"  NCCL kernel (per-GPU mean) over window = {nccl_s_pergpu:.4f}s")
        print(f"  wall_on={wall_on_s:.4f}s wall_off={wall_off_s:.4f}s (profiler OFF, mean of {args.wall_repeats}) exposed={exposed_s:.4f}s")
        print(f"  --> nccl_kernel/epoch={nccl_per_epoch:.3f}s  exposed_comm/epoch={exposed_per_epoch:.3f}s")
        print("  top NCCL device kernels:")
        for k,v in sorted(comm_names.items(),key=lambda kv:-kv[1])[:6]:
            print(f"    {v/1e6:8.4f}s  {k[:70]}")
        if args.dump_kernels:
            print("  ALL device kernels (top 12):")
            for k,v in sorted(all_names.items(),key=lambda kv:-kv[1])[:12]:
                print(f"    {v/1e6:8.4f}s  {k[:70]}")
        os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
        hdr=not os.path.exists(args.outfile)
        with open(args.outfile,'a',newline='') as f:
            w=csv.writer(f)
            if hdr: w.writerow(['scale_tag','method','K','world','seed','measured_steps','units',
                                'steps_per_epoch','nccl_kernel_s_window_pergpu','wall_on_s','wall_off_s',
                                'exposed_s_window','nccl_kernel_per_epoch_s','exposed_comm_per_epoch_s'])
            w.writerow([args.scale_tag,args.method,K,world,args.seed,measured_steps,units,steps_per_epoch,
                        f'{nccl_s_pergpu:.6f}',f'{wall_on_s:.6f}',f'{wall_off_s:.6f}',f'{exposed_s:.6f}',
                        f'{nccl_per_epoch:.6f}',f'{exposed_per_epoch:.6f}'])
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    main()
