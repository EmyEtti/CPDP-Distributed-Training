#!/usr/bin/env python3
"""
CPDP Scaling Experiments - Unified Training Script
Supports: DDP (K=1), LocalSGD (periodic avg, no SlowMo), CPDP (periodic avg + SlowMo)
Models: ResNet-50, ResNet-18, ViT-S
Implementation used for the FGCS experimental configurations.
"""

import os
import csv
import time
import argparse
import random
import json
from slowmo import SlowMoController
from diagnostics import compute_param_drift
from schedule import step_policy
from synchronization import gradient_context, reconcile_parameters

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


###############################################################################
# 1. ARGUMENTS
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description='CPDP Scaling Experiments')
    p.add_argument('--model', type=str, default='resnet50',
                   choices=['resnet50', 'resnet18', 'vit_s'])
    p.add_argument('--dataset', type=str, default='cifar100',
                   choices=['cifar100', 'tinyimagenet', 'cifar10'])
    p.add_argument('--method', type=str, default='ddp',
                   choices=['ddp', 'localsgd', 'cpdp'])
    p.add_argument('--sync-period', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--global-batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--slowmo-beta', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', type=str, default='results/default')
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--backend', type=str, default='nccl', choices=['nccl','gloo'])
    p.add_argument('--no-syncbn', action='store_true', default=False)
    p.add_argument('--enable-drift-diagnostics', action='store_true',
                   help='Extra diagnostic AllReduce at boundaries; OFF by default')
    args = p.parse_args()
    if args.sync_period < 1 or (args.method == 'ddp' and args.sync_period != 1):
        p.error('K must be positive; DDP requires K=1')
    if not 0 <= args.slowmo_beta < 1:
        p.error('SlowMo beta must be in [0, 1)')
    if args.epochs < 1 or not 0 <= args.warmup_epochs <= args.epochs:
        p.error('Require 0 <= warmup epochs <= epochs and epochs >= 1')
    if args.workers < 0 or args.batch_size < 1 or args.lr <= 0:
        p.error('Invalid workers, batch size or learning rate')
    return args


###############################################################################
# 2. REPRODUCIBILITY
###############################################################################

def set_seed(seed, rank):
    s = seed + rank
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################################
# 3. DATASETS
###############################################################################

def build_cifar100(root, train):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    return torchvision.datasets.CIFAR100(root, train=train, download=False,
                                          transform=transform)


def build_cifar10(root, train):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    return torchvision.datasets.CIFAR10(root, train=train, download=False,
                                         transform=transform)


def build_tinyimagenet(root, train):
    split = 'train' if train else 'val'
    path = os.path.join(root, 'tiny-imagenet-200', split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Tiny-ImageNet not found at {path}")
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
    return torchvision.datasets.ImageFolder(path, transform=transform)


def build_dataset(name, root, train):
    if name == 'cifar100':
        return build_cifar100(root, train)
    elif name == 'cifar10':
        return build_cifar10(root, train)
    elif name == 'tinyimagenet':
        return build_tinyimagenet(root, train)
    raise ValueError(f"Unknown dataset: {name}")


def get_num_classes(name):
    return {'cifar100': 100, 'tinyimagenet': 200, 'cifar10': 10}[name]


def get_dataset_size(name):
    return {'cifar100': 50000, 'tinyimagenet': 100000, 'cifar10': 50000}[name]


###############################################################################
# 4. MODELS
###############################################################################

class ViTSmall(nn.Module):
    """
    Small Vision Transformer for CIFAR-100 (32x32 images).
    Patch size 4 -> 8x8 = 64 patches.
    Embed dim 384, 6 heads, 12 layers (DeiT-Small scale).
    ~22M parameters.
    """
    def __init__(self, num_classes=100, img_size=32, patch_size=4,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.1):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size,
                                      stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate, activation='gelu',
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                              num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_model(name, num_classes, dataset):
    if name == 'resnet50':
        model = models.resnet50(weights=None, num_classes=num_classes)
    elif name == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif name == 'vit_s':
        img_size = 32 if dataset == 'cifar100' else 64
        return ViTSmall(num_classes=num_classes, img_size=img_size)
    else:
        raise ValueError(f"Unknown model: {name}")

    if dataset in ('cifar100', 'tinyimagenet', 'cifar10'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def is_transformer(model_name):
    return model_name in ('vit_s',)


###############################################################################
# 5. SLOWMO CONTROLLER
###############################################################################

###############################################################################
# 6. COMMUNICATION PROFILER
###############################################################################

class CommProfiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grad_sync_count = 0
        self.param_sync_count = 0
        self.comm_time_param = 0.0
        self._t = None
        self.diagnostic_collectives = 0
        self.diagnostic_seconds = 0.0

    def start(self):
        torch.cuda.synchronize()
        self._t = time.perf_counter()

    def stop_param(self):
        torch.cuda.synchronize()
        self.comm_time_param += time.perf_counter() - self._t
        self.param_sync_count += 1


###############################################################################
# 7. PARAMETER DRIFT
###############################################################################

###############################################################################
# 8. LEARNING RATE SCHEDULE
###############################################################################

def get_lr(epoch, args):
    """Cosine annealing with linear warmup."""
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(
        1, args.epochs - args.warmup_epochs)
    return args.lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


###############################################################################
# 9. TRAINING - Algorithm 1 (Paper-Verified)
###############################################################################

def train_one_epoch(model, loader, optimizer, criterion, epoch, args,
                    slowmo, prof):
    """Method-specific gradient and parameter events with optional diagnostics."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_steps = len(loader)
    in_warmup = epoch < args.warmup_epochs
    time_fwd = 0.0
    time_bwd = 0.0
    max_drift = 0.0

    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        policy = step_policy(args.method, args.sync_period, step + 1,
                             num_steps, in_warmup)
        ctx = gradient_context(model, policy)

        # Forward + Backward in single context block
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with ctx:
            output = model(inputs)
            loss = criterion(output, targets)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            time_fwd += t1 - t0

            loss.backward()

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        time_bwd += t2 - t1

        if policy.gradient_sync:
            prof.grad_sync_count += 1

        # Optimizer step EVERY iteration (local or sync)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if policy.reconcile:
            if args.enable_drift_diagnostics:
                torch.cuda.synchronize()
                diagnostic_start = time.perf_counter()
                drift_now = compute_param_drift(model)
                torch.cuda.synchronize()
                prof.diagnostic_seconds += time.perf_counter() - diagnostic_start
                prof.diagnostic_collectives += 1
                max_drift = max(max_drift, drift_now)
            prof.start()
            reconcile_parameters(slowmo, policy)
            prof.stop_param()

        # Accumulate metrics
        total_loss += loss.item() * inputs.size(0)
        _, pred = output.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    # Reset SlowMo reference at end of warmup
    if epoch == args.warmup_epochs - 1 and args.method != 'ddp':
        slowmo.reset_reference()

    return (total_loss / total, 100.0 * correct / total,
            time_fwd, time_bwd, max_drift)


###############################################################################
# 10. EVALUATION
###############################################################################

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in loader:
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        output = model(inputs)
        loss = criterion(output, targets)
        total_loss += loss.item() * inputs.size(0)
        _, pred = output.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    stats = torch.tensor([correct, total, total_loss], device='cuda',
                         dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (100.0 * stats[0].item() / stats[1].item(),
            stats[2].item() / stats[1].item())


###############################################################################
# 11. MAIN
###############################################################################

def main():
    args = parse_args()

    dist.init_process_group(backend=args.backend if hasattr(args, "backend") else "nccl")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    if args.global_batch != args.batch_size * world_size:
        raise ValueError('--global-batch must equal --batch-size * WORLD_SIZE')

    set_seed(args.seed, rank)

    if rank == 0:
        print("=" * 70)
        print(f"  Method:     {args.method} (K={args.sync_period})")
        print(f"  Model:      {args.model}")
        print(f"  Dataset:    {args.dataset}")
        print(f"  GPUs:       {world_size}")
        print(f"  Batch/GPU:  {args.batch_size}")
        print(f"  Global BS:  {args.batch_size * world_size}")
        print(f"  LR:         {args.lr}")
        print(f"  Epochs:     {args.epochs}")
        print(f"  Warmup:     {args.warmup_epochs}")
        print(f"  SlowMo b:   {args.slowmo_beta}")
        print(f"  Seed:       {args.seed}")
        print("=" * 70)

    num_classes = get_num_classes(args.dataset)
    train_ds = build_dataset(args.dataset, args.data_root, train=True)
    val_ds = build_dataset(args.dataset, args.data_root, train=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True,
                                        seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    dataset_size = get_dataset_size(args.dataset)

    model = build_model(args.model, num_classes, args.dataset).cuda()

    # SyncBatchNorm only for models with BatchNorm (not ViT)
    if not is_transformer(args.model) and not args.no_syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print(f"  Parameters:  {count_parameters(model):,}")
        print(f"  Steps/epoch: {len(train_loader)}")

    # Optimizer: SGD for ResNet, AdamW for ViT
    if is_transformer(args.model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=0.05)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                     momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    use_slowmo = (args.method == 'cpdp')
    slowmo = SlowMoController(model, beta=args.slowmo_beta,
                               enabled=use_slowmo)
    prof = CommProfiler()

    csv_path = os.path.join(args.outdir, 'metrics.csv')
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        if os.path.exists(csv_path):
            raise FileExistsError('Refusing to overwrite existing metrics.csv')
        with open(os.path.join(args.outdir, 'run_info.json'), 'w') as info:
            json.dump(dict(vars(args), world_size=world_size,
                           torch_version=torch.__version__, cuda=torch.version.cuda,
                           nccl=torch.cuda.nccl.version(),
                           implementation='CPDP FGCS release'),
                      info, indent=2)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'epoch', 'train_loss', 'train_acc', 'test_acc', 'test_loss',
                'epoch_time', 'time_fwd', 'time_bwd', 'time_comm_param',
                'lr', 'gradient_sync_events', 'parameter_reconciliation_events',
                'param_drift', 'throughput',
                'method', 'K', 'world_size', 'global_batch', 'seed',
                'model', 'diagnostic_collectives', 'diagnostic_seconds',
            ])

    best_acc = 0.0
    cumulative_time = 0.0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        lr = get_lr(epoch, args)
        set_lr(optimizer, lr)
        prof.reset()

        torch.cuda.synchronize()
        dist.barrier()
        t_start = time.perf_counter()

        train_loss, train_acc, t_fwd, t_bwd, drift = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, args,
            slowmo, prof)

        torch.cuda.synchronize()
        dist.barrier()
        t_end = time.perf_counter()
        epoch_time = t_end - t_start
        cumulative_time += epoch_time

        test_acc, test_loss = evaluate(model, val_loader)
        best_acc = max(best_acc, test_acc)
        throughput = dataset_size / epoch_time

        if rank == 0:
            print(f"Ep {epoch:3d}/{args.epochs} | "
                  f"LR {lr:.5f} | "
                  f"TrL {train_loss:.4f} TrA {train_acc:.1f}% | "
                  f"TeA {test_acc:.2f}% (best {best_acc:.2f}%) | "
                  f"T {epoch_time:.1f}s F {t_fwd:.1f}s B {t_bwd:.1f}s "
                  f"C {prof.comm_time_param:.1f}s | "
                  f"GS {prof.grad_sync_count} PS {prof.param_sync_count} | "
                  f"Drift {drift:.4f} | {throughput:.0f} img/s")

            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    epoch,
                    f'{train_loss:.6f}', f'{train_acc:.4f}',
                    f'{test_acc:.4f}', f'{test_loss:.6f}',
                    f'{epoch_time:.4f}', f'{t_fwd:.4f}', f'{t_bwd:.4f}',
                    f'{prof.comm_time_param:.4f}', f'{lr:.8f}',
                    prof.grad_sync_count, prof.param_sync_count,
                    f'{drift:.6f}', f'{throughput:.2f}',
                    args.method, args.sync_period, world_size,
                    args.batch_size * world_size, args.seed,
                    args.model, prof.diagnostic_collectives, prof.diagnostic_seconds,
                ])

    if rank == 0:
        print("=" * 70)
        print(f"DONE | Best: {best_acc:.2f}% | Time: {cumulative_time:.1f}s")
        print(f"Saved: {csv_path}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
