# Reproducibility

This repository contains the implementation and experiment configurations
associated with the FGCS paper:

**Controlled Periodic Synchronization for Efficient Data-Parallel Training**

## Implementation

The repository implements the three synchronization strategies evaluated in
the paper:

- DistributedDataParallel (DDP)
- LocalSGD
- Controlled Periodic Data Parallelism (CPDP)

For CPDP, synchronization boundaries occur when `t mod K = 0` or at the final
mini-batch of an epoch. At these boundaries, CPDP performs gradient
synchronization followed by parameter reconciliation and SlowMo correction.

LocalSGD performs local updates between synchronization boundaries and
periodically averages model parameters without gradient synchronization at the
averaging step.

During warmup, CPDP and LocalSGD use synchronous per-step gradient updates,
while periodic parameter reconciliation and SlowMo are disabled.

Optimizer internal states, including SGD momentum buffers and AdamW moments,
remain local and are not averaged during parameter reconciliation.

## Experimental configurations

Configuration files under `configs/` define the experimental protocols reported
in the paper, including:

- cross-site WAN experiments between Nancy and Sophia;
- intra-site scaling experiments;
- synchronization-period ablation;
- SlowMo coefficient sensitivity;
- ViT-S/CIFAR-100 experiments;
- ResNet-50/TinyImageNet experiments.

The experiments use the random seeds reported in the corresponding
configuration files.

## Software environment

The software environment reported in the paper is:

- Python 3.9
- PyTorch 2.4
- CUDA 11.8
- NCCL 2.18

Runtime versions can be recorded using:

    python scripts/environment_report.py

## Results

The CSV files under `results/paper_summary/` contain the experimental results
reported in the paper and are used by the plotting utilities.

New experimental runs are stored separately under `results/new_runs/` so that
the reported paper results are not overwritten.

## Reproduction

Experiment configurations can be launched using the scripts under `scripts/`.

For example:

    bash scripts/reproduce_table5_wan.sh
    bash scripts/reproduce_table10_scaling.sh --experiment g4_ddp --seed 42

The scripts print the corresponding distributed-training command by default.
Execution on allocated compute resources can be enabled using the options
described in the repository README and Grid'5000 reproduction guide.
