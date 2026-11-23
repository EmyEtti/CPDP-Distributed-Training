# Grid'5000 reproduction

Reserve resources through OAR yourself; no script submits jobs, kills processes,
or reserves nodes. Do not run training on a frontend. Confirm current availability
and access rules with Grid'5000; cluster descriptions below are paper-era.

Nancy graffiti uses RTX 2080 Ti GPUs and 10 Gb Ethernet; Sophia esterel provides
the other four RTX 2080 Ti GPUs over 1 Gb Ethernet in the WAN experiment. Use
4 GPUs per site (two nodes total), global batch 1024, per-GPU batch 128. The
paper measured approximately 16.6 ms inter-site RTT; it is not a guaranteed value.
Intra-site scaling: g4=1x4, g6=2x3, g8=2x4, g12=3x4, g16=4x4 GPUs.
Ensure the selected nodes match those resources; never infer GPU count from the
number of repeated lines in OAR_NODEFILE. g6 deliberately uses three GPUs/node.

Example reservation request for one Nancy node (adapt walltime/site):
`oarsub -I -l nodes=1,walltime=02:00:00 -p "cluster='graffiti'"`.
Reserve Sophia independently for WAN. Validate node GPU models with `nvidia-smi`.
Install the same environment and checkout on every node; data may be local but
must have identical contents. The two sites may have different home filesystems.

Launch one command per reserved node, concurrently, with the same seed/config:

```bash
export MASTER_ADDR=<reachable-address-of-rank-zero>
export MASTER_PORT=29500
export NODE_RANK=0  # 1 on the other node
export DATA_ROOT=/path/to/data
export NCCL_SOCKET_IFNAME=br0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
bash scripts/reproduce_table5_wan.sh --experiment cpdp_K4 --seed 42 --execute
```

Use an unused port; `MASTER_ADDR` must resolve/reach rank zero from all nodes.
`br0` is the socket interface used in the reported configuration; verify the interface on your allocation.
The wrapper emits the complete torchrun command and automatically applies those
WAN transport defaults only for cross-site configs. It does not SSH into nodes;
launch in separate OAR/SSH sessions with matching arguments. Intra-site wrappers
do not add WAN-only flags. The WAN config disables SyncBatchNorm as in the
cross-site configuration; intra-site ResNet enables it.

Datasets: CIFAR-100 (50,000 training / 10,000 test); TinyImageNet (200 classes,
100,000 training images) with validation organized into class directories.
ViT-S uses CIFAR-100. Storage needs vary with extraction; obtain datasets before
reserving long GPU runs. Transforms/architectures remain in src/train_scaling.py.

Every table wrapper documents its output root and points to explicit JSON resource
requirements. Profile wrappers run short GPU timing windows only when --execute
is explicitly requested. They do not generate full training accuracy results.
Seeds are 42/123/777 except the K sweep (42). Main beta is 0.3; the explicit
beta sweep and g16 diagnostic have their own values. Save environment_report.py
output with each run; do not assume installed packages equal the paper environment.
