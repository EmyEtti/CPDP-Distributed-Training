"""Optional diagnostic: ONE additional parameter AllReduce per invocation.
Never included in logical gradient or reconciliation counts.
"""
import torch
import torch.distributed as dist
@torch.no_grad()
def compute_param_drift(model):
    """L2 distance between local params and global average.
    MUST be called BEFORE parameter reconciliation."""
    world_size = dist.get_world_size()
    local_flat = torch.cat([p.data.reshape(-1)
                            for p in model.module.parameters()])
    avg_flat = local_flat.clone()
    dist.all_reduce(avg_flat, op=dist.ReduceOp.SUM)
    avg_flat.div_(world_size)
    return torch.norm(local_flat - avg_flat).item()
