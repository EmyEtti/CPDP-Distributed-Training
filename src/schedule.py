"""Logical synchronization events, not NCCL bucket counts."""
from dataclasses import dataclass

@dataclass(frozen=True)
class StepPolicy:
    boundary: bool
    gradient_sync: bool
    reconcile: bool
    slowmo: bool

def step_policy(method, period, step, steps_per_epoch, in_warmup=False):
    if method not in ('ddp', 'localsgd', 'cpdp'):
        raise ValueError('Unknown method: ' + method)
    if period < 1 or not 1 <= step <= steps_per_epoch:
        raise ValueError('Require K >= 1 and 1 <= step <= steps_per_epoch')
    if method == 'ddp' and period != 1:
        raise ValueError('DDP requires K=1')
    if in_warmup or method == 'ddp':
        return StepPolicy(True, True, False, False)
    boundary = step % period == 0 or step == steps_per_epoch
    return StepPolicy(boundary, method == 'cpdp' and boundary,
                      boundary, method == 'cpdp' and boundary)
