"""Shared method-specific context used by training, profiling, and tests."""
from contextlib import nullcontext

def gradient_context(model, policy):
    # no_sync encloses BOTH forward and backward. LocalSGD never reduces
    # gradients after warmup, including parameter-averaging boundaries.
    return nullcontext() if policy.gradient_sync else model.no_sync()

def reconcile_parameters(controller, policy):
    if policy.reconcile:
        if controller.enabled != policy.slowmo:
            raise ValueError('Controller and method disagree about SlowMo')
        controller.synchronize()
