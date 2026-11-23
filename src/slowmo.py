import torch
import torch.distributed as dist

class SlowMoController:
    """
    Periodic parameter synchronization.
    enabled=False -> LocalSGD (AllReduce average only)
    enabled=True  -> CPDP (AllReduce average + SlowMo momentum)

    At sync boundary (matches Algorithm 1 in paper):
      1. theta_avg = AllReduce_mean(theta_i)
      2. If CPDP:
         a. delta = theta_avg - theta_ref
         b. v = beta * v + delta
         c. theta_i = theta_ref + v
         d. theta_ref = theta_ref + v
      3. If LocalSGD:
         a. theta_i = theta_avg (already set)
         b. theta_ref = theta_avg

    Does NOT modify optimizer state (momentum buffers, Adam state).
    """

    def __init__(self, model, beta=0.3, enabled=True):
        self.model = model
        self.module = model.module if hasattr(model, 'module') else model
        self.beta = beta
        self.enabled = enabled
        self.velocity = {}
        self.reference = {}
        for name, param in self.module.named_parameters():
            self.reference[name] = param.data.clone()
            self.velocity[name] = torch.zeros_like(param.data)

    @torch.no_grad()
    def synchronize(self):
        world_size = dist.get_world_size()

        # Step 1: AllReduce parameter average (batched into single call)
        params = list(self.module.parameters())
        flat = torch.cat([p.data.reshape(-1) for p in params])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for p in params:
            numel = p.data.numel()
            p.data.copy_(flat[offset:offset+numel].reshape(p.data.shape))
            offset += numel

        if not self.enabled:
            # LocalSGD: parameters already averaged, just update reference
            for name, param in self.module.named_parameters():
                self.reference[name].copy_(param.data)
            return

        # Step 2: SlowMo momentum correction
        for name, param in self.module.named_parameters():
            delta = param.data - self.reference[name]          # theta_avg - theta_ref
            self.velocity[name].mul_(self.beta).add_(delta)    # v = beta*v + delta
            new_val = self.reference[name] + self.velocity[name]  # theta_ref + v
            param.data.copy_(new_val)                           # theta_i = new
            self.reference[name].copy_(new_val)                 # theta_ref = new

    @torch.no_grad()
    def reset_reference(self):
        """Call after warmup to reset reference and velocity."""
        for name, param in self.module.named_parameters():
            self.reference[name].copy_(param.data)
            self.velocity[name].zero_()
