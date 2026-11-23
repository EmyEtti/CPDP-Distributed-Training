import copy
import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from schedule import step_policy
from launch import validate, commands


class ScheduleTests(unittest.TestCase):
    def test_counts_and_partial_intervals(self):
        for total in [1, 5, 48, 97]:
            for method in ['ddp', 'localsgd', 'cpdp']:
                for k in ([1] if method == 'ddp' else [1, 2, 4, 64]):
                    plans = [step_policy(method, k, t, total) for t in range(1, total + 1)]
                    expected = math.ceil(total / k)
                    self.assertEqual(sum(p.gradient_sync for p in plans),
                                     total if method == 'ddp' else expected if method == 'cpdp' else 0)
                    self.assertEqual(sum(p.reconcile for p in plans), 0 if method == 'ddp' else expected)
                    self.assertEqual(sum(p.slowmo for p in plans), expected if method == 'cpdp' else 0)
                    self.assertTrue(plans[-1].boundary)

    def test_warmup(self):
        for method in ['ddp', 'localsgd', 'cpdp']:
            for t in range(1, 6):
                p = step_policy(method, 1 if method == 'ddp' else 4, t, 5, True)
                self.assertTrue(p.gradient_sync)
                self.assertFalse(p.reconcile or p.slowmo)

    def test_bad_inputs(self):
        for args in [('unknown', 1, 1, 5), ('ddp', 2, 1, 5), ('cpdp', 0, 1, 5), ('cpdp', 2, 0, 5)]:
            with self.assertRaises(ValueError): step_policy(*args)

    def test_all_config_commands(self):
        for path in ROOT.glob('configs/**/*.json'):
            cfg = json.loads(path.read_text())
            validate(cfg)
            for cmd, e in commands(cfg, '/data', '/outputs'):
                entry = next(x for x in cmd if x.endswith('.py'))
                self.assertTrue(Path(entry).is_file())
                self.assertNotIn('--enable-drift-diagnostics', cmd)
                self.assertIn(str(e['training']['slowmo_beta']), cmd)

    def test_imports(self):
        for name in ['slowmo', 'diagnostics', 'synchronization', 'train_scaling', 'profile_comm']:
            importlib.import_module(name)


def distributed_worker(rank, init_method):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from slowmo import SlowMoController
    from diagnostics import compute_param_drift
    from synchronization import gradient_context, reconcile_parameters
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=init_method, rank=rank, world_size=2)
    try:
        for optimizer_name in ['sgd', 'adamw']:
            for method, k in [('ddp', 1), ('localsgd', 1), ('localsgd', 4), ('cpdp', 1), ('cpdp', 4)]:
                torch.manual_seed(7)
                model = DDP(torch.nn.Linear(2, 1, bias=False))
                opt = (torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0005)
                       if optimizer_name == 'sgd' else torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.05))
                hook_calls = [0]
                def hook(state, bucket):
                    hook_calls[0] += 1
                    tensor = bucket.buffer()
                    dist.all_reduce(tensor)
                    tensor.div_(2)
                    future = torch.futures.Future()
                    future.set_result(tensor)
                    return future
                model.register_comm_hook(None, hook)
                controller = SlowMoController(model, beta=.3, enabled=method == 'cpdp')
                reconciliation_count = 0
                for warmup in [True, False]:
                    before_hooks = hook_calls[0]
                    total = 5  # includes final partial K=4 interval
                    for t in range(1, total + 1):
                        plan = step_policy(method, k, t, total, warmup)
                        opt.zero_grad(set_to_none=True)
                        with gradient_context(model, plan):
                            x = torch.tensor([[float(rank + 1), float(t)]])
                            loss = model(x).square().sum()
                            loss.backward()
                        opt.step()
                        states_before = copy.deepcopy(opt.state_dict())
                        if plan.reconcile:
                            average = model.module.weight.detach().clone()
                            dist.all_reduce(average); average.div_(2)
                            reference = controller.reference['weight'].clone()
                            velocity = controller.velocity['weight'].clone()
                            expected = reference + .3 * velocity + average - reference if plan.slowmo else average
                            reconcile_parameters(controller, plan)
                            torch.testing.assert_close(model.module.weight, expected)
                            reconciliation_count += 1
                            # No momentum buffer / Adam moment / counter is changed.
                            for index, state in states_before['state'].items():
                                for key, value in state.items():
                                    torch.testing.assert_close(value, opt.state_dict()['state'][index][key])
                            gathered = [torch.zeros_like(model.module.weight) for _ in range(2)]
                            dist.all_gather(gathered, model.module.weight.detach())
                            torch.testing.assert_close(gathered[0], gathered[1])
                        else:
                            reconcile_parameters(controller, plan)
                    expected_grad = total if warmup or method == 'ddp' else math.ceil(total/k) if method == 'cpdp' else 0
                    assert hook_calls[0] - before_hooks == expected_grad, (method, warmup, hook_calls, expected_grad)
                    if warmup:
                        assert reconciliation_count == 0
                        controller.reset_reference()
                        assert not controller.velocity['weight'].count_nonzero()
                        torch.testing.assert_close(controller.reference['weight'], model.module.weight)
                assert reconciliation_count == (0 if method == 'ddp' else math.ceil(total/k))
                weights = model.module.weight.detach().clone()
                drift = compute_param_drift(model)
                assert drift >= 0
                torch.testing.assert_close(weights, model.module.weight)
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_actual_gradient_hooks_and_optimizer_state(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            init = (Path(directory) / 'rendezvous').as_uri()
            mp.spawn(distributed_worker, args=(init,), nprocs=2, join=True)


if __name__ == '__main__': unittest.main()
