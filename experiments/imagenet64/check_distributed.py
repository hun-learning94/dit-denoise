"""Run with torchrun --nproc_per_node=2: accumulated DDP versus one full batch."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from .train import accumulated_backward


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank = int(os.environ['RANK'])
    world = dist.get_world_size()
    torch.manual_seed(42)
    reference = torch.nn.Linear(4, 2).double()
    actual = torch.nn.Linear(4, 2).double()
    actual.load_state_dict(reference.state_dict())
    ddp = DistributedDataParallel(actual)
    x = torch.randn(world * 6, 4, dtype=torch.float64)
    y = torch.randn(world * 6, 2, dtype=torch.float64)
    loss = lambda model, batch: (model(batch[0]) - batch[1]).square().mean()
    optimizer = torch.optim.AdamW(actual.parameters(), lr=.01, weight_decay=0)
    full_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01, weight_decay=0)
    local = [(x[rank*6+i:rank*6+i+3], y[rank*6+i:rank*6+i+3]) for i in (0, 3)]
    accumulated_backward(ddp, iter(local), loss, accumulation=2)
    loss(reference, (x, y)).backward()
    for a, b in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    optimizer.step(); full_optimizer.step()
    for a, b in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    sampler = DistributedSampler(range(40), num_replicas=world, rank=rank, seed=7, drop_last=True)
    indices = [None] * world
    dist.all_gather_object(indices, list(sampler))
    assert len(set(i for rows in indices for i in rows)) == sum(map(len, indices))
    if rank == 0:
        print('PASS: accumulated DDP gradients and AdamW update equal full-batch update; rank samples disjoint.')
    dist.destroy_process_group()


if __name__ == '__main__': main()
