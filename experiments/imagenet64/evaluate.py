"""Reuse established evaluation code with the held-out ImageNet64 memory map."""
import importlib
import sys
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from .data import ImageNet64

class NormalizedValidation(Dataset):
    def __init__(self, root): self.data = ImageNet64(root, 'validation')
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index].float().div_(127.5).sub_(1.), 0

def make_loader(args, max_images):
    data = NormalizedValidation(args.eval_data_root)
    if max_images: data = Subset(data, range(min(max_images, len(data))))
    return DataLoader(data, batch_size=args.eval_batch, num_workers=args.num_workers,
                      shuffle=False, pin_memory=torch.cuda.is_available())

def main():
    name = sys.argv.pop(1)
    if name not in ('fid', 'vp', 'denoising'): raise ValueError(name)
    import eval_model_comparison
    eval_model_comparison.make_loader = make_loader
    module = importlib.import_module(dict(fid='eval_fid50k_resumable', vp='eval_vp_velocity_loss',
                                         denoising='eval_denoising_paired')[name])
    module.make_loader = make_loader
    module.main()

if __name__ == '__main__': main()
