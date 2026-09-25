"""Read official channel-first ImageNet64 arrays without loading the bank into RAM."""
import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageNet64(Dataset):
    def __init__(self, root, split='train'):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.shards = self.manifest['splits'][split]
        self.ends = np.cumsum([s['count'] for s in self.shards]).tolist()
        self.arrays = {}
        if not self.ends:
            raise ValueError('No prepared shards')

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        if shard not in self.arrays:
            self.arrays[shard] = np.load(self.root / self.shards[shard]['path'], mmap_mode='r', allow_pickle=False)
        offset = index - (self.ends[shard - 1] if shard else 0)
        # Official flat ordering is R pixels, G pixels, B pixels.
        return torch.from_numpy(np.array(self.arrays[shard][offset], copy=True).reshape(3, 64, 64))

    def __getstate__(self):
        state = dict(self.__dict__)
        state['arrays'] = {}
        return state
