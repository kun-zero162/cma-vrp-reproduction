"""Dataset / dataloader over precomputed multimodal entity graphs.

Cache layout: `cache/{dataset}/{split}.npz` holds one padded numeric array per key
(no pickled Python objects, so `np.load` is called without `allow_pickle`), and
`cache/{dataset}/{split}.ids.json` holds the sample ids. Padding to the fixed
`max_text_nodes` / `max_visual_nodes` happens at build time, so `torch.stack` in the
collate function can never hit a shape mismatch.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FLOAT_KEYS = ('X_t', 'X_v', 'X_v_vis', 'A_t', 'A_v', 'A_tv', 'VRF_t', 'VRF_v', 'mask_t', 'mask_v')
LONG_KEYS = ('R_t', 'R_v')


class CMAVRPDataset(Dataset):
    def __init__(self, npz_path, ids_path, fp16=True):
        with np.load(npz_path) as data:
            # loaded once, decompressed once; keep storage dtype to halve RAM
            self.arrays = {k: data[k] for k in data.files}
        with open(ids_path, encoding='utf-8') as f:
            self.ids = json.load(f)
        self.fp16 = fp16
        n = len(self.ids)
        for k, v in self.arrays.items():
            if v.shape[0] != n:
                raise ValueError(f'Array {k} has {v.shape[0]} rows but the id list has {n}')
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        out = {}
        for k, v in self.arrays.items():
            t = torch.from_numpy(np.ascontiguousarray(v[idx]))
            if k in FLOAT_KEYS:
                out[k] = t.float()
            else:
                out[k] = t.long()
        out['y'] = torch.as_tensor(int(out['y'])) if 'y' in out else None
        out['id'] = self.ids[idx]
        return out


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        if k == 'id':
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def get_dataloader(config, split, shuffle=None, drop_last=False):
    npz_path = os.path.join(config.cache_path, f'{split}.npz')
    ids_path = os.path.join(config.cache_path, f'{split}.ids.json')
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f'Graph cache not found: {npz_path}\n'
            f'Build it first: python preprocess/build_dataset.py --datasets {config.datasets}')
    ds = CMAVRPDataset(npz_path, ids_path)
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=(drop_last and split == 'train'),
        collate_fn=collate_fn,
        pin_memory=False,
    )
