"""Test / evaluate a trained CMA-VRP checkpoint (Table 1 protocol)."""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import numpy as np
import torch

from configs.config_setting import setting_config as config
from datasets.graph_dataset import get_dataloader
from ease_bridge import get_logger, save_results, set_seed
from engine import test_one_epoch
from models.CMAVRP import CMAVRP

ALL_ABLATIONS = ('use_vrf', 'use_gcl_inter', 'use_gcl_intra', 'use_cmgaf', 'use_accil', 'use_vsca')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default=config.datasets, choices=['twitter', 'pheme', 'weibo'])
    p.add_argument('--vrp_backend', default=config.vrp_backend, choices=['mock', 'hf', 'agent'])
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--batch_size', type=int, default=config.batch_size)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--ablate', nargs='*', default=[], choices=list(ALL_ABLATIONS))
    p.add_argument('--tag', default='')
    p.add_argument('--seed', type=int, default=config.seed)
    return p.parse_args()


def main():
    args = parse_args()
    config.datasets = args.datasets
    config.batch_size = args.batch_size
    config.vrp_backend = args.vrp_backend
    for k in args.ablate:
        config.ablation[k] = False

    device = torch.device('cpu') if (args.cpu or not torch.cuda.is_available()) \
        else torch.device('cuda:0')
    config.device_ = device
    config.use_cuda = device.type == 'cuda'
    config.cache_path = os.path.join(_HERE, 'cache', config.datasets)

    name = f'{config.network}-{config.datasets}-{config.vrp_backend}{("-" + args.tag) if args.tag else ""}'
    ckpt_dir = os.path.join(_HERE, 'ckpt', name)
    ckpt = os.path.join(ckpt_dir, 'state_dict.pt')
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f'No checkpoint at {ckpt}. Run train.py first.')
    out_dir = os.path.join(_HERE, 'results', name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(os.path.join(_HERE, 'logs', name)):
        import shutil
        shutil.rmtree(os.path.join(_HERE, 'logs', name))

    logger = get_logger('test', os.path.join(_HERE, 'logs', name))
    set_seed(config.seed)

    loader = get_dataloader(config, args.split, shuffle=False)
    model = CMAVRP(config).to(device)
    # weights_only: a checkpoint is data, not code (and this refuses object payloads)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning('Missing keys: %s', missing)
    if unexpected:
        logger.warning('Unexpected keys: %s', unexpected)

    results, labels, preds, ids, (feats, flabels) = test_one_epoch(
        loader, model, logger, config, collect_features=True)
    save_results(results, labels, preds, ids, os.path.join(out_dir, f'{args.split}.json'))
    if feats is not None:
        np.save(os.path.join(out_dir, f'{args.split}_features.npy'), feats)
        np.save(os.path.join(out_dir, f'{args.split}_feature_labels.npy'), flabels)
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'ablation': config.ablation, 'split': args.split,
                   'n': int(len(labels)), **results}, f, indent=2)
    logger.info('Saved to %s', out_dir)
    print(json.dumps({k: round(float(v), 4) for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
