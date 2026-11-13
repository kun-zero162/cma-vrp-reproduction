"""Training entry point for the CMA-VRP reproduction.

Mirrors EASE's Expert/train.py so the two projects share conventions, but resolves the
device automatically (the paper's setup assumes a GPU; a 4 GB card only fits small graphs).
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

from configs.config_setting import setting_config as config
from ease_bridge import (Recorder, cal_params_flops, get_logger, get_optimizer,
                         get_scheduler, log_config_info, set_seed)
import engine
from datasets.graph_dataset import get_dataloader
from models.CMAVRP import CMAVRP

ALL_ABLATIONS = ('use_vrf', 'use_gcl_inter', 'use_gcl_intra', 'use_cmgaf', 'use_accil', 'use_vsca')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default=config.datasets, choices=['twitter', 'pheme', 'weibo'])
    p.add_argument('--batch_size', type=int, default=config.batch_size)
    p.add_argument('--epochs', type=int, default=config.epochs)
    p.add_argument('--early_stop', type=int, default=config.early_stop)
    p.add_argument('--lr', type=float, default=config.lr)
    p.add_argument('--seed', type=int, default=config.seed)
    p.add_argument('--gpu', default=None, help='CUDA device id, e.g. "0". Omit / --cpu for CPU.')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--num_workers', type=int, default=config.num_workers)
    p.add_argument('--vrp_backend', default=config.vrp_backend, choices=['mock', 'hf', 'agent'])
    p.add_argument('--ablate', nargs='*', default=[], choices=list(ALL_ABLATIONS),
                   help='Switches to DISABLE (Table 2 ablations)')
    p.add_argument('--tag', default='', help='suffix for the log / checkpoint directory')
    return p.parse_args()


def main():
    args = parse_args()
    config.datasets = args.datasets
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.early_stop = args.early_stop
    config.lr = args.lr
    config.seed = args.seed
    config.num_workers = args.num_workers
    config.vrp_backend = args.vrp_backend
    for k in args.ablate:
        config.ablation[k] = False

    if args.cpu or not torch.cuda.is_available():
        use_cuda, device = False, torch.device('cpu')
    else:
        use_cuda, device = True, torch.device(f'cuda:{args.gpu or 0}')
    config.use_cuda = use_cuda
    config.device_ = device

    root = os.path.dirname(_HERE)
    name = f'{config.network}-{config.datasets}-{config.vrp_backend}{("-" + args.tag) if args.tag else ""}'
    config.works_num = name
    log_dir = os.path.join(root, 'CMAVRP', 'logs', name)
    ckpt_dir = os.path.join(root, 'CMAVRP', 'ckpt', name)
    config.ckpt_path = os.path.join(ckpt_dir, 'state_dict.pt')
    config.cache_path = os.path.join(_HERE, 'cache', config.datasets)
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.isdir(log_dir):
        # EASE's get_logger() calls os.makedirs() without exist_ok
        shutil.rmtree(log_dir)

    logger = get_logger(name, log_dir)
    # EASE's get_logger() only writes to a file; echo to the console so a long CPU run
    # is observable without opening the log.
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, '_echo', False)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S'))
        sh._echo = True
        logger.addHandler(sh)
    logger.info(f'#----------Start to train {config.network} on {config.datasets}----------#')
    log_config_info(config, logger)
    if args.ablate:
        logger.info('Ablation (disabled): %s', ', '.join(args.ablate))

    set_seed(config.seed)
    if torch.cuda.is_available():
        logger.info('GPU available: %s | VRAM %.1f GB',
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    else:
        logger.info('No CUDA device -> CPU training (smoke-test scale only)')

    train_loader = get_dataloader(config, 'train', shuffle=True, drop_last=False)
    val_loader = get_dataloader(config, 'val', shuffle=False)
    logger.info('Train / Val batches: %d / %d', len(train_loader), len(val_loader))

    model = CMAVRP(config).to(device)
    cal_params_flops(model, config, logger)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    recorder = Recorder(config.early_stop)

    writer = engine.make_writer(log_dir)

    start = time.time()
    for epoch in range(config.epochs):
        step = epoch * len(train_loader) + 1
        step = engine.train_one_epoch(train_loader, model, optimizer, scheduler,
                                      epoch, step, logger, config, writer)
        results, _ = engine.val_one_epoch(val_loader, model, logger, config, epoch, writer)
        flag = recorder.add(results)
        if flag == 'save':
            torch.save(model.state_dict(), config.ckpt_path)
            logger.info('Saved checkpoint -> %s', config.ckpt_path)
            with open(os.path.join(ckpt_dir, 'history.json'), 'w', encoding='utf-8') as f:
                json.dump({'best': recorder.max, 'maxindex': recorder.maxindex,
                           'ablation': config.ablation}, f, indent=2)
        elif flag == 'esc':
            logger.info('Early stopping after %d epochs (no mac_f1 gain for %d epochs)',
                        epoch + 1, config.early_stop)
            break
    logger.info('Finished in %.1f s', time.time() - start)
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
