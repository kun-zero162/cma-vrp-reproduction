"""Sec.4.3 robustness study (Fig.4) and the t-SNE visualisation (Fig.5).

Perturbations are applied to the *cached graph tensors*, i.e. to what the model actually
sees after preprocessing:
  noise  : Gaussian noise added to the entity node features (intensity 0 -> 1e8, Fig.4a-d)
  mask   : a random fraction of visual entity nodes is removed (0% -> 50%, Fig.4e)
  wd     : Wasserstein distance between the aligned textual / visual node clouds, used as
           the cross-modal drift diagnostic (Fig.4f)
  tsne   : 2-D projection of x_hat coloured by ground truth (Fig.5)

DEVIATION: occluding the raw pixels instead would require re-running DETR + VRP for every
setting; the node-level corruption below is the same test at the representation level and
is what the published curves track (accuracy under increasing perturbation).
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

from configs.config_setting import setting_config as config
from datasets.graph_dataset import get_dataloader
from ease_bridge import calculate_metrics, get_logger, set_seed
from engine import _to_device
from models.CMAVRP import CMAVRP
from models.cycle_align import sinkhorn_log

NOISE_LEVELS = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
MASK_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def inject_noise(batch, sigma):
    if sigma <= 0:
        return batch
    out = dict(batch)
    for k in ('X_t', 'X_v', 'X_v_vis', 'VRF_t', 'VRF_v'):
        if k in out:
            out[k] = out[k] + sigma * torch.randn_like(out[k])
    return out


def occlude(batch, rate):
    """Drops visual entity nodes (mask_v -> 0) at the given rate."""
    if rate <= 0:
        return batch
    out = dict(batch)
    # work in float: the cached mask may arrive as bool, and `1 - bool_tensor` is
    # not supported, while the model only ever calls mask.to(dtype) on it
    mv = out['mask_v'].clone().to(torch.float32)
    drop = (torch.rand(mv.shape, device=mv.device) < rate).to(mv.dtype)
    mv = mv * (1 - drop)
    empty = (mv.sum(1) == 0) & (out['mask_v'].sum(1) > 0)   # only rows that had nodes
    if empty.any():                             # never leave a sample without nodes
        first = (out['mask_v'] > 0).long().argmax(dim=1)
        rows = torch.arange(mv.shape[0], device=mv.device)[empty]
        mv[rows, first[empty]] = 1
    out['mask_v'] = mv
    return out


@torch.no_grad()
def evaluate(loader, model, config, mode, value):
    model.eval()
    preds, labels = [], []
    wds = []
    for batch in loader:
        batch = _to_device(batch, config.device_)
        if mode == 'noise':
            batch = inject_noise(batch, value)
        elif mode == 'mask':
            batch = occlude(batch, value)
        res = model(batch)
        preds.append(res['classify_pred'].cpu().numpy())
        labels.append(batch['y'].cpu().numpy())
        # Fig.6 tracks the Wasserstein distance under every perturbation, not just a
        # dedicated sweep, so measure it for each condition
        wds.append(float(_wd(batch, model)))
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    r = calculate_metrics(labels, preds, config.threshold)
    r['wd'] = float(np.mean(wds)) if wds else float('nan')
    return r


@torch.no_grad()
def _wd(batch, model):
    """Sinkhorn-Wasserstein distance between the two aligned node clouds (Eq.8)."""
    enc = model._encode(batch)
    (H_t0, mask_t, Z_t, H_v0, mask_v, Z_v,
     A_t, A_v, A_tv, VRF_t, VRF_v) = enc
    H_t, H_v = model.fusion(H_t0, A_t, Z_t, mask_t, H_v0, A_v, Z_v, mask_v,
                            A_tv, VRF_t, VRF_v)
    cost = torch.cdist(H_t, H_v, p=2)
    pi = sinkhorn_log(cost, mask_t, mask_v, eps=config.adv_config['sinkhorn_eps'],
                      n_iters=config.adv_config['sinkhorn_iters'])
    return (pi * cost).sum(dim=(1, 2)).mean()


@torch.no_grad()
def save_tsne(loader, model, config, out_png):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as e:
        print(f'[skip t-SNE] {e}')
        return
    model.eval()
    feats, labels = [], []
    for batch in loader:
        batch = _to_device(batch, config.device_)
        res = model(batch)
        feats.append(res['final_feature'].cpu().numpy())
        labels.append(batch['y'].cpu().numpy())
    X = np.concatenate(feats)
    y = np.concatenate(labels)
    X = X[: min(len(X), 2000)]
    y = y[: min(len(y), 2000)]
    Z = TSNE(n_components=2, perplexity=min(30, max(5, len(y) // 4)),
             init='pca', random_state=config.seed).fit_transform(X)
    plt.figure(figsize=(6, 5))
    for c, name in ((0, 'real'), (1, 'fake')):
        m = y == c
        if m.sum():
            plt.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.7, label=name)
    plt.legend()
    plt.title('CMA-VRP fused representation (Fig.5)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f'[ok] {out_png}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default=config.datasets, choices=['twitter', 'pheme', 'weibo'])
    p.add_argument('--vrp_backend', default=config.vrp_backend, choices=['mock', 'hf', 'agent'])
    p.add_argument('--tag', default='')
    p.add_argument('--split', default='test')
    p.add_argument('--mode', default='all', choices=['all', 'noise', 'mask', 'wd', 'tsne'])
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--batch_size', type=int, default=config.batch_size)
    args = p.parse_args()

    config.batch_size = args.batch_size
    config.cache_path = os.path.join(_HERE, 'cache', config.datasets)
    config.device_ = torch.device('cpu') if (args.cpu or not torch.cuda.is_available()) \
        else torch.device('cuda:0')
    name = f'{config.network}-{config.datasets}-{config.vrp_backend}{("-" + args.tag) if args.tag else ""}'
    ckpt = os.path.join(_HERE, 'ckpt', name, 'state_dict.pt')
    out_dir = os.path.join(_HERE, 'results', name, 'robustness')
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger('robustness', os.path.join(_HERE, 'logs', name))
    set_seed(config.seed)
    loader = get_dataloader(config, args.split, shuffle=False)
    model = CMAVRP(config).to(config.device_)
    loaded = model.load_state_dict(torch.load(ckpt, map_location=config.device_,
                                              weights_only=True))
    if loaded.missing_keys or loaded.unexpected_keys:
        logger.warning('Checkpoint/key mismatch -- missing=%s unexpected=%s',
                       loaded.missing_keys, loaded.unexpected_keys)
    model.eval()

    table = {}
    if args.mode in ('all', 'noise', 'wd'):
        for s in NOISE_LEVELS:
            r = evaluate(loader, model, config, 'noise', s)
            table[f'noise_{s:g}'] = r
            logger.info('noise sigma=%-8g acc=%.4f mac_f1=%.4f wd=%s',
                        s, r['accuracy'], r['mac_f1'], f"{r['wd']:.3f}")
    if args.mode in ('all', 'mask'):
        for rate in MASK_RATES:
            r = evaluate(loader, model, config, 'mask', rate)
            table[f'mask_{rate:g}'] = r
            logger.info('occlusion rate=%-5.2f acc=%.4f mac_f1=%.4f',
                        rate, r['accuracy'], r['mac_f1'])
    if args.mode in ('all', 'tsne'):
        save_tsne(loader, model, config, os.path.join(out_dir, 'tsne.png'))

    with open(os.path.join(out_dir, f'robustness_{args.mode}.json'), 'w', encoding='utf-8') as f:
        json.dump(table, f, indent=2)
    print(f'[ok] {os.path.join(out_dir, f"robustness_{args.mode}.json")}')


if __name__ == '__main__':
    main()
