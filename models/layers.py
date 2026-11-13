"""Shared layers for CMA-VRP.

`ReverseLayerF` (gradient reversal) is inherited from the EASE Expert codebase; a local
equivalent is used only if the EASE module cannot be imported (it pulls in `timm`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:  # reuse the EASE implementation, loaded by path so sys.path stays clean
    from ease_bridge import load_expert_module
    ReverseLayerF = load_expert_module('models/layers.py').ReverseLayerF
except Exception:  # noqa: BLE001 - EASE module unavailable (e.g. timm not installed)
    class ReverseLayerF(Function):
        @staticmethod
        def forward(ctx, input_, alpha):
            ctx.alpha = alpha
            return input_

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.alpha, None


def build_mlp(in_dim, hidden_dim, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def masked_mean(x, mask, dim=1, eps=1e-6):
    """mask: [B, N] -> reduce [B, N, d] to [B, d]."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim) / m.sum(dim).clamp_min(eps)


def masked_max(x, mask, dim=1):
    m = mask.to(x.dtype).unsqueeze(-1)
    return x.masked_fill(m == 0, -1e4).max(dim=dim).values


def graph_readout(H_t, mask_t, H_v, mask_v, mode='mean'):
    """READOUT(.) over the multimodal graph G_m (Eq.1 / Sec.3.2.2)."""
    H = torch.cat([H_t, H_v], dim=1)
    mask = torch.cat([mask_t, mask_v], dim=1)
    if mode == 'mean':
        return masked_mean(H, mask)
    if mode == 'max':
        return masked_max(H, mask)
    if mode == 'mean_max':
        return torch.cat([masked_mean(H, mask), masked_max(H, mask)], dim=-1)
    raise ValueError(f'Unknown readout mode: {mode}')


def readout_dim(gnn_dim, mode='mean'):
    return gnn_dim * 2 if mode == 'mean_max' else gnn_dim


def grl(x, alpha, enabled=True):
    return ReverseLayerF.apply(x, alpha) if enabled else x
