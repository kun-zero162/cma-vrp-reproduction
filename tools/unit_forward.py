"""Shape / gradient unit check for CMA-VRP, run on random tensors (no data needed).

Validates that every branch of Eq.13 produces finite gradients and that the components
actually receive gradient from the objective (the min-max terms are the risky ones).
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from configs.config_setting import setting_config as config  # noqa: E402
from models.CMAVRP import CMAVRP  # noqa: E402


def random_batch(B=8, pad_empty=False):
    c = config
    Nt, Nv = c.max_text_nodes, c.max_visual_nodes
    rd = c.model_config['region_dim']
    g = torch.Generator().manual_seed(0)
    mask_t = torch.ones(B, Nt)
    mask_v = torch.ones(B, Nv)
    if pad_empty:                      # some samples with very few nodes
        mask_t[0, 6:] = 0
        mask_v[1, 3:] = 0
        mask_v[2] = 0                  # a sample with NO visual nodes at all
        mask_t[3] = 0
    batch = dict(
        X_t=torch.randn(B, 768, generator=g), X_v=torch.randn(B, 768, generator=g),
        X_v_vis=torch.randn(B, Nv, rd, generator=g),
        VRF_t=torch.randn(B, 768, generator=g), VRF_v=torch.randn(B, 768, generator=g),
    )
    batch['X_t'] = batch['X_t'].unsqueeze(1).repeat(1, Nt, 1)
    batch['X_v'] = batch['X_v'].unsqueeze(1).repeat(1, Nv, 1)
    batch['VRF_t'] = batch['VRF_t'].unsqueeze(1).repeat(1, Nt, 1)
    batch['VRF_v'] = batch['VRF_v'].unsqueeze(1).repeat(1, Nv, 1)
    batch['mask_t'], batch['mask_v'] = mask_t, mask_v
    batch['A_t'] = (torch.rand(B, Nt, Nt, generator=g) > 0.6).float()
    batch['A_v'] = (torch.rand(B, Nv, Nv, generator=g) > 0.6).float()
    batch['A_tv'] = (torch.rand(B, Nt, Nv, generator=g) > 0.85).float()
    batch['A_t'] = batch['A_t'] * mask_t.unsqueeze(2) * mask_t.unsqueeze(1)
    batch['A_v'] = batch['A_v'] * mask_v.unsqueeze(2) * mask_v.unsqueeze(1)
    batch['A_tv'] = batch['A_tv'] * mask_t.unsqueeze(2) * mask_v.unsqueeze(1)
    batch['R_t'] = torch.randint(0, 36, (B, Nt, Nt), generator=g)
    batch['R_v'] = torch.randint(0, 34, (B, Nv, Nv), generator=g)
    batch['y'] = torch.randint(0, 2, (B,), generator=g)
    batch['id'] = [str(i) for i in range(B)]
    return batch


def check_forward_backward(label, batch, model, expect_zero=()):
    model.train()
    out = model(batch)
    scalar = {k: (float(out[k].detach()) if torch.is_tensor(out[k]) else float(out[k]))
              for k in ('loss_total', 'loss_y', 'loss_cl', 'loss_adv',
                        'L_inter', 'L_intra', 'L_cycle', 'L_ot',
                        'L_D0', 'L_D1', 'J', 'theta')}
    assert all(v == v and abs(v) != float('inf') for v in scalar.values()), scalar
    # a disabled branch must report an exactly-zero loss, otherwise the ablation
    # switch is cosmetic and the component still feeds the objective
    for k in expect_zero:
        assert scalar[k] == 0.0, f'ablation disabled {k} but it is {scalar[k]:.4f}'
    model.zero_grad(set_to_none=True)
    out['loss_total'].backward()
    nonfinite = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    nograd = [n for n, p in model.named_parameters() if p.grad is None]
    print(f'  [{label}] losses: ' + ' '.join(f'{k}={v:.3f}' for k, v in scalar.items()))
    print(f'  [{label}] non-finite grads: {nonfinite or "none"} | '
          f'params without grad: {len(nograd)}/{len(list(model.parameters()))}')
    if nonfinite:
        raise AssertionError(f'non-finite gradients in {label}: {nonfinite}')
    model.eval()
    with torch.no_grad():
        o = model(batch)
    assert torch.isfinite(o['logits']).all(), 'non-finite logits at eval'
    assert o['classify_pred'].shape == (batch['y'].shape[0],), o['classify_pred'].shape
    print(f'  [{label}] eval ok | logits std {float(o["logits"].std()):.4f}, '
          f'x_hat {tuple(o["final_feature"].shape)}')


def main():
    torch.manual_seed(config.seed)
    model = CMAVRP(config)
    print(f'# CMA-VRP unit check ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)')
    check_forward_backward('dense', random_batch(), model)
    check_forward_backward('sparse/padding', random_batch(pad_empty=True), model)

    # theta must be trainable: dL/d(theta_raw) != 0 when the inter-graph loss is active
    # (the CL branch only runs in train mode, so re-arm it before this check)
    model.train()
    model.zero_grad(set_to_none=True)
    model(random_batch())['loss_total'].backward()
    g = model.cl.theta_raw.grad
    txt = 'None' if g is None else f'{float(g):.5f}'
    print(f'  [theta] grad wrt theta_raw = {txt} '
          f'(theta={float(model.cl.theta.detach()):.3f})')
    assert g is not None and abs(float(g)) > 0, 'the learnable smoothing theta gets no gradient'

    # Table 2 ablations must all run. NOTE: the model copies config.ablation in
    # __init__, so switches have to go through set_ablation -- mutating the config
    # here would silently test nothing.
    # Every switch that removes a loss term must zero *that* term, otherwise the row in
    # the ablation table is just the full model measured twice.
    zero_key = {'use_gcl_inter': 'L_inter', 'use_gcl_intra': 'L_intra',
                'use_vsca': 'L_cycle', 'use_accil': 'L_D0'}
    for switch in ('use_vrf', 'use_gcl_inter', 'use_gcl_intra', 'use_cmgaf',
                   'use_accil', 'use_vsca'):
        model.set_ablation(**{switch: False})
        try:
            check_forward_backward(f'ablate:{switch}', random_batch(), model,
                                   expect_zero=tuple(z for k, z in zero_key.items()
                                                     if k == switch))
        finally:
            model.set_ablation(**{switch: True})
    print('# ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
