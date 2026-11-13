"""CMA-VRP: Entity Graph Alignment and Visual Reasoning for Multimodal Fake News Detection.

Top-level model (Sec.3.2-3.4, Eq.13):
    x_hat = o (x) c_m
    L     = L_y + lambda * L_cl + beta * L_adv
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import GraphContrastiveLearning, mix_modality
from .cycle_align import ModalityInvariantLearning, VisualSemanticCycleAlignment
from .gnn import CrossModalGraphFusion
from .layers import build_mlp, graph_readout, masked_mean, readout_dim


class CMAVRP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        mc = config.model_config
        d = mc['gnn_dim']
        self.d = d
        self.ablation = dict(config.ablation)

        # node features are BERT-encoded entity strings [S4]
        self.node_proj_t = nn.Linear(mc['node_dim'], d)
        self.node_proj_v = nn.Linear(mc['node_dim'], d)
        self.region_proj = nn.Linear(mc['region_dim'], d)
        self.vrf_proj_t = nn.Linear(mc['node_dim'], d)     # VRF_t / VRF_v reasoning features
        self.vrf_proj_v = nn.Linear(mc['node_dim'], d)

        self.edge_t = nn.Embedding(mc['max_text_relations'], d, padding_idx=0)
        self.edge_v = nn.Embedding(mc['max_visual_relations'], d, padding_idx=0)

        self.fusion = CrossModalGraphFusion(
            d, n_layers=mc['n_fusion_layers'], ffn_dim=mc['ffn_dim'], dropout=mc['dropout'])
        self.readout_mode = mc['readout']
        o_dim = readout_dim(d, mc['readout'])

        cc = config.cl_config
        self.cl = GraphContrastiveLearning(
            dim=d, tau=cc['tau'], p_m=cc['p_m'], theta_init=cc['theta_init'],
            symmetric_inter=cc['symmetric_inter'], detach_soft_target=cc['detach_soft_target'])

        ac = config.adv_config
        self.vsca = VisualSemanticCycleAlignment(
            dim=d, vrf_dim=d, hidden=ac['hidden_dim'], latent=ac['latent_dim'],
            cyc_lambda=ac['cyc_lambda'], ot_lambda=ac['ot_lambda'], gp_lambda=ac['gp_lambda'],
            sinkhorn_eps=ac['sinkhorn_eps'], sinkhorn_iters=ac['sinkhorn_iters'],
            use_grl=ac['use_grl'])
        self.accil = ModalityInvariantLearning(
            latent=ac['latent_dim'], hidden=ac['hidden_dim'], num_classes=config.num_classes,
            use_grl=ac['use_grl'])
        self.vrf_dim = d

        self.classifier = nn.Sequential(
            build_mlp(o_dim + config.num_classes, d, d), nn.GELU(),
            nn.Dropout(mc['dropout']),
            nn.Linear(d, config.num_classes),
        )
        self._alpha = 0.0

    # Table 2 ablations
    def set_ablation(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.ablation:
                raise KeyError(f'Unknown ablation switch: {k}')
            self.ablation[k] = v

    def set_progress(self, epoch, total_epochs):
        """DANN-style ramp for the gradient-reversal layers."""
        p = (epoch + 1) / max(1, total_epochs)
        self._alpha = self.cfg.adv_config['grl_alpha_max'] * (2 / (1 + math.exp(-10 * p)) - 1)

    # ------------------------------------------------------------------
    def _encode(self, batch):
        """Raw cached tensors -> dense graph tensors (Eq.1 inputs)."""
        X_t = batch['X_t']
        X_v = batch['X_v']
        mask_t = batch['mask_t'].to(X_t.dtype)
        mask_v = batch['mask_v'].to(X_t.dtype)

        H_t0 = self.node_proj_t(X_t) * mask_t.unsqueeze(-1)
        H_v0 = self.node_proj_v(X_v) * mask_v.unsqueeze(-1)
        if 'X_v_vis' in batch and self.cfg.use_region_feats:
            H_v0 = H_v0 + self.region_proj(batch['X_v_vis']) * mask_v.unsqueeze(-1)

        Z_t = self.edge_t(batch['R_t'])                       # [B,Nt,Nt,d]
        Z_v = self.edge_v(batch['R_v'])

        VRF_t = self.vrf_proj_t(batch['VRF_t']) * mask_t.unsqueeze(-1)
        VRF_v = self.vrf_proj_v(batch['VRF_v']) * mask_v.unsqueeze(-1)
        if not self.ablation['use_vrf']:
            VRF_t = torch.zeros_like(VRF_t)
            VRF_v = torch.zeros_like(VRF_v)
        return H_t0, mask_t, Z_t, H_v0, mask_v, Z_v, batch['A_t'], batch['A_v'], batch['A_tv'], VRF_t, VRF_v

    def _graph_embeddings(self, H_t0, mask_t, Z_t, H_v0, mask_v, Z_v,
                          A_t, A_v, A_tv, VRF_t, VRF_v):
        """CMGAF + readouts; returns o (multimodal) and per-graph s_t / s_v."""
        if not self.ablation['use_cmgaf']:
            o = graph_readout(H_t0, mask_t, H_v0, mask_v, self.readout_mode)
            return o, masked_mean(H_t0, mask_t), masked_mean(H_v0, mask_v), H_t0, H_v0
        H_t, H_v = self.fusion(H_t0, A_t, Z_t, mask_t, H_v0, A_v, Z_v, mask_v,
                               A_tv, VRF_t, VRF_v)
        o = graph_readout(H_t, mask_t, H_v, mask_v, self.readout_mode)
        return o, masked_mean(H_t, mask_t), masked_mean(H_v, mask_v), H_t, H_v

    def _readout(self, *enc):
        """Fused multimodal graph representation o = READOUT(G_m), Eq.(1)."""
        return self._graph_embeddings(*enc)[0]

    def forward(self, batch):
        y = batch['y'].view(-1).long()
        enc = self._encode(batch)
        (H_t0, mask_t, Z_t, H_v0, mask_v, Z_v,
         A_t, A_v, A_tv, VRF_t, VRF_v) = enc

        o, s_t, s_v, H_t, H_v = self._graph_embeddings(*enc)

        zero = o.new_zeros(())
        L_inter = L_intra = zero
        loss_cl = zero
        if self.training and (self.ablation['use_gcl_inter'] or self.ablation['use_gcl_intra']):
            o_pos = o_neg = o                                  # reused if a branch is off
            if self.ablation['use_gcl_intra']:
                # positive view: entity-masked features, edges untouched (Eq.4)
                Hm_t = self.cl.entity_mask(H_t0, mask_t) * mask_t.unsqueeze(-1)
                Hm_v = self.cl.entity_mask(H_v0, mask_v) * mask_v.unsqueeze(-1)
                o_pos = self._readout(Hm_t, mask_t, Z_t, Hm_v, mask_v, Z_v,
                                      A_t, A_v, A_tv, VRF_t, VRF_v)
                # negative view: one modality swapped in from another sample
                o_neg = self._readout(*self._encode(mix_modality(batch)))
            cl = self.cl(s_t, s_v, o, o_pos, o_neg,
                         use_inter=self.ablation['use_gcl_inter'],
                         use_intra=self.ablation['use_gcl_intra'])
            loss_cl, L_inter, L_intra = cl['L_cl'], cl['L_inter'], cl['L_intra']

        # ---- VSCA + ACCIL ----
        # Each switch gates its own term: `use_accil` must not depend on the VSCA branch
        # having set e_t/e_v, otherwise -use_accil is a no-op while VSCA is on.
        loss_adv = zero
        L_cycle = L_ot = L_D0 = L_D1 = zero
        L_cycle_con = L_cycle_adv = J = zero
        c_m = o.new_zeros(o.shape[0], self.cfg.num_classes)
        if self.ablation['use_vsca']:
            e_t, e_v, _, _, vsca_losses = self.vsca(
                H_t, mask_t, VRF_t, H_v, mask_v, VRF_v, alpha=self._alpha)
            L_cycle, L_ot = vsca_losses['L_cycle'], vsca_losses['L_ot']
            L_cycle_con, L_cycle_adv = vsca_losses['L_cycle_con'], vsca_losses['L_cycle_adv']
            loss_adv = loss_adv + L_cycle
        if self.ablation['use_accil']:
            if not self.ablation['use_vsca']:
                e_t, e_v = masked_mean(H_t, mask_t), masked_mean(H_v, mask_v)
            c_m, _, accil_losses = self.accil(e_t, e_v, y, alpha=self._alpha)
            L_D0, L_D1, J = accil_losses['L_D0'], accil_losses['L_D1'], accil_losses['J']
            loss_adv = loss_adv + L_D0 + L_D1

        x_hat = torch.cat([o, c_m], dim=-1)                       # Eq.(13)
        logits = self.classifier(x_hat)
        loss_y = F.cross_entropy(logits, y)
        loss_total = loss_y + self.cfg.lambda_cl * loss_cl + self.cfg.beta_adv * loss_adv

        return {
            'logits': logits,
            'classify_pred': F.softmax(logits, dim=-1)[:, 1],
            'final_feature': x_hat,
            'o': o,
            'c_m': c_m,
            'loss_total': loss_total,
            'loss_y': loss_y,
            'loss_cl': loss_cl,
            'loss_adv': loss_adv,
            'theta': self.cl.theta.detach(),
            'alpha': torch.tensor(self._alpha, device=o.device),
            'L_inter': L_inter, 'L_intra': L_intra,
            'L_cycle': L_cycle, 'L_cycle_con': L_cycle_con, 'L_cycle_adv': L_cycle_adv,
            'L_ot': L_ot, 'L_D0': L_D0, 'L_D1': L_D1, 'J': J,
        }
