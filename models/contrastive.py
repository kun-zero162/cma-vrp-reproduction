"""Sec.3.2.1 Graph Contrastive Learning.

(i)   entity matching        -> preprocess/entity_matching.py (produces the cross links)
(ii)  inter-graph (Eq.1-3)  -> cross-modal InfoNCE between X^m_t and X^m_v with the
    soft alignment target  Ytil = theta*Y + (1-theta)*S, theta learnable, init 0.5
(iii) intra-graph (Eq.4)    -> SimCLR/InfoNCE over the fused representation g(.):
    anchor = original graph, positive = entity-masked graph,
    negative = modality-mixed graph  x- = {s_t^i, s_v^d(i)}  or  {s_t^d(i), s_v^i}
L_cl = L_intra + L_inter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_TEXT_KEYS = ('X_t', 'R_t', 'A_t', 'VRF_t', 'mask_t')
_VISUAL_KEYS = ('X_v', 'R_v', 'A_v', 'VRF_v', 'mask_v', 'X_v_vis')


def mix_modality(batch, mode='visual'):
    """[S3.2.1-iii] negative mining: keep one modality, swap the other from another sample."""
    perm = torch.randperm(batch['X_t'].shape[0], device=batch['X_t'].device)
    out = dict(batch)
    keys = _VISUAL_KEYS if mode == 'visual' else _TEXT_KEYS
    for k in keys:
        out[k] = batch[k][perm]
    out['A_tv'] = torch.zeros_like(batch['A_tv'])     # no valid cross links after mixing
    out['y'] = batch['y']
    return out


class GraphContrastiveLearning(nn.Module):
    def __init__(self, dim, tau=0.07, p_m=0.2, theta_init=0.5, symmetric_inter=False,
                 detach_soft_target=True):
        super().__init__()
        self.tau = tau
        self.p_m = p_m
        self.symmetric_inter = symmetric_inter
        self.detach_soft_target = detach_soft_target
        # f(.) and g(.) are the projection heads of Eq.(3) and Eq.(4)
        self.proj_f = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.proj_g = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        init = 0.0 if theta_init <= 0 else torch.logit(torch.tensor(float(theta_init))).item()
        self.theta_raw = nn.Parameter(torch.tensor(init))

    @property
    def theta(self):
        return torch.sigmoid(self.theta_raw)

    def entity_mask(self, X, mask):
        """V+(G) = (M (x) V_m, E_m): replace masked entity features by random embeddings."""
        bern = torch.bernoulli(torch.full(X.shape[:2] + (1,), self.p_m, device=X.device))
        return X * (1 - bern) + torch.randn_like(X) * bern

    # ---------------- Eq.(2)(3) ----------------
    def inter_graph_loss(self, Xm_t, Xm_v):
        a = F.normalize(self.proj_f(Xm_t), dim=-1)
        b = F.normalize(self.proj_f(Xm_v), dim=-1)
        logits = a @ b.T / self.tau                              # [B,B], i anchored on text
        S = torch.softmax(logits.detach() if self.detach_soft_target else logits, dim=-1)
        Y = torch.eye(logits.shape[0], device=logits.device)     # j == i is the true match
        Ytil = self.theta * Y + (1 - self.theta) * S
        loss = -(Ytil * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        if self.symmetric_inter:
            logits2 = b @ a.T / self.tau
            S2 = torch.softmax(logits2.detach(), dim=-1)
            Ytil2 = self.theta * Y + (1 - self.theta) * S2
            loss = 0.5 * (loss + -(Ytil2 * torch.log_softmax(logits2, dim=-1)).sum(-1).mean())
        return loss

    # ---------------- Eq.(4) ----------------
    def intra_graph_loss(self, z_a, z_p, z_n):
        """anchor / positive / fused representations, all after g(.)."""
        z_a, z_p, z_n = (F.normalize(z, dim=-1) for z in (z_a, z_p, z_n))
        pos = (z_a * z_p).sum(-1) / self.tau                      # [B]
        neg_mixed = z_a @ z_n.T / self.tau                        # [B,B] own + others' mixes
        neg_anchor = z_a @ z_a.T / self.tau                       # [B,B] other anchors
        eye = torch.eye(z_a.shape[0], device=z_a.device, dtype=torch.bool)
        neg_anchor = neg_anchor.masked_fill(eye, -1e4)            # exclude the anchor itself
        logZ = torch.logsumexp(torch.cat([neg_mixed, neg_anchor], dim=-1), dim=-1)
        return -(pos - logZ).mean()

    def forward(self, Xm_t, Xm_v, o, o_pos, o_neg, use_inter=True, use_intra=True):
        L_inter = self.inter_graph_loss(Xm_t, Xm_v) if use_inter else o.new_zeros(())
        L_intra = (self.intra_graph_loss(self.proj_g(o), self.proj_g(o_pos), self.proj_g(o_neg))
                   if use_intra else o.new_zeros(()))
        return {'L_cl': L_inter + L_intra, 'L_inter': L_inter, 'L_intra': L_intra}
