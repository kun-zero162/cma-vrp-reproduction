"""Sec.3.3 VSCA (Eq.8-9) and Sec.3.4 ACCIL (Eq.10-12).

The adversarial terms cannot be expressed as a single scalar objective (min_I max_D1),
so they are trained through EASE's `ReverseLayerF` gradient-reversal layer: the critic
weights are pushed up while the encoder weights receive the negated gradient. This lets
one optimiser drive both games, which is what `engine.py` assumes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import build_mlp, grl, masked_mean


def sinkhorn_log(cost, mask_a, mask_b, eps=0.05, n_iters=50):
    """Entropic OT in the log domain. cost [B,I,J] -> transfer plan pi [B,I,J].

    mask_a [B,I] selects valid source rows, mask_b [B,J] valid target columns; invalid
    rows/columns are dropped from the marginals by adding log(mask) to the kernel.
    """
    B, I, J = cost.shape
    log_u = torch.zeros(B, I, 1, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(B, 1, J, device=cost.device, dtype=cost.dtype)
    row_log = torch.log(mask_a.to(cost.dtype).clamp_min(1e-9)).unsqueeze(-1)   # [B,I,1]
    col_log = torch.log(mask_b.to(cost.dtype).clamp_min(1e-9)).unsqueeze(1)    # [B,1,J]
    logK = -cost / eps + row_log + col_log
    for _ in range(n_iters):
        log_u = -torch.logsumexp(logK + log_v, dim=2, keepdim=True)            # [B,I,1]
        log_v = -torch.logsumexp(logK + log_u, dim=1, keepdim=True)            # [B,1,J]
    pi = torch.exp(logK + log_u + log_v)
    return torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)


def _gradient_penalty(discriminator, real, fake, dim=-1):
    B = min(real.shape[0], fake.shape[0])
    if B == 0:
        return real.new_zeros(())
    real, fake = real[:B], fake[:B]
    alpha = torch.rand(B, 1, device=real.device, dtype=real.dtype)
    xhat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d = discriminator(xhat)
    grad = torch.autograd.grad(d.sum(), xhat, create_graph=True)[0]
    return (grad.norm(2, dim=dim) - 1.0).pow(2).mean()


class VisualSemanticCycleAlignment(nn.Module):
    """VSCA: Wasserstein cycle alignment between textual and visual entity graphs."""

    def __init__(self, dim, vrf_dim, hidden, latent, cyc_lambda=10.0, ot_lambda=1.0,
                 gp_lambda=10.0, sinkhorn_eps=0.1, sinkhorn_iters=50, use_grl=True,
                 critic_nodes=512):
        super().__init__()
        self.cyc_lambda, self.ot_lambda = cyc_lambda, ot_lambda
        self.gp_lambda, self.eps, self.iters = gp_lambda, sinkhorn_eps, sinkhorn_iters
        self.use_grl, self.critic_nodes = use_grl, critic_nodes

        self.proj_vrf_t = nn.Linear(vrf_dim, dim)
        self.proj_vrf_v = nn.Linear(vrf_dim, dim)
        self.G_tv = build_mlp(dim * 2, hidden, latent)     # (f_t, f_v) -> cycle latent
        self.G_vt = build_mlp(dim * 2, hidden, latent)
        # Wasserstein critic D_s over (aligned entity rep, VRF rep), Eq.(9)
        self.D_s = nn.Sequential(build_mlp(dim * 2, hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, H_t, mask_t, VRF_t, H_v, mask_v, VRF_v, alpha=1.0):
        B, Nt, _ = H_t.shape
        Nv = H_v.shape[1]

        # ---- Eq.(8): Wasserstein distance weighted by VRF similarity wd(u,w) ----
        with torch.no_grad():
            wd = F.cosine_similarity(VRF_t.unsqueeze(2), VRF_v.unsqueeze(1), dim=-1)   # [B,Nt,Nv]
        dist = torch.cdist(H_t, H_v, p=1)
        invalid = (~mask_t.unsqueeze(2).bool()) | (~mask_v.unsqueeze(1).bool())
        cost = (1.0 - wd) * dist + 1e3 * invalid.to(dist.dtype)
        pi = sinkhorn_log(cost, mask_t, mask_v, self.eps, self.iters)
        row = pi / pi.sum(-1, keepdim=True).clamp_min(1e-9)
        T_t = torch.bmm(row, H_v)                                # domain-transferred entities
        T_v = torch.bmm(row.transpose(1, 2), H_t)
        L_ot = self.ot_lambda * (
            masked_mean(H_t - T_t, mask_t).norm(dim=-1).mean()
            + masked_mean(H_v - T_v, mask_v).norm(dim=-1).mean()
        )
        H_t_al, H_v_al = H_t + T_t, H_v + T_v

        # ---- cycle consistency through G_tv / G_vt ----
        f_t, f_v = masked_mean(H_t_al, mask_t), masked_mean(H_v_al, mask_v)
        z_tv = self.G_tv(torch.cat([f_t, f_v], dim=-1))
        z_vt = self.G_vt(torch.cat([f_v, f_t], dim=-1))
        L_con = self.cyc_lambda * (
            (self.G_vt(torch.cat([z_tv, f_t], dim=-1)) - f_v).pow(2).mean()
            + (self.G_tv(torch.cat([z_vt, f_v], dim=-1)) - f_t).pow(2).mean()
        )

        # ---- Eq.(9): critic separates aligned textual from aligned visual entity nodes ----
        node_t = torch.cat([H_t_al, self.proj_vrf_t(VRF_t)], dim=-1)
        node_v = torch.cat([H_v_al, self.proj_vrf_v(VRF_v)], dim=-1)
        flat_t = node_t.reshape(B * Nt, -1)[mask_t.reshape(-1).bool()]
        flat_v = node_v.reshape(B * Nv, -1)[mask_v.reshape(-1).bool()]
        if flat_t.shape[0] == 0:
            flat_t = node_t[:, 0]
        if flat_v.shape[0] == 0:
            flat_v = node_v[:, 0]
        k = self.critic_nodes
        d_rt = self.D_s(grl(flat_t[:k], alpha, self.use_grl))
        d_fv = self.D_s(grl(flat_v[:k], alpha, self.use_grl))
        L_Ds = -(F.softplus(d_rt).mean() - F.softplus(d_fv).mean())
        L_Ds_v = -(F.softplus(d_fv).mean() - F.softplus(d_rt).mean())
        L_adv = 0.5 * (L_Ds + L_Ds_v)
        if self.gp_lambda > 0 and torch.is_grad_enabled():
            # the penalty needs a real backward pass; it is meaningless under no_grad()
            pen = 0.5 * (_gradient_penalty(self.D_s, flat_t[:k], flat_v[:k])
                         + _gradient_penalty(self.D_s, flat_v[:k], flat_t[:k]))
            L_adv = L_adv + self.gp_lambda * pen

        losses = {'L_cycle': L_con + L_adv, 'L_cycle_adv': L_adv,
                  'L_cycle_con': L_con, 'L_ot': L_ot}
        return masked_mean(H_t_al, mask_t), masked_mean(H_v_al, mask_v), H_t_al, H_v_al, losses


class ModalityInvariantLearning(nn.Module):
    """ACCIL Eq.(10-12): invariant projection + dual-modality critics D0 / D1."""

    def __init__(self, latent, hidden, num_classes=2, use_grl=True):
        super().__init__()
        self.use_grl = use_grl
        self.I = nn.Sequential(build_mlp(latent, hidden, latent), nn.GELU())
        self.D0 = nn.Linear(latent * 2, num_classes)      # fused class critic -> sample weights
        self.D1 = nn.Linear(latent, num_classes)          # per-modality invariance critic
        self.head = nn.Linear(latent, num_classes)

    @staticmethod
    def _jsd(p, q):
        p, q = p.clamp_min(1e-8), q.clamp_min(1e-8)
        m = 0.5 * (p + q)
        lm = m.log()
        return 0.5 * (p * (p.log() - lm)).sum(-1) + 0.5 * (q * (q.log() - lm)).sum(-1)

    def forward(self, e_t, e_v, y, alpha=1.0):
        z_t, z_v = self.I(e_t), self.I(e_v)                       # Eq.(10) invariant features

        z = torch.cat([grl(z_t, alpha, self.use_grl),
                       grl(z_v, alpha, self.use_grl)], dim=-1)
        w_logits = self.D0(z)
        L_D0 = F.cross_entropy(w_logits, y)                       # Eq.(11)
        with torch.no_grad():
            w = 1.0 - F.softmax(w_logits, dim=-1).gather(
                1, y.view(-1, 1) if y.dim() > 1 else y.long().view(-1, 1))   # w_i(x)=1-D0^i(x)

        p_t = F.softmax(self.D1(grl(z_t, alpha, self.use_grl)), dim=-1)
        p_v = F.softmax(self.D1(grl(z_v, alpha, self.use_grl)), dim=-1)
        J = (w * self._jsd(p_t, p_v)).mean()                      # Eq.(12), weighted JSD
        L_D1 = -J                                                 # min_I max_D1 via the GRL

        c_t, c_v = F.softmax(self.head(e_t), dim=-1), F.softmax(self.head(e_v), dim=-1)
        c_m = torch.maximum(w * c_t, w * c_v)                     # Eq.(12) invariant prediction
        return c_m, z, {'L_D0': L_D0, 'L_D1': L_D1, 'J': J}
