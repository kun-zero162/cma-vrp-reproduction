"""Sec.3.2.2 Cross-Modal Graph Fusion (CMGAF): Uni-Att (Eq.5) + cross-modal gating (Eq.6-7)."""
import math

import torch
import torch.nn as nn


class UniAtt(nn.Module):
    """Unimodal Attention, Eq.(5).

    C_t^(l) = Softmax(A_t * Q K^T / sqrt(d)) * V, with K = W_K H + phi(Z_t),
    V = W_V H + phi(Z_t). phi(Z) is a per-edge term, so K/V become pairwise.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.phi = nn.Linear(dim, dim)            # learnable edge transformation phi(Z)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, H, A, Z, mask):
        # H [B,N,d] | A [B,N,N] | Z [B,N,N,d] | mask [B,N]
        Q = self.Wq(H)
        K = self.Wk(H)
        V = self.Wv(H)
        E = self.phi(Z)                                            # [B,N,N,d]
        K_ij = K.unsqueeze(1) + E                                  # [B,i,j,d]
        V_ij = V.unsqueeze(1) + E
        scores = (Q.unsqueeze(2) * K_ij).sum(-1) / math.sqrt(self.dim)   # [B,N,N]

        allowed = (A > 0) & mask.unsqueeze(1).bool() & mask.unsqueeze(2).bool()
        scores = scores.masked_fill(~allowed, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)                     # isolated nodes
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * V_ij).sum(2)                   # [B,N,d]
        return self.norm(H + out)


class CrossModalGating(nn.Module):
    """Inter-modal fusion, Eq.(6)(7).

    R_tu = C_tu + sum_{j in N_v(u)} alpha_ij * C_vj + W_vt * VRF_t(u)
    alpha_ij = gate(W1 * C_tu + W2 * C_vj)
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        self.W_vrf = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, C_src, C_tgt, A_cross, VRF_src):
        # C_src [B,Ns,d] | C_tgt [B,Nm,d] | A_cross [B,Ns,Nm] | VRF_src [B,Ns,d]
        alpha = torch.sigmoid(
            self.gate(self.W1(C_src)).unsqueeze(2) + self.gate(self.W2(C_tgt)).unsqueeze(1)
        ).squeeze(-1)                                              # [B,Ns,Nm]
        A = (A_cross > 0).to(alpha.dtype)
        alpha = self.dropout(alpha * A)
        denom = alpha.sum(-1, keepdim=True).clamp_min(1e-6)
        agg = torch.bmm(alpha, C_tgt) / denom                      # normalised for stability
        return self.norm(C_src + agg + self.W_vrf(VRF_src))


class FFN(nn.Module):
    """Position-wise feed-forward network, H_t^(l) = FFN(R_t^(l))."""

    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class CrossModalGraphFusion(nn.Module):
    """L stacked fusion layers over the multimodal graph G_m."""

    def __init__(self, dim, n_layers=2, ffn_dim=600, dropout=0.2):
        super().__init__()
        self.t_att = nn.ModuleList([UniAtt(dim, dropout) for _ in range(n_layers)])
        self.v_att = nn.ModuleList([UniAtt(dim, dropout) for _ in range(n_layers)])
        self.t_gate = nn.ModuleList([CrossModalGating(dim, dropout) for _ in range(n_layers)])
        self.v_gate = nn.ModuleList([CrossModalGating(dim, dropout) for _ in range(n_layers)])
        self.t_ffn = nn.ModuleList([FFN(dim, ffn_dim, dropout) for _ in range(n_layers)])
        self.v_ffn = nn.ModuleList([FFN(dim, ffn_dim, dropout) for _ in range(n_layers)])

    def forward(self, H_t, A_t, Z_t, mask_t, H_v, A_v, Z_v, mask_v, A_tv, VRF_t, VRF_v):
        for l in range(len(self.t_att)):
            C_t = self.t_att[l](H_t, A_t, Z_t, mask_t)
            C_v = self.v_att[l](H_v, A_v, Z_v, mask_v)
            R_t = self.t_gate[l](C_t, C_v, A_tv, VRF_t)
            R_v = self.v_gate[l](C_v, C_t, A_tv.transpose(1, 2), VRF_v)
            H_t = self.t_ffn[l](R_t) * mask_t.unsqueeze(-1)
            H_v = self.v_ffn[l](R_v) * mask_v.unsqueeze(-1)
        return H_t, H_v
