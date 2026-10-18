import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    base: int = 64
    fusion_hidden: int = 128
    norm_mode: str = "instance"
    modality_dropout: float = 0.2
    fa_scale: float = 0.7
    fo_scale: float = 1.0
    fc_scale: float = 1.0
    accel: Tuple[int, int] = (0, 6)
    orbit: Tuple[int, int] = (6, 8)
    acoustic: Tuple[int, int] = (8, 10)


def _instance_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=2, keepdim=True)
    var = (x - mean).pow(2).mean(dim=2, keepdim=True)
    std = (var + eps).sqrt()
    return (x - mean) / std


class CNNBackbone(nn.Module):
    def __init__(self, in_ch: int, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, padding=3),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base, base * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = base * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.squeeze(-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FusionFaultSetMTLNet(nn.Module):
    """
    对外保持兼容：
      out["logits"] : [B,3]   # 给旧 evaluate / 统一评估脚本继续用
      out["reg"]    : [B,7]
    同时新增：
      out["set_logits"] : [B,8]
      out["set_prob"]   : [B,8]
      out["feats"]      : dict
    """
    def __init__(
        self,
        num_labels: int = 3,
        reg_dim: int = 7,
        cfg: Optional[FusionConfig] = None,
        fault_hidden: int = 128,
        set_hidden: int = 64,
    ):
        super().__init__()
        assert num_labels == 3, "当前最小版先固定 3 标签"

        self.cfg = cfg or FusionConfig()

        self.backbone_accel = CNNBackbone(
            in_ch=self.cfg.accel[1] - self.cfg.accel[0],
            base=self.cfg.base
        )
        self.backbone_orbit = CNNBackbone(
            in_ch=self.cfg.orbit[1] - self.cfg.orbit[0],
            base=self.cfg.base
        )
        self.backbone_acoustic = CNNBackbone(
            in_ch=self.cfg.acoustic[1] - self.cfg.acoustic[0],
            base=self.cfg.base
        )

        d = self.backbone_accel.out_dim
        assert d == self.backbone_orbit.out_dim == self.backbone_acoustic.out_dim
        self.d = d

        # 保留 shared gate
        self.shared_gate = nn.Sequential(
            nn.Linear(d * 3, self.cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.fusion_hidden, 3),
        )

        # 3 个 fault-wise gate
        self.fault_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d * 3, self.cfg.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.cfg.fusion_hidden, 3),
            )
            for _ in range(3)
        ])

        # 每个 fault 每个模态一个 adapter
        self.adapt_a = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3)])
        self.adapt_o = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3)])
        self.adapt_c = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3)])

        # unary
        self.unary_heads = nn.ModuleList([
            MLP(d * 2, set_hidden, 1) for _ in range(3)
        ])

        # pairwise
        self.pair01 = MLP(d * 3, set_hidden, 1)
        self.pair02 = MLP(d * 3, set_hidden, 1)
        self.pair12 = MLP(d * 3, set_hidden, 1)

        # cardinality bias
        self.card_bias = nn.Parameter(torch.zeros(4))

        # 回归头：吃 [z0,z1,z2,h_shared]
        self.reg_head = nn.Sequential(
            nn.Linear(d * 4, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, reg_dim),
        )
        self.bce_head = nn.Sequential(
            nn.Linear(d * 4, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, 3),
        )
        # 枚举 8 个集合，bit 顺序固定为 [y0,y1,y2]
        # set_id = y0 + 2*y1 + 4*y2
        all_sets = []
        for sid in range(8):
            y0 = (sid >> 0) & 1
            y1 = (sid >> 1) & 1
            y2 = (sid >> 2) & 1
            all_sets.append([y0, y1, y2])
        self.register_buffer("all_sets", torch.tensor(all_sets, dtype=torch.float32))  # [8,3]

    def split_modalities(self, x: torch.Tensor):
        if x.size(1) < self.cfg.acoustic[1]:
            raise ValueError(f"FusionFaultSetMTLNet requires all10 input. Got C={x.size(1)}")
        a0, a1 = self.cfg.accel
        o0, o1 = self.cfg.orbit
        c0, c1 = self.cfg.acoustic
        xa = x[:, a0:a1, :]
        xo = x[:, o0:o1, :]
        xc = x[:, c0:c1, :]
        return xa, xo, xc

    def _apply_modality_dropout(self, ea, eo, ec):
        p = float(self.cfg.modality_dropout)
        if (not self.training) or p <= 0:
            keep = torch.ones((ea.size(0), 3), device=ea.device, dtype=ea.dtype)
            return ea, eo, ec, keep

        keep = (torch.rand((ea.size(0), 3), device=ea.device) > p).to(ea.dtype)
        all_drop = keep.sum(dim=1, keepdim=True) < 0.5
        if all_drop.any():
            keep[all_drop.squeeze(1), 0] = 1.0

        ea = ea * keep[:, 0:1]
        eo = eo * keep[:, 1:2]
        ec = ec * keep[:, 2:3]
        return ea, eo, ec, keep

    def _masked_softmax3(self, logits3: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.tensor(-1e9, device=logits3.device, dtype=logits3.dtype)
        logits3 = torch.where(keep > 0.5, logits3, neg_inf)
        return F.softmax(logits3, dim=1)

    def _build_fault_token(self, idx: int, ea, eo, ec, keep):
        g_in = torch.cat([ea, eo, ec], dim=1)
        g = self.fault_gates[idx](g_in)
        w = self._masked_softmax3(g, keep)

        za = self.adapt_a[idx](ea)
        zo = self.adapt_o[idx](eo)
        zc = self.adapt_c[idx](ec)

        z = w[:, 0:1] * za + w[:, 1:2] * zo + w[:, 2:3] * zc
        return z, w

    def _set_logits_from_tokens(self, z0, z1, z2, h_shared):
        u0 = self.unary_heads[0](torch.cat([z0, h_shared], dim=1)).squeeze(1)
        u1 = self.unary_heads[1](torch.cat([z1, h_shared], dim=1)).squeeze(1)
        u2 = self.unary_heads[2](torch.cat([z2, h_shared], dim=1)).squeeze(1)

        p01 = self.pair01(torch.cat([z0, z1, h_shared], dim=1)).squeeze(1)
        p02 = self.pair02(torch.cat([z0, z2, h_shared], dim=1)).squeeze(1)
        p12 = self.pair12(torch.cat([z1, z2, h_shared], dim=1)).squeeze(1)

        B = z0.size(0)
        Y = self.all_sets.to(z0.device)  # [8,3]
        y0 = Y[:, 0].view(1, 8)
        y1 = Y[:, 1].view(1, 8)
        y2 = Y[:, 2].view(1, 8)

        card = (y0 + y1 + y2).long()  # [1,8]
        b = self.card_bias[card]      # [1,8]

        # broadcast 到 [B,8]
        set_logits = (
            u0.unsqueeze(1) * y0
            + u1.unsqueeze(1) * y1
            + u2.unsqueeze(1) * y2
            + p01.unsqueeze(1) * (y0 * y1)
            + p02.unsqueeze(1) * (y0 * y2)
            + p12.unsqueeze(1) * (y1 * y2)
            + b
        )
        return set_logits

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        xa, xo, xc = self.split_modalities(x)

        nm = (self.cfg.norm_mode or "none").lower().strip()
        if nm == "instance":
            xa = _instance_norm(xa)
            xo = _instance_norm(xo)
            xc = _instance_norm(xc)
        elif nm == "none":
            pass
        else:
            raise ValueError(f"Unknown norm_mode={self.cfg.norm_mode}")

        ea = self.backbone_accel(xa)
        eo = self.backbone_orbit(xo)
        ec = self.backbone_acoustic(xc)

        ea = float(self.cfg.fa_scale) * ea
        eo = float(self.cfg.fo_scale) * eo
        ec = float(self.cfg.fc_scale) * ec

        ea, eo, ec, keep = self._apply_modality_dropout(ea, eo, ec)

        # shared context
        g_in = torch.cat([ea, eo, ec], dim=1)
        g_shared = self.shared_gate(g_in)
        w_shared = self._masked_softmax3(g_shared, keep)
        h_shared = w_shared[:, 0:1] * ea + w_shared[:, 1:2] * eo + w_shared[:, 2:3] * ec

        # 3 fault tokens
        z0, w0 = self._build_fault_token(0, ea, eo, ec, keep)
        z1, w1 = self._build_fault_token(1, ea, eo, ec, keep)
        z2, w2 = self._build_fault_token(2, ea, eo, ec, keep)

        # A1: 保留 token，但主分类退回 BCE
        cls_in = torch.cat([z0, z1, z2, h_shared], dim=1)
        logits = self.bce_head(cls_in)

        # set 分支先保留，但不再作为主损失
        set_logits = self._set_logits_from_tokens(z0, z1, z2, h_shared)  # [B,8]
        set_prob = F.softmax(set_logits, dim=1)

        reg_in = cls_in
        reg = self.reg_head(reg_in)

        out = {
            "logits": logits,  # A1 主分类直接走 BCE logits
            "reg": reg,
            "set_logits": set_logits,  # 先保留，便于后续分析，但这次训练不用它
            "set_prob": set_prob,
            "feats": {
                "ea": ea, "eo": eo, "ec": ec,
                "h_shared": h_shared,
                "z0": z0, "z1": z1, "z2": z2,
                "w_shared": w_shared,
                "w0": w0, "w1": w1, "w2": w2,
            }
        }
        return out