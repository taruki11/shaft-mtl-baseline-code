# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone(nn.Module):
    def __init__(self, in_ch: int, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base, 11, padding=5),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base, base * 2, 9, padding=4),
            nn.BatchNorm1d(base * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base * 2, base * 4, 7, padding=3),
            nn.BatchNorm1d(base * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = base * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class FusionConfig:
    base: int = 64
    fusion_hidden: int = 128
    norm_mode: str = "instance"
    modality_dropout: float = 0.2

    # 新增：三路特征缩放
    fa_scale: float = 1.0
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


class FusionGatedMTLNet(nn.Module):
    def __init__(
        self,
        num_labels: int = 3,
        reg_dim: int = 7,
        cfg: Optional[FusionConfig] = None,
    ):
        super().__init__()
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

        self.gate = nn.Sequential(
            nn.Linear(d * 3, self.cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.fusion_hidden, 3),
        )

        self.cls = nn.Linear(d, num_labels)
        self.reg = nn.Linear(d, reg_dim)

    def split_modalities(self, x: torch.Tensor):
        if x.size(1) < self.cfg.acoustic[1]:
            raise ValueError(
                f"FusionGatedMTLNet requires all10 input (C>=10). Got C={x.size(1)}"
            )
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
            raise ValueError(f"Unknown norm_mode={self.cfg.norm_mode}. Use none|instance")

        ea = self.backbone_accel(xa)
        eo = self.backbone_orbit(xo)
        ec = self.backbone_acoustic(xc)

        # 新增：fusion 前特征缩放
        ea = float(self.cfg.fa_scale) * ea
        eo = float(self.cfg.fo_scale) * eo
        ec = float(self.cfg.fc_scale) * ec

        ea, eo, ec, keep = self._apply_modality_dropout(ea, eo, ec)

        g_in = torch.cat([ea, eo, ec], dim=1)
        g_logits = self.gate(g_in)

        neg_inf = torch.tensor(-1e9, device=g_logits.device, dtype=g_logits.dtype)
        g_logits = torch.where(keep > 0.5, g_logits, neg_inf)
        w = F.softmax(g_logits, dim=1)

        h = w[:, 0:1] * ea + w[:, 1:2] * eo + w[:, 2:3] * ec

        logits = self.cls(h)
        reg = self.reg(h)

        if return_feats:
            return logits, reg, ea, eo, ec, h, w
        return logits, reg


def masked_scaled_huber_loss(pred, target, y_multi, scale=None, beta=1.0):
    device = pred.device
    m = torch.zeros_like(pred, device=device)
    m[:, 0] = y_multi[:, 0]
    m[:, 1:4] = y_multi[:, 1].unsqueeze(1)
    m[:, 4:7] = y_multi[:, 2].unsqueeze(1)

    diff = pred - target
    if scale is not None:
        sc = torch.as_tensor(scale, device=device, dtype=diff.dtype).view(1, -1)
        sc = torch.clamp(sc, min=1e-6)
        diff = diff / sc

    loss_mat = F.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        beta=beta,
        reduction="none"
    )
    loss_mat = loss_mat * m
    denom = m.sum().clamp_min(1.0)
    return loss_mat.sum() / denom