
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Optional, Tuple

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


class CNNBackboneMap(nn.Module):
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
        )
        self.out_dim = base * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class MultiBandPool(nn.Module):
    def __init__(self, in_dim: int, num_bands: int = 3):
        super().__init__()
        assert num_bands == 3, "当前最小版固定 3 个 band"
        self.num_bands = num_bands
        self.proj1 = nn.Linear(in_dim, in_dim)
        self.proj2 = nn.Linear(in_dim, in_dim)
        self.proj3 = nn.Linear(in_dim, in_dim)

    def forward(self, fmap: torch.Tensor):
        b1 = fmap.mean(dim=2)
        b2 = F.adaptive_max_pool1d(fmap, 1).squeeze(-1)
        if fmap.size(2) >= 2:
            diff = torch.abs(fmap[:, :, 1:] - fmap[:, :, :-1])
            b3 = diff.mean(dim=2)
        else:
            b3 = fmap.mean(dim=2)
        return [self.proj1(b1), self.proj2(b2), self.proj3(b3)]


class SpeedEncoder(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(5, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, rpm: torch.Tensor) -> torch.Tensor:
        if rpm.dim() == 2 and rpm.size(1) == 1:
            rpm = rpm[:, 0]
        rpm = rpm.float()
        f = torch.stack([
            rpm / 600.0,
            rpm / 1200.0,
            rpm / 1800.0,
            rpm / 2400.0,
            torch.log1p(torch.clamp(rpm, min=0.0)) / 8.0,
        ], dim=1)
        return self.net(f)


class OrderAwareStat(nn.Module):
    """
    轻量“order-like”分支：
    - 对 raw signal 做 rFFT
    - 自适应压到若干频带
    - 结合 rpm token 投影到 d 维
    """
    def __init__(self, speed_dim: int, out_dim: int, bins: int = 8, hidden: int = 128):
        super().__init__()
        self.bins = bins
        self.net = nn.Sequential(
            nn.Linear(bins + speed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, speed_emb: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        spec = torch.fft.rfft(x, dim=2).abs().mean(dim=1)  # [B,F]
        bins = F.adaptive_avg_pool1d(spec.unsqueeze(1), self.bins).squeeze(1)
        return self.net(torch.cat([bins, speed_emb], dim=1))


class FusionMultiBandFaultRPMOrderMTLNet(nn.Module):
    """
    在 FusionMultiBandFaultMTLNet 基础上加入：
    1) rpm conditioning
    2) 轻量 order-like spectral branch
    输出保持兼容旧 BCE 评估口径。
    """
    def __init__(
        self,
        num_labels: int = 3,
        reg_dim: int = 7,
        cfg: Optional[FusionConfig] = None,
        fault_hidden: int = 128,
        num_bands: int = 3,
        speed_embed_dim: int = 32,
        order_bins: int = 8,
        order_hidden: int = 128,
        order_scale: float = 0.35,
    ):
        super().__init__()
        assert num_labels == 3
        assert num_bands == 3

        self.cfg = cfg or FusionConfig()
        self.num_bands = num_bands
        self.order_scale = float(order_scale)

        self.backbone_accel = CNNBackboneMap(
            in_ch=self.cfg.accel[1] - self.cfg.accel[0], base=self.cfg.base
        )
        self.backbone_orbit = CNNBackboneMap(
            in_ch=self.cfg.orbit[1] - self.cfg.orbit[0], base=self.cfg.base
        )
        self.backbone_acoustic = CNNBackboneMap(
            in_ch=self.cfg.acoustic[1] - self.cfg.acoustic[0], base=self.cfg.base
        )

        d = self.backbone_accel.out_dim
        self.d = d

        self.pool_accel = MultiBandPool(d, num_bands=num_bands)
        self.pool_orbit = MultiBandPool(d, num_bands=num_bands)
        self.pool_acoustic = MultiBandPool(d, num_bands=num_bands)

        self.speed_encoder = SpeedEncoder(speed_embed_dim)
        self.order_accel = OrderAwareStat(speed_embed_dim, d, bins=order_bins, hidden=order_hidden)
        self.order_orbit = OrderAwareStat(speed_embed_dim, d, bins=order_bins, hidden=order_hidden)
        self.order_acoustic = OrderAwareStat(speed_embed_dim, d, bins=order_bins, hidden=order_hidden)

        self.shared_gate = nn.Sequential(
            nn.Linear(d * 3 + speed_embed_dim, self.cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.fusion_hidden, 3),
        )

        self.fault_mod_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d * 3 + speed_embed_dim, self.cfg.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.cfg.fusion_hidden, 3),
            ) for _ in range(3)
        ])

        self.band_gate_a = nn.ModuleList([MLP(d * 3 + speed_embed_dim, fault_hidden, num_bands) for _ in range(3)])
        self.band_gate_o = nn.ModuleList([MLP(d * 3 + speed_embed_dim, fault_hidden, num_bands) for _ in range(3)])
        self.band_gate_c = nn.ModuleList([MLP(d * 3 + speed_embed_dim, fault_hidden, num_bands) for _ in range(3)])

        self.adapt_a = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3 * num_bands)])
        self.adapt_o = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3 * num_bands)])
        self.adapt_c = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3 * num_bands)])

        self.bce_head = nn.Sequential(
            nn.Linear(d * 4, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, 3),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d * 4, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, reg_dim),
        )

        # 8-way structured label-set decoder:
        # 000,001,010,011,100,101,110,111
        self.set_head = nn.Sequential(
            nn.Linear(d * 4, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, 8),
        )

    def split_modalities(self, x: torch.Tensor):
        if x.size(1) < self.cfg.acoustic[1]:
            raise ValueError(f"Requires all10 input. Got C={x.size(1)}")
        a0, a1 = self.cfg.accel
        o0, o1 = self.cfg.orbit
        c0, c1 = self.cfg.acoustic
        return x[:, a0:a1, :], x[:, o0:o1, :], x[:, c0:c1, :]

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

    def _fuse_bands(self, tokens, band_gate, adapt_list, fault_idx, speed_emb):
        gate_in = torch.cat(tokens + [speed_emb], dim=1)
        bw = F.softmax(band_gate[fault_idx](gate_in), dim=1)
        outs = [adapt_list[fault_idx * self.num_bands + j](tokens[j]) for j in range(self.num_bands)]
        z = bw[:, 0:1] * outs[0] + bw[:, 1:2] * outs[1] + bw[:, 2:3] * outs[2]
        return z, bw

    def forward(self, x: torch.Tensor, rpm: Optional[torch.Tensor] = None, return_feats: bool = False):
        xa, xo, xc = self.split_modalities(x)
        nm = (self.cfg.norm_mode or "none").lower().strip()
        if nm == "instance":
            xa = _instance_norm(xa)
            xo = _instance_norm(xo)
            xc = _instance_norm(xc)
        elif nm != "none":
            raise ValueError(f"Unknown norm_mode={self.cfg.norm_mode}")

        if rpm is None:
            rpm = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
        speed_emb = self.speed_encoder(rpm)

        fa_map = self.backbone_accel(xa)
        fo_map = self.backbone_orbit(xo)
        fc_map = self.backbone_acoustic(xc)

        ea = fa_map.mean(dim=2)
        eo = fo_map.mean(dim=2)
        ec = fc_map.mean(dim=2)

        oa = self.order_accel(xa, speed_emb)
        oo = self.order_orbit(xo, speed_emb)
        oc = self.order_acoustic(xc, speed_emb)

        ea = float(self.cfg.fa_scale) * ea + self.order_scale * oa
        eo = float(self.cfg.fo_scale) * eo + self.order_scale * oo
        ec = float(self.cfg.fc_scale) * ec + self.order_scale * oc

        ea, eo, ec, keep = self._apply_modality_dropout(ea, eo, ec)

        g_in = torch.cat([ea, eo, ec, speed_emb], dim=1)
        w_shared = self._masked_softmax3(self.shared_gate(g_in), keep)
        h_shared = w_shared[:, 0:1] * ea + w_shared[:, 1:2] * eo + w_shared[:, 2:3] * ec

        ta = self.pool_accel(fa_map)
        to = self.pool_orbit(fo_map)
        tc = self.pool_acoustic(fc_map)

        ta = [float(self.cfg.fa_scale) * t + self.order_scale * oa for t in ta]
        to = [float(self.cfg.fo_scale) * t + self.order_scale * oo for t in to]
        tc = [float(self.cfg.fc_scale) * t + self.order_scale * oc for t in tc]

        z_list, w_mod_list = [], []
        bw_a_list, bw_o_list, bw_c_list = [], [], []

        mod_in = torch.cat([ea, eo, ec, speed_emb], dim=1)
        for k in range(3):
            w_mod = self._masked_softmax3(self.fault_mod_gates[k](mod_in), keep)
            za, bwa = self._fuse_bands(ta, self.band_gate_a, self.adapt_a, k, speed_emb)
            zo, bwo = self._fuse_bands(to, self.band_gate_o, self.adapt_o, k, speed_emb)
            zc, bwc = self._fuse_bands(tc, self.band_gate_c, self.adapt_c, k, speed_emb)
            zk = w_mod[:, 0:1] * za + w_mod[:, 1:2] * zo + w_mod[:, 2:3] * zc
            z_list.append(zk)
            w_mod_list.append(w_mod)
            bw_a_list.append(bwa)
            bw_o_list.append(bwo)
            bw_c_list.append(bwc)

        z0, z1, z2 = z_list
        fault_concat = torch.cat([z0, z1, z2], dim=1)
        fused = torch.cat([z0, z1, z2, h_shared], dim=1)

        logits = self.bce_head(fused)
        reg = self.reg_head(fused)
        set_logits = self.set_head(fused)
        return {
            "logits": logits,
            "reg": reg,
            "set_logits": set_logits,
            "feats": {
                "ea": ea, "eo": eo, "ec": ec,
                "oa": oa, "oo": oo, "oc": oc,
                "speed_emb": speed_emb,
                "h_shared": h_shared,
                "z0": z0, "z1": z1, "z2": z2,
                "fault_concat": fault_concat,
                "fused": fused,
                "w_shared": w_shared,
                "w0": w_mod_list[0], "w1": w_mod_list[1], "w2": w_mod_list[2],
                "bw_a0": bw_a_list[0], "bw_a1": bw_a_list[1], "bw_a2": bw_a_list[2],
                "bw_o0": bw_o_list[0], "bw_o1": bw_o_list[1], "bw_o2": bw_o_list[2],
                "bw_c0": bw_c_list[0], "bw_c1": bw_c_list[1], "bw_c2": bw_c_list[2],
            }
        }
