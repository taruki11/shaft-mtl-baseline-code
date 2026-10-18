# -*- coding: utf-8 -*-
"""Multi-band fault-wise fusion with factorized set decoder.

保留已经验证有效的 multiband + fault-wise 表征，
把最终组合决策改成：
- empty score
- 3 unary scores
- 3 pairwise scores
- 1 triple score
然后对 8 个组合统一打分。
"""
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


class FusionMultiBandFactorizedMTLNet(nn.Module):
    """
    输出兼容旧评估：
      out["logits"]      : [B,3] 由 factorized set posterior 边缘化得到
      out["reg"]         : [B,7]
    新增：
      out["comb_logits"] : [B,8]
      out["set_logits"]  : [B,8] 兼容别的脚本命名
      out["set_prob"]    : [B,8]
      out["aux_logits"]  : [B,3] 辅助 BCE
      out["factor_scores"]: dict(empty/unary/pair/triple)
    """
    def __init__(
        self,
        num_labels: int = 3,
        reg_dim: int = 7,
        cfg: Optional[FusionConfig] = None,
        fault_hidden: int = 128,
        num_bands: int = 3,
        factor_hidden: int = 128,
    ):
        super().__init__()
        assert num_labels == 3
        assert num_bands == 3

        self.cfg = cfg or FusionConfig()
        self.num_bands = num_bands

        self.backbone_accel = CNNBackboneMap(self.cfg.accel[1] - self.cfg.accel[0], base=self.cfg.base)
        self.backbone_orbit = CNNBackboneMap(self.cfg.orbit[1] - self.cfg.orbit[0], base=self.cfg.base)
        self.backbone_acoustic = CNNBackboneMap(self.cfg.acoustic[1] - self.cfg.acoustic[0], base=self.cfg.base)

        d = self.backbone_accel.out_dim
        self.d = d

        self.pool_accel = MultiBandPool(d, num_bands=num_bands)
        self.pool_orbit = MultiBandPool(d, num_bands=num_bands)
        self.pool_acoustic = MultiBandPool(d, num_bands=num_bands)

        self.shared_gate = nn.Sequential(
            nn.Linear(d * 3, self.cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.fusion_hidden, 3),
        )
        self.fault_mod_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d * 3, self.cfg.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.cfg.fusion_hidden, 3),
            )
            for _ in range(3)
        ])

        self.band_gate_a = nn.ModuleList([MLP(d * 3, fault_hidden, num_bands) for _ in range(3)])
        self.band_gate_o = nn.ModuleList([MLP(d * 3, fault_hidden, num_bands) for _ in range(3)])
        self.band_gate_c = nn.ModuleList([MLP(d * 3, fault_hidden, num_bands) for _ in range(3)])

        self.adapt_a = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3 * num_bands)])
        self.adapt_o = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3 * num_bands)])
        self.adapt_c = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3 * num_bands)])

        fused_dim = d * 4
        self.factor_mlp = nn.Sequential(
            nn.Linear(fused_dim, factor_hidden),
            nn.ReLU(inplace=True),
        )
        self.empty_head = nn.Linear(factor_hidden, 1)
        self.unary_head = nn.Linear(factor_hidden, 3)
        self.pair_head = nn.Linear(factor_hidden, 3)   # 01, 02, 12
        self.triple_head = nn.Linear(factor_hidden, 1)

        self.aux_bce_head = nn.Sequential(
            nn.Linear(fused_dim, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, 3),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(fused_dim, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, reg_dim),
        )

        table = []
        pair_mask = []
        triple_mask = []
        for sid in range(8):
            y0 = (sid >> 0) & 1
            y1 = (sid >> 1) & 1
            y2 = (sid >> 2) & 1
            table.append([y0, y1, y2])
            pair_mask.append([y0 * y1, y0 * y2, y1 * y2])
            triple_mask.append([y0 * y1 * y2])
        self.register_buffer("all_sets", torch.tensor(table, dtype=torch.float32))
        self.register_buffer("pair_mask", torch.tensor(pair_mask, dtype=torch.float32))
        self.register_buffer("triple_mask", torch.tensor(triple_mask, dtype=torch.float32))
        self.register_buffer("empty_mask", torch.tensor([[1.0 if sid == 0 else 0.0] for sid in range(8)], dtype=torch.float32))

    def split_modalities(self, x: torch.Tensor):
        if x.size(1) < self.cfg.acoustic[1]:
            raise ValueError(f"FusionMultiBandFactorizedMTLNet requires all10 input. Got C={x.size(1)}")
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
        return ea * keep[:, 0:1], eo * keep[:, 1:2], ec * keep[:, 2:3], keep

    def _masked_softmax3(self, logits3: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.tensor(-1e9, device=logits3.device, dtype=logits3.dtype)
        logits3 = torch.where(keep > 0.5, logits3, neg_inf)
        return F.softmax(logits3, dim=1)

    def _fuse_bands(self, tokens, band_gate, adapt_list, fault_idx):
        gate_in = torch.cat(tokens, dim=1)
        bw = F.softmax(band_gate[fault_idx](gate_in), dim=1)
        outs = [adapt_list[fault_idx * self.num_bands + j](tokens[j]) for j in range(self.num_bands)]
        z = bw[:, 0:1] * outs[0] + bw[:, 1:2] * outs[1] + bw[:, 2:3] * outs[2]
        return z, bw

    def _compose_set_logits(self, empty_score, unary, pair, triple):
        # empty_score: [B,1], unary:[B,3], pair:[B,3], triple:[B,1]
        comb = (
            empty_score * self.empty_mask.t() +
            unary @ self.all_sets.t() +
            pair @ self.pair_mask.t() +
            triple @ self.triple_mask.t()
        )
        return comb

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        xa, xo, xc = self.split_modalities(x)
        nm = (self.cfg.norm_mode or "none").lower().strip()
        if nm == "instance":
            xa = _instance_norm(xa)
            xo = _instance_norm(xo)
            xc = _instance_norm(xc)
        elif nm != "none":
            raise ValueError(f"Unknown norm_mode={self.cfg.norm_mode}")

        fa_map = self.backbone_accel(xa)
        fo_map = self.backbone_orbit(xo)
        fc_map = self.backbone_acoustic(xc)

        ea = float(self.cfg.fa_scale) * fa_map.mean(dim=2)
        eo = float(self.cfg.fo_scale) * fo_map.mean(dim=2)
        ec = float(self.cfg.fc_scale) * fc_map.mean(dim=2)
        ea, eo, ec, keep = self._apply_modality_dropout(ea, eo, ec)

        g_in = torch.cat([ea, eo, ec], dim=1)
        w_shared = self._masked_softmax3(self.shared_gate(g_in), keep)
        h_shared = w_shared[:, 0:1] * ea + w_shared[:, 1:2] * eo + w_shared[:, 2:3] * ec

        ta = [float(self.cfg.fa_scale) * t for t in self.pool_accel(fa_map)]
        to = [float(self.cfg.fo_scale) * t for t in self.pool_orbit(fo_map)]
        tc = [float(self.cfg.fc_scale) * t for t in self.pool_acoustic(fc_map)]

        z_list = []
        w_mod_list = []
        bw_a_list = []
        bw_o_list = []
        bw_c_list = []
        for k in range(3):
            w_mod = self._masked_softmax3(self.fault_mod_gates[k](torch.cat([ea, eo, ec], dim=1)), keep)
            za, bwa = self._fuse_bands(ta, self.band_gate_a, self.adapt_a, k)
            zo, bwo = self._fuse_bands(to, self.band_gate_o, self.adapt_o, k)
            zc, bwc = self._fuse_bands(tc, self.band_gate_c, self.adapt_c, k)
            zk = w_mod[:, 0:1] * za + w_mod[:, 1:2] * zo + w_mod[:, 2:3] * zc
            z_list.append(zk)
            w_mod_list.append(w_mod)
            bw_a_list.append(bwa)
            bw_o_list.append(bwo)
            bw_c_list.append(bwc)

        z0, z1, z2 = z_list
        fused = torch.cat([z0, z1, z2, h_shared], dim=1)

        factor_h = self.factor_mlp(fused)
        empty_score = self.empty_head(factor_h)
        unary = self.unary_head(factor_h)
        pair = self.pair_head(factor_h)
        triple = self.triple_head(factor_h)
        comb_logits = self._compose_set_logits(empty_score, unary, pair, triple)
        set_prob = F.softmax(comb_logits, dim=1)
        marg = torch.clamp(set_prob @ self.all_sets, min=1e-4, max=1.0 - 1e-4)
        logits = torch.log(marg / (1.0 - marg))

        aux_logits = self.aux_bce_head(fused)
        reg = self.reg_head(fused)

        out = {
            "logits": logits,
            "aux_logits": aux_logits,
            "reg": reg,
            "comb_logits": comb_logits,
            "set_logits": comb_logits,
            "set_prob": set_prob,
            "factor_scores": {
                "empty": empty_score,
                "unary": unary,
                "pair": pair,
                "triple": triple,
            },
            "feats": {
                "ea": ea, "eo": eo, "ec": ec,
                "h_shared": h_shared,
                "z0": z0, "z1": z1, "z2": z2,
                "factor_h": factor_h,
                "w_shared": w_shared,
                "w0": w_mod_list[0], "w1": w_mod_list[1], "w2": w_mod_list[2],
                "bw_a0": bw_a_list[0], "bw_a1": bw_a_list[1], "bw_a2": bw_a_list[2],
                "bw_o0": bw_o_list[0], "bw_o1": bw_o_list[1], "bw_o2": bw_o_list[2],
                "bw_c0": bw_c_list[0], "bw_c1": bw_c_list[1], "bw_c2": bw_c_list[2],
            }
        }
        return out
