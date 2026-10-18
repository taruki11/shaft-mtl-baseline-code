# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(num_channels: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        k = int(kernel_size)
        d = int(dilation)
        pad = ((k - 1) * d) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, dilation=d, bias=False)
        self.norm1 = _make_gn(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, dilation=d, bias=False)
        self.norm2 = _make_gn(channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return x + h


class TCNProbeNet(nn.Module):
    def __init__(
        self,
        num_labels: int = 3,
        reg_dim: int = 7,
        in_ch: int = 10,
        base_channels: int = 32,
        tcn_levels: int = 4,
        tcn_kernel_size: int = 5,
        tcn_dropout: float = 0.1,
        speed_feat_dim: int = 16,
        head_hidden: int = 64,
        head_dropout: float = 0.0,
        rpm_center: float = 1200.0,
        rpm_scale: float = 600.0,
        temporal_pool: int = 8,
        order_align: bool = False,
        order_align_ref_rpm: float = 1200.0,
        order_align_mix: float = 1.0,
        order_align_min_scale: float = 0.67,
        order_align_max_scale: float = 2.0,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.base_channels = int(base_channels)
        self.rpm_center = float(rpm_center)
        self.rpm_scale = float(max(rpm_scale, 1e-6))
        self.temporal_pool = max(int(temporal_pool), 1)
        self.order_align = bool(order_align)
        self.order_align_ref_rpm = float(max(order_align_ref_rpm, 1e-6))
        self.order_align_mix = float(order_align_mix)
        self.order_align_min_scale = float(max(order_align_min_scale, 1e-3))
        self.order_align_max_scale = float(max(order_align_max_scale, self.order_align_min_scale))

        self.stem = nn.Sequential(
            nn.Conv1d(self.in_ch, self.base_channels, kernel_size=7, padding=3, bias=False),
            _make_gn(self.base_channels),
            nn.GELU(),
        )

        blocks = []
        for i in range(int(tcn_levels)):
            blocks.append(
                TemporalBlock(
                    channels=self.base_channels,
                    kernel_size=int(tcn_kernel_size),
                    dilation=2 ** i,
                    dropout=float(tcn_dropout),
                )
            )
        self.tcn = nn.Sequential(*blocks)

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, speed_feat_dim),
            nn.ReLU(inplace=True),
        )

        pooled_dim = self.base_channels * 2
        feat_dim = pooled_dim + speed_feat_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
        )
        self.cls_head = nn.Linear(head_hidden, num_labels)
        self.reg_head = nn.Linear(head_hidden, reg_dim)

    def _pseudo_order_align(self, x: torch.Tensor, rpm: torch.Tensor | None) -> torch.Tensor:
        if (not self.order_align) or rpm is None:
            return x
        if x.dim() != 3:
            return x

        B, C, T = x.shape
        if T <= 4:
            return x

        rpm = rpm.float().view(-1)
        aligned = []
        for i in range(B):
            xi = x[i:i + 1]
            sp = float(rpm[i].item()) if i < rpm.numel() else self.order_align_ref_rpm
            scale = sp / self.order_align_ref_rpm
            scale = min(max(scale, self.order_align_min_scale), self.order_align_max_scale)
            new_len = int(round(T * scale))
            new_len = max(new_len, 8)
            yi = F.interpolate(xi, size=new_len, mode="linear", align_corners=False)
            if new_len == T:
                pass
            elif new_len > T:
                start = max((new_len - T) // 2, 0)
                yi = yi[:, :, start:start + T]
            else:
                pad_left = max((T - new_len) // 2, 0)
                pad_right = max(T - new_len - pad_left, 0)
                yi = F.pad(yi, (pad_left, pad_right), mode="replicate")
            aligned.append(yi)
        x_aligned = torch.cat(aligned, dim=0)
        mix = min(max(self.order_align_mix, 0.0), 1.0)
        if mix <= 0.0:
            return x
        if mix >= 1.0:
            return x_aligned
        return (1.0 - mix) * x + mix * x_aligned

    def _rpm_norm(self, rpm: torch.Tensor) -> torch.Tensor:
        return (rpm - self.rpm_center) / self.rpm_scale

    def forward(self, x: torch.Tensor, rpm: torch.Tensor | None = None):
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")

        x = self._pseudo_order_align(x, rpm)

        if self.temporal_pool > 1:
            x = F.avg_pool1d(x, kernel_size=self.temporal_pool, stride=self.temporal_pool, ceil_mode=False)

        h = self.stem(x)
        h = self.tcn(h)

        h_avg = h.mean(dim=-1)
        h_max = h.amax(dim=-1)
        h_shared = torch.cat([h_avg, h_max], dim=1)

        if rpm is None:
            rpm = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        rpm = rpm.float().view(-1, 1)
        rpm_norm = self._rpm_norm(rpm)
        speed_feat = self.speed_mlp(rpm_norm)

        fault_concat = torch.cat([h_shared, speed_feat], dim=1)
        fused = self.fusion_head(fault_concat)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused)

        feats = {
            "h_shared": h_shared,
            "fault_concat": fault_concat,
            "fused": fused,
            "speed_emb": speed_feat,
            "fa": h_shared,
            "fo": h_shared,
            "fc": h_shared,
            "ea": h_shared,
            "eo": h_shared,
            "ec": h_shared,
            "oa": h_shared,
            "oo": h_shared,
            "oc": h_shared,
        }
        return {"logits": logits, "reg": reg, "feats": feats}
