# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ncps.torch import CfC
except Exception as e:
    raise ImportError('Failed to import ncps.torch.CfC. Install with: pip install -U ncps') from e


class CFCProbeLiteNet(nn.Module):
    def __init__(
        self,
        num_labels: int = 3,
        reg_dim: int = 7,
        in_ch: int = 10,
        input_proj: int = 16,
        cfc_units: int = 24,
        cfc_mode: str = 'default',
        cfc_backbone_units: int = 32,
        cfc_backbone_layers: int = 1,
        cfc_backbone_dropout: float = 0.0,
        cfc_mixed_memory: bool = False,
        speed_feat_dim: int = 8,
        head_hidden: int = 32,
        head_dropout: float = 0.0,
        rpm_center: float = 1200.0,
        rpm_scale: float = 600.0,
        temporal_pool: int = 8,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.rpm_center = float(rpm_center)
        self.rpm_scale = float(max(rpm_scale, 1e-6))
        self.temporal_pool = max(int(temporal_pool), 1)

        self.input_proj = nn.Sequential(
            nn.Linear(self.in_ch + 1, input_proj),
            nn.LayerNorm(input_proj),
            nn.GELU(),
        )

        self.cfc = CfC(
            input_size=input_proj,
            units=cfc_units,
            return_sequences=False,
            batch_first=True,
            mixed_memory=bool(cfc_mixed_memory),
            mode=cfc_mode,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            backbone_dropout=cfc_backbone_dropout,
        )

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, speed_feat_dim),
            nn.ReLU(inplace=True),
        )

        feat_dim = cfc_units + speed_feat_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
        )
        self.cls_head = nn.Linear(head_hidden, num_labels)
        self.reg_head = nn.Linear(head_hidden, reg_dim)

    def _rpm_norm(self, rpm: torch.Tensor) -> torch.Tensor:
        return (rpm - self.rpm_center) / self.rpm_scale

    def forward(self, x: torch.Tensor, rpm: torch.Tensor | None = None):
        if x.dim() != 3:
            raise ValueError(f'Expected x with shape [B, C, T], got {tuple(x.shape)}')

        if self.temporal_pool > 1:
            x = F.avg_pool1d(x, kernel_size=self.temporal_pool, stride=self.temporal_pool, ceil_mode=False)

        x = x.transpose(1, 2).contiguous()

        if rpm is None:
            rpm = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        rpm = rpm.float().view(-1, 1)
        rpm_norm = self._rpm_norm(rpm)

        rpm_seq = rpm_norm.unsqueeze(1).expand(-1, x.size(1), -1)
        z = torch.cat([x, rpm_seq], dim=-1)
        z = self.input_proj(z)

        out, _ = self.cfc(z)
        h_last = out[:, -1, :] if out.dim() == 3 else out

        speed_feat = self.speed_mlp(rpm_norm)
        fault_concat = torch.cat([h_last, speed_feat], dim=1)
        fused = self.fusion_head(fault_concat)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused)

        feats = {
            'h_shared': h_last,
            'fault_concat': fault_concat,
            'fused': fused,
            'speed_emb': speed_feat,
            'fa': h_last, 'fo': h_last, 'fc': h_last,
            'ea': h_last, 'eo': h_last, 'ec': h_last,
            'oa': h_last, 'oo': h_last, 'oc': h_last,
        }
        return {'logits': logits, 'reg': reg, 'feats': feats}
