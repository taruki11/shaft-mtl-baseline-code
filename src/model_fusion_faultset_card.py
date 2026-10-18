import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


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
        return self.net(x).squeeze(-1)


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


class FusionFaultSetCardMTLNet(nn.Module):
    def __init__(
        self,
        num_labels: int = 3,
        reg_dim: int = 7,
        cfg: Optional[FusionConfig] = None,
        fault_hidden: int = 128,
        card_hidden: int = 64,
    ):
        super().__init__()
        assert num_labels == 3, "当前固定 3 标签"
        self.cfg = cfg or FusionConfig()

        self.backbone_accel = CNNBackbone(self.cfg.accel[1] - self.cfg.accel[0], base=self.cfg.base)
        self.backbone_orbit = CNNBackbone(self.cfg.orbit[1] - self.cfg.orbit[0], base=self.cfg.base)
        self.backbone_acoustic = CNNBackbone(self.cfg.acoustic[1] - self.cfg.acoustic[0], base=self.cfg.base)

        d = self.backbone_accel.out_dim
        assert d == self.backbone_orbit.out_dim == self.backbone_acoustic.out_dim

        self.shared_gate = nn.Sequential(
            nn.Linear(d * 3, self.cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.fusion_hidden, 3),
        )

        self.fault_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d * 3, self.cfg.fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.cfg.fusion_hidden, 3),
            )
            for _ in range(3)
        ])

        self.adapt_a = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3)])
        self.adapt_o = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3)])
        self.adapt_c = nn.ModuleList([MLP(d, fault_hidden, d) for _ in range(3)])

        self.bce_head = nn.Sequential(
            nn.Linear(d * 4, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, 3),
        )
        self.card_head = nn.Sequential(
            nn.Linear(d * 4, card_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(card_hidden, 4),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d * 4, fault_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fault_hidden, reg_dim),
        )

    def split_modalities(self, x: torch.Tensor):
        if x.size(1) < self.cfg.acoustic[1]:
            raise ValueError(f"FusionFaultSetCardMTLNet requires all10 input. Got C={x.size(1)}")
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

    def _build_fault_token(self, idx: int, ea, eo, ec, keep):
        g_in = torch.cat([ea, eo, ec], dim=1)
        g = self.fault_gates[idx](g_in)
        w = self._masked_softmax3(g, keep)
        za = self.adapt_a[idx](ea)
        zo = self.adapt_o[idx](eo)
        zc = self.adapt_c[idx](ec)
        z = w[:, 0:1] * za + w[:, 1:2] * zo + w[:, 2:3] * zc
        return z, w

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        xa, xo, xc = self.split_modalities(x)
        nm = (self.cfg.norm_mode or "none").lower().strip()
        if nm == "instance":
            xa = _instance_norm(xa)
            xo = _instance_norm(xo)
            xc = _instance_norm(xc)
        elif nm != "none":
            raise ValueError(f"Unknown norm_mode={self.cfg.norm_mode}")

        ea = self.backbone_accel(xa)
        eo = self.backbone_orbit(xo)
        ec = self.backbone_acoustic(xc)

        ea = float(self.cfg.fa_scale) * ea
        eo = float(self.cfg.fo_scale) * eo
        ec = float(self.cfg.fc_scale) * ec

        ea, eo, ec, keep = self._apply_modality_dropout(ea, eo, ec)

        g_in = torch.cat([ea, eo, ec], dim=1)
        g_shared = self.shared_gate(g_in)
        w_shared = self._masked_softmax3(g_shared, keep)
        h_shared = w_shared[:, 0:1] * ea + w_shared[:, 1:2] * eo + w_shared[:, 2:3] * ec

        z0, w0 = self._build_fault_token(0, ea, eo, ec, keep)
        z1, w1 = self._build_fault_token(1, ea, eo, ec, keep)
        z2, w2 = self._build_fault_token(2, ea, eo, ec, keep)

        fused = torch.cat([z0, z1, z2, h_shared], dim=1)
        logits = self.bce_head(fused)
        card_logits = self.card_head(fused)
        kprob = F.softmax(card_logits, dim=1)
        reg = self.reg_head(fused)

        return {
            "logits": logits,
            "reg": reg,
            "card_logits": card_logits,
            "kprob": kprob,
            "feats": {
                "ea": ea, "eo": eo, "ec": ec,
                "h_shared": h_shared,
                "z0": z0, "z1": z1, "z2": z2,
                "w_shared": w_shared,
                "w0": w0, "w1": w1, "w2": w2,
            }
        }
