# -*- coding: utf-8 -*-
"""src/model_level2_dtfe.py

Level2 ("2026-style") model for this project:

Backbone naming (paper-friendly):
  - DTFE: Deep Temporal Feature Extractor (multi-scale residual 1D encoder)
  - Attention-guided CNN: SE blocks inside residual blocks
  - Dual-Stream: optional Frequency-Domain Pathway (rFFT magnitude)
  - CNN-Transformer Hybrid: optional lightweight Transformer encoder head

Fusion:
  - 3-branch (accel/orbit/acoustic) gated fusion on all10 layout
  - Optional speed-conditioned fusion using rpm embedding (domain-conditioned)

I/O:
  x: [B,C,T] (all10: C=10)
  speed: [B] float/long, optional
  return: logits [B,3], reg [B,7]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _instance_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample, per-channel normalization over time dimension."""
    mean = x.mean(dim=2, keepdim=True)
    var = (x - mean).pow(2).mean(dim=2, keepdim=True)
    std = (var + eps).sqrt()
    return (x - mean) / std


class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        b, c, _ = x.shape
        s = self.pool(x).view(b, c)
        w = self.fc(s).view(b, c, 1)
        return x * w


class ConvBNAct1D(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 7, s: int = 1, p: Optional[int] = None, act: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualSEBlock1D(nn.Module):
    def __init__(self, channels: int, k: int = 7, reduction: int = 16, dilation: int = 1):
        super().__init__()
        p = (k // 2) * dilation
        self.f = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=k, padding=p, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=k, padding=p, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se = SE1D(channels, reduction=reduction)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.f(x)
        y = self.se(y)
        return self.act(x + y)


class MultiScaleBlock1D(nn.Module):
    """Light multi-scale temporal block.

    It is intentionally simple/stable (for your dataset size) but looks "2026" in the paper:
      - parallel convs with different receptive fields
      - then fuse with 1x1 conv
    """

    def __init__(self, channels: int, ks: Tuple[int, int, int] = (3, 5, 7)):
        super().__init__()
        c = channels
        self.branches = nn.ModuleList([
            ConvBNAct1D(c, c, k=ks[0]),
            ConvBNAct1D(c, c, k=ks[1]),
            ConvBNAct1D(c, c, k=ks[2]),
        ])
        self.fuse = ConvBNAct1D(c * 3, c, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [b(x) for b in self.branches]
        y = torch.cat(ys, dim=1)
        return self.fuse(y)


class FFTPathway(nn.Module):
    """Frequency-domain pathway using rFFT magnitude (log1p).

    Input:  x_time (B,C,T)
    Output: z_freq (B, out_dim)
    """

    def __init__(self, in_channels: int, n_fft_bins: int = 512, out_dim: int = 128):
        super().__init__()
        self.n_fft_bins = int(n_fft_bins)
        self.proj = nn.Sequential(
            nn.Linear(in_channels * self.n_fft_bins, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, out_dim),
        )

    def forward(self, x_time: torch.Tensor) -> torch.Tensor:
        # rFFT -> (B,C,F)
        X = torch.fft.rfft(x_time, dim=-1)
        mag = torch.abs(X)
        # select low-frequency bins
        f = mag.size(-1)
        k = min(self.n_fft_bins, f)
        mag = mag[..., :k]
        if k < self.n_fft_bins:
            # pad to fixed length for linear layer
            pad = self.n_fft_bins - k
            mag = F.pad(mag, (0, pad), mode="constant", value=0.0)
        mag = torch.log1p(mag)
        b, c, _ = mag.shape
        mag = mag.reshape(b, c * self.n_fft_bins)
        return self.proj(mag)


class HybridTransformerHead(nn.Module):
    """Lightweight Transformer head on top of CNN features (CNN-Transformer hybrid)."""

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B,d,T') -> tokens (B,T',d)
        x = feat.transpose(1, 2)
        x = self.enc(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)


class DTFEBackbone(nn.Module):
    """DTFE backbone: multi-scale residual encoder + optional dual-stream + optional hybrid head."""

    def __init__(
        self,
        in_ch: int,
        base: int = 64,
        out_dim: int = 256,
        use_se: bool = True,
        use_multiscale: bool = True,
        use_fft: bool = True,
        n_fft_bins: int = 512,
        use_hybrid: bool = False,
    ):
        super().__init__()
        self.use_fft = bool(use_fft)
        self.use_hybrid = bool(use_hybrid)

        # stem (downsample)
        self.stem = nn.Sequential(
            ConvBNAct1D(in_ch, base, k=11, s=2),
            ConvBNAct1D(base, base * 2, k=9, s=2),
            ConvBNAct1D(base * 2, base * 2, k=7, s=1),
        )
        c = base * 2

        blocks = []
        if use_multiscale:
            blocks.append(MultiScaleBlock1D(c))
        # residual encoder (SE inside)
        if use_se:
            blocks += [ResidualSEBlock1D(c, k=7, dilation=1), ResidualSEBlock1D(c, k=5, dilation=2)]
        else:
            # fallback: no SE, keep residual but plain
            blocks += [ResidualSEBlock1D(c, k=7, reduction=10**9, dilation=1), ResidualSEBlock1D(c, k=5, reduction=10**9, dilation=2)]

        self.encoder = nn.Sequential(*blocks)
        self.proj_feat = ConvBNAct1D(c, 128, k=1, p=0)

        self.hybrid = HybridTransformerHead(d_model=128, nhead=4, num_layers=2) if self.use_hybrid else None
        self.pool = nn.AdaptiveAvgPool1d(1)

        if self.use_fft:
            self.fft = FFTPathway(in_channels=in_ch, n_fft_bins=n_fft_bins, out_dim=128)
            self.fuse = nn.Sequential(
                nn.Linear(128 + 128, out_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1),
            )
        else:
            self.fuse = nn.Sequential(
                nn.Linear(128, out_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1),
            )

        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        feat = self.proj_feat(self.encoder(self.stem(x)))  # (B,128,T')
        if self.use_hybrid:
            z_time = self.hybrid(feat)  # (B,128)
        else:
            z_time = self.pool(feat).squeeze(-1)  # (B,128)

        if self.use_fft:
            z_freq = self.fft(x)  # (B,128)
            z = self.fuse(torch.cat([z_time, z_freq], dim=1))
        else:
            z = self.fuse(z_time)
        return z


@dataclass
class Level2Config:
    # backbone
    base: int = 64
    emb_dim: int = 256
    use_se: bool = True
    use_multiscale: bool = True
    use_fft: bool = True
    n_fft_bins: int = 512
    use_hybrid: bool = False

    # fusion
    fusion_hidden: int = 256
    norm_mode: str = "instance"  # none|instance
    modality_dropout: float = 0.0

    # speed conditioning
    use_speedcond: bool = True
    speed_emb_dim: int = 32
    speed_values: Tuple[int, ...] = (600, 900, 1200, 1300, 1500, 1800)

    # channel splits for all10
    accel: Tuple[int, int] = (0, 6)
    orbit: Tuple[int, int] = (6, 8)
    acoustic: Tuple[int, int] = (8, 10)


class Level2SpeedCondFusionMTLNet(nn.Module):
    """Level2 3-branch DTFE + (optional) speed-conditioned gated fusion."""

    def __init__(self, num_labels: int = 3, reg_dim: int = 7, cfg: Optional[Level2Config] = None):
        super().__init__()
        self.cfg = cfg or Level2Config()

        a0, a1 = self.cfg.accel
        o0, o1 = self.cfg.orbit
        c0, c1 = self.cfg.acoustic
        self.backbone_accel = DTFEBackbone(
            in_ch=a1 - a0,
            base=self.cfg.base,
            out_dim=self.cfg.emb_dim,
            use_se=self.cfg.use_se,
            use_multiscale=self.cfg.use_multiscale,
            use_fft=self.cfg.use_fft,
            n_fft_bins=self.cfg.n_fft_bins,
            use_hybrid=self.cfg.use_hybrid,
        )
        self.backbone_orbit = DTFEBackbone(
            in_ch=o1 - o0,
            base=self.cfg.base,
            out_dim=self.cfg.emb_dim,
            use_se=self.cfg.use_se,
            use_multiscale=self.cfg.use_multiscale,
            use_fft=self.cfg.use_fft,
            n_fft_bins=self.cfg.n_fft_bins,
            use_hybrid=self.cfg.use_hybrid,
        )
        self.backbone_acoustic = DTFEBackbone(
            in_ch=c1 - c0,
            base=self.cfg.base,
            out_dim=self.cfg.emb_dim,
            use_se=self.cfg.use_se,
            use_multiscale=self.cfg.use_multiscale,
            use_fft=self.cfg.use_fft,
            n_fft_bins=self.cfg.n_fft_bins,
            use_hybrid=self.cfg.use_hybrid,
        )
        d = self.cfg.emb_dim

        # speed embedding (domain-conditioned)
        self.use_speedcond = bool(self.cfg.use_speedcond)
        if self.use_speedcond:
            self.speed_table = torch.tensor(list(self.cfg.speed_values), dtype=torch.long)
            self.speed_emb = nn.Embedding(len(self.cfg.speed_values), self.cfg.speed_emb_dim)
            # FiLM per modality
            self.film_a = nn.Linear(self.cfg.speed_emb_dim, d * 2)
            self.film_o = nn.Linear(self.cfg.speed_emb_dim, d * 2)
            self.film_c = nn.Linear(self.cfg.speed_emb_dim, d * 2)
            gate_in_dim = d * 3 + self.cfg.speed_emb_dim
        else:
            self.speed_table = None
            self.speed_emb = None
            self.film_a = self.film_o = self.film_c = None
            gate_in_dim = d * 3

        # gating network
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, self.cfg.fusion_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.cfg.fusion_hidden, 3),
        )

        self.cls = nn.Linear(d, num_labels)
        self.reg = nn.Linear(d, reg_dim)

    def split_modalities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.size(1) < self.cfg.acoustic[1]:
            raise ValueError(f"Level2SpeedCondFusionMTLNet requires all10 input (C>=10). Got C={x.size(1)}")
        a0, a1 = self.cfg.accel
        o0, o1 = self.cfg.orbit
        c0, c1 = self.cfg.acoustic
        return x[:, a0:a1, :], x[:, o0:o1, :], x[:, c0:c1, :]

    def _speed_to_id(self, speed: torch.Tensor) -> torch.Tensor:
        # speed: (B,) float or long
        s = speed
        if s.dtype.is_floating_point:
            s = s.round().long()
        else:
            s = s.long()

        table = self.speed_table.to(s.device)
        # nearest-match (robust to float)
        diff = (s.unsqueeze(1) - table.unsqueeze(0)).abs()
        idx = diff.argmin(dim=1)
        return idx

    def _apply_film(self, z: torch.Tensor, film: nn.Linear, e: torch.Tensor) -> torch.Tensor:
        gb = film(e)  # (B, 2d)
        g, b = gb.chunk(2, dim=1)
        g = torch.tanh(g)  # stabilize
        return (1.0 + g) * z + b

    def forward(self, x: torch.Tensor, speed: Optional[torch.Tensor] = None):
        xa, xo, xc = self.split_modalities(x)

        nm = (self.cfg.norm_mode or "none").lower().strip()
        if nm == "instance":
            xa = _instance_norm(xa)
            xo = _instance_norm(xo)
            xc = _instance_norm(xc)
        elif nm != "none":
            raise ValueError(f"Unknown norm_mode={self.cfg.norm_mode}. Use none|instance")

        ea = self.backbone_accel(xa)
        eo = self.backbone_orbit(xo)
        ec = self.backbone_acoustic(xc)

        # speed conditioning
        if self.use_speedcond and (speed is not None):
            sid = self._speed_to_id(speed)
            e = self.speed_emb(sid)
            ea = self._apply_film(ea, self.film_a, e)
            eo = self._apply_film(eo, self.film_o, e)
            ec = self._apply_film(ec, self.film_c, e)
            g_in = torch.cat([ea, eo, ec, e], dim=1)
        else:
            g_in = torch.cat([ea, eo, ec], dim=1)

        w = F.softmax(self.gate(g_in), dim=1)  # (B,3)
        h = w[:, 0:1] * ea + w[:, 1:2] * eo + w[:, 2:3] * ec
        logits = self.cls(h)
        reg = self.reg(h)
        return logits, reg
# ---- backward-compatible alias (keep training script import unchanged) ----
Level2DTFESpeedCondFusion = Level2SpeedCondFusionMTLNet
__all__ = ["Level2Config", "Level2DTFESpeedCondFusion", "Level2SpeedCondFusionMTLNet", "masked_scaled_huber_loss"]


def masked_scaled_huber_loss(pred, target, y_multi, scale=None, beta=1.0):
    """Same loss utility as src/model.py, kept here for convenience."""
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

    loss_mat = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction="none")
    loss_mat = loss_mat * m
    denom = m.sum().clamp_min(1.0)
    return loss_mat.sum() / denom
