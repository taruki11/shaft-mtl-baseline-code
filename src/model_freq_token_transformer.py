# -*- coding: utf-8 -*-
"""Frequency-domain Token Transformer for rotating machinery fault diagnosis.

Core idea:
  Time-domain signal [B, C, T]  →  RFFT  →  frequency bins [B, F, C]
  Each frequency bin = one token (C channels → d_model features)
  RPM-conditioned encoding injected into every token
  Transformer encoder learns cross-frequency relationships (harmonics, sidebands)
  Self-attention weights are physically interpretable frequency coupling maps.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(num_channels: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class FreqTokenTransformer(nn.Module):
    """Frequency Token Transformer (FreqTokenFormer).

    Architecture:
      x [B, C, T] → RFFT(n_fft) → mag [B, C, F_bins]
      → permute → [B, F_bins, C]
      → token_proj: Linear(C → d_model) → [B, F_bins, d_model]
      → + freq_pe (learnable frequency position encoding)
      → + rpm_enc (RPM-conditioned encoding injected per token)
      → TransformerEncoder(n_layers, n_heads)
      → pool (avg or cls) → fusion_head → cls_head + reg_head
    """

    def __init__(
        self,
        num_labels: int = 3,
        reg_dim: int = 7,
        in_ch: int = 10,
        # ---- FFT ----
        n_fft: int = 128,
        # ---- Transformer ----
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 512,
        trans_dropout: float = 0.1,
        # ---- Speed ----
        speed_feat_dim: int = 32,
        rpm_center: float = 1200.0,
        rpm_scale: float = 600.0,
        # ---- Heads ----
        head_hidden: int = 128,
        head_dropout: float = 0.0,
        pool_mode: str = "avg",
        # ---- Order-aware frequency encoding ----
        use_order_rpm_encoding: bool = True,
        # ---- Token mode: "freq" (FFT) or "time" (patch) — for ablation ----
        token_mode: str = "freq",
        time_patch_len: int = 32,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.n_fft = int(n_fft)
        self.d_model = int(d_model)
        self.speed_feat_dim = int(speed_feat_dim)
        self.rpm_center = float(rpm_center)
        self.rpm_scale = float(max(rpm_scale, 1e-6))
        self.pool_mode = str(pool_mode).lower()
        self.use_order_rpm_encoding = bool(use_order_rpm_encoding)
        self.token_mode = str(token_mode).lower().strip()
        self.time_patch_len = int(time_patch_len)

        # FFT produces n_fft // 2 + 1 frequency bins (rfft)
        n_freq_bins = self.n_fft // 2 + 1
        self.n_freq_bins = n_freq_bins

        # ---- Token embedding ----
        if self.token_mode == "time":
            # Time-domain patch embedding: Conv1d as patch projection (ViT-style)
            self.time_patch_conv = nn.Conv1d(
                self.in_ch, self.d_model,
                kernel_size=self.time_patch_len,
                stride=self.time_patch_len,
            )
            # Position encoding for time tokens (same count as freq bins for fair comparison)
            self.freq_pe = nn.Parameter(torch.randn(1, n_freq_bins, self.d_model) * 0.02)
        else:
            # Frequency-domain: C-channel features per freq bin → d_model
            self.token_proj = nn.Linear(self.in_ch, self.d_model)
            # ---- Learnable frequency position encoding ----
            self.freq_pe = nn.Parameter(torch.randn(1, n_freq_bins, self.d_model) * 0.02)

        # ---- RPM encoding ----
        speed_in_dim = 4 if self.use_order_rpm_encoding else 1
        self.speed_mlp = nn.Sequential(
            nn.Linear(speed_in_dim, self.speed_feat_dim),
            nn.ReLU(inplace=True),
        )
        # Project speed feat to d_model for token addition
        self.speed_proj = nn.Linear(self.speed_feat_dim, self.d_model)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(trans_dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))

        # ---- Fusion head (h_pooled + speed_feat → fused) ----
        fusion_in_dim = self.d_model + self.speed_feat_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, int(head_hidden)),
            nn.LayerNorm(int(head_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
        )

        # ---- Output heads ----
        self.cls_head = nn.Linear(int(head_hidden), int(num_labels))
        self.reg_head = nn.Linear(int(head_hidden), int(reg_dim))

        # ---- CLS token (if pool_mode == "cls") ----
        if self.pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # ---- Init ----
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _rpm_norm(self, rpm: torch.Tensor) -> torch.Tensor:
        return (rpm - self.rpm_center) / self.rpm_scale

    def _speed_basis(self, rpm_norm: torch.Tensor) -> torch.Tensor:
        """Build speed basis: linear or rich (with harmonics)."""
        if self.use_order_rpm_encoding:
            # Rich encoding: rpm_norm, sin(pi*rpm_norm), cos(pi*rpm_norm), rpm^2
            r = rpm_norm.view(-1, 1)
            return torch.cat([r, torch.sin(math.pi * r), torch.cos(math.pi * r), r * r], dim=1)
        return rpm_norm.view(-1, 1)

    def _to_frequency_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Convert time-domain signal to frequency-domain tokens via STFT averaging.

        Uses STFT with sliding window over the full signal → magnitude → average
        over time → each frequency bin becomes one token. This captures the
        frequency content of the ENTIRE signal, not just the head.

        Args:
            x: [B, C, T] time-domain signal, T can be any length
        Returns:
            tokens: [B, n_freq_bins, d_model]
        """
        B, C, T = x.shape
        hop_length = max(1, self.n_fft // 2)
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)

        # Reshape: [B, C, T] → [B*C, T] for single STFT call
        x_flat = x.reshape(B * C, T)

        X = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )  # [B*C, F, n_frames]

        # Magnitude → average over time frames → [B*C, F]
        mag = X.abs().mean(dim=-1)

        # Reshape back: [B*C, F] → [B, C, F] → [B, F, C] → [B, F, d_model]
        x_mag = mag.reshape(B, C, self.n_freq_bins)  # [B, C, F]
        tokens = x_mag.permute(0, 2, 1)  # [B, F, C]
        tokens = self.token_proj(tokens)  # [B, F, d_model]

        return tokens

    def _to_time_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Convert time-domain signal to time-domain patch tokens (ablation mode).

        Uses Conv1d patch embedding (ViT-style) + adaptive pooling to match
        the same token count as frequency mode for fair comparison.

        Args:
            x: [B, C, T] time-domain signal
        Returns:
            tokens: [B, n_freq_bins, d_model]
        """
        # Conv1d: [B, C, T] → [B, d_model, T//patch_len]
        tokens = self.time_patch_conv(x)

        # Adaptive pool to fixed token count (same as freq mode for fair comparison)
        tokens = F.adaptive_avg_pool1d(tokens, self.n_freq_bins)  # [B, d_model, n_freq_bins]
        tokens = tokens.transpose(1, 2)  # [B, n_freq_bins, d_model]
        return tokens

    def forward(self, x: torch.Tensor, rpm: torch.Tensor | None = None):
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")

        if rpm is None:
            rpm = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        rpm = rpm.float().view(-1, 1)

        # ---- 1. Tokenization (freq or time) ----
        if self.token_mode == "time":
            tokens = self._to_time_tokens(x)  # [B, n_freq_bins, d_model]
        else:
            tokens = self._to_frequency_tokens(x)  # [B, n_freq_bins, d_model]

        # ---- 2. Speed encoding ----
        rpm_norm = self._rpm_norm(rpm)
        speed_feat = self.speed_mlp(self._speed_basis(rpm_norm))  # [B, speed_feat_dim]
        rpm_emb = self.speed_proj(speed_feat).unsqueeze(1)  # [B, 1, d_model]

        # ---- 3. Positional + RPM encoding ----
        tokens = tokens + self.freq_pe + rpm_emb

        # ---- 4. CLS token (optional) ----
        if self.pool_mode == "cls":
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, 1+F, d_model]

        # ---- 5. Transformer encoder ----
        encoded = self.transformer(tokens)  # [B, N, d_model]

        # ---- 6. Pool ----
        if self.pool_mode == "cls":
            h_pooled = encoded[:, 0, :]  # CLS token
        else:
            h_pooled = encoded.mean(dim=1)  # avg pooling

        # ---- 7. Fusion + prediction ----
        fault_concat = torch.cat([h_pooled, speed_feat], dim=1)
        fused = self.fusion_head(fault_concat)
        logits = self.cls_head(fused)
        reg = self.reg_head(fused)

        # ---- 8. Build feats dict (compatible with existing training scripts) ----
        feats = {
            "h_shared": h_pooled,
            "fault_concat": fault_concat,
            "fused": fused,
            "speed_emb": speed_feat,
            "diag_feat": h_pooled,
            "order_feat": h_pooled,
            "orbit_img": h_pooled,
            "fa": h_pooled, "fo": h_pooled, "fc": h_pooled,
            "ea": h_pooled, "eo": h_pooled, "ec": h_pooled,
            "oa": h_pooled, "oo": h_pooled, "oc": h_pooled,
        }

        return {"logits": logits, "reg": reg, "feats": feats}
