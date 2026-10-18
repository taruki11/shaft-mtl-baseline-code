# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, Optional

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
        speed_basis: str = "linear",
        speed_film: int = 0,
        speed_film_strength: float = 0.1,
        causal_graph_alpha: float = 0.0,
        order_resample_enable: int = 0,
        order_ref_rpm: float = 1200.0,
        order_resample_min_scale: float = 0.5,
        order_resample_max_scale: float = 1.5,
        order_resample_pad_mode: str = "replicate",
        diag_feat_enable: int = 0,
        diag_feat_hidden: int = 64,
        diag_fft_bins: int = 8,
        diag_fft_size: int = 1024,
        diag_feat_zero_init: int = 1,
        order_feat_enable: int = 0,
        order_feat_hidden: int = 32,
        order_feat_max_order: int = 8,
        order_feat_fft_size: int = 8192,
        order_feat_fs: float = 25000.0,
        order_feat_band: int = 2,
        order_feat_scale: float = 1.0,
        order_feat_zero_init: int = 1,
        orbit_img_enable: int = 0,
        orbit_img_size: int = 32,
        orbit_img_hidden: int = 32,
        orbit_img_clip: float = 3.0,
        orbit_img_blur: int = 1,
        orbit_img_zero_init: int = 0,
        orbit_img_fusion_enable: int = 1,
        orbit_img_logit_enable: int = 0,
        orbit_img_logit_labels: str = "",
        orbit_img_logit_scale: float = 1.0,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.base_channels = int(base_channels)
        self.rpm_center = float(rpm_center)
        self.rpm_scale = float(max(rpm_scale, 1e-6))
        self.temporal_pool = max(int(temporal_pool), 1)
        self.speed_basis = str(speed_basis).lower()
        if self.speed_basis not in {"linear", "rich"}:
            raise ValueError(f"Unsupported speed_basis={speed_basis!r}; expected 'linear' or 'rich'.")
        self.speed_film = bool(int(speed_film))
        self.speed_film_strength = float(speed_film_strength)
        self.causal_graph_alpha = float(causal_graph_alpha)
        self.order_resample_enable = bool(int(order_resample_enable))
        self.order_ref_rpm = float(max(order_ref_rpm, 1e-6))
        self.order_resample_min_scale = float(max(order_resample_min_scale, 1e-3))
        self.order_resample_max_scale = float(max(order_resample_max_scale, self.order_resample_min_scale))
        self.order_resample_pad_mode = str(order_resample_pad_mode or "replicate").lower()
        if self.order_resample_pad_mode not in {"replicate", "reflect", "zero"}:
            raise ValueError(
                f"Unsupported order_resample_pad_mode={order_resample_pad_mode!r}; "
                "expected replicate, reflect, or zero."
            )
        self.diag_feat_enable = bool(int(diag_feat_enable))
        self.diag_fft_bins = max(int(diag_fft_bins), 1)
        self.diag_fft_size = max(int(diag_fft_size), 128)
        self.diag_feat_zero_init = bool(int(diag_feat_zero_init))
        self.order_feat_enable = bool(int(order_feat_enable))
        self.order_feat_max_order = max(int(order_feat_max_order), 1)
        self.order_feat_fft_size = max(int(order_feat_fft_size), 512)
        self.order_feat_fs = float(max(order_feat_fs, 1e-6))
        self.order_feat_band = max(int(order_feat_band), 0)
        self.order_feat_scale = float(order_feat_scale)
        self.order_feat_zero_init = bool(int(order_feat_zero_init))
        self.orbit_img_enable = bool(int(orbit_img_enable))
        self.orbit_img_size = max(int(orbit_img_size), 8)
        self.orbit_img_clip = float(max(orbit_img_clip, 0.5))
        self.orbit_img_blur = max(int(orbit_img_blur), 0)
        self.orbit_img_zero_init = bool(int(orbit_img_zero_init))
        self.orbit_img_fusion_enable = bool(int(orbit_img_fusion_enable)) and self.orbit_img_enable
        self.orbit_img_logit_scale = float(orbit_img_logit_scale)
        self.register_buffer("causal_adj", torch.eye(self.in_ch), persistent=False)

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
        self.tcn = nn.ModuleList(blocks)

        speed_in_dim = 4 if self.speed_basis == "rich" else 1
        self.speed_mlp = nn.Sequential(
            nn.Linear(speed_in_dim, speed_feat_dim),
            nn.ReLU(inplace=True),
        )
        if self.speed_film:
            self.film_stem = nn.Linear(speed_feat_dim, self.base_channels * 2)
            self.film_blocks = nn.ModuleList(
                [nn.Linear(speed_feat_dim, self.base_channels * 2) for _ in range(int(tcn_levels))]
            )
            for layer in [self.film_stem, *self.film_blocks]:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        pooled_dim = self.base_channels * 2
        diag_out_dim = int(diag_feat_hidden) if self.diag_feat_enable else 0
        if self.diag_feat_enable:
            diag_in_dim = self.in_ch * (8 + self.diag_fft_bins) + 6
            self.diag_mlp = nn.Sequential(
                nn.LayerNorm(diag_in_dim),
                nn.Linear(diag_in_dim, int(diag_feat_hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(head_dropout)),
            )

        order_out_dim = int(order_feat_hidden) if self.order_feat_enable else 0
        if self.order_feat_enable:
            order_in_dim = self.in_ch * self.order_feat_max_order
            self.order_feat_mlp = nn.Sequential(
                nn.LayerNorm(order_in_dim),
                nn.Linear(order_in_dim, int(order_feat_hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(head_dropout)),
            )

        orbit_out_dim = int(orbit_img_hidden) if self.orbit_img_enable else 0
        self.orbit_geom_dim = 12
        if self.orbit_img_enable:
            self.orbit_img_encoder = nn.Sequential(
                nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
                _make_gn(8),
                nn.GELU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                _make_gn(16),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
                _make_gn(16),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.orbit_img_mlp = nn.Sequential(
                nn.LayerNorm(16 + self.orbit_geom_dim),
                nn.Linear(16 + self.orbit_geom_dim, int(orbit_img_hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(head_dropout)),
            )

        orbit_fusion_dim = orbit_out_dim if self.orbit_img_fusion_enable else 0
        feat_dim = pooled_dim + speed_feat_dim + diag_out_dim + order_out_dim + orbit_fusion_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
        )
        with torch.no_grad():
            offset = pooled_dim + speed_feat_dim
            if self.diag_feat_enable:
                if self.diag_feat_zero_init:
                    self.fusion_head[0].weight[:, offset : offset + diag_out_dim].zero_()
                offset += diag_out_dim
            if self.order_feat_enable and self.order_feat_zero_init:
                self.fusion_head[0].weight[:, offset : offset + order_out_dim].zero_()
            offset += order_out_dim
            if self.orbit_img_fusion_enable and self.orbit_img_zero_init:
                self.fusion_head[0].weight[:, offset : offset + orbit_fusion_dim].zero_()
        self.cls_head = nn.Linear(head_hidden, num_labels)
        self.reg_head = nn.Linear(head_hidden, reg_dim)
        logit_mask = torch.zeros(int(num_labels), dtype=torch.float32)
        for tok in str(orbit_img_logit_labels or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            idx = int(tok)
            if 0 <= idx < int(num_labels):
                logit_mask[idx] = 1.0
        self.register_buffer("orbit_img_logit_mask", logit_mask, persistent=False)
        self.orbit_img_logit_enable = bool(int(orbit_img_logit_enable)) and self.orbit_img_enable and bool(logit_mask.any())
        if self.orbit_img_logit_enable:
            self.orbit_img_logit_head = nn.Linear(orbit_out_dim, num_labels)
            nn.init.zeros_(self.orbit_img_logit_head.weight)
            nn.init.zeros_(self.orbit_img_logit_head.bias)

    def _rpm_norm(self, rpm: torch.Tensor) -> torch.Tensor:
        return (rpm - self.rpm_center) / self.rpm_scale

    def _speed_basis(self, rpm_norm: torch.Tensor) -> torch.Tensor:
        if self.speed_basis == "linear":
            return rpm_norm
        r = torch.clamp(rpm_norm, min=-2.0, max=2.0)
        return torch.cat([r, r * r, torch.sin(math.pi * r), torch.cos(math.pi * r)], dim=1)

    def _apply_speed_film(self, h: torch.Tensor, speed_feat: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        gamma_beta = layer(speed_feat).to(dtype=h.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        strength = self.speed_film_strength
        return h * (1.0 + strength * gamma) + strength * beta

    @torch.no_grad()
    def set_causal_adj(self, adj: torch.Tensor) -> None:
        if adj.shape != (self.in_ch, self.in_ch):
            raise ValueError(f"Expected causal adj shape {(self.in_ch, self.in_ch)}, got {tuple(adj.shape)}")
        adj = adj.to(device=self.causal_adj.device, dtype=self.causal_adj.dtype)
        row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.causal_adj.copy_(adj / row_sum)

    def _apply_causal_graph(self, x: torch.Tensor) -> torch.Tensor:
        alpha = float(self.causal_graph_alpha)
        if alpha <= 0.0:
            return x
        mixed = torch.einsum("ij,bjt->bit", self.causal_adj.to(dtype=x.dtype), x)
        return (1.0 - alpha) * x + alpha * mixed

    def _fit_length(self, y: torch.Tensor, target_len: int) -> torch.Tensor:
        cur_len = int(y.size(-1))
        if cur_len == int(target_len):
            return y
        if cur_len > int(target_len):
            start = max((cur_len - int(target_len)) // 2, 0)
            return y[:, :, start : start + int(target_len)]

        pad_left = max((int(target_len) - cur_len) // 2, 0)
        pad_right = max(int(target_len) - cur_len - pad_left, 0)
        if self.order_resample_pad_mode == "zero":
            return F.pad(y, (pad_left, pad_right), mode="constant", value=0.0)
        if self.order_resample_pad_mode == "reflect" and pad_left < cur_len and pad_right < cur_len:
            return F.pad(y, (pad_left, pad_right), mode="reflect")
        return F.pad(y, (pad_left, pad_right), mode="replicate")

    def _apply_order_resample(self, x: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        if not self.order_resample_enable:
            return x
        if x.dim() != 3:
            return x
        target_len = int(x.size(-1))
        if target_len <= 8:
            return x

        scale = (rpm.float().view(-1) / self.order_ref_rpm).clamp(
            min=self.order_resample_min_scale,
            max=self.order_resample_max_scale,
        )
        new_lens = torch.round(scale * float(target_len)).to(dtype=torch.long).clamp_min(8)
        out = torch.empty_like(x)
        for new_len_t in torch.unique(new_lens):
            new_len = int(new_len_t.item())
            mask = new_lens == new_len_t
            y = F.interpolate(x[mask], size=new_len, mode="linear", align_corners=False)
            out[mask] = self._fit_length(y, target_len)
        return out

    def _diagnostic_features(self, x: torch.Tensor, rpm_norm: torch.Tensor) -> torch.Tensor:
        z = x.float()
        eps = 1e-6
        mean = z.mean(dim=-1)
        centered = z - mean.unsqueeze(-1)
        std = centered.pow(2).mean(dim=-1).add(eps).sqrt()
        rms = z.pow(2).mean(dim=-1).add(eps).sqrt()
        abs_mean = z.abs().mean(dim=-1)
        max_abs = z.abs().amax(dim=-1)
        peak_to_peak = z.amax(dim=-1) - z.amin(dim=-1)
        skew = centered.pow(3).mean(dim=-1) / std.clamp_min(eps).pow(3)
        kurt = centered.pow(4).mean(dim=-1) / std.clamp_min(eps).pow(4)
        stats = torch.stack([mean, std, rms, abs_mean, max_abs, peak_to_peak, skew, kurt], dim=-1)

        target_size = min(int(self.diag_fft_size), int(z.size(-1)))
        if int(z.size(-1)) != target_size:
            z_fft = F.interpolate(z, size=target_size, mode="linear", align_corners=False)
        else:
            z_fft = z
        mag = torch.fft.rfft(z_fft, dim=-1).abs()
        mag = torch.log1p(mag)
        mag = mag[:, :, 1:] if mag.size(-1) > 1 else mag
        band_feats = []
        n_freq = int(mag.size(-1))
        for bi in range(self.diag_fft_bins):
            lo = int(round(bi * n_freq / self.diag_fft_bins))
            hi = int(round((bi + 1) * n_freq / self.diag_fft_bins))
            hi = max(hi, lo + 1)
            band_feats.append(mag[:, :, lo:hi].mean(dim=-1))
        bands = torch.stack(band_feats, dim=-1)

        # Lightweight "diagnostic prompt" tokens: channel statistics, spectrum bands, and modality summaries.
        ch_feat = torch.cat([stats, bands], dim=-1).flatten(1)
        accel_rms = rms[:, 0:6].mean(dim=1, keepdim=True)
        orbit_rms = rms[:, 6:8].mean(dim=1, keepdim=True)
        acoustic_rms = rms[:, 8:10].mean(dim=1, keepdim=True)
        group_feat = torch.cat(
            [
                accel_rms,
                orbit_rms,
                acoustic_rms,
                orbit_rms / accel_rms.clamp_min(eps),
                acoustic_rms / accel_rms.clamp_min(eps),
                rpm_norm.float(),
            ],
            dim=1,
        )
        return torch.cat([ch_feat, group_feat], dim=1).to(dtype=x.dtype)

    def _order_features(self, x: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        z = x.float()
        eps = 1e-6
        n_fft = min(int(self.order_feat_fft_size), int(z.size(-1)))
        if int(z.size(-1)) > n_fft:
            z = z[..., :n_fft]
        elif int(z.size(-1)) < n_fft:
            z = F.pad(z, (0, n_fft - int(z.size(-1))), mode="replicate")
        z = z - z.mean(dim=-1, keepdim=True)

        mag = torch.fft.rfft(z, n=n_fft, dim=-1).abs() / float(max(n_fft, 1))
        n_freq = int(mag.size(-1))
        if n_freq > 8:
            noise_hi = min(n_freq, 400)
            noise = mag[:, :, 5:noise_hi].median(dim=-1).values.clamp_min(eps)
        else:
            noise = mag[:, :, 1:].median(dim=-1).values.clamp_min(eps)

        base_bin = (rpm.float().view(-1, 1) / 60.0) * (float(n_fft) / self.order_feat_fs)
        order_feats = []
        for order in range(1, self.order_feat_max_order + 1):
            band_vals = []
            for off in range(-self.order_feat_band, self.order_feat_band + 1):
                idx = torch.round(base_bin * float(order) + float(off)).long().clamp(min=1, max=max(n_freq - 1, 1))
                gather_idx = idx.view(-1, 1, 1).expand(-1, z.size(1), 1)
                band_vals.append(torch.gather(mag, dim=2, index=gather_idx).squeeze(-1))
            amp = torch.stack(band_vals, dim=0).mean(dim=0)
            order_feats.append(torch.log1p(amp / noise))
        return torch.stack(order_feats, dim=-1).flatten(1).to(dtype=x.dtype)

    def _rasterize_orbit_image(self, xo: torch.Tensor, yo: torch.Tensor) -> torch.Tensor:
        """Rasterize orbit trajectory into a 2-channel image (points + path)."""
        eps = 1e-6
        grid = self.orbit_img_size
        coord_scale = float(grid - 1) / (2.0 * self.orbit_img_clip)
        xi = torch.round((xo + self.orbit_img_clip) * coord_scale).long().clamp_(0, grid - 1)
        yi = torch.round((yo + self.orbit_img_clip) * coord_scale).long().clamp_(0, grid - 1)

        batch_offsets = (
            torch.arange(xo.size(0), device=xo.device, dtype=torch.long).view(-1, 1) * (grid * grid)
        )
        flat_points = (yi * grid + xi + batch_offsets).reshape(-1)
        img_points = torch.bincount(flat_points, minlength=xo.size(0) * grid * grid).float().view(xo.size(0), grid, grid)

        dx = xo[:, 1:] - xo[:, :-1]
        dy = yo[:, 1:] - yo[:, :-1]
        xm = 0.5 * (xo[:, 1:] + xo[:, :-1])
        ym = 0.5 * (yo[:, 1:] + yo[:, :-1])
        xmi = torch.round((xm + self.orbit_img_clip) * coord_scale).long().clamp_(0, grid - 1)
        ymi = torch.round((ym + self.orbit_img_clip) * coord_scale).long().clamp_(0, grid - 1)
        seg_w = torch.sqrt(dx * dx + dy * dy + eps)
        flat_seg = (ymi * grid + xmi + batch_offsets[:, : xmi.size(1)]).reshape(-1)
        img_path = torch.zeros(xo.size(0) * grid * grid, device=xo.device, dtype=seg_w.dtype)
        img_path.scatter_add_(0, flat_seg, seg_w.reshape(-1))
        img_path = img_path.view(xo.size(0), grid, grid)

        img = torch.stack([img_points, img_path], dim=1)
        img = img / img.flatten(2).sum(dim=-1, keepdim=True).clamp_min(eps).view(xo.size(0), 2, 1, 1)
        if self.orbit_img_blur > 0:
            kernel = 2 * self.orbit_img_blur + 1
            img = F.avg_pool2d(img, kernel_size=kernel, stride=1, padding=self.orbit_img_blur)
        return img

    def _compute_orbit_geometry(self, xo: torch.Tensor, yo: torch.Tensor) -> torch.Tensor:
        """Compute 12-dim geometric features from orbit trajectory."""
        eps = 1e-6
        cov_xx = (xo * xo).mean(dim=-1)
        cov_yy = (yo * yo).mean(dim=-1)
        cov_xy = (xo * yo).mean(dim=-1)
        trace = cov_xx + cov_yy
        det = (cov_xx * cov_yy - cov_xy * cov_xy).clamp_min(0.0)
        delta = (trace * trace - 4.0 * det).clamp_min(0.0).sqrt()
        eig1 = 0.5 * (trace + delta)
        eig2 = 0.5 * (trace - delta).clamp_min(eps)
        axis_ratio = eig1 / eig2.clamp_min(eps)
        corr = cov_xy / (cov_xx.sqrt() * cov_yy.sqrt()).clamp_min(eps)
        angle = 0.5 * torch.atan2(2.0 * cov_xy, cov_xx - cov_yy + eps)

        radius_n = torch.sqrt(xo * xo + yo * yo + eps)
        signed_area = 0.5 * (xo[:, :-1] * yo[:, 1:] - xo[:, 1:] * yo[:, :-1]).sum(dim=-1)
        bbox_x = xo.amax(dim=-1) - xo.amin(dim=-1)
        bbox_y = yo.amax(dim=-1) - yo.amin(dim=-1)
        dx = xo[:, 1:] - xo[:, :-1]
        dy = yo[:, 1:] - yo[:, :-1]
        path_len = torch.sqrt(dx * dx + dy * dy + eps).mean(dim=-1)

        return torch.stack(
            [
                eig1,
                eig2,
                axis_ratio,
                corr,
                torch.sin(angle),
                torch.cos(angle),
                radius_n.mean(dim=-1),
                radius_n.std(dim=-1, unbiased=False),
                radius_n.amax(dim=-1),
                path_len,
                torch.abs(signed_area),
                bbox_x / bbox_y.clamp_min(eps),
            ],
            dim=1,
        )

    def _orbit_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute orbit image features: rasterize trajectory + geometric descriptors."""
        if x.size(1) < 8:
            return torch.zeros(
                (x.size(0), self.orbit_img_mlp[1].out_features),
                device=x.device,
                dtype=x.dtype,
            )

        xo = x[:, 6, :].float()
        yo = x[:, 7, :].float()
        eps = 1e-6

        xo = xo - xo.mean(dim=-1, keepdim=True)
        yo = yo - yo.mean(dim=-1, keepdim=True)
        radius = torch.sqrt(xo * xo + yo * yo + eps)
        scale = radius.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        xo = torch.clamp(xo / scale, min=-self.orbit_img_clip, max=self.orbit_img_clip)
        yo = torch.clamp(yo / scale, min=-self.orbit_img_clip, max=self.orbit_img_clip)

        img = self._rasterize_orbit_image(xo, yo)
        img_feat = self.orbit_img_encoder(img).flatten(1)
        geom = self._compute_orbit_geometry(xo, yo)
        return self.orbit_img_mlp(torch.cat([img_feat, geom], dim=1)).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, rpm: torch.Tensor | None = None):
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")

        if rpm is None:
            rpm = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        rpm = rpm.float().view(-1, 1)
        rpm_norm = self._rpm_norm(rpm)
        speed_feat = self.speed_mlp(self._speed_basis(rpm_norm))

        x = self._apply_order_resample(x, rpm)
        x = self._apply_causal_graph(x)
        diag_feat = self.diag_mlp(self._diagnostic_features(x, rpm_norm)) if self.diag_feat_enable else None
        order_feat = self.order_feat_mlp(self._order_features(x, rpm)) if self.order_feat_enable else None
        orbit_img_feat = self._orbit_image_features(x) if self.orbit_img_enable else None
        order_feat_for_fusion = order_feat
        if order_feat_for_fusion is not None and self.order_feat_scale != 1.0:
            order_feat_for_fusion = order_feat_for_fusion * self.order_feat_scale

        if self.temporal_pool > 1:
            x = F.avg_pool1d(x, kernel_size=self.temporal_pool, stride=self.temporal_pool, ceil_mode=False)

        h = self.stem(x)
        if self.speed_film:
            h = self._apply_speed_film(h, speed_feat, self.film_stem)
        for i, block in enumerate(self.tcn):
            h = block(h)
            if self.speed_film:
                h = self._apply_speed_film(h, speed_feat, self.film_blocks[i])

        h_avg = h.mean(dim=-1)
        h_max = h.amax(dim=-1)
        h_shared = torch.cat([h_avg, h_max], dim=1)

        fault_parts = [h_shared, speed_feat]
        if diag_feat is not None:
            fault_parts.append(diag_feat)
        if order_feat_for_fusion is not None:
            fault_parts.append(order_feat_for_fusion)
        if orbit_img_feat is not None and self.orbit_img_fusion_enable:
            fault_parts.append(orbit_img_feat)
        fault_concat = torch.cat(fault_parts, dim=1)
        fused = self.fusion_head(fault_concat)
        logits = self.cls_head(fused)
        if self.orbit_img_logit_enable and orbit_img_feat is not None:
            orbit_resid = self.orbit_img_logit_head(orbit_img_feat)
            logits = logits + self.orbit_img_logit_scale * orbit_resid * self.orbit_img_logit_mask.view(1, -1)
        reg = self.reg_head(fused)

        feats: Dict[str, torch.Tensor] = {
            "h_shared": h_shared,
            "fault_concat": fault_concat,
            "fused": fused,
            "speed_emb": speed_feat,
            "diag_feat": diag_feat if diag_feat is not None else h_shared,
            "order_feat": order_feat if order_feat is not None else h_shared,
            "orbit_img": orbit_img_feat if orbit_img_feat is not None else h_shared,
        }
        return {"logits": logits, "reg": reg, "feats": feats}
