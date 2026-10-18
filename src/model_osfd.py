# -*- coding: utf-8 -*-
"""Order-Sparse Fault Dictionary (OSFD) — 阶次稀疏故障字典分解算法.

Core hypothesis:
  A compound fault sample's order-spectrum can be decomposed as:
      S ≈ B_speed + a0·D0 + a1·D1 + a2·D2 + residual

where D0/D1/D2 are three fault dictionaries (imbalance / misalignment / rubbing)
and each dictionary is a set of sparse, physically-interpretable order atoms.

This is fundamentally different from "another Transformer":
  - Faults are explicitly modelled as additive, orthogonal components
  - The dictionaries are cross-speed invariant
  - Activation of each dictionary gives direct interpretability
  - Composition consistency constraint enables zero-shot combination generalization
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Speed encoding utilities (same as TCN baseline)
# ──────────────────────────────────────────────

def _speed_basis(rpm: torch.Tensor, center: float, scale: float, variant: str = "rich") -> torch.Tensor:
    """Build speed basis from normalized RPM."""
    r = ((rpm.view(-1, 1) - center) / max(scale, 1e-6)).float()
    if variant == "rich":
        return torch.cat([r, torch.sin(math.pi * r), torch.cos(math.pi * r), r * r], dim=1)
    return r


# ──────────────────────────────────────────────
# Order Spectrum Extractor
# ──────────────────────────────────────────────

class OrderSpectrumExtractor(nn.Module):
    """Deterministic, physics-based order spectrum extraction (VECTORIZED).

    For each sample:
      1. FFT magnitude spectrum → [B, C, n_freq]
      2. Convert freq axis → order axis (order = f / (rpm/60))
      3. Interpolate to fixed order grid → [B, C, n_orders]

    Fully vectorized across B, C — no per-sample loops.
    """

    def __init__(
        self,
        n_orders: int = 64,
        min_order: float = 0.5,
        max_order: float = 15.0,
        n_fft: int = 1024,
        fs: float = 5000.0,
        log_mag: bool = True,
    ):
        super().__init__()
        self.n_orders = int(n_orders)
        self.min_order = float(min_order)
        self.max_order = float(max_order)
        self.n_fft = int(n_fft)
        self.fs = float(fs)
        self.log_mag = bool(log_mag)

        # Fixed target order grid
        tgt_order = torch.linspace(self.min_order, self.max_order, self.n_orders)
        self.register_buffer("tgt_order", tgt_order, persistent=True)

        n_freq = self.n_fft // 2 + 1
        freq_hz = torch.arange(n_freq, dtype=torch.float32) * (self.fs / float(self.n_fft))
        self.register_buffer("freq_hz", freq_hz, persistent=True)
        self.n_freq = n_freq

    def forward(self, x: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        """Extract order spectrum (vectorized).

        Args:
            x:   [B, C, T]  time-domain signal
            rpm: [B]        speed in RPM

        Returns:
            S: [B, C * n_orders]  order spectrum features
        """
        B, C, T = x.shape
        device, dtype = x.device, x.dtype

        # 1. FFT magnitude — done in one shot
        mag = torch.fft.rfft(x, n=self.n_fft, dim=-1).abs()  # [B, C, n_freq]
        if self.log_mag:
            mag = torch.log1p(mag)

        # 2. Compute source order axis for each sample
        shaft_hz = rpm.clamp(min=100.0) / 60.0  # [B]
        src_order = (self.freq_hz.unsqueeze(0) / shaft_hz.unsqueeze(1)).contiguous()  # [B, F]

        # 3. For each (b, o), find interpolation neighbors via searchsorted
        #    tgt_order: [O]  → expand to [B, O]
        tgt_order_exp = self.tgt_order.unsqueeze(0).expand(B, -1).contiguous()  # [B, O]
        idx_r = torch.searchsorted(src_order, tgt_order_exp)  # [B, O]
        idx_r = idx_r.clamp(1, self.n_freq - 1)
        idx_l = idx_r - 1  # [B, O]

        # 4. Interpolation weights
        o_l = src_order.gather(1, idx_l)  # [B, O]
        o_r = src_order.gather(1, idx_r)
        w = (tgt_order_exp - o_l) / (o_r - o_l + 1e-8)
        w = w.clamp(0, 1).unsqueeze(1)  # [B, 1, O]

        # 5. Gather values and interpolate
        #    mag: [B, C, F]
        idx_l_exp = idx_l.unsqueeze(1).expand(B, C, -1)  # [B, C, O]
        idx_r_exp = idx_r.unsqueeze(1).expand(B, C, -1)
        v_l = mag.gather(2, idx_l_exp)  # [B, C, O]
        v_r = mag.gather(2, idx_r_exp)
        result = v_l + w * (v_r - v_l)  # [B, C, O]

        return result.reshape(B, C * self.n_orders)


# ──────────────────────────────────────────────
# Fault Dictionary Module
# ──────────────────────────────────────────────

class FaultDictionary(nn.Module):
    """A single fault dictionary: K sparse atoms, each a feat_dim vector.

    Each atom represents a specific order-spectrum pattern characteristic
    of that fault type (e.g., "1X dominant" for imbalance D0).
    """

    def __init__(self, name: str, dict_size: int, feat_dim: int):
        super().__init__()
        self.name = name
        self.dict_size = int(dict_size)
        self.feat_dim = int(feat_dim)
        # Initialize orthogonally (near-identity, then perturb)
        atoms = torch.randn(dict_size, feat_dim) * 0.1
        # Start with distinct atoms via orthogonal init
        if dict_size <= feat_dim:
            q, _ = torch.linalg.qr(torch.randn(feat_dim, dict_size))
            atoms = q.T
        self.atoms = nn.Parameter(atoms)

    def forward(self, f: torch.Tensor, temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute activation of this dictionary on feature f.

        Args:
            f: [B, feat_dim]  normalized feature vector

        Returns:
            activation: [B, 1]  scalar activation (0~1, how much this fault is present)
            reconstruction: [B, feat_dim]  reconstructed component
            atom_weights: [B, dict_size]  attention over atoms
        """
        # Cosine similarity between f and each atom
        f_norm = F.normalize(f, dim=1)  # [B, feat_dim]
        atoms_norm = F.normalize(self.atoms, dim=1)  # [K, feat_dim]

        sim = f_norm @ atoms_norm.T  # [B, K]

        # Softmax over atoms to get atom weights (with temperature for sparsity)
        atom_weights = F.softmax(sim / temperature, dim=1)  # [B, K]

        # Activation: weighted max similarity
        # Use a smooth approximation: softmax-weighted average of similarities
        activation = (atom_weights * sim).sum(dim=1, keepdim=True)  # [B, 1]
        # Squash to [0, 1] for interpretability as "fault severity"
        activation = torch.sigmoid((activation - 0.5) * 5.0)

        # Reconstruction: weighted combination of atoms
        recon = atom_weights @ self.atoms  # [B, feat_dim]

        return activation, recon, atom_weights


# ──────────────────────────────────────────────
# OSFD Main Model
# ──────────────────────────────────────────────

class OSFDNet(nn.Module):
    """Order-Sparse Fault Dictionary Network.

    Architecture:
      x [B, C, T] + rpm [B]
        │
        ▼
      OrderSpectrumExtractor → S [B, C*N_orders]
        │
        ▼
      Feature MLP → f [B, feat_dim]
        │
        ├── Speed Background MLP → b [B, feat_dim]
        │
        └── f_clean = f - b
            │
            ├── FaultDictionary D0 → a0, recon0, attn0
            ├── FaultDictionary D1 → a1, recon1, attn1
            └── FaultDictionary D2 → a2, recon2, attn2
                │
                ▼
            Activations a = [a0, a1, a2] [B, 3]
                │
                ├── cls_head → logits [B, 3]
                └── reg_head → reg [B, 7]

    Reconstructed feature:
        f_recon = b + a0·recon0 + a1·recon1 + a2·recon2
    """

    def __init__(
        self,
        # Order spectrum
        n_orders: int = 64,
        min_order: float = 0.5,
        max_order: float = 15.0,
        n_fft: int = 1024,
        fs: float = 5000.0,
        log_mag: bool = True,
        in_ch: int = 10,
        # Feature extraction
        feat_dim: int = 128,
        feat_hidden: int = 256,
        feat_dropout: float = 0.1,
        # Dictionary
        dict_size: int = 8,
        dict_temperature: float = 0.1,
        # Speed
        speed_feat_dim: int = 32,
        rpm_center: float = 1200.0,
        rpm_scale: float = 600.0,
        # Heads
        num_labels: int = 3,
        reg_dim: int = 7,
        head_hidden: int = 64,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.dict_size = int(dict_size)
        self.dict_temperature = float(dict_temperature)
        self.rpm_center = float(rpm_center)
        self.rpm_scale = float(max(rpm_scale, 1e-6))
        self.in_ch = int(in_ch)
        self.n_orders = int(n_orders)

        # 1. Order Spectrum Extractor (deterministic, no learnable params)
        self.order_extractor = OrderSpectrumExtractor(
            n_orders=n_orders,
            min_order=min_order,
            max_order=max_order,
            n_fft=n_fft,
            fs=fs,
            log_mag=log_mag,
        )
        order_feat_dim = in_ch * n_orders

        # 2. Feature Extractor MLP
        self.feat_net = nn.Sequential(
            nn.LayerNorm(order_feat_dim),
            nn.Linear(order_feat_dim, feat_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(feat_dropout)),
            nn.Linear(feat_hidden, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # 3. Speed Background MLP
        speed_in_dim = 4  # rich basis
        self.bg_net = nn.Sequential(
            nn.Linear(speed_in_dim, speed_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(speed_feat_dim, feat_dim),
        )

        # 4. Fault Dictionaries
        self.dict_y0 = FaultDictionary("D0_imbalance", dict_size, feat_dim)
        self.dict_y1 = FaultDictionary("D1_misalign", dict_size, feat_dim)
        self.dict_y2 = FaultDictionary("D2_rubbing", dict_size, feat_dim)

        # 5. Activation → Prediction Heads
        self.cls_head = nn.Sequential(
            nn.Linear(3, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Linear(head_hidden, num_labels),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(3, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Linear(head_hidden, reg_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, rpm: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)

        # 1. Order spectrum extraction
        S = self.order_extractor(x, rpm)  # [B, C*n_orders]

        # 2. Feature extraction
        f = self.feat_net(S)  # [B, feat_dim]

        # 3. Speed-dependent background
        speed_basis = _speed_basis(rpm, self.rpm_center, self.rpm_scale, "rich")  # [B, 4]
        b = self.bg_net(speed_basis)  # [B, feat_dim]

        # 4. Remove background → clean fault features
        f_clean = f - b  # [B, feat_dim]

        # 5. Dictionary decomposition
        a0, recon0, attn0 = self.dict_y0(f_clean, self.dict_temperature)  # each: [B,1], [B,feat_dim]
        a1, recon1, attn1 = self.dict_y1(f_clean, self.dict_temperature)
        a2, recon2, attn2 = self.dict_y2(f_clean, self.dict_temperature)

        # 6. Activations → predictions
        a = torch.cat([a0, a1, a2], dim=1)  # [B, 3]
        logits = self.cls_head(a)  # [B, 3]
        reg = self.reg_head(a)  # [B, 7]

        # 7. Reconstruction
        f_recon = b + a0 * recon0 + a1 * recon1 + a2 * recon2  # [B, feat_dim]

        return {
            "logits": logits,
            "reg": reg,
            "activations": a,
            "feat": f,
            "feat_clean": f_clean,
            "feat_recon": f_recon,
            "background": b,
            "dict_recon": [recon0, recon1, recon2],
            "dict_attn": [attn0, attn1, attn2],
            "dict_atoms": [
                self.dict_y0.atoms,
                self.dict_y1.atoms,
                self.dict_y2.atoms,
            ],
        }


# ──────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────

def osfd_loss(
    out: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    y_reg: torch.Tensor,
    cls_w: float = 1.0,
    reg_w: float = 0.2,
    lambda_recon: float = 0.5,
    lambda_act: float = 0.5,
    lambda_ortho: float = 0.1,
    lambda_sparse: float = 0.05,
    lambda_combo: float = 0.0,  # composition consistency (needs batch-level sampling)
) -> Dict[str, torch.Tensor]:
    """Total OSFD loss with all constraints.

    L_total = cls_w * L_cls + reg_w * L_reg
            + lambda_recon * L_recon
            + lambda_ortho * L_ortho
            + lambda_sparse * L_sparse
            + lambda_combo * L_combo
    """
    logits = out["logits"]
    reg = out["reg"]
    a = out["activations"]  # [B, 3]
    f = out["feat"]  # [B, feat_dim]
    f_recon = out["feat_recon"]  # [B, feat_dim]
    dict_atoms = out["dict_atoms"]  # list of 3 × [K, feat_dim]
    dict_attn = out["dict_attn"]  # list of 3 × [B, K]

    B = y_true.size(0)

    # ── L_cls: Binary cross-entropy for multi-label classification ──
    cls_loss = F.binary_cross_entropy_with_logits(logits, y_true)

    # ── L_reg: Masked Huber loss ──
    # Use mask from y_true: only regress for active faults
    mask = y_true.sum(dim=1) > 0  # [B]
    if mask.sum() > 0:
        reg_loss = F.smooth_l1_loss(
            reg[mask], y_reg[mask], beta=0.7, reduction="mean"
        )
    else:
        reg_loss = reg.new_tensor(0.0)

    # ── L_recon: Feature reconstruction ──
    recon_loss = F.mse_loss(f_recon, f, reduction="mean")
    act_loss = F.binary_cross_entropy(
        a.clamp(min=1e-6, max=1.0 - 1e-6),
        y_true,
        reduction="mean",
    )

    # ── L_ortho: Dictionaries should be mutually orthogonal ──
    d0 = F.normalize(dict_atoms[0], dim=1)  # [K0, feat_dim]
    d1 = F.normalize(dict_atoms[1], dim=1)  # [K1, feat_dim]
    d2 = F.normalize(dict_atoms[2], dim=1)  # [K2, feat_dim]

    ortho_01 = (d0 @ d1.T).pow(2).mean()
    ortho_02 = (d0 @ d2.T).pow(2).mean()
    ortho_12 = (d1 @ d2.T).pow(2).mean()
    ortho_loss = (ortho_01 + ortho_02 + ortho_12) / 3.0

    # ── L_sparse: Attention over atoms should be sparse (low entropy) ──
    sparse_loss = 0.0
    for attn in dict_attn:
        # attn: [B, K]  atom weights
        # Encourage low entropy → sparse activation
        eps = 1e-8
        ent = -(attn * (attn + eps).log()).sum(dim=1).mean()  # high entropy = bad
        # We want LOW entropy → add negative entropy penalty
        sparse_loss += ent
    sparse_loss /= 3.0

    # ── L_combo: Composition consistency (optional) ──
    # If we have both (100, 010) and 110 in the same batch, enforce:
    #   a_i(110) ≈ a_i(100) + a_i(010)
    # This requires batch-level pair sampling, simplified version:
    #   Encourage near-linear additivity of activations
    if lambda_combo > 0:
        # Simple proxy: logits should be consistent with activations
        # (handled implicitly by cls_head mapping a → logits)
        combo_loss = F.mse_loss(
            torch.sigmoid(logits), a,
            reduction="mean",
        )
    else:
        combo_loss = logits.new_tensor(0.0)

    # ── Total ──
    total = (
        cls_w * cls_loss
        + reg_w * reg_loss
        + lambda_recon * recon_loss
        + lambda_act * act_loss
        + lambda_ortho * ortho_loss
        + lambda_sparse * sparse_loss
        + lambda_combo * combo_loss
    )

    return {
        "total": total,
        "cls": cls_loss,
        "reg": reg_loss,
        "recon": recon_loss,
        "act": act_loss,
        "ortho": ortho_loss,
        "sparse": sparse_loss,
        "combo": combo_loss,
    }
