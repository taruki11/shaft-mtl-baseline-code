# -*- coding: utf-8 -*-
"""Training utilities for TCN probe and other models.

Extracted from scripts/11_train_loso_tcn_probe_py38.py to reduce duplication
and improve maintainability.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from src.metrics import masked_regression_metrics, multilabel_metrics


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class SpeedAdversary(nn.Module):
    def __init__(self, in_dim: int, n_speed: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_speed),
        )

    def forward(self, x):
        return self.net(x)


def apply_modality_dropout(x, ch_mode: str, p_accel: float, p_orbit: float, p_acoustic: float):
    if p_accel <= 0 and p_orbit <= 0 and p_acoustic <= 0:
        return x
    if x is None:
        return x

    maps = {
        "accel6": {"accel": list(range(0, 6))},
        "accel_orbit": {"accel": list(range(0, 6)), "orbit": list(range(6, 8))},
        "accel_acoustic": {"accel": list(range(0, 6)), "acoustic": list(range(6, 8))},
        "all10": {"accel": list(range(0, 6)), "orbit": list(range(6, 8)), "acoustic": list(range(8, 10))},
    }
    cm = maps.get(ch_mode, None)
    if cm is None:
        return x

    B = x.shape[0]
    device = x.device

    def drop_group(idxs, p):
        if p <= 0:
            return
        m = (torch.rand(B, 1, 1, device=device) < p).float()
        if len(idxs) > 0:
            x[:, idxs, :] = x[:, idxs, :] * (1.0 - m)

    if "accel" in cm:
        drop_group(cm["accel"], float(p_accel))
    if "orbit" in cm:
        drop_group(cm["orbit"], float(p_orbit))
    if "acoustic" in cm:
        drop_group(cm["acoustic"], float(p_acoustic))
    return x


def to_all10_layout(x: torch.Tensor, ch_mode: str) -> torch.Tensor:
    if x is None:
        return x
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if ch_mode == "all10":
        return x

    B, C, T = x.shape
    out = torch.zeros((B, 10, T), device=x.device, dtype=x.dtype)
    if ch_mode == "accel6":
        out[:, 0:6, :] = x[:, 0:6, :]
    elif ch_mode == "accel_orbit":
        out[:, 0:6, :] = x[:, 0:6, :]
        out[:, 6:8, :] = x[:, 6:8, :]
    elif ch_mode == "accel_acoustic":
        out[:, 0:6, :] = x[:, 0:6, :]
        out[:, 8:10, :] = x[:, 6:8, :]
    else:
        out[:, 0 : min(C, 10), :] = x[:, 0 : min(C, 10), :]
    return out


def build_reg_scale_from_train(tr_df, y_cols, reg_cols):
    tr_y = tr_df[y_cols].values.astype(np.float64)
    tr_reg = tr_df[reg_cols].values.astype(np.float64)
    mask = np.zeros_like(tr_reg, dtype=np.float64)
    mask[:, 0] = tr_y[:, 0]
    mask[:, 1:4] = tr_y[:, 1:2]
    mask[:, 4:7] = tr_y[:, 2:3]
    scale = np.ones(7, dtype=np.float64)
    for j in range(7):
        idx = mask[:, j] > 0.5
        if idx.sum() >= 2:
            t = tr_reg[idx, j]
            std = float(np.std(t))
            rng = float(np.max(t) - np.min(t))
            scale[j] = std if std > 1e-6 else max(rng, 1e-6)
        else:
            scale[j] = 1.0
    return scale


def masked_mse_reg(reg_pred, reg_true, y_true):
    m = np.zeros_like(reg_true)
    m[:, 0] = y_true[:, 0]
    m[:, 1:4] = y_true[:, 1:2]
    m[:, 4:7] = y_true[:, 2:3]
    denom = max(m.sum(), 1.0)
    return float((((reg_pred - reg_true) * m) ** 2).sum() / denom)


def get_k_sample_weight(y_true: torch.Tensor, w0: float, w1: float, w2: float, w3: float) -> torch.Tensor:
    k = y_true.sum(dim=1).long()
    lut = torch.tensor([w0, w1, w2, w3], dtype=torch.float32, device=y_true.device)
    k = torch.clamp(k, 0, 3)
    return lut[k]


def weighted_bce_loss(logits: torch.Tensor, y_true: torch.Tensor, sample_w: torch.Tensor) -> torch.Tensor:
    bce_elem = F.binary_cross_entropy_with_logits(logits, y_true, reduction="none")
    per_sample = bce_elem.mean(dim=1)
    return (per_sample * sample_w).mean()


def masked_scaled_huber_loss_weighted(reg_pred, reg_true, y_true, scale, sample_w, beta=1.0):
    diff = (reg_pred - reg_true) / torch.clamp(scale, min=1e-6)
    abs_diff = torch.abs(diff)
    huber = torch.where(abs_diff <= beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    mask = torch.zeros_like(reg_true)
    mask[:, 0] = y_true[:, 0]
    mask[:, 1:4] = y_true[:, 1:2]
    mask[:, 4:7] = y_true[:, 2:3]
    masked = huber * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    per_sample = masked.sum(dim=1) / denom
    return (per_sample * sample_w).mean()


def cardinality_regularization_loss(logits: torch.Tensor, y_true: torch.Tensor, sample_w: Optional[torch.Tensor] = None, beta: float = 0.25) -> torch.Tensor:
    pred_k_soft = torch.sigmoid(logits).sum(dim=1)
    true_k = y_true.sum(dim=1)
    diff = pred_k_soft - true_k
    abs_diff = diff.abs()
    if beta is not None and float(beta) > 0:
        beta_t = float(beta)
        per_sample = torch.where(abs_diff <= beta_t, 0.5 * diff * diff / beta_t, abs_diff - 0.5 * beta_t)
    else:
        per_sample = abs_diff
    if sample_w is not None:
        return (per_sample * sample_w).mean()
    return per_sample.mean()


def labelset_id_from_y(y_true: torch.Tensor) -> torch.Tensor:
    yb = (y_true > 0.5).long()
    return yb[:, 0] + 2 * yb[:, 1] + 4 * yb[:, 2]


def _fetch_feat(feats: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name in feats:
        return feats[name]
    if name == "h_fault":
        return feats["fault_concat"]
    raise KeyError(f"Unknown feat name: {name}; available={list(feats.keys())}")


def proto_consistency_loss(feat: torch.Tensor, y_true: torch.Tensor, proto_bank: torch.Tensor, proto_seen: torch.Tensor):
    feat_n = F.normalize(feat, dim=1)
    sid = labelset_id_from_y(y_true)
    valid = proto_seen[sid] > 0.5
    if valid.sum() <= 0:
        return feat_n.new_tensor(0.0)
    proto = proto_bank[sid[valid]]
    return (1.0 - F.cosine_similarity(feat_n[valid], proto, dim=1)).mean()


@torch.no_grad()
def update_proto_bank(feat: torch.Tensor, y_true: torch.Tensor, proto_bank: torch.Tensor, proto_seen: torch.Tensor, momentum: float = 0.9):
    feat_n = F.normalize(feat.detach(), dim=1)
    sid = labelset_id_from_y(y_true)
    uniq = sid.unique()
    for k in uniq.tolist():
        m = sid == int(k)
        if m.sum() <= 0:
            continue
        cur = F.normalize(feat_n[m].mean(dim=0, keepdim=False), dim=0)
        if proto_seen[k] > 0.5:
            newv = F.normalize(momentum * proto_bank[k] + (1.0 - momentum) * cur, dim=0)
            proto_bank[k] = newv
        else:
            proto_bank[k] = cur
            proto_seen[k] = 1.0


@torch.no_grad()
def evaluate(model, loader, device, ch_mode, use_amp, max_batches: int = 0):
    model.eval()
    ys, ps, regs, regts = [], [], [], []
    for bi, (x, y, yreg, speed) in enumerate(loader):
        if int(max_batches) > 0 and bi >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        x = to_all10_layout(x, ch_mode)
        speed = torch.as_tensor(speed, dtype=torch.float32, device=device)
        with autocast(enabled=use_amp):
            out = model(x, rpm=speed)
            logits = out["logits"]
            reg = out["reg"]
        prob = torch.sigmoid(logits).float().cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
        regs.append(reg.float().cpu().numpy())
        regts.append(yreg.numpy())

    y_true = np.concatenate(ys, 0)
    y_prob = np.concatenate(ps, 0)
    met = multilabel_metrics(y_true, y_prob, thr=0.5)
    reg_pred = np.concatenate(regs, 0)
    reg_true = np.concatenate(regts, 0)
    reg_mse = masked_mse_reg(reg_pred, reg_true, y_true)
    reg_report = masked_regression_metrics(reg_true, reg_pred, y_true, use_range=False)
    pred_k = (y_prob >= 0.5).astype(np.int32).sum(axis=1)
    true_k = y_true.astype(np.int32).sum(axis=1)
    met["avg_pred_k@0.5"] = float(pred_k.mean())
    met["avg_true_k"] = float(true_k.mean())
    met["frac_multi@0.5"] = float((pred_k > 1).mean())
    met["frac_zero@0.5"] = float((pred_k == 0).mean())
    return met, reg_mse, reg_report


def _round_nested(obj, digits: int = 4):
    if isinstance(obj, dict):
        return {k: _round_nested(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_nested(v, digits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_round_nested(v, digits) for v in obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), digits)
    return obj


def compute_epoch_lr_scale(ep: int, total_epochs: int, warmup_epochs: int = 0, sched: str = "none", min_lr_ratio: float = 0.1) -> float:
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs > 0 and ep <= warmup_epochs:
        return max(float(ep) / float(max(warmup_epochs, 1)), 1e-8)
    sched = str(sched or "none").lower()
    if sched == "none":
        return 1.0
    remain_total = max(total_epochs - warmup_epochs, 1)
    remain_ep = max(ep - warmup_epochs, 0)
    progress = min(float(remain_ep) / float(remain_total), 1.0)
    floor = float(min_lr_ratio)
    if sched == "cosine":
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


def set_optimizer_epoch_lr(opt: torch.optim.Optimizer, base_lr: float, ep: int, total_epochs: int, warmup_epochs: int = 0, sched: str = "none", min_lr_ratio: float = 0.1) -> float:
    scale = compute_epoch_lr_scale(ep=ep, total_epochs=total_epochs, warmup_epochs=warmup_epochs, sched=sched, min_lr_ratio=min_lr_ratio)
    lr = float(base_lr) * float(scale)
    for pg in opt.param_groups:
        pg["lr"] = lr
    return lr


def build_compact_epoch_summary(
    ep: int,
    test_speed: int,
    train_loss: float,
    train_cls: float,
    train_reg: float,
    train_adv: float,
    train_cons: float,
    train_kreg: float,
    met_va: Dict[str, Any],
    va_reg_mse: float,
    met_te: Dict[str, Any],
    te_reg_mse: float,
    best_key: str,
    cur_score: float,
    best_score: float,
    improved: bool,
    no_improve: int,
    best_epoch: int,
    lr: float,
    digits: int = 4,
):
    return _round_nested(
        {
            "speed": int(test_speed),
            "epoch": int(ep),
            "train": {
                "loss": train_loss,
                "cls": train_cls,
                "reg": train_reg,
                "adv": train_adv,
                "cons": train_cons,
                "kreg": train_kreg,
            },
            "val": {
                "exact": float(met_va.get("exact_match", 0.0)),
                "macro_f1": float(met_va.get("macro_f1", 0.0)),
                "micro_f1": float(met_va.get("micro_f1", 0.0)),
                "pred_k": float(met_va.get("avg_pred_k@0.5", 0.0)),
                "frac_zero": float(met_va.get("frac_zero@0.5", 0.0)),
                "reg_mse": float(va_reg_mse),
            },
            "test": {
                "exact": float(met_te.get("exact_match", 0.0)),
                "macro_f1": float(met_te.get("macro_f1", 0.0)),
                "micro_f1": float(met_te.get("micro_f1", 0.0)),
                "pred_k": float(met_te.get("avg_pred_k@0.5", 0.0)),
                "frac_zero": float(met_te.get("frac_zero@0.5", 0.0)),
                "reg_mse": float(te_reg_mse),
            },
            "pick": {
                "best_key": best_key,
                "cur": float(cur_score),
                "best": float(best_score),
                "improved": bool(improved),
                "no_improve": int(no_improve),
                "best_epoch": int(best_epoch),
                "lr": float(lr),
            },
        },
        digits,
    )


def format_compact_epoch_log(compact: Dict[str, Any]) -> str:
    tr = compact["train"]
    va = compact["val"]
    te = compact["test"]
    pk = compact["pick"]
    tag = "*" if pk["improved"] else "-"
    return (
        f"[sp{compact['speed']}] ep {compact['epoch']:02d} {tag} "
        f"| tr loss {tr['loss']:.4f} cls {tr['cls']:.4f} reg {tr['reg']:.4f} adv {tr['adv']:.4f} cons {tr['cons']:.4f} kreg {tr['kreg']:.4f} "
        f"| va ex {va['exact']:.4f} ma {va['macro_f1']:.4f} mi {va['micro_f1']:.4f} pk {va['pred_k']:.3f} reg {va['reg_mse']:.4f} "
        f"| te ex {te['exact']:.4f} ma {te['macro_f1']:.4f} mi {te['micro_f1']:.4f} pk {te['pred_k']:.3f} reg {te['reg_mse']:.4f} "
        f"| pick {pk['best_key']} cur {pk['cur']:.4f} best {pk['best']:.4f} be {pk['best_epoch']} bad {pk['no_improve']} lr {pk['lr']:.6f}"
    )
