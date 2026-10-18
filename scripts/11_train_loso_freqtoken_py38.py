# -*- coding: utf-8 -*-
"""LOSO training script for Frequency Token Transformer.

Architecture: time signal → RFFT → frequency tokens → Transformer → cls+reg
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # 修复：确保 CWD 是项目根目录，避免相对路径找不到文件

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset import SliceDataset
from src.metrics import masked_regression_metrics, multilabel_metrics
from src.model import masked_scaled_huber_loss
from src.model_freq_token_transformer import FreqTokenTransformer
from src.sampler import GroupedBatchSampler


LABEL_NAMES = ["y0", "y1", "y2"]


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


def _normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_y_cols(df: pd.DataFrame):
    cols = list(df.columns)
    norm = {_normalize_colname(c): c for c in cols}
    cand = []
    for k in ["y0", "y1", "y2"]:
        if k in norm:
            cand.append(norm[k])
    if len(cand) == 3:
        return cand
    cand = []
    for k in ["label0", "label1", "label2"]:
        if k in norm:
            cand.append(norm[k])
    if len(cand) == 3:
        return cand
    raise KeyError(f"Cannot find y columns in meta.csv. Existing columns: {cols}")


def find_reg_cols(df: pd.DataFrame, reg_dim=7):
    cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in cols}
    patterns = [
        [f"reg{i}" for i in range(reg_dim)],
        [f"reg_{i}" for i in range(reg_dim)],
        [f"r{i}" for i in range(reg_dim)],
    ]
    for pat in patterns:
        found = []
        ok = True
        for k in pat:
            nk = _normalize_colname(k)
            if nk in norm_map:
                found.append(norm_map[nk])
            else:
                ok = False
                break
        if ok and len(found) == reg_dim:
            return found
    reg_like = []
    for c in cols:
        n = _normalize_colname(c)
        m = re.match(r"(reg|r)(\d+)$", n)
        if m:
            reg_like.append((int(m.group(2)), c))
    reg_like = sorted(reg_like, key=lambda x: x[0])
    if reg_like and len(reg_like) >= reg_dim:
        reg_like = reg_like[:reg_dim]
        return [c for _, c in reg_like]
    raise KeyError("Cannot find 7 regression columns in meta.csv")


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


def _make_dataset(meta_like, ch_mode: str):
    return SliceDataset(meta_like, ch_mode=ch_mode)


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
    if name == "concat":
        parts = [k for k in ["ea", "eo", "ec"] if k in feats]
        if len(parts) == 3:
            return torch.cat([feats[p] for p in parts], dim=1)
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


def run_one_split(args, test_speed):
    set_seed(args.seed)
    out_dir = os.path.join(args.out_dir, f"test_{test_speed}rpm")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.meta_csv)
    df = df[df["speed_rpm"] != -1].copy()
    y_cols = find_y_cols(df)
    reg_cols = find_reg_cols(df, reg_dim=7)

    test_df = df[df["speed_rpm"] == test_speed].copy().reset_index(drop=True)
    if len(test_df) == 0:
        raise ValueError(f"test_speed={test_speed} 在 meta_csv 中不存在，请检查 speed_rpm 列。")
    train_df = df[df["speed_rpm"] != test_speed].copy().reset_index(drop=True)

    serials = train_df["serial"].unique()
    # Stratify by dominant label per serial so val covers all label combos
    serial_label = train_df.groupby("serial")[["y0", "y1", "y2"]].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    )
    serial_label_strat = serial_label.apply(
        lambda r: f"{int(r.y0)}{int(r.y1)}{int(r.y2)}", axis=1
    )
    # Reindex to serials order and ensure each strat class has >=2 serials
    strat_labels = serial_label_strat.reindex(serials).fillna("000")
    vc = strat_labels.value_counts()
    rare = vc[vc < 2].index.tolist()
    strat_labels_fixed = strat_labels.apply(lambda x: "rare" if x in rare else x)
    tr_ser, va_ser = train_test_split(
        serials, test_size=0.2, random_state=args.seed,
        stratify=strat_labels_fixed.values,
    )
    tr_df = train_df[train_df["serial"].isin(tr_ser)].reset_index(drop=True).copy()
    va_df = train_df[train_df["serial"].isin(va_ser)].reset_index(drop=True).copy()

    reg_scale = build_reg_scale_from_train(tr_df, y_cols, reg_cols)

    tmp = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    tr_csv = os.path.join(tmp, "tr.csv")
    va_csv = os.path.join(tmp, "va.csv")
    te_csv = os.path.join(tmp, "te.csv")
    tr_df.to_csv(tr_csv, index=False)
    va_df.to_csv(va_csv, index=False)
    test_df.to_csv(te_csv, index=False)

    tr_ds = _make_dataset(tr_csv, ch_mode=args.ch_mode)
    va_ds = _make_dataset(va_csv, ch_mode=args.ch_mode)
    te_ds = _make_dataset(te_csv, ch_mode=args.ch_mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (args.amp == 1)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model = FreqTokenTransformer(
        num_labels=len(y_cols),
        reg_dim=len(reg_cols),
        in_ch=10,
        n_fft=args.n_fft,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        trans_dropout=args.trans_dropout,
        speed_feat_dim=args.speed_feat_dim,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        rpm_center=args.rpm_center,
        rpm_scale=args.rpm_scale,
        pool_mode=args.pool_mode,
        use_order_rpm_encoding=bool(args.use_order_rpm_encoding),
        token_mode=str(getattr(args, "token_mode", "freq")),
        time_patch_len=int(getattr(args, "time_patch_len", 32)),
    ).to(device)

    dummy_x, _, _, dummy_speed = tr_ds[0]
    dummy_x = dummy_x.unsqueeze(0).to(device)
    dummy_x = to_all10_layout(dummy_x, args.ch_mode)
    dummy_speed = torch.as_tensor([float(dummy_speed)], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy_x, rpm=dummy_speed)

    adv_feat = _fetch_feat(out["feats"], args.adv_on)
    adv_in_dim = int(adv_feat.shape[1])
    cons_feat = _fetch_feat(out["feats"], args.cons_on)
    cons_in_dim = int(cons_feat.shape[1])

    speed_values = sorted([int(s) for s in df["speed_rpm"].unique() if int(s) != -1])
    speed_to_idx = {sp: i for i, sp in enumerate(speed_values)}
    n_speed = len(speed_values)
    adv_head = SpeedAdversary(adv_in_dim, n_speed=n_speed, hidden=args.adv_hidden).to(device) if float(args.lambda_adv) > 0 else None

    proto_bank = F.normalize(torch.randn(8, cons_in_dim, device=device), dim=1) if float(args.lambda_cons) > 0 else None
    proto_seen = torch.zeros(8, device=device) if float(args.lambda_cons) > 0 else None

    params = list(model.parameters()) + (list(adv_head.parameters()) if adv_head is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    base_lr = float(args.lr)
    scaler = GradScaler(enabled=use_amp)
    ce_speed = nn.CrossEntropyLoss()
    reg_scale_t = torch.as_tensor(reg_scale, dtype=torch.float32, device=device)

    if not args.no_group_batch:
        bs = GroupedBatchSampler(tr_df, batch_size=args.batch_size, group_col=args.group_col, seed=args.seed, drop_last=True, shuffle=True)
        dl_kwargs = dict(num_workers=args.num_workers, pin_memory=(device == "cuda"), persistent_workers=(args.num_workers > 0))
        if args.num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2
        tr_loader = DataLoader(tr_ds, batch_sampler=bs, **dl_kwargs)
    else:
        tr_loader = DataLoader(
            tr_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=(device == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    best: Dict[str, Any] = {args.best_key: -1.0}
    best_path = os.path.join(out_dir, "best.pt")
    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")  # Full training state
    epoch_log_jsonl = os.path.join(out_dir, "epoch_log_full.jsonl")
    epoch_log_compact = os.path.join(out_dir, "epoch_log_compact.txt")
    best_score = -1e18
    no_improve = 0
    start_epoch = 1

    # Resume from checkpoint if specified
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"[RESUME] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if adv_head is not None and ckpt.get("adv_head") is not None:
            adv_head.load_state_dict(ckpt["adv_head"])
        if proto_bank is not None and ckpt.get("proto_bank") is not None:
            proto_bank = ckpt["proto_bank"].to(device)
        if proto_seen is not None and ckpt.get("proto_seen") is not None:
            proto_seen = ckpt["proto_seen"].to(device)
        # Restore optimizer state if available
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            print(f"[RESUME] Optimizer state restored")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
            print(f"[RESUME] GradScaler state restored")
        # Restore training state
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
            print(f"[RESUME] Resuming from epoch {start_epoch}")
        if "best_score" in ckpt:
            best_score = float(ckpt["best_score"])
            print(f"[RESUME] Best score restored: {best_score:.4f}")
        if "no_improve" in ckpt:
            no_improve = int(ckpt["no_improve"])
            print(f"[RESUME] No improve counter: {no_improve}")
        if "summary_best" in ckpt:
            best = ckpt["summary_best"]
            print(f"[RESUME] Best validation summary restored")
        # Remove old logs if resuming to append
        if os.path.isfile(epoch_log_jsonl):
            # Backup old log
            backup = epoch_log_jsonl + f".backup_ep{start_epoch-1}"
            os.rename(epoch_log_jsonl, backup)
        if os.path.isfile(epoch_log_compact):
            backup = epoch_log_compact + f".backup_ep{start_epoch-1}"
            os.rename(epoch_log_compact, backup)
    else:
        if os.path.isfile(epoch_log_jsonl):
            os.remove(epoch_log_jsonl)
        if os.path.isfile(epoch_log_compact):
            os.remove(epoch_log_compact)

    # --- EVAL ONLY: skip training, load best.pt and evaluate ---
    if getattr(args, "eval_only", False):
        best_path = os.path.join(out_dir, "best.pt")
        if not os.path.isfile(best_path):
            best_path = os.path.join(out_dir, "checkpoint.pt")
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"No checkpoint found in {out_dir}")
        print(f"[EVAL_ONLY] Loading {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        best_val = ckpt.get("summary_best", {"exact_match": 0.0, "epoch": 0})
        met_te, te_reg_mse, te_reg = evaluate(model, te_loader, device, args.ch_mode, use_amp, max_batches=0)
        summary = {
            "test_speed": int(test_speed),
            "model_type": "freqtoken_transformer",
            "args": vars(args),
            "picked_cols": {"y_cols": y_cols, "reg_cols": reg_cols},
            "reg_scale": reg_scale.tolist(),
            "huber_beta": float(args.huber_beta),
            "best_val": best_val,
            "test": met_te,
            "test_reg_mse": te_reg_mse,
            "test_reg": te_reg,
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[EVAL_ONLY] Wrote summary.json for test_speed={test_speed}")
        return summary

    for ep in range(start_epoch, args.epochs + 1):
        current_lr = set_optimizer_epoch_lr(opt, base_lr=base_lr, ep=ep, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs, sched=args.sched, min_lr_ratio=args.min_lr_ratio)
        model.train()
        if adv_head is not None:
            adv_head.train()

        loss_sum = 0.0
        cls_sum = 0.0
        reg_sum = 0.0
        adv_sum = 0.0
        cons_sum = 0.0
        kreg_sum = 0.0
        sw_sum = 0.0

        for bi, (x, y, yreg, speed) in enumerate(tqdm(tr_loader, desc=f"train ep{ep} sp{test_speed}", leave=False)):
            if int(args.max_train_batches) > 0 and bi >= int(args.max_train_batches):
                break
            x = x.to(device, non_blocking=True)
            x = apply_modality_dropout(x, args.ch_mode, args.drop_accel, args.drop_orbit, args.drop_acoustic)
            x = to_all10_layout(x, args.ch_mode)
            y = y.to(device, non_blocking=True)
            yreg = yreg.to(device, non_blocking=True)
            speed_t = torch.as_tensor(speed, dtype=torch.float32, device=device)

            speed_np = np.asarray(speed, dtype=int)
            speed_idx = torch.as_tensor([speed_to_idx[int(s)] for s in speed_np], dtype=torch.long, device=device)

            if int(args.kbal_enable) == 1:
                sample_w = get_k_sample_weight(y, args.k_w0, args.k_w1, args.k_w2, args.k_w3)
            else:
                sample_w = torch.ones(y.size(0), dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = model(x, rpm=speed_t)
                logits = out["logits"]
                reg = out["reg"]
                feats = out["feats"]

                cls_loss = weighted_bce_loss(logits, y, sample_w)

                if int(args.kbal_weight_reg) == 1:
                    r_loss = masked_scaled_huber_loss_weighted(reg, yreg, y, scale=reg_scale_t, sample_w=sample_w, beta=args.huber_beta)
                else:
                    r_loss = masked_scaled_huber_loss(reg, yreg, y, scale=reg_scale_t, beta=args.huber_beta)

                if adv_head is not None and float(args.lambda_adv) > 0:
                    adv_feat = _fetch_feat(feats, args.adv_on)
                    speed_logits = adv_head(grad_reverse(adv_feat, 1.0))
                    adv_loss = ce_speed(speed_logits, speed_idx)
                else:
                    adv_loss = logits.new_tensor(0.0)

                if proto_bank is not None and float(args.lambda_cons) > 0:
                    cons_feat = _fetch_feat(feats, args.cons_on)
                    cons_loss = proto_consistency_loss(cons_feat, y, proto_bank, proto_seen)
                else:
                    cons_feat = None
                    cons_loss = logits.new_tensor(0.0)

                if float(args.kreg_lambda) > 0:
                    kreg_sample_w = sample_w if int(args.kreg_use_sample_weight) == 1 else None
                    kreg_loss = cardinality_regularization_loss(logits, y, sample_w=kreg_sample_w, beta=args.kreg_beta)
                else:
                    kreg_loss = logits.new_tensor(0.0)

                loss = (
                    args.cls_w * cls_loss
                    + args.reg_w * r_loss
                    + float(args.lambda_adv) * adv_loss
                    + float(args.lambda_cons) * cons_loss
                    + float(args.kreg_lambda) * kreg_loss
                )

            if not torch.isfinite(loss):
                print("[WARN] non-finite loss detected, skip batch")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if args.clip_grad and float(args.clip_grad) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(args.clip_grad))
            scaler.step(opt)
            scaler.update()

            if proto_bank is not None and cons_feat is not None:
                update_proto_bank(cons_feat, y, proto_bank, proto_seen, momentum=args.proto_momentum)

            bsz = x.size(0)
            loss_sum += float(loss.item()) * bsz
            cls_sum += float(cls_loss.item()) * bsz
            reg_sum += float(r_loss.item()) * bsz
            adv_sum += float(adv_loss.item()) * bsz
            cons_sum += float(cons_loss.item()) * bsz
            kreg_sum += float(kreg_loss.item()) * bsz
            sw_sum += bsz

        met_va, va_reg_mse, va_reg = evaluate(model, va_loader, device, args.ch_mode, use_amp, max_batches=args.max_eval_batches)
        if int(args.eval_test_each_epoch) == 1:
            met_te, te_reg_mse, te_reg = evaluate(model, te_loader, device, args.ch_mode, use_amp, max_batches=args.max_eval_batches)
        else:
            met_te = {"exact_match": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "avg_pred_k@0.5": 0.0, "frac_zero@0.5": 0.0}
            te_reg_mse = 0.0
            te_reg = {}

        epoch_summary = {
            "epoch": ep,
            "train_loss": loss_sum / max(sw_sum, 1.0),
            "train_cls": cls_sum / max(sw_sum, 1.0),
            "train_reg": reg_sum / max(sw_sum, 1.0),
            "train_adv": adv_sum / max(sw_sum, 1.0),
            "train_cons": cons_sum / max(sw_sum, 1.0),
            "train_kreg": kreg_sum / max(sw_sum, 1.0),
            "val": met_va,
            "val_reg_mse": va_reg_mse,
            "val_reg": va_reg,
            "test": met_te,
            "test_reg_mse": te_reg_mse,
            "test_reg": te_reg,
        }

        cur = float(met_va.get(args.best_key, -1e18))
        improved = cur > best_score + float(args.min_delta)
        next_best_score = max(best_score, cur) if improved else best_score
        next_no_improve = 0 if improved else (no_improve + 1)

        compact_epoch = build_compact_epoch_summary(
            ep=ep,
            test_speed=test_speed,
            train_loss=loss_sum / max(sw_sum, 1.0),
            train_cls=cls_sum / max(sw_sum, 1.0),
            train_reg=reg_sum / max(sw_sum, 1.0),
            train_adv=adv_sum / max(sw_sum, 1.0),
            train_cons=cons_sum / max(sw_sum, 1.0),
            train_kreg=kreg_sum / max(sw_sum, 1.0),
            met_va=met_va,
            va_reg_mse=va_reg_mse,
            met_te=met_te,
            te_reg_mse=te_reg_mse,
            best_key=args.best_key,
            cur_score=cur,
            best_score=next_best_score,
            improved=improved,
            no_improve=next_no_improve,
            best_epoch=(ep if improved else int(best.get("epoch", 0))),
            lr=current_lr,
            digits=args.log_digits,
        )
        compact_line = format_compact_epoch_log(compact_epoch)
        print(compact_line)
        with open(epoch_log_compact, "a", encoding="utf-8") as f:
            f.write(compact_line + "\n")
        with open(epoch_log_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_summary, ensure_ascii=False) + "\n")

        if args.log_full_json == 1:
            print(json.dumps(_round_nested(epoch_summary, args.log_digits), ensure_ascii=False))

        if improved:
            best_score = cur
            no_improve = 0
            best = {
                args.best_key: cur,
                "epoch": ep,
                "val": met_va,
                "val_reg_mse": va_reg_mse,
                "val_reg": va_reg,
                "test": met_te,
                "test_reg_mse": te_reg_mse,
                "test_reg": te_reg,
            }
            ckpt = {
                "model": model.state_dict(),
                "model_type": "freqtoken_transformer",
                "args": vars(args),
                "picked_cols": {"y_cols": y_cols, "reg_cols": reg_cols},
                "reg_scale": reg_scale.tolist(),
                "speed_values": speed_values,
                "adv_head": (adv_head.state_dict() if adv_head is not None else None),
                "proto_bank": (proto_bank.detach().cpu() if proto_bank is not None else None),
                "proto_seen": (proto_seen.detach().cpu() if proto_seen is not None else None),
                "summary_best": best,
                # Resume support
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "best_score": best_score,
                "no_improve": 0,  # Will be 0 since we just improved
            }
            torch.save(ckpt, best_path)
            # Also save a full checkpoint for resume safety
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "best_score": best_score,
                "no_improve": 0,
                "summary_best": best,
                "args": vars(args),
                "picked_cols": {"y_cols": y_cols, "reg_cols": reg_cols},
                "reg_scale": reg_scale.tolist(),
                "speed_values": speed_values,
                "adv_head": (adv_head.state_dict() if adv_head is not None else None),
                "proto_bank": (proto_bank.detach().cpu() if proto_bank is not None else None),
                "proto_seen": (proto_seen.detach().cpu() if proto_seen is not None else None),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[CHECKPOINT] Saved full checkpoint to {checkpoint_path}")
        else:
            no_improve = next_no_improve
            # Save checkpoint every epoch for resume safety
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "best_score": best_score,
                "no_improve": no_improve,
                "summary_best": best,
                "args": vars(args),
                "picked_cols": {"y_cols": y_cols, "reg_cols": reg_cols},
                "reg_scale": reg_scale.tolist(),
                "speed_values": speed_values,
                "adv_head": (adv_head.state_dict() if adv_head is not None else None),
                "proto_bank": (proto_bank.detach().cpu() if proto_bank is not None else None),
                "proto_seen": (proto_seen.detach().cpu() if proto_seen is not None else None),
            }
            torch.save(checkpoint, checkpoint_path)
            if ep % 5 == 0:
                print(f"[CHECKPOINT] Saved checkpoint at epoch {ep} to {checkpoint_path}")

        if no_improve >= int(args.patience):
            print(f"[EARLY STOP] test_speed={test_speed} at epoch={ep}")
            break

    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    met_te, te_reg_mse, te_reg = evaluate(model, te_loader, device, args.ch_mode, use_amp, max_batches=0)
    summary = {
        "test_speed": int(test_speed),
        "model_type": "freqtoken_transformer",
        "args": vars(args),
        "picked_cols": {"y_cols": y_cols, "reg_cols": reg_cols},
        "reg_scale": reg_scale.tolist(),
        "huber_beta": float(args.huber_beta),
        "best_val": best,
        "test": met_te,
        "test_reg_mse": te_reg_mse,
        "test_reg": te_reg,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_speed", type=int, default=None)
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--cls_w", type=float, default=1.0)
    ap.add_argument("--reg_w", type=float, default=0.2)
    ap.add_argument("--no_group_batch", action="store_true")
    ap.add_argument("--group_col", default="npz_path")
    ap.add_argument("--huber_beta", type=float, default=0.7)
    ap.add_argument("--best_key", default="exact_match")

    ap.add_argument("--ch_mode", default="all10", choices=["all10"])
    ap.add_argument("--norm_mode", default="instance", choices=["none", "instance"])
    ap.add_argument("--amp", type=int, default=0)

    ap.add_argument("--modality_dropout", type=float, default=0.0)
    ap.add_argument("--drop_accel", type=float, default=0.0)
    ap.add_argument("--drop_orbit", type=float, default=0.0)
    ap.add_argument("--drop_acoustic", type=float, default=0.1)
    ap.add_argument("--clip_grad", type=float, default=0.5)

    # ---- FreqTokenTransformer model args ----
    ap.add_argument("--n_fft", type=int, default=128, help="FFT size for frequency tokenization")
    ap.add_argument("--d_model", type=int, default=128, help="Transformer hidden dim")
    ap.add_argument("--n_layers", type=int, default=4, help="Transformer encoder layers")
    ap.add_argument("--n_heads", type=int, default=8, help="Attention heads")
    ap.add_argument("--ffn_dim", type=int, default=512, help="FFN hidden dim")
    ap.add_argument("--trans_dropout", type=float, default=0.1, help="Transformer dropout")
    ap.add_argument("--pool_mode", type=str, default="avg", choices=["avg", "cls"], help="Token pooling mode")
    ap.add_argument("--use_order_rpm_encoding", type=int, default=1, help="Use rich RPM encoding")
    ap.add_argument("--token_mode", type=str, default="freq", choices=["freq", "time"], help="Token mode: freq (FFT) or time (patch) — for ablation")
    ap.add_argument("--time_patch_len", type=int, default=32, help="Patch length for time-domain token mode")
    ap.add_argument("--speed_feat_dim", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=128)
    ap.add_argument("--head_dropout", type=float, default=0.0)
    ap.add_argument("--rpm_center", type=float, default=1200.0)
    ap.add_argument("--rpm_scale", type=float, default=600.0)

    ap.add_argument("--lambda_adv", type=float, default=0.0)
    ap.add_argument(
        "--adv_on",
        type=str,
        default="h_shared",
        choices=["fa", "fo", "fc", "concat", "h_shared", "fault_concat", "fused", "speed_emb", "oa", "oo", "oc"],
    )
    ap.add_argument("--adv_hidden", type=int, default=128)

    ap.add_argument("--lambda_cons", type=float, default=0.0)
    ap.add_argument(
        "--cons_on",
        type=str,
        default="fault_concat",
        choices=["h_shared", "fault_concat", "fused", "ea", "eo", "ec", "oa", "oo", "oc"],
    )
    ap.add_argument("--proto_momentum", type=float, default=0.9)

    ap.add_argument("--kbal_enable", type=int, default=1)
    ap.add_argument("--kbal_weight_reg", type=int, default=1)
    ap.add_argument("--kreg_lambda", type=float, default=0.05)
    ap.add_argument("--kreg_beta", type=float, default=0.25)
    ap.add_argument("--kreg_use_sample_weight", type=int, default=1)
    ap.add_argument("--k_w0", type=float, default=1.5)
    ap.add_argument("--k_w1", type=float, default=1.3)
    ap.add_argument("--k_w2", type=float, default=1.0)
    ap.add_argument("--k_w3", type=float, default=1.3)

    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--sched", type=str, default="cosine", choices=["none", "cosine"])
    ap.add_argument("--min_lr_ratio", type=float, default=0.2)
    ap.add_argument("--eval_test_each_epoch", type=int, default=0)
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_eval_batches", type=int, default=0)
    ap.add_argument("--log_digits", type=int, default=4)
    ap.add_argument("--log_full_json", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None,
                   help="Resume training from a checkpoint (path to best.pt or checkpoint.pt)")
    ap.add_argument("--eval_only", action="store_true",
                   help="Eval only: load best.pt and run evaluation, skip training")
    ap.add_argument("--fix_summary", action="store_true",
                   help="Fix all_summary.json by scanning test_*/summary.json")

    args = ap.parse_args()

    if args.fix_summary:
        root = args.out_dir
        import glob
        summary_files = sorted(glob.glob(os.path.join(root, "test_*rpm", "summary.json")))
        all_summ = []
        for sf in summary_files:
            try:
                with open(sf, "r", encoding="utf-8") as f:
                    summ = json.load(f)
                all_summ.append(summ)
                print(f"[FIX] Loaded {sf}")
            except Exception as e:
                print(f"[FIX] ERROR loading {sf}: {e}")
        out_path = os.path.join(root, "all_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_summ, f, ensure_ascii=False, indent=2)
        print(f"[FIX] Wrote {len(all_summ)} entries to {out_path}")
        return

    df = pd.read_csv(args.meta_csv)
    speeds = sorted([int(s) for s in df["speed_rpm"].unique() if int(s) != -1])
    if args.test_speed is not None:
        speeds = [args.test_speed]

    root = args.out_dir
    os.makedirs(root, exist_ok=True)
    all_summ = []
    all_summ_path = os.path.join(root, "all_summary.json")
    # Resume: load existing results if present
    if os.path.isfile(all_summ_path):
        try:
            with open(all_summ_path, "r", encoding="utf-8") as f:
                all_summ = json.load(f)
            print(f"[RESUME] Loaded {len(all_summ)} existing results from {all_summ_path}")
        except Exception:
            all_summ = []
    done_speeds = {int(s["test_speed"]) for s in all_summ}
    for sp in speeds:
        if int(sp) in done_speeds:
            print(f"[SKIP] test_speed={sp} already completed, skipping.")
            continue
        try:
            summ = run_one_split(args, sp)
            all_summ.append(summ)
            # Save incrementally after each speed
            with open(all_summ_path, "w", encoding="utf-8") as f:
                json.dump(all_summ, f, ensure_ascii=False, indent=2)
            print(f"[OK] Saved results after test_speed={sp}")
        except Exception as e:
            print(f"[ERROR] test_speed={sp} failed with: {e}")
            import traceback
            traceback.print_exc()
            # Save what we have so far
            with open(all_summ_path, "w", encoding="utf-8") as f:
                json.dump(all_summ, f, ensure_ascii=False, indent=2)
            print(f"[WARN] Saved partial results after error on test_speed={sp}")
            continue

    print(f"[OK] LOSO finished: {root}")


if __name__ == "__main__":
    main()
