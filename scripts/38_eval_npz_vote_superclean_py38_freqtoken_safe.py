# -*- coding: utf-8 -*-
"""Safe eval + calibration for FreqTokenTransformer checkpoints.

Mirror of 38_eval_npz_vote_superclean_py38_tcn_safe.py but for FreqToken model.
Outputs:
- *_summary.csv/json
- *_per_label.csv
- *_npz_details.csv
- *_labelset_confusion.csv
- *_calib_sweep.csv / *_calib_best.json when calib_meta_csv is provided
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import SliceDataset
from src.model_freq_token_transformer import FreqTokenTransformer

LABEL_NAMES = ["unbalance", "misalign", "rub"]


def parse_list_float(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if str(x).strip()]


def parse_bool_int(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {v}")


def find_speed_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in ["speed_rpm", "speed", "rpm", "Speed", "SPEED"]:
        if c in cols:
            return c
    for c in cols:
        if re.search(r"speed|rpm", str(c), flags=re.I):
            return c
    raise KeyError("Cannot find speed column.")


def to_all10_layout(x: torch.Tensor, ch_mode: str) -> torch.Tensor:
    """Map SliceDataset output to all10 channel layout expected by FreqTokenTransformer.

    Copied from 11_train_loso_freqtoken_py38.py to keep eval consistent with training.
    """
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


def multilabel_metrics_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    exact = float(np.mean(np.all(y_pred == y_true.astype(int), axis=1)))

    precs, recs, f1s = [], [], []
    for j in range(y_true.shape[1]):
        t = y_true[:, j].astype(int)
        p = y_pred[:, j].astype(int)
        tp = np.sum((t == 1) & (p == 1))
        fp = np.sum((t == 0) & (p == 1))
        fn = np.sum((t == 1) & (p == 0))
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    macro_precision = float(np.mean(precs))
    macro_recall = float(np.mean(recs))
    macro_f1 = float(np.mean(f1s))

    t = y_true.astype(int)
    p = y_pred.astype(int)
    tp = np.sum((t == 1) & (p == 1))
    fp = np.sum((t == 0) & (p == 1))
    fn = np.sum((t == 1) & (p == 0))
    micro_precision = float(tp / (tp + fp + eps))
    micro_recall = float(tp / (tp + fn + eps))
    micro_f1 = float(2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12))
    pred_k = y_pred.sum(axis=1)
    true_k = y_true.sum(axis=1)
    return {
        "exact_match": exact,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "avg_pred_k": float(pred_k.mean()),
        "avg_true_k": float(true_k.mean()),
        "frac_multi": float((pred_k > 1).mean()),
        "frac_zero": float((pred_k == 0).mean()),
    }


def prf_per_label(y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict[str, Any]]:
    eps = 1e-12
    rows = []
    for j, name in enumerate(LABEL_NAMES):
        t = y_true[:, j].astype(int)
        p = y_pred[:, j].astype(int)
        tp = int(np.sum((t == 1) & (p == 1)))
        fp = int(np.sum((t == 0) & (p == 1)))
        fn = int(np.sum((t == 1) & (p == 0)))
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        rows.append({"label": name, "tp": tp, "fp": fp, "fn": fn, "precision": float(prec), "recall": float(rec), "f1": float(f1)})
    return rows


def build_model_from_ckpt(ckpt: Dict[str, Any], device: str) -> FreqTokenTransformer:
    """Build FreqTokenTransformer from checkpoint args.

    All constructor params are read from ckpt['args'] so the eval script
    stays in sync with whatever training config was saved.
    """
    args_cfg = ckpt.get("args", {}) or {}
    model = FreqTokenTransformer(
        num_labels=int(args_cfg.get("num_labels", 3)) if "num_labels" in args_cfg else 3,
        reg_dim=int(args_cfg.get("reg_dim", 7)) if "reg_dim" in args_cfg else 7,
        in_ch=int(args_cfg.get("in_ch", 10)) if "in_ch" in args_cfg else 10,
        n_fft=int(args_cfg.get("n_fft", 256)),
        d_model=int(args_cfg.get("d_model", 128)),
        n_layers=int(args_cfg.get("n_layers", 4)),
        n_heads=int(args_cfg.get("n_heads", 8)),
        ffn_dim=int(args_cfg.get("ffn_dim", 512)),
        trans_dropout=float(args_cfg.get("trans_dropout", 0.1)),
        speed_feat_dim=int(args_cfg.get("speed_feat_dim", 32)),
        head_hidden=int(args_cfg.get("head_hidden", 128)),
        head_dropout=float(args_cfg.get("head_dropout", 0.0)),
        rpm_center=float(args_cfg.get("rpm_center", 1200.0)),
        rpm_scale=float(args_cfg.get("rpm_scale", 600.0)),
        pool_mode=str(args_cfg.get("pool_mode", "avg")),
        use_order_rpm_encoding=bool(int(args_cfg.get("use_order_rpm_encoding", 1))),
        # Fix: read token_mode and time_patch_len so time-token ablation eval works
        token_mode=str(args_cfg.get("token_mode", "freq")),
        time_patch_len=int(args_cfg.get("time_patch_len", 32)),
    ).to(device)
    return model


def load_model_state(model: torch.nn.Module, state: Dict[str, Any], strict_load: bool) -> Dict[str, Any]:
    msg = model.load_state_dict(state, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    report = {
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict_load": bool(strict_load),
    }
    if strict_load and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint/model mismatch under strict_load=1. "
            f"missing_keys={missing[:20]} unexpected_keys={unexpected[:20]}"
        )
    return report


@torch.no_grad()
def infer_cache(
    best_pt: str,
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: str,
    ch_mode: str,
    strict_load: bool,
):
    """Run FreqTokenTransformer forward over all rows in df, return per-row sigmoid probs."""
    ckpt = torch.load(best_pt, map_location=device)
    args_cfg = ckpt.get("args", {}) or {}
    # FreqToken uses SliceDataset without order_resample; pass through ch_mode only.
    ds = SliceDataset(df, ch_mode=ch_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
    model = build_model_from_ckpt(ckpt, device)
    state = ckpt.get("model", ckpt)
    load_report = load_model_state(model, state, strict_load=strict_load)
    model.eval()

    probs: List[np.ndarray] = []
    for x, _y, _yreg, speed in dl:
        x = x.to(device, non_blocking=True)
        x = to_all10_layout(x, ch_mode)
        speed = torch.as_tensor(speed, dtype=torch.float32, device=device)
        out = model(x, rpm=speed)
        prob = torch.sigmoid(out["logits"]).float().cpu().numpy()
        probs.append(prob)
    return np.concatenate(probs, 0), ckpt, load_report


def vote_pool(arr_rows: np.ndarray, idxs: List[List[int]], vote: str) -> np.ndarray:
    out = []
    for g in idxs:
        p = arr_rows[g]
        if vote == "majority":
            y = (p >= 0.5).astype(np.int32)
            y = (y.sum(axis=0) >= (len(g) / 2.0)).astype(np.int32)
            out.append(y.astype(np.float32))
        elif vote == "max":
            out.append(np.max(p, axis=0).astype(np.float32))
        elif vote == "meanprob":
            out.append(np.mean(p, axis=0).astype(np.float32))
        else:
            raise ValueError(f"Unknown vote mode: {vote}")
    return np.stack(out, axis=0)


def smooth_pool_by_npz(df: pd.DataFrame, idxs: List[List[int]], y_pool: np.ndarray, mode: str, window: int) -> np.ndarray:
    mode = str(mode or "none").strip().lower()
    if mode in {"", "none", "raw"}:
        return y_pool
    if mode not in {"rolling", "causal", "file_mean"}:
        raise ValueError(f"Unknown smooth mode: {mode}")
    if mode != "file_mean" and int(window) <= 1:
        return y_pool

    rows = []
    has_start = "start" in df.columns
    has_end = "end" in df.columns
    for gi, g in enumerate(idxs):
        first = df.iloc[g[0]]
        rows.append({
            "gi": gi,
            "npz_path": str(first.get("npz_path", "")),
            "start": float(first.get("start", gi)) if has_start else float(gi),
            "end": float(first.get("end", gi)) if has_end else float(gi),
        })

    meta = pd.DataFrame(rows).sort_values(["npz_path", "start", "end", "gi"], kind="mergesort")
    out = np.array(y_pool, copy=True)
    w = int(window)

    for _key, sub in meta.groupby("npz_path", sort=False):
        order = sub["gi"].to_numpy(dtype=np.int64)
        p = np.asarray(y_pool[order], dtype=np.float32)
        n = p.shape[0]
        if n == 0:
            continue

        sm = np.empty_like(p, dtype=np.float32)
        if mode == "file_mean":
            sm[:] = p.mean(axis=0, keepdims=True)
        elif mode == "causal":
            for i in range(n):
                lo = max(0, i - w + 1)
                sm[i] = p[lo:i + 1].mean(axis=0)
        elif mode == "rolling":
            half = w // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                sm[i] = p[lo:hi].mean(axis=0)
        out[order] = sm

    return out


def group_indices(df: pd.DataFrame) -> Tuple[List[List[int]], np.ndarray, List[str]]:
    cols = df.columns
    if "sample_id" in cols:
        key = df["sample_id"].astype(str)
    else:
        key = df["npz_path"].astype(str) + "|" + df["start"].astype(str) + "|" + df["end"].astype(str)
    groups: Dict[str, List[int]] = {}
    for i, k in enumerate(key.tolist()):
        groups.setdefault(k, []).append(i)
    group_keys = list(groups.keys())
    idxs = list(groups.values())
    y_true = df[["y0", "y1", "y2"]].values.astype(np.int32)
    y_true_g = np.stack([y_true[g[0]] for g in idxs], axis=0)
    return idxs, y_true_g, group_keys


def bitstr_from_row(row: Sequence[int]) -> str:
    return "".join(map(str, np.asarray(row).astype(int).tolist()))


def _apply_threshold(y_prob, t0, t1, t2, cap_k=0, cap_delta=0.0, tau_none=0.0):
    y_prob = np.asarray(y_prob, dtype=np.float32)
    n = y_prob.shape[0]
    y_pred = np.zeros((n, 3), dtype=np.int32)
    y_pred[:, 0] = (y_prob[:, 0] >= t0).astype(np.int32)
    y_pred[:, 1] = (y_prob[:, 1] >= t1).astype(np.int32)
    y_pred[:, 2] = (y_prob[:, 2] >= t2).astype(np.int32)
    thr_vec = np.array([t0, t1, t2], dtype=np.float32)

    if tau_none is not None and float(tau_none) > 0:
        mx = y_prob.max(axis=1)
        none_mask = mx < float(tau_none)
        y_pred[none_mask, :] = 0

    if cap_k and int(cap_k) > 0:
        cap_k = int(cap_k)
        cd = float(cap_delta) if cap_delta is not None else 0.0
        for i in range(n):
            pos = np.where(y_pred[i] > 0)[0]
            if len(pos) <= cap_k:
                continue
            margins = y_prob[i, pos] - thr_vec[pos]
            order = np.argsort(-margins)
            keep = pos[order[:cap_k]]
            if cd and cd > 0 and len(order) > cap_k:
                best_extra_margin = float(margins[order[cap_k]])
                if best_extra_margin >= cd:
                    continue
            new_row = np.zeros(3, dtype=np.int32)
            new_row[keep] = 1
            y_pred[i] = new_row
    return y_pred.astype(np.int32)


def eval_threshold(y_true_g, y_pool, t0, t1, t2, cap_k=0, cap_delta=0.0, tau_none=0.0):
    y_pred = _apply_threshold(y_pool, t0, t1, t2, cap_k=cap_k, cap_delta=cap_delta, tau_none=tau_none)
    met = multilabel_metrics_from_binary(y_true_g, y_pred)
    return met, y_pred


def build_npz_detail_df(df, group_keys, idxs, y_true_g, y_pool, y_pred):
    rows = []
    speed_col = find_speed_col(df)
    for gi, g in enumerate(idxs):
        first = df.iloc[g[0]]
        rows.append({
            "sample_id": group_keys[gi],
            "n_windows": len(g),
            "npz_path": first.get("npz_path", ""),
            "speed": first.get(speed_col, ""),
            "true_y0": int(y_true_g[gi, 0]),
            "true_y1": int(y_true_g[gi, 1]),
            "true_y2": int(y_true_g[gi, 2]),
            "prob_y0": float(y_pool[gi, 0]),
            "prob_y1": float(y_pool[gi, 1]),
            "prob_y2": float(y_pool[gi, 2]),
            "pred_y0": int(y_pred[gi, 0]),
            "pred_y1": int(y_pred[gi, 1]),
            "pred_y2": int(y_pred[gi, 2]),
            "true_k": int(y_true_g[gi].sum()),
            "pred_k": int(y_pred[gi].sum()),
            "true_bits": bitstr_from_row(y_true_g[gi]),
            "pred_bits": bitstr_from_row(y_pred[gi]),
            "is_exact": int(np.all(y_true_g[gi].astype(int) == y_pred[gi].astype(int))),
        })
    return pd.DataFrame(rows)


def build_labelset_confusion_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    true_bits = [bitstr_from_row(r) for r in y_true]
    pred_bits = [bitstr_from_row(r) for r in y_pred]
    uniq = sorted(set(true_bits) | set(pred_bits))
    for tb in uniq:
        for pb in uniq:
            count = sum((t == tb and p == pb) for t, p in zip(true_bits, pred_bits))
            if count > 0:
                rows.append({"true_bits": tb, "pred_bits": pb, "count": int(count)})
    rows.sort(key=lambda x: (-x["count"], x["true_bits"], x["pred_bits"]))
    return pd.DataFrame(rows)


def maybe_filter_speed(df: pd.DataFrame, speeds: str):
    if not str(speeds or "").strip():
        return df.reset_index(drop=True)
    speed_col = find_speed_col(df)
    sp_list = [int(x) for x in str(speeds).split(",") if str(x).strip()]
    return df[df[speed_col].astype(int).isin(sp_list)].reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: str):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def maybe_warn_overlap(eval_group_keys: Sequence[str], calib_group_keys: Sequence[str], out_prefix: str) -> Dict[str, Any]:
    eval_set = set(map(str, eval_group_keys))
    calib_set = set(map(str, calib_group_keys))
    overlap = sorted(eval_set & calib_set)
    report = {
        "eval_groups": len(eval_set),
        "calib_groups": len(calib_set),
        "overlap_groups": len(overlap),
        "overlap_examples": overlap[:20],
    }
    with open(out_prefix + "_calib_overlap_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if overlap:
        print(f"[WARN] calib/eval group overlap detected: {len(overlap)} overlapping groups. See _calib_overlap_check.json")
    return report


def main():
    ap = argparse.ArgumentParser(description="Safe eval for FreqTokenTransformer checkpoints")
    ap.add_argument("--best_pt", required=True, help="Path to best.pt checkpoint")
    ap.add_argument("--meta_csv", required=True, help="meta csv with y0/y1/y2 + start/end + npz_path")
    ap.add_argument("--calib_meta_csv", default="", help="Optional calib meta csv for threshold sweep")
    ap.add_argument("--out_prefix", required=True, help="Output prefix; _summary.csv etc appended")
    ap.add_argument("--speeds", default="", help="Comma-separated speed_rpm filter, e.g. 600,900")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ch_mode", default="all10", choices=["accel6", "accel_orbit", "accel_acoustic", "all10"])
    ap.add_argument("--vote", default="meanprob", choices=["meanprob", "max", "majority"])
    ap.add_argument("--smooth_mode", default="none", choices=["none", "rolling", "causal", "file_mean"])
    ap.add_argument("--smooth_window", type=int, default=1)
    ap.add_argument("--thr", default="0.5,0.5,0.5")
    ap.add_argument("--tau_none", type=float, default=0.0)
    ap.add_argument("--cap_k", type=int, default=0)
    ap.add_argument("--thr0_list", default="")
    ap.add_argument("--thr1_list", default="")
    ap.add_argument("--thr2_list", default="")
    ap.add_argument("--tau_none_list", default="0.0")
    ap.add_argument("--cap_k_list", default="0")
    ap.add_argument("--cap_delta", type=float, default=0.0)
    ap.add_argument("--calib_key", default="exact_match")
    ap.add_argument("--strict_load", default="0", help="Default 0: FreqToken ckpt has extra keys (proto_bank etc).")
    args = ap.parse_args()

    strict_load = parse_bool_int(args.strict_load)

    df = maybe_filter_speed(pd.read_csv(args.meta_csv), args.speeds)
    prob_rows, ckpt, load_report = infer_cache(
        args.best_pt,
        df,
        args.batch_size,
        args.num_workers,
        args.device,
        args.ch_mode,
        strict_load,
    )
    idxs, y_true_g, group_keys = group_indices(df)
    y_pool = vote_pool(prob_rows, idxs, args.vote)
    y_pool = smooth_pool_by_npz(df, idxs, y_pool, args.smooth_mode, args.smooth_window)

    thr = parse_list_float(args.thr)
    if len(thr) != 3:
        raise ValueError("--thr must be 3 floats")

    base_met, y_pred = eval_threshold(
        y_true_g, y_pool, thr[0], thr[1], thr[2],
        cap_k=args.cap_k, cap_delta=args.cap_delta, tau_none=args.tau_none,
    )

    best_calib = {
        "t0": thr[0], "t1": thr[1], "t2": thr[2],
        "tau_none": float(args.tau_none), "cap_k": int(args.cap_k),
        "score": float(base_met.get(args.calib_key, -1e18)),
    }
    overlap_report = None

    if str(args.calib_meta_csv).strip():
        dfc = pd.read_csv(args.calib_meta_csv)
        prob_rows_c, _ckpt_c, _load_report_c = infer_cache(
            args.best_pt, dfc, args.batch_size, args.num_workers, args.device, args.ch_mode, strict_load,
        )
        idxs_c, y_true_c, group_keys_c = group_indices(dfc)
        y_pool_c = vote_pool(prob_rows_c, idxs_c, args.vote)
        y_pool_c = smooth_pool_by_npz(dfc, idxs_c, y_pool_c, args.smooth_mode, args.smooth_window)

        overlap_report = maybe_warn_overlap(group_keys, group_keys_c, args.out_prefix)

        thr0_list = parse_list_float(args.thr0_list) or [thr[0]]
        thr1_list = parse_list_float(args.thr1_list) or [thr[1]]
        thr2_list = parse_list_float(args.thr2_list) or [thr[2]]
        tau_list = parse_list_float(args.tau_none_list) or [float(args.tau_none)]
        cap_k_list_raw = parse_list_float(args.cap_k_list)
        cap_k_list = [int(v) for v in cap_k_list_raw] or [int(args.cap_k)]

        sweep_rows = []
        best_score = -1e18
        for t0 in thr0_list:
            for t1 in thr1_list:
                for t2 in thr2_list:
                    for tau in tau_list:
                        for cap_k in cap_k_list:
                            met_c, _ = eval_threshold(y_true_c, y_pool_c, t0, t1, t2, cap_k=cap_k, cap_delta=args.cap_delta, tau_none=tau)
                            row = {"t0": t0, "t1": t1, "t2": t2, "tau_none": tau, "cap_k": int(cap_k), **met_c}
                            sweep_rows.append(row)
                            score = float(met_c.get(args.calib_key, -1e18))
                            if score > best_score:
                                best_score = score
                                best_calib = {"t0": t0, "t1": t1, "t2": t2, "tau_none": tau, "cap_k": int(cap_k), "score": score}
        save_csv(pd.DataFrame(sweep_rows), args.out_prefix + "_calib_sweep.csv")

        base_met, y_pred = eval_threshold(
            y_true_g, y_pool, best_calib["t0"], best_calib["t1"], best_calib["t2"],
            cap_k=best_calib["cap_k"], cap_delta=args.cap_delta, tau_none=best_calib["tau_none"],
        )
        with open(args.out_prefix + "_calib_best.json", "w", encoding="utf-8") as f:
            json.dump(best_calib, f, ensure_ascii=False, indent=2)

    per_label = prf_per_label(y_true_g, y_pred)
    labelset_conf = build_labelset_confusion_df(y_true_g, y_pred)

    summary_row = {
        "best_pt": args.best_pt,
        "model_type": ckpt.get("model_type", "freqtoken_transformer"),
        "meta_csv": args.meta_csv,
        "calib_meta_csv": args.calib_meta_csv,
        "vote": args.vote,
        "smooth_mode": args.smooth_mode,
        "smooth_window": int(args.smooth_window),
        "thr": f"{best_calib['t0']},{best_calib['t1']},{best_calib['t2']}",
        "tau_none": best_calib["tau_none"],
        "cap_k": best_calib["cap_k"],
        "cap_delta": args.cap_delta,
        "strict_load": int(strict_load),
        "missing_keys_count": load_report["missing_keys_count"],
        "unexpected_keys_count": load_report["unexpected_keys_count"],
        **base_met,
    }
    if overlap_report is not None:
        summary_row["calib_eval_overlap_groups"] = overlap_report["overlap_groups"]

    save_csv(pd.DataFrame([summary_row]), args.out_prefix + "_summary.csv")
    save_csv(pd.DataFrame(per_label), args.out_prefix + "_per_label.csv")
    save_csv(pd.DataFrame([load_report]), args.out_prefix + "_load_report.csv")
    save_csv(build_npz_detail_df(df, group_keys, idxs, y_true_g, y_pool, y_pred), args.out_prefix + "_npz_details.csv")
    save_csv(labelset_conf, args.out_prefix + "_labelset_confusion.csv")

    with open(args.out_prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_row, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
