# -*- coding: utf-8 -*-
"""Build strict/quasi-strict order-domain fixed-revolution windows from source signals.

This version supports both:
1) per-sample .npz/.npy windows
2) raw continuous .npy/.npz files referenced by meta rows with start/end columns

Key upgrades over v1
--------------------
- Supports .npy arrays directly (np.load returning ndarray)
- Supports object .npy saved as dict-like payloads
- Supports slicing from continuous raw arrays via start/end columns in meta
- Adds simple file cache so repeated rows from the same raw file do not reload every time
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_X_KEYS = ["x", "signal", "signals", "data", "arr_0"]


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    norm_map = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        nc = normalize_name(cand)
        if nc in norm_map:
            return norm_map[nc]
    if required:
        raise KeyError(f"Cannot find required column among candidates={list(candidates)}. Existing columns={list(df.columns)}")
    return None


def find_y_cols(df: pd.DataFrame) -> List[str]:
    return [find_col(df, [f"y{i}"]) for i in range(3)]


def find_reg_cols(df: pd.DataFrame, reg_dim: int = 7) -> List[str]:
    norm_map = {normalize_name(c): c for c in df.columns}
    patterns = [[f"reg{i}" for i in range(reg_dim)], [f"r{i}" for i in range(reg_dim)], [f"reg_{i}" for i in range(reg_dim)]]
    for pat in patterns:
        out = []
        ok = True
        for p in pat:
            npat = normalize_name(p)
            if npat not in norm_map:
                ok = False
                break
            out.append(norm_map[npat])
        if ok:
            return out
    raise KeyError("Cannot find regression columns reg0..reg6 / r0..r6 in meta CSV")


def _normalize_2d_signal(x: np.ndarray, src_desc: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Signal must be 2D [C,T] or [T,C], got shape={x.shape} from {src_desc}")
    # Prefer channel-first when one dim looks like channel count
    if x.shape[0] > x.shape[1] and x.shape[1] <= 64:
        x = x.T
    if x.shape[0] > 64 and x.shape[1] <= 64:
        x = x.T
    return x.astype(np.float32, copy=False)


def load_signal_payload(src_path: str, x_key: Optional[str] = None) -> Tuple[np.ndarray, str]:
    obj = np.load(src_path, allow_pickle=True)

    # Case A: npz file
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        chosen = None
        if x_key is not None:
            if x_key not in obj:
                raise KeyError(f"Requested x_key={x_key} not in keys={keys} for file={src_path}")
            chosen = x_key
        else:
            for k in DEFAULT_X_KEYS:
                if k in obj:
                    chosen = k
                    break
            if chosen is None and len(keys) == 1:
                chosen = keys[0]
        if chosen is None:
            raise KeyError(f"Cannot infer signal key for NPZ keys={keys} file={src_path}")
        return _normalize_2d_signal(np.asarray(obj[chosen]), f"{src_path}:{chosen}"), chosen

    # Case B: npy saved dict/object
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        payload = obj.item()
        if isinstance(payload, dict):
            keys = list(payload.keys())
            chosen = None
            if x_key is not None:
                if x_key not in payload:
                    raise KeyError(f"Requested x_key={x_key} not in object payload keys={keys} file={src_path}")
                chosen = x_key
            else:
                for k in DEFAULT_X_KEYS:
                    if k in payload:
                        chosen = k
                        break
                if chosen is None and len(keys) == 1:
                    chosen = keys[0]
            if chosen is None:
                raise KeyError(f"Cannot infer signal key for object payload keys={keys} file={src_path}")
            return _normalize_2d_signal(np.asarray(payload[chosen]), f"{src_path}:{chosen}"), chosen

    # Case C: plain ndarray .npy
    if isinstance(obj, np.ndarray):
        return _normalize_2d_signal(obj, src_path), "__array__"

    raise TypeError(f"Unsupported source type from np.load({src_path}): {type(obj)}")


def resample_to_order_domain(x: np.ndarray, rpm: float, fs: float, ppr: int) -> np.ndarray:
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    samples_per_rev = fs * 60.0 / float(rpm)
    T = x.shape[1]
    rev_pos = np.arange(T, dtype=np.float32) / np.float32(samples_per_rev)
    total_revs = float(rev_pos[-1]) if T > 1 else 0.0
    if total_revs <= (1.0 / ppr):
        return np.zeros((x.shape[0], 1), dtype=np.float32)
    target_rev_pos = np.arange(0.0, total_revs, 1.0 / float(ppr), dtype=np.float32)
    y = np.zeros((x.shape[0], len(target_rev_pos)), dtype=np.float32)
    for c in range(x.shape[0]):
        y[c] = np.interp(target_rev_pos, rev_pos, x[c]).astype(np.float32)
    return y


def slice_order_windows(x_order: np.ndarray, ppr: int, revs_per_win: int, step_revs: int) -> List[Tuple[np.ndarray, int, int]]:
    win_len = int(ppr * revs_per_win)
    step_len = int(ppr * step_revs)
    if step_len <= 0:
        raise ValueError("step_revs must produce positive step_len")
    if x_order.shape[1] < win_len:
        return []
    out = []
    for s in range(0, x_order.shape[1] - win_len + 1, step_len):
        e = s + win_len
        out.append((x_order[:, s:e].copy(), s, e))
    return out


def make_relpath(path: str, root: Optional[str]) -> str:
    if root is None:
        return path
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def maybe_resolve_source_path(src_path: str, meta_dir: str, project_root: Optional[str]) -> str:
    cands = []
    if os.path.isabs(src_path):
        cands.append(src_path)
    else:
        cands.append(os.path.normpath(os.path.join(meta_dir, src_path)))
        if project_root is not None:
            cands.append(os.path.normpath(os.path.join(project_root, src_path)))
        cands.append(os.path.normpath(src_path))
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot resolve source path={src_path}. Tried={cands}")


def _safe_int(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        return int(round(float(v)))
    except Exception:
        return None


def extract_row_signal(full_x: np.ndarray,
                       row: pd.Series,
                       start_col: Optional[str],
                       end_col: Optional[str]) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    start = _safe_int(row[start_col]) if start_col else None
    end = _safe_int(row[end_col]) if end_col else None

    if start is None or end is None:
        return full_x, start, end

    if end <= start:
        raise ValueError(f"Invalid start/end: start={start}, end={end}")

    T = full_x.shape[1]
    start = max(0, min(start, T))
    end = max(0, min(end, T))
    if end <= start:
        raise ValueError(f"Clamped start/end invalid for signal length {T}: start={start}, end={end}")
    return full_x[:, start:end], start, end


def build_rows(df: pd.DataFrame,
               meta_csv: str,
               out_root: str,
               out_meta_csv: str,
               project_root: Optional[str],
               fs: float,
               ppr: int,
               revs_per_win: int,
               step_revs: int,
               x_key: Optional[str],
               source_path_col: str,
               rpm_col: str,
               serial_col: str,
               start_col: Optional[str],
               end_col: Optional[str],
               verbose_every: int = 200) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    skipped_short = 0
    skipped_missing = 0
    skipped_bad_slice = 0
    signal_key_hist: Dict[str, int] = {}
    per_speed_counts: Dict[str, int] = {}
    source_cache: Dict[str, Tuple[np.ndarray, str]] = {}
    cache_hits = 0

    os.makedirs(out_root, exist_ok=True)
    meta_dir = os.path.dirname(os.path.abspath(meta_csv))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="build_order_windows"):
        try:
            src_path = maybe_resolve_source_path(str(row[source_path_col]), meta_dir, project_root)
        except FileNotFoundError:
            skipped_missing += 1
            continue

        if src_path in source_cache:
            full_x, used_key = source_cache[src_path]
            cache_hits += 1
        else:
            full_x, used_key = load_signal_payload(src_path, x_key=x_key)
            source_cache[src_path] = (full_x, used_key)
        signal_key_hist[used_key] = signal_key_hist.get(used_key, 0) + 1

        try:
            x, src_start, src_end = extract_row_signal(full_x, row, start_col=start_col, end_col=end_col)
        except Exception:
            skipped_bad_slice += 1
            continue

        rpm = float(row[rpm_col])
        x_order = resample_to_order_domain(x=x, rpm=rpm, fs=fs, ppr=ppr)
        wins = slice_order_windows(x_order, ppr=ppr, revs_per_win=revs_per_win, step_revs=step_revs)
        if len(wins) == 0:
            skipped_short += 1
            continue

        src_stem = Path(src_path).stem
        speed_tag = f"{int(round(rpm))}rpm"
        subdir = os.path.join(out_root, speed_tag)
        os.makedirs(subdir, exist_ok=True)

        base_serial = str(row[serial_col])
        for wi, (x_win, s, e) in enumerate(wins):
            out_name = f"{src_stem}__ord_ppr{ppr}_r{revs_per_win}_w{wi:03d}.npz"
            out_npz = os.path.join(subdir, out_name)
            np.savez_compressed(
                out_npz,
                x=x_win.astype(np.float32),
                rpm=np.float32(rpm),
                order_ppr=np.int32(ppr),
                order_revs=np.int32(revs_per_win),
                order_step_revs=np.int32(step_revs),
                order_start_idx=np.int32(s),
                order_end_idx=np.int32(e),
                src_signal_path=str(src_path),
                src_start=np.int32(-1 if src_start is None else src_start),
                src_end=np.int32(-1 if src_end is None else src_end),
            )

            new_row = dict(row)
            new_row[source_path_col] = make_relpath(out_npz, project_root) if project_root else out_npz
            new_row["src_signal_path"] = str(src_path)
            new_row["order_domain"] = 1
            new_row["order_ppr"] = int(ppr)
            new_row["order_revs"] = int(revs_per_win)
            new_row["order_step_revs"] = int(step_revs)
            new_row["order_start_idx"] = int(s)
            new_row["order_end_idx"] = int(e)
            new_row["derived_window_id"] = int(wi)
            if src_start is not None:
                new_row["src_start"] = int(src_start)
            if src_end is not None:
                new_row["src_end"] = int(src_end)
            new_row[serial_col] = f"{base_serial}__ordppr{ppr}_r{revs_per_win}_w{wi:03d}"
            rows.append(new_row)
            per_speed_counts[speed_tag] = per_speed_counts.get(speed_tag, 0) + 1

        if verbose_every > 0 and (idx + 1) % verbose_every == 0:
            print(f"[build] processed={idx+1} rows={len(rows)} skipped_short={skipped_short} skipped_missing={skipped_missing} skipped_bad_slice={skipped_bad_slice} cache_hits={cache_hits}")

    out_df = pd.DataFrame(rows)
    summary = {
        "src_rows": int(len(df)),
        "out_rows": int(len(out_df)),
        "skipped_short": int(skipped_short),
        "skipped_missing": int(skipped_missing),
        "skipped_bad_slice": int(skipped_bad_slice),
        "signal_key_hist": signal_key_hist,
        "per_speed_counts": per_speed_counts,
        "ppr": int(ppr),
        "revs_per_win": int(revs_per_win),
        "step_revs": int(step_revs),
        "fs": float(fs),
        "start_col": start_col,
        "end_col": end_col,
        "cache_files": int(len(source_cache)),
        "cache_hits": int(cache_hits),
    }
    return out_df, summary


def main(args):
    df = pd.read_csv(args.meta_csv)

    source_path_col = find_col(df, [args.source_path_col, "npz_path", "npy_path", "path"])
    rpm_col = find_col(df, [args.rpm_col, "speed_rpm", "rpm"])
    serial_col = find_col(df, [args.serial_col, "serial"])
    start_col = find_col(df, [args.start_col, "start", "start_idx", "begin"], required=False)
    end_col = find_col(df, [args.end_col, "end", "end_idx", "stop"], required=False)

    _ = find_y_cols(df)
    _ = find_reg_cols(df, reg_dim=7)

    out_df, summary = build_rows(
        df=df,
        meta_csv=args.meta_csv,
        out_root=args.out_root,
        out_meta_csv=args.out_meta_csv,
        project_root=args.project_root,
        fs=args.fs,
        ppr=args.ppr,
        revs_per_win=args.revs_per_win,
        step_revs=args.step_revs,
        x_key=args.x_key,
        source_path_col=source_path_col,
        rpm_col=rpm_col,
        serial_col=serial_col,
        start_col=start_col,
        end_col=end_col,
        verbose_every=args.verbose_every,
    )

    if len(out_df) == 0:
        raise RuntimeError("No output rows were generated. Check source path / start-end columns / fs / ppr / revs_per_win.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_meta_csv)), exist_ok=True)
    out_df.to_csv(args.out_meta_csv, index=False, encoding="utf-8-sig")

    summary_path = os.path.splitext(args.out_meta_csv)[0] + "__build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "out_meta_csv": args.out_meta_csv,
        "summary_json": summary_path,
        **summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--out_meta_csv", type=str, required=True)
    ap.add_argument("--project_root", type=str, default=None)
    ap.add_argument("--source_path_col", type=str, default="npz_path")
    ap.add_argument("--rpm_col", type=str, default="speed_rpm")
    ap.add_argument("--serial_col", type=str, default="serial")
    ap.add_argument("--start_col", type=str, default="start")
    ap.add_argument("--end_col", type=str, default="end")
    ap.add_argument("--x_key", type=str, default=None)
    ap.add_argument("--fs", type=float, default=5000.0)
    ap.add_argument("--ppr", type=int, default=256)
    ap.add_argument("--revs_per_win", type=int, default=8)
    ap.add_argument("--step_revs", type=int, default=4)
    ap.add_argument("--verbose_every", type=int, default=200)
    args = ap.parse_args()
    main(args)
