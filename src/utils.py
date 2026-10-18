import os, re, yaml
from typing import Any, Dict, List, Optional

import pandas as pd


def find_y_cols(df: pd.DataFrame) -> List[str]:
    """Find y0/y1/y2 label columns in a DataFrame (case-insensitive)."""
    cols = list(df.columns)
    norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in cols}
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


def find_reg_cols(df: pd.DataFrame, reg_dim: int = 7) -> List[str]:
    """Find regression columns (r0-r6, reg0-reg6, or reg_0-reg_6) in a DataFrame."""
    cols = list(df.columns)
    norm_map = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in cols}
    patterns = [
        [f"reg{i}" for i in range(reg_dim)],
        [f"reg_{i}" for i in range(reg_dim)],
        [f"r{i}" for i in range(reg_dim)],
    ]
    for pat in patterns:
        found = []
        ok = True
        for k in pat:
            nk = re.sub(r"[^a-z0-9]+", "", k.lower())
            if nk in norm_map:
                found.append(norm_map[nk])
            else:
                ok = False
                break
        if ok and len(found) == reg_dim:
            return found
    reg_like = []
    for c in cols:
        n = re.sub(r"[^a-z0-9]+", "", c.lower())
        m = re.match(r"(reg|r)(\d+)$", n)
        if m:
            reg_like.append((int(m.group(2)), c))
    reg_like = sorted(reg_like, key=lambda x: x[0])
    if reg_like and len(reg_like) >= reg_dim:
        reg_like = reg_like[:reg_dim]
        return [c for _, c in reg_like]
    raise KeyError("Cannot find 7 regression columns in meta.csv")


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_alias_map(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("alias", {}) or {}

def stem_without_ext(p: str) -> str:
    base = os.path.basename(p)
    return os.path.splitext(base)[0]

def resolve_npz_path(npz_dir: str, serial: str, alias_map: Dict[str, str]) -> Optional[str]:
    """Try to locate an npz file by serial or alias. Returns path or None."""
    candidates = []
    serial = str(serial).strip()
    if not serial:
        return None
    candidates.append(serial)
    if serial in alias_map:
        candidates.append(alias_map[serial])
    # Also try normalize: remove spaces
    candidates = list(dict.fromkeys([c.strip() for c in candidates if c and str(c).strip()]))
    for c in candidates:
        p = os.path.join(npz_dir, f"{c}.npz")
        if os.path.exists(p):
            return p
    # If not found, try fuzzy: any file that startswith serial
    try:
        files = [f for f in os.listdir(npz_dir) if f.lower().endswith('.npz')]
    except FileNotFoundError:
        return None
    for f in files:
        st = os.path.splitext(f)[0]
        if st == serial or st.startswith(serial):
            return os.path.join(npz_dir, f)
    return None

def parse_tuple3(s: str):
    """Parse '(0.2, 0 , 0 )' -> (0.2,0,0). Returns (0,0,0) if invalid or '—'."""
    if s is None:
        return (0.0, 0.0, 0.0)
    s = str(s).strip()
    if not s or s == '—' or s.lower() == 'nan':
        return (0.0, 0.0, 0.0)
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    if len(m) >= 3:
        return tuple(float(x) for x in m[:3])
    return (0.0, 0.0, 0.0)

def parse_unbalance_screws(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s).strip()
    if not s or s == '—' or s.lower() == 'nan':
        return 0.0
    m = re.findall(r"\d+", s)
    return float(m[0]) if m else 0.0

def parse_rub_onehot(s: str):
    """'轴承2碰磨' -> [0,1,0]. Unknown or '—' -> [0,0,0]."""
    if s is None:
        return (0.0, 0.0, 0.0)
    s = str(s).strip()
    if not s or s == '—' or s.lower() == 'nan':
        return (0.0, 0.0, 0.0)
    if '轴承1' in s:
        return (1.0, 0.0, 0.0)
    if '轴承2' in s:
        return (0.0, 1.0, 0.0)
    if '轴承3' in s:
        return (0.0, 0.0, 1.0)
    return (0.0, 0.0, 0.0)
