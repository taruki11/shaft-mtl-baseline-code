import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .npz_reader import load_npz_array
from .utils import find_reg_cols

# 每个进程各自维护缓存
_LAST_NPZ_PATH = None
_LAST_NPZ_ARR = None


def _load_with_cache(npz_path):
    global _LAST_NPZ_PATH, _LAST_NPZ_ARR
    if _LAST_NPZ_PATH == npz_path and _LAST_NPZ_ARR is not None:
        return _LAST_NPZ_ARR
    arr = load_npz_array(npz_path)
    _LAST_NPZ_PATH = npz_path
    _LAST_NPZ_ARR = arr
    return arr


class SliceDataset(Dataset):
    def __init__(
        self,
        meta_csv,
        ch_mode: str = "all10",
        order_source_resample_enable: int = 0,
        order_source_ref_rpm: float = 1200.0,
        order_source_min_scale: float = 0.5,
        order_source_max_scale: float = 2.0,
        order_source_pad_mode: str = "edge",
    ):
        if isinstance(meta_csv, pd.DataFrame):
            self.df = meta_csv.reset_index(drop=True)
        else:
            self.df = pd.read_csv(meta_csv)

        self.ch_mode = (ch_mode or "all10").lower().strip()
        self.order_source_resample_enable = bool(int(order_source_resample_enable))
        self.order_source_ref_rpm = float(max(order_source_ref_rpm, 1e-6))
        self.order_source_min_scale = float(max(order_source_min_scale, 1e-3))
        self.order_source_max_scale = float(max(order_source_max_scale, self.order_source_min_scale))
        self.order_source_pad_mode = str(order_source_pad_mode or "edge").lower()
        if self.order_source_pad_mode not in {"edge", "reflect", "zero"}:
            raise ValueError(
                f"Unsupported order_source_pad_mode={order_source_pad_mode!r}; "
                "expected edge, reflect, or zero."
            )

        # 动态识别回归列 (r0-r6 或 reg0-reg6) — 统一使用共享实现
        self.reg_cols = find_reg_cols(self.df, reg_dim=7)

        # 通道选择逻辑
        if self.ch_mode == "accel6":
            self.ch_idx = slice(0, 6)
        elif self.ch_mode in ["accel_orbit", "accel8"]:
            self.ch_idx = slice(0, 8)
        elif self.ch_mode in ["accel_acoustic", "accel6_acoustic2"]:
            self.ch_idx = [0, 1, 2, 3, 4, 5, 8, 9]
        else:
            self.ch_idx = None

    def __len__(self):
        return len(self.df)

    def _slice_source_window(self, x, s: int, e: int, speed_rpm: float):
        target_len = max(int(e) - int(s), 1)
        if not self.order_source_resample_enable:
            return x[:, s:e], target_len

        scale = self.order_source_ref_rpm / max(float(speed_rpm), 1e-6)
        scale = min(max(scale, self.order_source_min_scale), self.order_source_max_scale)
        source_len = max(int(round(target_len * scale)), 8)
        center = (int(s) + int(e)) // 2
        ss = center - source_len // 2
        ee = ss + source_len
        total_len = int(x.shape[1])

        pad_left = max(0, -ss)
        pad_right = max(0, ee - total_len)
        ss_clip = max(ss, 0)
        ee_clip = min(ee, total_len)
        xs = x[:, ss_clip:ee_clip]
        if pad_left > 0 or pad_right > 0:
            if self.order_source_pad_mode == "zero":
                xs = np.pad(xs, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0.0)
            elif self.order_source_pad_mode == "reflect" and xs.shape[1] > 1:
                xs = np.pad(xs, ((0, 0), (pad_left, pad_right)), mode="reflect")
            else:
                xs = np.pad(xs, ((0, 0), (pad_left, pad_right)), mode="edge")
        return xs, target_len

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        x = _load_with_cache(r["npz_path"])
        s, e = int(r["start"]), int(r["end"])
        speed = np.float32(r["speed_rpm"])

        xs, target_len = self._slice_source_window(x, s, e, float(speed))
        if self.ch_idx is not None:
            xs = xs[self.ch_idx, :]

        if (not xs.flags["C_CONTIGUOUS"]) or (not xs.flags["WRITEABLE"]):
            xs = np.ascontiguousarray(xs, dtype=np.float32).copy()
        else:
            xs = xs.astype(np.float32, copy=False)

        if self.order_source_resample_enable and xs.shape[1] != target_len:
            xt = torch.from_numpy(xs).unsqueeze(0)
            xt = F.interpolate(xt, size=int(target_len), mode="linear", align_corners=False)
            xs_t = xt.squeeze(0).contiguous()
        else:
            xs_t = torch.from_numpy(xs)

        # 标签处理
        y = np.array([r["y0"], r["y1"], r["y2"]], dtype=np.float32)
        # 使用动态找出的回归列名
        yreg = np.array([r[col] for col in self.reg_cols], dtype=np.float32)

        return xs_t, torch.from_numpy(y), torch.from_numpy(yreg), speed


# 必须定义此函数，供主脚本调用
def _make_dataset(meta_csv, ch_mode):
    return SliceDataset(meta_csv, ch_mode=ch_mode)
