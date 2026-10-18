import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def multilabel_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    out["exact_match"] = float((y_pred == y_true).all(axis=1).mean())
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return out
import numpy as np

def masked_regression_metrics(y_true_reg, y_pred_reg, y_true_cls, eps=1e-12, use_range=True):
    """
    y_true_reg/y_pred_reg: [N,7]
    y_true_cls: [N,3]
    返回：
      - 每维 MAE/RMSE（原单位）
      - 每维 NMAE/NRMSE（无量纲，按range或std归一化）
      - 总体 mean_nmae / mean_nrmse（方便做一个回归总分）
      - r4-6 额外 acc（因为本质是0/1）
    """
    y_true_reg = np.asarray(y_true_reg, dtype=np.float64)
    y_pred_reg = np.asarray(y_pred_reg, dtype=np.float64)
    y_true_cls = np.asarray(y_true_cls, dtype=np.float64)

    mask = np.zeros_like(y_true_reg, dtype=np.float64)
    mask[:, 0] = y_true_cls[:, 0]
    mask[:, 1:4] = y_true_cls[:, 1:2]
    mask[:, 4:7] = y_true_cls[:, 2:3]

    out = {}
    per_dim_mae = []
    per_dim_rmse = []
    per_dim_nmae = []
    per_dim_nrmse = []

    # 归一化尺度：用range或std（都从“masked有效样本”里估计）
    for j in range(7):
        idx = mask[:, j] > 0.5
        if idx.sum() == 0:
            out[f"r{j}_mae"] = np.nan
            out[f"r{j}_rmse"] = np.nan
            out[f"r{j}_nmae"] = np.nan
            out[f"r{j}_nrmse"] = np.nan
            continue

        t = y_true_reg[idx, j]
        p = y_pred_reg[idx, j]
        err = p - t

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))

        if use_range:
            sc = float(np.max(t) - np.min(t))
        else:
            sc = float(np.std(t))
        sc = max(sc, eps)

        nmae = float(mae / sc)
        nrmse = float(rmse / sc)

        out[f"r{j}_mae"] = mae
        out[f"r{j}_rmse"] = rmse
        out[f"r{j}_nmae"] = nmae
        out[f"r{j}_nrmse"] = nrmse

        per_dim_mae.append(mae)
        per_dim_rmse.append(rmse)
        per_dim_nmae.append(nmae)
        per_dim_nrmse.append(nrmse)

    # 汇总（忽略nan）
    out["reg_mae_mean"] = float(np.nanmean(per_dim_mae)) if len(per_dim_mae) else np.nan
    out["reg_rmse_mean"] = float(np.nanmean(per_dim_rmse)) if len(per_dim_rmse) else np.nan
    out["reg_nmae_mean"] = float(np.nanmean(per_dim_nmae)) if len(per_dim_nmae) else np.nan
    out["reg_nrmse_mean"] = float(np.nanmean(per_dim_nrmse)) if len(per_dim_nrmse) else np.nan

    # r4-6: 0/1位置，额外给acc（在y2=1时）
    for j in [4, 5, 6]:
        idx = mask[:, j] > 0.5
        if idx.sum() == 0:
            out[f"r{j}_acc"] = np.nan
        else:
            pred_bin = (y_pred_reg[idx, j] >= 0.5).astype(int)
            true_bin = (y_true_reg[idx, j] >= 0.5).astype(int)
            out[f"r{j}_acc"] = float((pred_bin == true_bin).mean())

    return out
