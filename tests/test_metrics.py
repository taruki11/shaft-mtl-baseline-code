"""Tests for metrics functions."""
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import masked_regression_metrics, multilabel_metrics


def test_multilabel_metrics_perfect():
    """Perfect predictions give exact_match=1.0 and macro_f1=1.0."""
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_prob = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.9, 0.9, 0.1]])
    met = multilabel_metrics(y_true, y_prob, thr=0.5)
    assert met["exact_match"] == 1.0
    assert abs(met["macro_f1"] - 1.0) < 1e-6
    assert abs(met["micro_f1"] - 1.0) < 1e-6


def test_multilabel_metrics_wrong():
    """All-wrong predictions give exact_match=0.0."""
    y_true = np.array([[1, 0, 0]])
    y_prob = np.array([[0.1, 0.9, 0.9]])
    met = multilabel_metrics(y_true, y_prob, thr=0.5)
    assert met["exact_match"] == 0.0


def test_masked_regression_metrics_basic():
    """Masked regression metrics handle masked dimensions correctly."""
    y_true_reg = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    y_pred_reg = y_true_reg.copy()  # Perfect prediction
    y_true_cls = np.array([[1.0, 1.0, 1.0]])
    out = masked_regression_metrics(y_true_reg, y_pred_reg, y_true_cls)
    assert abs(out["r0_mae"]) < 1e-6
    assert abs(out["reg_mae_mean"]) < 1e-6


def test_masked_regression_metrics_partial_mask():
    """Only active labels contribute to regression metrics."""
    y_true_reg = np.array([[1.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]])
    y_pred_reg = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y_true_cls = np.array([[1.0, 0.0, 0.0]])  # Only r0 active
    out = masked_regression_metrics(y_true_reg, y_pred_reg, y_true_cls)
    assert abs(out["r0_mae"]) < 1e-6
    assert np.isnan(out["r1_mae"])  # Masked => NaN
    assert abs(out["reg_mae_mean"]) < 1e-6  # Only r0 counts


if __name__ == "__main__":
    test_multilabel_metrics_perfect()
    test_multilabel_metrics_wrong()
    test_masked_regression_metrics_basic()
    test_masked_regression_metrics_partial_mask()
    print("All metrics tests passed.")
