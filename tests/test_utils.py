"""Tests for utility functions."""
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import find_reg_cols, find_y_cols


def test_find_y_cols_standard():
    """find_y_cols finds standard y0/y1/y2 columns."""
    df = pd.DataFrame({"y0": [1], "y1": [0], "y2": [1]})
    cols = find_y_cols(df)
    assert cols == ["y0", "y1", "y2"]


def test_find_y_cols_label_prefix():
    """find_y_cols falls back to label0/label1/label2."""
    df = pd.DataFrame({"label0": [1], "label1": [0], "label2": [1]})
    cols = find_y_cols(df)
    assert cols == ["label0", "label1", "label2"]


def test_find_y_cols_case_insensitive():
    """find_y_cols is case-insensitive."""
    df = pd.DataFrame({"Y0": [1], "Y1": [0], "Y2": [1]})
    cols = find_y_cols(df)
    assert cols == ["Y0", "Y1", "Y2"]


def test_find_y_cols_missing_raises():
    """find_y_cols raises KeyError when columns are missing."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    try:
        find_y_cols(df)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_find_reg_cols_r_prefix():
    """find_reg_cols finds r0-r6 columns."""
    df = pd.DataFrame({f"r{i}": [i] for i in range(7)})
    cols = find_reg_cols(df)
    assert cols == [f"r{i}" for i in range(7)]


def test_find_reg_cols_reg_prefix():
    """find_reg_cols finds reg0-reg6 columns."""
    df = pd.DataFrame({f"reg{i}": [i] for i in range(7)})
    cols = find_reg_cols(df)
    assert cols == [f"reg{i}" for i in range(7)]


def test_find_reg_cols_reg_underscore_prefix():
    """find_reg_cols finds reg_0-reg_6 columns."""
    df = pd.DataFrame({f"reg_{i}": [i] for i in range(7)})
    cols = find_reg_cols(df)
    assert cols == [f"reg_{i}" for i in range(7)]


if __name__ == "__main__":
    test_find_y_cols_standard()
    test_find_y_cols_label_prefix()
    test_find_y_cols_case_insensitive()
    test_find_y_cols_missing_raises()
    test_find_reg_cols_r_prefix()
    test_find_reg_cols_reg_prefix()
    test_find_reg_cols_reg_underscore_prefix()
    print("All utils tests passed.")
