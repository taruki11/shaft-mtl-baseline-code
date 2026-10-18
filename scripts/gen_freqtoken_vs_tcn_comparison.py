# -*- coding: utf-8 -*-
"""Generate FreqToken vs TCN comparison table + cross-speed generalization curves.

Reads:
- scripts/runs/paper_loso_freqtoken_v4_strat_s2026/all_summary.json  (FreqToken, list of 6)
- scripts/final_freeze_2026_04_14/test_{speed}rpm/summary.json       (TCN, 6 files)

Writes (under reports/freqtoken_vs_tcn/):
- comparison.csv       — flat table, one row per (model, speed)
- comparison_wide.csv  — pivoted: speed as rows, model metrics as columns
- comparison.md        — markdown table for paper
- generalization_curve.png — exact_match / macro_f1 / reg_nmae vs speed, two lines per chart
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREQTOKEN_SUMMARY = PROJECT_ROOT / "scripts" / "runs" / "paper_loso_freqtoken_v4_strat_s2026" / "all_summary.json"
TCN_FREEZE_DIR = PROJECT_ROOT / "scripts" / "final_freeze_2026_04_14"
OUT_DIR = PROJECT_ROOT / "reports" / "freqtoken_vs_tcn"
SPEEDS = [600, 900, 1200, 1300, 1500, 1800]


def load_freqtoken_rows() -> list[dict]:
    with open(FREQTOKEN_SUMMARY, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        t = r["test"]
        tr = r["test_reg"]
        rows.append({
            "model": "FreqToken",
            "test_speed": int(r["test_speed"]),
            "exact_match": float(t["exact_match"]),
            "macro_f1": float(t["macro_f1"]),
            "micro_f1": float(t.get("micro_f1", 0.0)),
            "macro_precision": float(t.get("macro_precision", 0.0)),
            "macro_recall": float(t.get("macro_recall", 0.0)),
            "reg_nmae_mean": float(tr["reg_nmae_mean"]),
            "reg_nrmse_mean": float(tr.get("reg_nrmse_mean", 0.0)),
            "r4_acc": float(tr.get("r4_acc", 0.0)),
            "r5_acc": float(tr.get("r5_acc", 0.0)),
            "r6_acc": float(tr.get("r6_acc", 0.0)),
        })
    return rows


def load_tcn_rows() -> list[dict]:
    rows = []
    for sp in SPEEDS:
        p = TCN_FREEZE_DIR / f"test_{sp}rpm" / "summary.json"
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        t = r["test"]
        tr = r["test_reg"]
        rows.append({
            "model": "TCN",
            "test_speed": int(r["test_speed"]),
            "exact_match": float(t["exact_match"]),
            "macro_f1": float(t["macro_f1"]),
            "micro_f1": float(t.get("micro_f1", 0.0)),
            "macro_precision": float(t.get("macro_precision", 0.0)),
            "macro_recall": float(t.get("macro_recall", 0.0)),
            "reg_nmae_mean": float(tr["reg_nmae_mean"]),
            "reg_nrmse_mean": float(tr.get("reg_nrmse_mean", 0.0)),
            "r4_acc": float(tr.get("r4_acc", 0.0)),
            "r5_acc": float(tr.get("r5_acc", 0.0)),
            "r6_acc": float(tr.get("r6_acc", 0.0)),
        })
    return rows


def build_wide_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = speed, columns = <model>_<metric>."""
    rows = []
    for sp in SPEEDS:
        row = {"test_speed": sp}
        for _, r in df_long[df_long["test_speed"] == sp].iterrows():
            m = r["model"]
            for metric in ["exact_match", "macro_f1", "micro_f1", "macro_precision", "macro_recall",
                           "reg_nmae_mean", "reg_nrmse_mean", "r4_acc", "r5_acc", "r6_acc"]:
                row[f"{m}_{metric}"] = float(r[metric])
        rows.append(row)
    wide = pd.DataFrame(rows)
    # add delta columns
    for metric in ["exact_match", "macro_f1", "reg_nmae_mean"]:
        wide[f"delta_{metric}"] = wide[f"FreqToken_{metric}"] - wide[f"TCN_{metric}"]
    # add averages row
    avg_row = {"test_speed": "avg"}
    for c in wide.columns:
        if c == "test_speed":
            continue
        vals = wide[c].astype(float)
        avg_row[c] = float(vals.mean())
    wide_with_avg = pd.concat([wide, pd.DataFrame([avg_row])], ignore_index=True)
    return wide_with_avg


def write_markdown(df_wide: pd.DataFrame, out_path: Path):
    """Write a paper-friendly markdown comparison table."""
    lines = []
    lines.append("# FreqToken vs TCN — LOSO Cross-Speed Generalization\n")
    lines.append("Setup: Leave-One-Speed-Out, 6 test speeds (600/900/1200/1300/1500/1800 rpm).")
    lines.append("Both models trained on the other 5 speeds, evaluated on the held-out speed.\n")

    lines.append("## Classification (exact_match / macro_f1)\n")
    lines.append("| Test Speed | TCN exact_match | FreqToken exact_match | Δ exact_match | TCN macro_f1 | FreqToken macro_f1 | Δ macro_f1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df_wide.iterrows():
        sp = r["test_speed"]
        lines.append(
            f"| {sp} | {r['TCN_exact_match']:.4f} | {r['FreqToken_exact_match']:.4f} | "
            f"{r['delta_exact_match']:+.4f} | {r['TCN_macro_f1']:.4f} | {r['FreqToken_macro_f1']:.4f} | "
            f"{r['delta_macro_f1']:+.4f} |"
        )
    lines.append("")

    lines.append("## Regression (reg_nmae_mean, lower is better)\n")
    lines.append("| Test Speed | TCN reg_nmae | FreqToken reg_nmae | Δ reg_nmae |")
    lines.append("|---|---|---|---|")
    for _, r in df_wide.iterrows():
        sp = r["test_speed"]
        lines.append(
            f"| {sp} | {r['TCN_reg_nmae_mean']:.4f} | {r['FreqToken_reg_nmae_mean']:.4f} | "
            f"{r['delta_reg_nmae_mean']:+.4f} |"
        )
    lines.append("")

    lines.append("## Key Findings\n")
    avg_em_tcn = df_wide[df_wide["test_speed"] == "avg"]["TCN_exact_match"].iloc[0]
    avg_em_ft = df_wide[df_wide["test_speed"] == "avg"]["FreqToken_exact_match"].iloc[0]
    avg_em_delta = avg_em_ft - avg_em_tcn
    lines.append(f"- **Average exact_match**: TCN {avg_em_tcn:.4f} → FreqToken {avg_em_ft:.4f} ({avg_em_delta:+.4f})")

    # biggest improvement
    delta_rows = df_wide[df_wide["test_speed"] != "avg"].copy()
    delta_rows["abs_delta"] = delta_rows["delta_exact_match"].abs()
    best_idx = delta_rows["delta_exact_match"].idxmax()
    best_sp = delta_rows.loc[best_idx, "test_speed"]
    best_d = delta_rows.loc[best_idx, "delta_exact_match"]
    lines.append(f"- **Largest gain**: {best_sp} rpm, +{best_d:.4f} exact_match")

    # only regression
    worst_idx = delta_rows["delta_reg_nmae_mean"].idxmin()  # most negative = biggest improvement (lower is better)
    worst_sp = delta_rows.loc[worst_idx, "test_speed"]
    worst_d = delta_rows.loc[worst_idx, "delta_reg_nmae_mean"]
    lines.append(f"- **Best regression improvement**: {worst_sp} rpm, {worst_d:+.4f} reg_nmae (lower is better)")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_path}")


def plot_generalization_curve(df_long: pd.DataFrame, out_path: Path):
    """Three subplots: exact_match, macro_f1, reg_nmae vs speed. Two lines per chart (TCN, FreqToken)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    tcn_color = "#1f77b4"   # blue
    ft_color = "#d62728"    # red
    tcn_marker = "o"
    ft_marker = "s"

    metrics = [
        ("exact_match", "Exact Match (↑)", axes[0]),
        ("macro_f1", "Macro F1 (↑)", axes[1]),
        ("reg_nmae_mean", "Reg NMAE mean (↓)", axes[2]),
    ]

    for metric, ylabel, ax in metrics:
        tcn_df = df_long[df_long["model"] == "TCN"].sort_values("test_speed")
        ft_df = df_long[df_long["model"] == "FreqToken"].sort_values("test_speed")

        ax.plot(tcn_df["test_speed"], tcn_df[metric], color=tcn_color, marker=tcn_marker,
                linewidth=2, markersize=8, label="TCN")
        ax.plot(ft_df["test_speed"], ft_df[metric], color=ft_color, marker=ft_marker,
                linewidth=2, markersize=8, label="FreqToken")

        ax.set_xlabel("Test Speed (rpm)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(SPEEDS)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=10)

        # annotate values
        for _, r in tcn_df.iterrows():
            ax.annotate(f"{r[metric]:.3f}", (r["test_speed"], r[metric]),
                        textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=tcn_color)
        for _, r in ft_df.iterrows():
            ax.annotate(f"{r[metric]:.3f}", (r["test_speed"], r[metric]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=ft_color)

    fig.suptitle("FreqToken vs TCN — LOSO Cross-Speed Generalization",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_delta_bar(df_wide: pd.DataFrame, out_path: Path):
    """Bar chart: Δ exact_match per speed (FreqToken - TCN). Positive = FreqToken better."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = df_wide[df_wide["test_speed"] != "avg"].copy()
    speeds = plot_df["test_speed"].astype(int).tolist()
    deltas = plot_df["delta_exact_match"].astype(float).tolist()
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]

    bars = ax.bar([str(s) for s in speeds], deltas, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Test Speed (rpm)", fontsize=11)
    ax.set_ylabel("Δ exact_match (FreqToken − TCN)", fontsize=11)
    ax.set_title("FreqToken Improvement over TCN per Speed", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    for bar, d in zip(bars, deltas):
        h = bar.get_height()
        ax.annotate(f"{d:+.3f}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4 if h >= 0 else -12),
                    ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ft_rows = load_freqtoken_rows()
    tcn_rows = load_tcn_rows()
    df_long = pd.DataFrame(ft_rows + tcn_rows).sort_values(["test_speed", "model"]).reset_index(drop=True)

    # sanity: both models have all 6 speeds
    for m in ["TCN", "FreqToken"]:
        sub = df_long[df_long["model"] == m]
        missing = set(SPEEDS) - set(sub["test_speed"].tolist())
        if missing:
            print(f"[WARN] model {m} missing speeds: {sorted(missing)}", file=sys.stderr)

    # save long table
    long_path = OUT_DIR / "comparison.csv"
    df_long.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {long_path}")

    # build and save wide table
    df_wide = build_wide_table(df_long)
    wide_path = OUT_DIR / "comparison_wide.csv"
    df_wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {wide_path}")

    # markdown
    md_path = OUT_DIR / "comparison.md"
    write_markdown(df_wide, md_path)

    # plots
    curve_path = OUT_DIR / "generalization_curve.png"
    plot_generalization_curve(df_long, curve_path)

    delta_path = OUT_DIR / "delta_exact_match_bar.png"
    plot_delta_bar(df_wide, delta_path)

    # print summary to stdout
    print("\n=== Summary (avg row) ===")
    avg_row = df_wide[df_wide["test_speed"] == "avg"].iloc[0]
    print(f"  TCN       avg exact_match: {avg_row['TCN_exact_match']:.4f}")
    print(f"  FreqToken avg exact_match: {avg_row['FreqToken_exact_match']:.4f}")
    print(f"  Δ                          : {avg_row['delta_exact_match']:+.4f}")
    print(f"  TCN       avg macro_f1    : {avg_row['TCN_macro_f1']:.4f}")
    print(f"  FreqToken avg macro_f1    : {avg_row['FreqToken_macro_f1']:.4f}")
    print(f"  Δ                          : {avg_row['delta_macro_f1']:+.4f}")
    print(f"  TCN       avg reg_nmae    : {avg_row['TCN_reg_nmae_mean']:.4f}")
    print(f"  FreqToken avg reg_nmae    : {avg_row['FreqToken_reg_nmae_mean']:.4f}")
    print(f"  Δ                          : {avg_row['delta_reg_nmae_mean']:+.4f}")


if __name__ == "__main__":
    main()
