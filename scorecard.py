#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end credit scorecard for consumer lending (FinTech risk modeling).

Pipeline
--------
1. Load the credit dataset (``target`` column: 1 = default, 0 = good).
2. Bin every feature and compute Weight of Evidence (WOE) + Information Value (IV).
3. Keep features above an IV threshold.
4. Fit a logistic regression on the WOE-transformed features.
5. Scale the model into a point-based scorecard (base score + points per bin).
6. Evaluate with KS and AUC; write the scorecard table and metrics.

Usage
-----
::

    python scorecard.py --input data/german_credit.csv --output-dir output

Note: this is an in-sample demonstration (bins + model + evaluation on one
dataset). A production pipeline would add train / validation / out-of-time splits,
reject inference and PSI monitoring -- see README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

TARGET = "target"  # 1 = default (bad), 0 = good

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Binning / WOE / IV
# --------------------------------------------------------------------------- #

def bin_series(s: pd.Series, max_bins: int) -> pd.Series:
    """Bin a single feature into string labels.

    * Categorical (object dtype) and low-cardinality numeric features keep each
      value as its own bin.
    * High-cardinality numeric features are cut into ``max_bins`` equal-frequency
      bins (qcut).
    * Missing values become a dedicated ``missing`` bin.
    """
    if s.dtype == object or s.nunique() <= max_bins:
        return s.fillna("missing").astype(str)
    try:
        binned = pd.qcut(s, q=max_bins, duplicates="drop")
    except ValueError:
        return s.fillna("missing").astype(str)
    # Label missing values before stringifying: astype(str) would otherwise turn
    # NaN into the literal bin "nan" and the following fillna would be a no-op.
    binned = binned.astype(object)
    return binned.where(binned.notna(), "missing").astype(str)


def woe_table(binned: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Aggregate counts and WOE per bin."""
    df = pd.DataFrame({"bin": binned, "target": y})
    tbl = (
        df.groupby("bin", as_index=False)
        .agg(n=("target", "size"), bad=("target", "sum"))
    )
    tbl["good"] = tbl["n"] - tbl["bad"]
    total_good = int(y.size - y.sum())
    total_bad = int(y.sum())
    tbl["pct_good"] = tbl["good"] / (total_good + _EPS)
    tbl["pct_bad"] = tbl["bad"] / (total_bad + _EPS)
    tbl["woe"] = np.log((tbl["pct_good"] + _EPS) / (tbl["pct_bad"] + _EPS))
    return tbl


def iv_from_table(tbl: pd.DataFrame) -> float:
    """Information Value = sum((pct_good - pct_bad) * woe)."""
    return float(((tbl["pct_good"] - tbl["pct_bad"]) * tbl["woe"]).sum())


def build_woe_map(X: pd.DataFrame, y: pd.Series, iv_threshold: float, max_bins: int):
    """Select features by IV; return WOE map, binned features and the IV table."""
    iv_rows = []
    woe_map: dict[str, dict[str, float]] = {}
    binned_map: dict[str, pd.Series] = {}

    for feature in X.columns:
        binned = bin_series(X[feature], max_bins)
        tbl = woe_table(binned, y)
        iv = iv_from_table(tbl)
        iv_rows.append({"feature": feature, "iv": round(iv, 4)})
        if iv >= iv_threshold:
            woe_map[feature] = dict(zip(tbl["bin"], tbl["woe"]))
            binned_map[feature] = binned

    iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return woe_map, binned_map, iv_table


def transform_woe(binned_map: dict[str, pd.Series],
                  woe_map: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Map binned labels to WOE values (unknown bins -> 0)."""
    out = pd.DataFrame(index=next(iter(binned_map.values())).index)
    for feature in binned_map:
        out[feature] = binned_map[feature].map(woe_map[feature]).fillna(0.0)
    return out


# --------------------------------------------------------------------------- #
# Scorecard scaling
# --------------------------------------------------------------------------- #

def scorecard_scaling(model: LogisticRegression,
                      woe_map: dict[str, dict[str, float]],
                      base_score: float, pdo: float, base_odds: float):
    """Convert a WOE logistic model into point-based scorecard (Siddiqi).

    Higher score = lower default risk:

        Score = Offset + Factor * ln(P_good / P_bad)
        Factor = PDO / ln(2)
        Offset = Base_Score - Factor * ln(Base_Odds)

    With logit(P_bad) = b0 + sum(b_j * woe_j), this yields

        points(bin of feature j) = -Factor * b_j * woe_j
        base_points               = Offset - Factor * b0
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    intercept = float(model.intercept_[0])
    coef = dict(zip(model.feature_names_in_, model.coef_[0]))

    rows = []
    for feature, bin_map in woe_map.items():
        beta = coef[feature]
        for b, woe in bin_map.items():
            rows.append({
                "feature": feature,
                "bin": b,
                "woe": round(woe, 4),
                "coef": round(beta, 4),
                "points": round(-factor * beta * woe, 2),
            })

    scorecard = pd.DataFrame(rows)
    base_points = round(offset - factor * intercept, 2)
    return scorecard, factor, base_points


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(y_true: pd.Series, proba: pd.Series) -> dict:
    """KS and AUC."""
    auc = float(roc_auc_score(y_true, proba))
    fpr, tpr, _ = roc_curve(y_true, proba)
    ks = float(np.max(tpr - fpr))
    return {"auc": round(auc, 4), "ks": round(ks, 4)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="信用评分卡：WOE/IV + 逻辑回归 + 评分刻度 + KS/AUC")
    p.add_argument("--input", default="data/german_credit.csv", help="信贷数据 CSV")
    p.add_argument("--target", default=TARGET, help="目标列名（1=违约，0=正常）")
    p.add_argument("--iv-threshold", type=float, default=0.02, help="IV 筛选阈值")
    p.add_argument("--max-bins", type=int, default=5, help="数值变量分箱数")
    p.add_argument("--base-score", type=float, default=600, help="基准分")
    p.add_argument("--pdo", type=float, default=20, help="Points to Double Odds")
    p.add_argument("--base-odds", type=float, default=50, help="基准分对应的好坏比 (good:bad)")
    p.add_argument("--output-dir", default="output", help="结果输出目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    woe_map, binned_map, iv_table = build_woe_map(X, y, args.iv_threshold, args.max_bins)
    X_woe = transform_woe(binned_map, woe_map)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_woe, y)
    proba = model.predict_proba(X_woe)[:, 1]

    scorecard, factor, base_points = scorecard_scaling(
        model, woe_map, args.base_score, args.pdo, args.base_odds
    )
    metrics = evaluate(y, proba)

    # ---- write outputs ----
    iv_table.to_csv(output_dir / "iv_table.csv", index=False, encoding="utf-8-sig")

    woe_rows = []
    for feature in woe_map:
        tbl = woe_table(binned_map[feature], y)
        tbl.insert(0, "feature", feature)
        woe_rows.append(tbl)
    pd.concat(woe_rows, ignore_index=True).to_csv(
        output_dir / "woe_table.csv", index=False, encoding="utf-8-sig"
    )

    scorecard.to_csv(output_dir / "scorecard.csv", index=False, encoding="utf-8-sig")

    summary = {
        "样本数": int(len(df)),
        "坏账率": round(float(y.mean()), 4),
        "入模变量数": int(len(woe_map)),
        "KS": metrics["ks"],
        "AUC": metrics["auc"],
        "base_score": args.base_score,
        "PDO": args.pdo,
        "base_odds": args.base_odds,
        "factor": round(factor, 4),
        "base_points": base_points,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- console summary ----
    print(f"样本数: {summary['样本数']} | 坏账率: {summary['坏账率']:.1%}")
    print(f"入模变量: {summary['入模变量数']} 个（IV >= {args.iv_threshold}）")
    print(f"KS = {metrics['ks']} | AUC = {metrics['auc']}")
    print(f"基准分 {args.base_score} / PDO {args.pdo} / base_points {base_points}")
    print("\nIV 排名前 8:")
    for row in iv_table.head(8).itertuples(index=False):
        print(f"  {row.feature:<18} IV={row.iv}")
    print("\n评分卡片段（前 12 行）:")
    for row in scorecard.head(12).itertuples(index=False):
        print(f"  {row.feature:<18} {row.bin:<22} {row.points:>8} 分")
    print(f"\n结果已写入: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
