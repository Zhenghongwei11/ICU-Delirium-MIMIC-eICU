#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from evaluate_mimic_prediction_models import OUTCOME, cross_validated_predictions, prepare_data


RANDOM_STATE = 20260608


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, iterations: int) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    briers = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        pred_b = pred[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, pred_b))
        briers.append(brier_score_loss(y_b, pred_b))
    return {
        "auc_ci_lower": float(np.quantile(aucs, 0.025)),
        "auc_ci_upper": float(np.quantile(aucs, 0.975)),
        "brier_ci_lower": float(np.quantile(briers, 0.025)),
        "brier_ci_upper": float(np.quantile(briers, 0.975)),
        "bootstrap_iterations_used": len(aucs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Add bootstrap CIs for internally cross-validated prediction metrics.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--base-eval", type=Path, default=Path("results/benchmarks/prediction_eval.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/benchmarks/prediction_eval_ci.tsv"))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data = prepare_data(args.matrix, args.classes)
    y = data[OUTCOME].astype(int).to_numpy()
    _, cv_preds = cross_validated_predictions(data, args.folds)
    base = pd.read_csv(args.base_eval, sep="\t")
    base = base[base["split_or_cohort"] == f"MIMIC_internal_{args.folds}fold_cv"].copy()

    rows = []
    for _, row in base.iterrows():
        model = row["model"]
        if model not in cv_preds:
            continue
        ci = bootstrap_ci(y, np.asarray(cv_preds[model], dtype=float), args.iterations)
        out = row.to_dict()
        out.update(ci)
        rows.append(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
