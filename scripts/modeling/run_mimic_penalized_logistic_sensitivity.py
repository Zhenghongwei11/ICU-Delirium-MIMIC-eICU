#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_ID = "MIMIC-IV-2.2"
OUTCOME = "incident_post_landmark_delirium"
RANDOM_STATE = 20260608


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def prepare_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    matrix = read_tsv(args.matrix)
    classes = read_tsv(args.classes)
    covariates = read_tsv(args.covariates)
    data = matrix.merge(classes[["stay_id", "class_label"]], on="stay_id", how="inner").merge(
        covariates, on=["subject_id", "hadm_id", "stay_id"], how="left"
    )
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data[OUTCOME] = data[OUTCOME].astype(int)
    data["high_intensity_class"] = data["class_label"].str.contains("high care intensity", na=False).astype(int)
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)
    for col in ["dementia_flag", "sepsis_flag", "sedative_exposure", "opioid_exposure", "vasopressor_exposure", "emergency_admission", "micu_unit"]:
        data[col] = data[col].fillna(False).astype(int)

    nursing_cols = [
        "nursing_skin_braden_pressure_n_records_48h",
        "nursing_rass_sedation_n_records_48h",
        "nursing_mobility_turning_n_records_48h",
        "nursing_pain_n_records_48h",
        "nursing_restraints_n_records_48h",
        "nursing_vitals_n_records_48h",
        "nursing_orientation_n_records_48h",
        "nursing_cam_icu_delirium_n_records_48h",
        "nursing_oral_suction_airway_n_records_48h",
    ]
    event_cols = [
        "event_all_total_count_48h",
        "event_all_night_fraction_48h",
        "event_all_day_06_22_count_48h",
        "event_all_night_22_06_count_48h",
    ]
    clinical_cols = [
        "high_intensity_class",
        "anchor_age",
        "male",
        "diagnosis_count",
        "dementia_flag",
        "sepsis_flag",
        "sedative_exposure",
        "opioid_exposure",
        "vasopressor_exposure",
        "emergency_admission",
        "micu_unit",
    ]
    features = clinical_cols + [c for c in nursing_cols + event_cols if c in data.columns]
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    return data, features


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, iterations: int = 500) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    briers = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], pred[idx]))
        briers.append(brier_score_loss(y[idx], pred[idx]))
    return {
        "auc_ci_lower": float(np.quantile(aucs, 0.025)),
        "auc_ci_upper": float(np.quantile(aucs, 0.975)),
        "brier_ci_lower": float(np.quantile(briers, 0.025)),
        "brier_ci_upper": float(np.quantile(briers, 0.975)),
        "bootstrap_iterations_used": len(aucs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run penalized logistic sensitivity models for MIMIC screening.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--covariates", type=Path, default=Path("results/covariates/mimic_clinical_covariates_48h.tsv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/benchmarks/penalized_logistic_sensitivity.tsv"))
    parser.add_argument("--coef-out", type=Path, default=Path("results/benchmarks/penalized_logistic_coefficients.tsv"))
    args = parser.parse_args()

    data, features = prepare_data(args)
    x = data[features].copy()
    y = data[OUTCOME].astype(int).to_numpy()
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "l2_penalized_logistic": LogisticRegressionCV(
            Cs=10,
            cv=3,
            penalty="l2",
            solver="lbfgs",
            scoring="roc_auc",
            class_weight=None,
            max_iter=2000,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "elastic_net_logistic": LogisticRegressionCV(
            Cs=8,
            cv=3,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.1, 0.5, 0.9],
            scoring="roc_auc",
            class_weight=None,
            max_iter=3000,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }
    summary_rows = []
    coef_rows = []
    for model_name, estimator in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        pred = cross_val_predict(pipe, x, y, cv=outer_cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, pred))
        brier = float(brier_score_loss(y, pred))
        ci = bootstrap_ci(y, pred)
        summary_rows.append(
            {
                "dataset_id": DATASET_ID,
                "model": model_name,
                "split_or_cohort": "MIMIC_internal_5fold_cv_nested_penalized",
                "outcome": OUTCOME,
                "auc_or_cstat": auc,
                "brier": brier,
                "n": int(len(y)),
                "events": int(y.sum()),
                "n_features": len(features),
                "analysis_note": "Penalized logistic sensitivity without class weighting; not the primary interpretable phenotype model.",
                **ci,
            }
        )

        pipe.fit(x, y)
        fitted = pipe.named_steps["model"]
        coefs = fitted.coef_[0]
        for feature, coef in zip(features, coefs):
            coef_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "model": model_name,
                    "feature": feature,
                    "coefficient_standardized": float(coef),
                    "absolute_coefficient": float(abs(coef)),
                    "selected_nonzero": bool(abs(coef) > 1e-8),
                }
            )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.summary_out, sep="\t", index=False)
    pd.DataFrame(coef_rows).sort_values(["model", "absolute_coefficient"], ascending=[True, False]).to_csv(
        args.coef_out, sep="\t", index=False
    )
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.coef_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
