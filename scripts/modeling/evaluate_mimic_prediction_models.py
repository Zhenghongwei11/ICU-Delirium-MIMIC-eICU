#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


DATASET_ID = "MIMIC-IV-2.2"
OUTCOME = "incident_post_landmark_delirium"
RANDOM_STATE = 20260608


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def zscore(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    std = x.std(ddof=0)
    if not std or np.isnan(std):
        return x * 0
    return (x - x.mean()) / std


def prepare_data(matrix_path: Path, classes_path: Path) -> pd.DataFrame:
    matrix = read_tsv(matrix_path)
    classes = read_tsv(classes_path)
    data = matrix.merge(
        classes[["stay_id", "assigned_class", "class_label", "posterior_probability"]],
        on="stay_id",
        how="inner",
    )
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data[OUTCOME] = data[OUTCOME].astype(int)

    data["age_z"] = zscore(data["anchor_age"])
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)
    data["night_all_z"] = zscore(data["event_all_night_22_06_count_48h"])
    data["day_all_z"] = zscore(data["event_all_day_06_22_count_48h"])
    data["total_all_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])
    data["braden_skin_records_z"] = zscore(data.get("nursing_skin_braden_pressure_n_records_48h", 0))
    data["rass_records_z"] = zscore(data.get("nursing_rass_sedation_n_records_48h", 0))
    data["mobility_records_z"] = zscore(data.get("nursing_mobility_turning_n_records_48h", 0))
    data["pain_records_z"] = zscore(data.get("nursing_pain_n_records_48h", 0))
    data["restraint_records_z"] = zscore(data.get("nursing_restraints_n_records_48h", 0))
    data["high_intensity_class"] = data["class_label"].str.contains("high care intensity", na=False).astype(int)
    return data


def model_specs() -> dict[str, list[str]]:
    return {
        "demographic_baseline": ["age_z", "male"],
        "care_intensity_total": ["age_z", "male", "total_all_z"],
        "day_vs_night_counts": ["age_z", "male", "day_all_z", "night_all_z"],
        "night_fraction": ["age_z", "male", "total_all_z", "night_fraction_z"],
        "nursing_vulnerability_density": [
            "age_z",
            "male",
            "braden_skin_records_z",
            "rass_records_z",
            "mobility_records_z",
            "pain_records_z",
            "restraint_records_z",
        ],
        "joint_density_screen": [
            "age_z",
            "male",
            "total_all_z",
            "night_fraction_z",
            "braden_skin_records_z",
            "rass_records_z",
            "mobility_records_z",
            "pain_records_z",
            "restraint_records_z",
        ],
        "dynamic_phenotype_unadjusted": ["high_intensity_class"],
        "dynamic_phenotype_demographic_adjusted": ["high_intensity_class", "age_z", "male"],
        "dynamic_phenotype_care_intensity_adjusted": [
            "high_intensity_class",
            "age_z",
            "male",
            "total_all_z",
            "night_fraction_z",
        ],
    }


def design_matrix(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    x = data[predictors].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    return sm.add_constant(x, has_constant="add")


def fit_predict(train: pd.DataFrame, predict_on: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    y = train[OUTCOME].astype(int)
    x_train = design_matrix(train, predictors)
    x_pred = design_matrix(predict_on, predictors)
    result = sm.Logit(y, x_train).fit(disp=False, maxiter=200)
    pred = result.predict(x_pred)
    return np.asarray(pred, dtype=float)


def calibration(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    lp = np.log(pred / (1 - pred))
    x = sm.add_constant(pd.DataFrame({"linear_predictor": lp}), has_constant="add")
    try:
        result = sm.Logit(y, x).fit(disp=False, maxiter=200)
        return float(result.params["const"]), float(result.params["linear_predictor"])
    except Exception:
        return np.nan, np.nan


def evaluate_predictions(
    y: np.ndarray,
    pred: np.ndarray,
    model: str,
    split_or_cohort: str,
    note: str,
) -> dict[str, object]:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    intercept, slope = calibration(y, pred)
    return {
        "dataset_id": DATASET_ID,
        "split_or_cohort": split_or_cohort,
        "outcome": OUTCOME,
        "model": model,
        "auc_or_cstat": float(roc_auc_score(y, pred)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "brier": float(brier_score_loss(y, pred)),
        "mean_predicted_risk": float(np.mean(pred)),
        "observed_event_rate": float(np.mean(y)),
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "evaluation_note": note,
    }


def apparent_predictions(data: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows = []
    preds = {}
    y = data[OUTCOME].astype(int).to_numpy()
    for model, predictors in model_specs().items():
        pred = fit_predict(data, data, predictors)
        preds[model] = pred
        rows.append(
            evaluate_predictions(
                y,
                pred,
                model,
                "MIMIC_derivation_apparent",
                "Apparent derivation performance; not external validation.",
            )
        )
    return rows, preds


def cross_validated_predictions(data: pd.DataFrame, folds: int) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    y = data[OUTCOME].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    preds = {}
    for model, predictors in model_specs().items():
        oof = np.full(len(data), np.nan, dtype=float)
        for train_idx, test_idx in splitter.split(data, y):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            oof[test_idx] = fit_predict(train, test, predictors)
        preds[model] = oof
        rows.append(
            evaluate_predictions(
                y,
                oof,
                model,
                f"MIMIC_internal_{folds}fold_cv",
                "Outcome-model internal cross-validation; unsupervised phenotype assignments were fixed from the derivation cohort.",
            )
        )
    return rows, preds


def decision_curve_rows(y: np.ndarray, predictions: dict[str, np.ndarray], split_or_cohort: str) -> list[dict[str, object]]:
    thresholds = np.round(np.arange(0.01, 0.301, 0.01), 2)
    rows: list[dict[str, object]] = []
    n = len(y)
    prevalence = float(np.mean(y))
    for threshold in thresholds:
        weight = threshold / (1 - threshold)
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "split_or_cohort": split_or_cohort,
                "outcome": OUTCOME,
                "model": "treat_none",
                "threshold": threshold,
                "net_benefit": 0.0,
                "n": n,
                "events": int(np.sum(y)),
            }
        )
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "split_or_cohort": split_or_cohort,
                "outcome": OUTCOME,
                "model": "treat_all",
                "threshold": threshold,
                "net_benefit": prevalence - (1 - prevalence) * weight,
                "n": n,
                "events": int(np.sum(y)),
            }
        )
        for model, pred in predictions.items():
            classified = pred >= threshold
            tp = int(((y == 1) & classified).sum())
            fp = int(((y == 0) & classified).sum())
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "split_or_cohort": split_or_cohort,
                    "outcome": OUTCOME,
                    "model": model,
                    "threshold": threshold,
                    "net_benefit": tp / n - fp / n * weight,
                    "n": n,
                    "events": int(np.sum(y)),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MIMIC screening models with discrimination, calibration, Brier score, and decision curves.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--prediction-out", type=Path, default=Path("results/benchmarks/prediction_eval.tsv"))
    parser.add_argument("--decision-out", type=Path, default=Path("results/clinical_utility/decision_curve.tsv"))
    parser.add_argument("--figure4-out", type=Path, default=Path("results/figures/figure4_screening_utility_anchor.tsv"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data = prepare_data(args.matrix, args.classes)
    y = data[OUTCOME].astype(int).to_numpy()
    apparent_rows, apparent_preds = apparent_predictions(data)
    cv_rows, cv_preds = cross_validated_predictions(data, args.folds)

    prediction = pd.DataFrame(apparent_rows + cv_rows)
    args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.prediction_out, sep="\t", index=False)

    dca = pd.DataFrame(
        decision_curve_rows(y, apparent_preds, "MIMIC_derivation_apparent")
        + decision_curve_rows(y, cv_preds, f"MIMIC_internal_{args.folds}fold_cv")
    )
    args.decision_out.parent.mkdir(parents=True, exist_ok=True)
    dca.to_csv(args.decision_out, sep="\t", index=False)

    args.figure4_out.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.figure4_out, sep="\t", index=False)
    print(f"Wrote {args.prediction_out} ({len(prediction)} rows)")
    print(f"Wrote {args.decision_out} ({len(dca)} rows)")
    print(f"Evaluation cohort excluding pre-landmark delirium: n={len(data)}, events={int(y.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
