#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATASET_ID = "MIMIC-IV-2.2"
LANDMARK_HOURS = 48


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
    data = matrix.merge(classes[["stay_id", "class_label"]], on="stay_id", how="inner")
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data["high_intensity_class"] = data["class_label"].str.contains("high care intensity", na=False).astype(int)
    data["age_z"] = zscore(data["anchor_age"])
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)
    data["total_event_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])
    death_time = pd.to_numeric(data["death_time_from_icu_admit_hours"], errors="coerce")
    data["post_landmark_death"] = death_time.notna() & (death_time > LANDMARK_HOURS)
    data["incident_post_landmark_delirium"] = data["incident_post_landmark_delirium"].astype(bool)
    data["death_or_delirium"] = data["incident_post_landmark_delirium"] | data["post_landmark_death"]
    data["icu_los_days"] = pd.to_numeric(data["icu_los_hours"], errors="coerce") / 24
    data["log_icu_los_days"] = np.log1p(data["icu_los_days"])
    data["log_hospital_los_days"] = np.log1p(pd.to_numeric(data["hospital_los_days"], errors="coerce"))
    data["hospital_expire_flag"] = pd.to_numeric(data["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    return data


def fit_logit(data: pd.DataFrame, outcome: str, model_name: str, covariate_set: str) -> dict[str, object]:
    predictors = ["high_intensity_class", "age_z", "male", "total_event_z", "night_fraction_z"]
    model_data = data[[outcome] + predictors].dropna().copy()
    y = model_data[outcome].astype(int)
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    x = sm.add_constant(x, has_constant="add")
    result = sm.Logit(y, x).fit(disp=False, maxiter=200)
    coef = result.params["high_intensity_class"]
    se = result.bse["high_intensity_class"]
    return {
        "dataset_id": DATASET_ID,
        "analysis": model_name,
        "outcome": outcome,
        "model_type": "logistic",
        "effect_type": "odds_ratio",
        "effect": float(np.exp(coef)),
        "ci_lower": float(np.exp(coef - 1.96 * se)),
        "ci_upper": float(np.exp(coef + 1.96 * se)),
        "pvalue": float(result.pvalues["high_intensity_class"]),
        "n": int(len(model_data)),
        "events": int(y.sum()),
        "covariate_set": covariate_set,
    }


def fit_ols(data: pd.DataFrame, outcome: str, model_name: str, covariate_set: str) -> dict[str, object]:
    predictors = ["high_intensity_class", "age_z", "male", "total_event_z", "night_fraction_z"]
    model_data = data[[outcome] + predictors].dropna().copy()
    y = pd.to_numeric(model_data[outcome], errors="coerce")
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    x = sm.add_constant(x, has_constant="add")
    result = sm.OLS(y, x).fit()
    coef = result.params["high_intensity_class"]
    se = result.bse["high_intensity_class"]
    return {
        "dataset_id": DATASET_ID,
        "analysis": model_name,
        "outcome": outcome,
        "model_type": "linear_ols",
        "effect_type": "mean_difference_on_log1p_scale",
        "effect": float(coef),
        "ci_lower": float(coef - 1.96 * se),
        "ci_upper": float(coef + 1.96 * se),
        "pvalue": float(result.pvalues["high_intensity_class"]),
        "n": int(len(model_data)),
        "events": np.nan,
        "covariate_set": covariate_set,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run death competing-event sensitivity and feasible secondary outcome models.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--competing-out", type=Path, default=Path("results/sensitivity/death_competing_sensitivity.tsv"))
    parser.add_argument("--secondary-out", type=Path, default=Path("results/effect_sizes/secondary_outcomes.tsv"))
    args = parser.parse_args()

    data = prepare_data(args.matrix, args.classes)
    covariate_set = "high_intensity_class,age_z,male,total_event_z,night_fraction_z"

    competing_rows = [
        fit_logit(data, "incident_post_landmark_delirium", "primary_all_patients", covariate_set),
        fit_logit(data[~data["post_landmark_death"]].copy(), "incident_post_landmark_delirium", "exclude_post_landmark_death", covariate_set),
        fit_logit(data, "death_or_delirium", "composite_death_or_delirium", covariate_set),
    ]
    args.competing_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(competing_rows).to_csv(args.competing_out, sep="\t", index=False)

    secondary_rows = [
        fit_logit(data, "hospital_expire_flag", "secondary_hospital_mortality", covariate_set),
        fit_ols(data, "log_icu_los_days", "secondary_icu_los", covariate_set),
        fit_ols(data, "log_hospital_los_days", "secondary_hospital_los", covariate_set),
    ]
    args.secondary_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(secondary_rows).to_csv(args.secondary_out, sep="\t", index=False)

    print(f"Wrote {args.competing_out}")
    print(f"Wrote {args.secondary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
