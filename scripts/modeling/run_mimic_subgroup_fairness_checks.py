#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score


DATASET_ID = "MIMIC-IV-2.2"
OUTCOME = "incident_post_landmark_delirium"


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


def broad_race(value: object) -> str:
    text = str(value).upper()
    if "WHITE" in text:
        return "White"
    if "BLACK" in text or "AFRICAN" in text:
        return "Black"
    if "ASIAN" in text:
        return "Asian"
    if "HISPANIC" in text or "LATINO" in text:
        return "Hispanic/Latino"
    return "Other/Unknown"


def prepare_data(args: argparse.Namespace) -> pd.DataFrame:
    matrix = read_tsv(args.matrix)
    classes = read_tsv(args.classes)
    covariates = read_tsv(args.covariates)
    data = matrix.merge(classes[["stay_id", "class_label"]], on="stay_id", how="inner").merge(
        covariates, on=["subject_id", "hadm_id", "stay_id"], how="left"
    )
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data[OUTCOME] = data[OUTCOME].astype(int)
    data["high_intensity_class"] = data["class_label"].str.contains("high care intensity", na=False).astype(int)
    data["age_z"] = zscore(data["anchor_age"])
    data["age_group"] = np.where(pd.to_numeric(data["anchor_age"], errors="coerce") >= 80, "age_ge_80", "age_65_79")
    data["sex_group"] = np.where(data["gender"].astype(str).str.upper() == "M", "male", "female")
    data["male"] = (data["sex_group"] == "male").astype(int)
    data["race_group"] = data["race"].map(broad_race)
    data["total_event_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])
    data["diagnosis_count_z"] = zscore(data["diagnosis_count"])
    data["joint_density_score"] = (
        zscore(data.get("nursing_skin_braden_pressure_n_records_48h", 0))
        + zscore(data.get("nursing_rass_sedation_n_records_48h", 0))
        + zscore(data.get("nursing_mobility_turning_n_records_48h", 0))
        + zscore(data.get("nursing_pain_n_records_48h", 0))
        + zscore(data.get("nursing_restraints_n_records_48h", 0))
        + data["total_event_z"]
    )
    for col in ["dementia_flag", "sepsis_flag", "sedative_exposure", "opioid_exposure", "vasopressor_exposure", "emergency_admission", "micu_unit"]:
        data[col] = data[col].fillna(False).astype(int)
    data["sedative_group"] = np.where(data["sedative_exposure"] == 1, "sedative_exposed", "no_sedative_exposure")
    return data


def fit_association(sub: pd.DataFrame, subgroup_name: str, subgroup_level: str) -> dict[str, object]:
    base_predictors = [
        "high_intensity_class",
        "age_z",
        "male",
        "total_event_z",
        "night_fraction_z",
        "diagnosis_count_z",
        "dementia_flag",
        "sepsis_flag",
        "sedative_exposure",
        "opioid_exposure",
        "vasopressor_exposure",
        "emergency_admission",
        "micu_unit",
    ]
    predictors = []
    for predictor in base_predictors:
        values = pd.to_numeric(sub[predictor], errors="coerce")
        if values.nunique(dropna=True) > 1:
            predictors.append(predictor)
    model_data = sub[[OUTCOME] + predictors].dropna().copy()
    y = model_data[OUTCOME].astype(int)
    if y.nunique() < 2 or "high_intensity_class" not in predictors:
        return {
            "dataset_id": DATASET_ID,
            "analysis_type": "association",
            "subgroup_name": subgroup_name,
            "subgroup_level": subgroup_level,
            "model": "clinical_covariate_extended_within_subgroup",
            "effect_type": "odds_ratio",
            "effect": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "auc_or_cstat": np.nan,
            "brier": np.nan,
            "n": int(len(model_data)),
            "events": int(y.sum()),
            "status": "not_estimable",
        }
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    x = sm.add_constant(x, has_constant="add")
    try:
        result = sm.Logit(y, x).fit(disp=False, maxiter=300)
        pred = np.asarray(result.predict(x), dtype=float)
        coef = result.params["high_intensity_class"]
        se = result.bse["high_intensity_class"]
        return {
            "dataset_id": DATASET_ID,
            "analysis_type": "association",
            "subgroup_name": subgroup_name,
            "subgroup_level": subgroup_level,
            "model": "clinical_covariate_extended_within_subgroup",
            "effect_type": "odds_ratio",
            "effect": float(np.exp(coef)),
            "ci_lower": float(np.exp(coef - 1.96 * se)),
            "ci_upper": float(np.exp(coef + 1.96 * se)),
            "pvalue": float(result.pvalues["high_intensity_class"]),
            "auc_or_cstat": float(roc_auc_score(y, pred)),
            "brier": float(brier_score_loss(y, pred)),
            "n": int(len(model_data)),
            "events": int(y.sum()),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "dataset_id": DATASET_ID,
            "analysis_type": "association",
            "subgroup_name": subgroup_name,
            "subgroup_level": subgroup_level,
            "model": "clinical_covariate_extended_within_subgroup",
            "effect_type": "odds_ratio",
            "effect": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "auc_or_cstat": np.nan,
            "brier": np.nan,
            "n": int(len(model_data)),
            "events": int(y.sum()),
            "status": f"not_estimable_{type(exc).__name__}",
        }


def performance_row(sub: pd.DataFrame, subgroup_name: str, subgroup_level: str) -> dict[str, object]:
    y = sub[OUTCOME].astype(int)
    score = pd.to_numeric(sub["joint_density_score"], errors="coerce")
    keep = y.notna() & score.notna()
    y = y[keep]
    score = score[keep]
    if y.nunique() < 2:
        auc = np.nan
        brier = np.nan
        status = "not_estimable"
    else:
        # Rank-score performance for subgroup fairness comparison; not a calibrated probability model.
        auc = float(roc_auc_score(y, score))
        prevalence_scaled = 1 / (1 + np.exp(-(score - score.mean()) / score.std(ddof=0)))
        brier = float(brier_score_loss(y, prevalence_scaled))
        status = "score_auc_only"
    return {
        "dataset_id": DATASET_ID,
        "analysis_type": "screening_score_performance",
        "subgroup_name": subgroup_name,
        "subgroup_level": subgroup_level,
        "model": "joint_density_rank_score",
        "effect_type": "auc_or_cstat",
        "effect": auc,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "pvalue": np.nan,
        "auc_or_cstat": auc,
        "brier": brier,
        "n": int(len(y)),
        "events": int(y.sum()),
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run subgroup/fairness checks for association and screening score performance.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--covariates", type=Path, default=Path("results/covariates/mimic_clinical_covariates_48h.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity/subgroup_fairness_checks.tsv"))
    args = parser.parse_args()

    data = prepare_data(args)
    subgroup_specs = {
        "sex": "sex_group",
        "age_group": "age_group",
        "race_group": "race_group",
        "sedative_exposure": "sedative_group",
    }
    rows = []
    for subgroup_name, col in subgroup_specs.items():
        for level, sub in data.groupby(col, dropna=False):
            if len(sub) < 200:
                continue
            rows.append(fit_association(sub, subgroup_name, str(level)))
            rows.append(performance_row(sub, subgroup_name, str(level)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
