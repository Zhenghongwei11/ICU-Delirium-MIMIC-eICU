#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)
    data["total_event_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])
    data["diagnosis_count_z"] = zscore(data["diagnosis_count"])
    data["post_landmark_negative_assessments"] = pd.to_numeric(data["post_landmark_negative_assessments"], errors="coerce").fillna(0)
    data["post_landmark_delirium_positive"] = data["post_landmark_delirium_positive"].fillna(False).astype(bool)
    data["post_landmark_assessment_proxy_count"] = data["post_landmark_negative_assessments"] + data["post_landmark_delirium_positive"].astype(int)
    data["log1p_post_landmark_assessment_proxy_count"] = np.log1p(data["post_landmark_assessment_proxy_count"])
    data["any_post_landmark_assessment"] = (data["post_landmark_assessment_proxy_count"] > 0).astype(int)
    for col in ["dementia_flag", "sepsis_flag", "sedative_exposure", "opioid_exposure", "vasopressor_exposure", "emergency_admission", "micu_unit"]:
        data[col] = data[col].fillna(False).astype(int)
    return data


def base_predictors(include_assessment_count: bool = False) -> list[str]:
    predictors = [
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
    if include_assessment_count:
        predictors.append("log1p_post_landmark_assessment_proxy_count")
    return predictors


def fit_logit(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    analysis: str,
    weights: pd.Series | None = None,
    note: str = "",
) -> dict[str, object]:
    model_data = data[[outcome] + predictors].dropna().copy()
    y = model_data[outcome].astype(int)
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    keep_predictors = [col for col in x.columns if x[col].nunique(dropna=True) > 1]
    x = sm.add_constant(x[keep_predictors], has_constant="add")
    if weights is not None:
        model_data = model_data.assign(_weight=weights.loc[model_data.index])
    if y.nunique() < 2 or "high_intensity_class" not in keep_predictors:
        return {
            "dataset_id": DATASET_ID,
            "analysis": analysis,
            "outcome": outcome,
            "effect_type": "odds_ratio",
            "effect": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "n": int(len(model_data)),
            "events": int(y.sum()),
            "covariate_set": ",".join(keep_predictors),
            "analysis_note": note,
            "status": "not_estimable",
        }
    try:
        if weights is None:
            result = sm.Logit(y, x).fit(disp=False, maxiter=300)
        else:
            result = sm.GLM(y, x, family=sm.families.Binomial(), freq_weights=model_data["_weight"]).fit(maxiter=300)
        coef = result.params["high_intensity_class"]
        se = result.bse["high_intensity_class"]
        return {
            "dataset_id": DATASET_ID,
            "analysis": analysis,
            "outcome": outcome,
            "effect_type": "odds_ratio",
            "effect": float(np.exp(coef)),
            "ci_lower": float(np.exp(coef - 1.96 * se)),
            "ci_upper": float(np.exp(coef + 1.96 * se)),
            "pvalue": float(result.pvalues["high_intensity_class"]),
            "n": int(len(model_data)),
            "events": int(y.sum()),
            "covariate_set": ",".join(keep_predictors),
            "analysis_note": note,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "dataset_id": DATASET_ID,
            "analysis": analysis,
            "outcome": outcome,
            "effect_type": "odds_ratio",
            "effect": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "n": int(len(model_data)),
            "events": int(y.sum()),
            "covariate_set": ",".join(keep_predictors),
            "analysis_note": note,
            "status": f"not_estimable_{type(exc).__name__}",
        }


def assessment_ipw(data: pd.DataFrame) -> pd.Series:
    predictors = [
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
    model_data = data[["any_post_landmark_assessment"] + predictors].dropna().copy()
    y = model_data["any_post_landmark_assessment"].astype(int)
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    x = sm.add_constant(x, has_constant="add")
    result = sm.Logit(y, x).fit(disp=False, maxiter=300)
    propensity = pd.Series(result.predict(x), index=model_data.index).clip(0.05, 0.95)
    stabilized = y.mean() / propensity
    out = pd.Series(np.nan, index=data.index)
    out.loc[model_data.index] = stabilized
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-landmark delirium ascertainment-bias sensitivity analyses.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--covariates", type=Path, default=Path("results/covariates/mimic_clinical_covariates_48h.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity/outcome_ascertainment_bias_models.tsv"))
    args = parser.parse_args()

    data = prepare_data(args)
    rows = []
    rows.append(
        fit_logit(
            data,
            OUTCOME,
            base_predictors(),
            "primary_extended_all_patients",
            note="Reference clinical-covariate-extended model; patients without post-landmark assessment are treated as no documented delirium.",
        )
    )
    rows.append(
        fit_logit(
            data[data["any_post_landmark_assessment"] == 1].copy(),
            OUTCOME,
            base_predictors(),
            "restricted_to_any_post_landmark_assessment",
            note="Restricts to patients with at least one post-landmark delirium-positive or negative assessment proxy.",
        )
    )
    rows.append(
        fit_logit(
            data,
            OUTCOME,
            base_predictors(include_assessment_count=True),
            "adjusted_for_post_landmark_assessment_count",
            note="Adjusts for post-landmark assessment proxy count; sensitivity only because assessment count may occur after outcome onset.",
        )
    )
    rows.append(
        fit_logit(
            data,
            "any_post_landmark_assessment",
            base_predictors(),
            "assessment_ascertainment_model",
            note="Models whether patients received any post-landmark delirium assessment proxy; estimates association of phenotype with ascertainment.",
        )
    )
    weights = assessment_ipw(data)
    assessed = data[data["any_post_landmark_assessment"] == 1].copy()
    rows.append(
        fit_logit(
            assessed,
            OUTCOME,
            base_predictors(),
            "ipw_restricted_to_assessed",
            weights=weights,
            note="Inverse-probability weighted model among assessed patients using probability of having any post-landmark assessment.",
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
