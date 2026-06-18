#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from build_mimic_covariate_extended_models import DATASET_ID, LANDMARK_HOURS, OUTCOME, read_tsv, zscore


def log1p_z(values: pd.Series) -> pd.Series:
    return zscore(np.log1p(pd.to_numeric(values, errors="coerce").fillna(0)))


def build_analysis_frame(matrix_path: Path, classes_path: Path, covariates_path: Path) -> pd.DataFrame:
    matrix = read_tsv(matrix_path)
    classes = read_tsv(classes_path)
    covariates = read_tsv(covariates_path)
    data = (
        matrix.merge(classes[["stay_id", "class_label"]], on="stay_id", how="inner")
        .merge(covariates, on=["subject_id", "hadm_id", "stay_id"], how="left")
    )
    data = data[data["pre_landmark_delirium_positive"] == False].copy()

    data["high_intensity_class"] = data["class_label"].str.contains("high care intensity", na=False).astype(int)
    data["age_z"] = zscore(data["anchor_age"])
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)
    data["total_event_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])
    data["diagnosis_count_z"] = zscore(data["diagnosis_count"])
    data["medication_infusion_total_z"] = log1p_z(data["event_medication_infusion_total_count_48h"])

    for col in [
        "dementia_flag",
        "sepsis_flag",
        "sedative_exposure",
        "opioid_exposure",
        "vasopressor_exposure",
        "emergency_admission",
        "micu_unit",
    ]:
        data[col] = data[col].fillna(False).astype(int)

    min_rass = pd.to_numeric(data["nursing_rass_sedation_min_valuenum_48h"], errors="coerce")
    data["deep_sedation_or_unarousable_48h"] = (min_rass <= -3).fillna(False).astype(int)
    data["rass_min_z"] = zscore(min_rass.fillna(min_rass.median()))

    high_dx = pd.to_numeric(data["diagnosis_count"], errors="coerce").fillna(0) >= data["diagnosis_count"].quantile(0.75)
    high_med_infusion = pd.to_numeric(data["event_medication_infusion_total_count_48h"], errors="coerce").fillna(0) >= data[
        "event_medication_infusion_total_count_48h"
    ].quantile(0.75)
    proxy_components = pd.DataFrame(
        {
            "high_diagnosis_burden": high_dx.astype(int),
            "sepsis_flag": data["sepsis_flag"],
            "sedative_exposure": data["sedative_exposure"],
            "vasopressor_exposure": data["vasopressor_exposure"],
            "deep_sedation_or_unarousable_48h": data["deep_sedation_or_unarousable_48h"],
            "high_medication_infusion_burden": high_med_infusion.astype(int),
            "micu_unit": data["micu_unit"],
            "emergency_admission": data["emergency_admission"],
        },
        index=data.index,
    )
    data["early_bedside_acuity_proxy"] = proxy_components.sum(axis=1)
    data["early_bedside_acuity_proxy_z"] = zscore(data["early_bedside_acuity_proxy"])
    return data


def fit_logit(data: pd.DataFrame, model_name: str, predictors: list[str]) -> list[dict[str, object]]:
    model_data = data[[OUTCOME] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = model_data[OUTCOME].astype(int)
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce")
    x = sm.add_constant(x, has_constant="add")
    result = sm.Logit(y, x).fit(disp=False, maxiter=300)
    rows = []
    for term in result.params.index:
        coef = result.params[term]
        se = result.bse[term]
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "outcome": OUTCOME,
                "landmark_time": f"{LANDMARK_HOURS}h",
                "model": model_name,
                "term": term,
                "effect_type": "odds_ratio",
                "effect": float(np.exp(coef)),
                "ci_lower": float(np.exp(coef - 1.96 * se)),
                "ci_upper": float(np.exp(coef + 1.96 * se)),
                "pvalue": float(result.pvalues[term]),
                "n": int(len(model_data)),
                "events": int(y.sum()),
                "covariate_set": ",".join(predictors),
                "aic": float(result.aic),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run severity-proxy sensitivity models for the MIMIC nursing phenotype analysis.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--covariates", type=Path, default=Path("results/covariates/mimic_clinical_covariates_48h.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity/severity_proxy_sensitivity.tsv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/tables/supplement_severity_proxy_sensitivity.tsv"))
    args = parser.parse_args()

    data = build_analysis_frame(args.matrix, args.classes, args.covariates)

    reference = [
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
    proxy_adjusted = [
        "high_intensity_class",
        "age_z",
        "male",
        "total_event_z",
        "night_fraction_z",
        "early_bedside_acuity_proxy_z",
    ]
    clinical_plus_bedside = reference + [
        "deep_sedation_or_unarousable_48h",
        "rass_min_z",
        "medication_infusion_total_z",
    ]

    rows = []
    rows.extend(fit_logit(data, "clinical_covariate_extended_reference", reference))
    rows.extend(fit_logit(data, "early_bedside_acuity_proxy_adjusted", proxy_adjusted))
    rows.extend(fit_logit(data, "clinical_plus_bedside_acuity_markers", clinical_plus_bedside))
    effects = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.out, sep="\t", index=False)

    summary = effects[effects["term"] == "high_intensity_class"].copy()
    summary["interpretation"] = [
        "Reference clinical covariate-extended model.",
        "Adds a composite early bedside-acuity proxy instead of individual clinical markers.",
        "Adds deep-sedation/minimum-RASS and medication-infusion burden to the clinical covariate-extended model.",
    ]
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, sep="\t", index=False)

    for _, row in summary.iterrows():
        print(
            f"{row['model']}: OR {row['effect']:.2f} "
            f"({row['ci_lower']:.2f}-{row['ci_upper']:.2f}), p={row['pvalue']:.3g}, n={int(row['n'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
