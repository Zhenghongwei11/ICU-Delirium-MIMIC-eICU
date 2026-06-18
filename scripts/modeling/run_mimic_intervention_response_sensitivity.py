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


def log1p_z(values: pd.Series) -> pd.Series:
    return zscore(np.log1p(pd.to_numeric(values, errors="coerce").fillna(0)))


def prepare_data(matrix_path: Path, classes_path: Path, covariates_path: Path) -> pd.DataFrame:
    matrix = read_tsv(matrix_path)
    classes = read_tsv(classes_path)
    covariates = read_tsv(covariates_path)
    data = (
        matrix.merge(classes[["stay_id", "class_label"]], on="stay_id", how="inner")
        .merge(covariates, on=["subject_id", "hadm_id", "stay_id"], how="left")
    )
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data[OUTCOME] = data[OUTCOME].astype(int)
    data["high_intensity_class"] = data["class_label"].str.contains("high care intensity", na=False).astype(int)
    data["age_z"] = zscore(data["anchor_age"])
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)
    data["total_event_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])
    data["diagnosis_count_z"] = zscore(data["diagnosis_count"])
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

    record_cols = {
        "mobility_turning_records_z": "nursing_mobility_turning_n_records_48h",
        "orientation_records_z": "nursing_orientation_n_records_48h",
        "moisture_hygiene_records_z": "nursing_moisture_hygiene_n_records_48h",
        "oral_suction_airway_records_z": "nursing_oral_suction_airway_n_records_48h",
        "nutrition_records_z": "nursing_nutrition_n_records_48h",
        "pain_records_z": "nursing_pain_n_records_48h",
        "rass_sedation_records_z": "nursing_rass_sedation_n_records_48h",
        "restraint_records_z": "nursing_restraints_n_records_48h",
    }
    for new_col, source_col in record_cols.items():
        data[new_col] = log1p_z(data[source_col])

    active_response_sources = [
        "nursing_mobility_turning_n_records_48h",
        "nursing_orientation_n_records_48h",
        "nursing_moisture_hygiene_n_records_48h",
        "nursing_oral_suction_airway_n_records_48h",
        "nursing_nutrition_n_records_48h",
        "nursing_pain_n_records_48h",
    ]
    data["active_nursing_response_proxy_z"] = log1p_z(data[active_response_sources].sum(axis=1))
    data["sedation_restraint_documentation_proxy_z"] = log1p_z(
        data["nursing_rass_sedation_n_records_48h"] + data["nursing_restraints_n_records_48h"]
    )
    return data


def fit_logit(data: pd.DataFrame, model_name: str, predictors: list[str], interpretation: str) -> dict[str, object]:
    model_data = data[[OUTCOME] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = model_data[OUTCOME].astype(int)
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce")
    x = sm.add_constant(x, has_constant="add")
    result = sm.Logit(y, x).fit(disp=False, maxiter=300)
    coef = result.params["high_intensity_class"]
    se = result.bse["high_intensity_class"]
    return {
        "dataset_id": DATASET_ID,
        "outcome": OUTCOME,
        "analysis": model_name,
        "effect_type": "odds_ratio",
        "effect": float(np.exp(coef)),
        "ci_lower": float(np.exp(coef - 1.96 * se)),
        "ci_upper": float(np.exp(coef + 1.96 * se)),
        "pvalue": float(result.pvalues["high_intensity_class"]),
        "n": int(len(model_data)),
        "events": int(y.sum()),
        "covariate_set": ",".join(predictors),
        "interpretation": interpretation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run intervention-response proxy sensitivity models for the nursing phenotype analysis.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--covariates", type=Path, default=Path("results/covariates/mimic_clinical_covariates_48h.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity/intervention_response_proxy_sensitivity.tsv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/tables/supplement_intervention_response_sensitivity.tsv"))
    args = parser.parse_args()

    data = prepare_data(args.matrix, args.classes, args.covariates)
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
    active_response = reference + ["active_nursing_response_proxy_z"]
    sedation_restraint = reference + ["sedation_restraint_documentation_proxy_z"]
    domain_response = reference + [
        "mobility_turning_records_z",
        "orientation_records_z",
        "moisture_hygiene_records_z",
        "oral_suction_airway_records_z",
        "nutrition_records_z",
        "pain_records_z",
    ]

    rows = [
        fit_logit(
            data,
            "clinical_covariate_reference",
            reference,
            "Reference clinical covariate-extended model.",
        ),
        fit_logit(
            data,
            "clinical_plus_active_nursing_response_proxy",
            active_response,
            "Adds a composite proxy for active bedside nursing-response documentation; interpreted as an overadjustment/proxy sensitivity, not as causal control.",
        ),
        fit_logit(
            data,
            "clinical_plus_sedation_restraint_documentation_proxy",
            sedation_restraint,
            "Adds a proxy for sedation/restraint documentation density to assess overlap with agitation, device protection, and immobility-related care.",
        ),
        fit_logit(
            data,
            "clinical_plus_individual_response_domain_counts",
            domain_response,
            "Adds individual mobility, orientation, hygiene, airway/suction, nutrition, and pain documentation counts as nursing-response proxies.",
        ),
    ]
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.summary_out, sep="\t", index=False)
    for row in rows:
        print(
            f"{row['analysis']}: OR {row['effect']:.2f} "
            f"({row['ci_lower']:.2f}-{row['ci_upper']:.2f}), p={row['pvalue']:.3g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
