#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def summarize_binary(data: pd.DataFrame, col: str) -> str:
    n = int(data[col].sum())
    pct = 100 * data[col].mean()
    return f"{n} ({pct:.1f}%)"


def summarize_cont(data: pd.DataFrame, col: str) -> str:
    values = pd.to_numeric(data[col], errors="coerce")
    return f"{values.median():.1f} [{values.quantile(0.25):.1f}, {values.quantile(0.75):.1f}]"


def table1(matrix: pd.DataFrame, classes: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    data = matrix.merge(classes[["stay_id", "class_label"]], on="stay_id", how="inner").merge(
        covariates, on=["subject_id", "hadm_id", "stay_id"], how="left"
    )
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data["phenotype"] = data["class_label"].map(
        lambda x: "High documentation/care-process intensity"
        if "high" in str(x)
        else "Lower documentation/care-process intensity"
    )
    data["male"] = data["gender"].astype(str).str.upper().eq("M")
    data["incident_post_landmark_delirium"] = data["incident_post_landmark_delirium"].astype(bool)
    rows = []
    variables = [
        ("Age, years", "anchor_age", "continuous"),
        ("Male sex", "male", "binary"),
        ("Dementia diagnosis", "dementia_flag", "binary"),
        ("Sepsis diagnosis", "sepsis_flag", "binary"),
        ("Sedative exposure, 0-48h", "sedative_exposure", "binary"),
        ("Opioid exposure, 0-48h", "opioid_exposure", "binary"),
        ("Vasopressor exposure, 0-48h", "vasopressor_exposure", "binary"),
        ("Incident post-landmark delirium", "incident_post_landmark_delirium", "binary"),
        ("ICU length of stay, days", "icu_los_days", "continuous"),
    ]
    data["icu_los_days"] = pd.to_numeric(data["icu_los_hours"], errors="coerce") / 24
    for label, col, kind in variables:
        row = {"variable": label}
        for group, sub in data.groupby("phenotype"):
            if kind == "binary":
                row[group] = summarize_binary(sub, col)
            else:
                row[group] = summarize_cont(sub, col)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build publication-ready source tables from analysis outputs.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matrix = read_tsv(Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    classes = read_tsv(Path("results/phenotypes/trajectory_classes.tsv"))
    covariates = read_tsv(Path("results/covariates/mimic_clinical_covariates_48h.tsv"))
    table1(matrix, classes, covariates).to_csv(args.out_dir / "table1_baseline_characteristics.tsv", sep="\t", index=False)

    fit = read_tsv(Path("results/phenotypes/trajectory_model_fit.tsv"))
    sensitivity_fit = read_tsv(Path("results/phenotypes/trajectory_sensitivity_model_fit.tsv"))
    pd.concat(
        [
            fit.assign(table_section="primary_dynamic_phenotype"),
            sensitivity_fit.assign(table_section="sensitivity_trajectory_models"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(args.out_dir / "table2_trajectory_model_fit.tsv", sep="\t", index=False)

    primary = read_tsv(Path("results/effect_sizes/claim_effects.tsv")).assign(table_section="primary_models")
    extended = read_tsv(Path("results/effect_sizes/claim_effects_extended.tsv")).assign(table_section="clinical_covariate_extended")
    secondary = read_tsv(Path("results/effect_sizes/secondary_outcomes.tsv")).assign(table_section="secondary_outcomes")
    pd.concat([primary, extended, secondary], ignore_index=True, sort=False).to_csv(
        args.out_dir / "table3_effect_estimates.tsv", sep="\t", index=False
    )

    replication = read_tsv(Path("results/replication/combined_summary.tsv")).assign(table_section="partial_construct_validation")
    landmark = read_tsv(Path("results/sensitivity/landmark_window_comparison.tsv")).assign(table_section="landmark_sensitivity")
    death = read_tsv(Path("results/sensitivity/death_competing_sensitivity.tsv")).assign(table_section="death_competing_sensitivity")
    pd.concat([replication, landmark, death], ignore_index=True, sort=False).to_csv(
        args.out_dir / "table4_validation_sensitivity.tsv", sep="\t", index=False
    )

    read_tsv(Path("results/benchmarks/prediction_eval_ci.tsv")).to_csv(
        args.out_dir / "table5_screening_performance_with_ci.tsv", sep="\t", index=False
    )
    read_tsv(Path("results/missingness/outcome_ascertainment_by_phenotype.tsv")).to_csv(
        args.out_dir / "supplement_outcome_ascertainment.tsv", sep="\t", index=False
    )
    read_tsv(Path("results/sensitivity/subgroup_fairness_checks.tsv")).to_csv(
        args.out_dir / "supplement_subgroup_fairness_checks.tsv", sep="\t", index=False
    )
    read_tsv(Path("results/sensitivity/outcome_ascertainment_bias_models.tsv")).to_csv(
        args.out_dir / "supplement_outcome_ascertainment_bias_models.tsv", sep="\t", index=False
    )
    read_tsv(Path("results/benchmarks/penalized_logistic_sensitivity.tsv")).to_csv(
        args.out_dir / "supplement_penalized_logistic_sensitivity.tsv", sep="\t", index=False
    )
    read_tsv(Path("results/benchmarks/penalized_logistic_coefficients.tsv")).to_csv(
        args.out_dir / "supplement_penalized_logistic_coefficients.tsv", sep="\t", index=False
    )

    print(f"Wrote publication source tables to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
