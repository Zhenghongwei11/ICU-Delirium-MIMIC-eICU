#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATASET_ID = "MIMIC-IV-2.2"


def read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing feature matrix: {path}")
    return pd.read_csv(path, sep="\t")


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not std or np.isnan(std):
        return values * 0
    return (values - values.mean()) / std


def fit_logit(data: pd.DataFrame, outcome: str, predictors: list[str], model_name: str) -> dict[str, object]:
    cols = [outcome] + predictors
    model_data = data[cols].dropna().copy()
    y = model_data[outcome].astype(int)
    x = model_data[predictors].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    x = sm.add_constant(x, has_constant="add")
    try:
        result = sm.Logit(y, x).fit(disp=False, maxiter=200)
        llf = result.llf
        aic = result.aic
        rows = []
        for term in result.params.index:
            coef = result.params[term]
            se = result.bse[term]
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "model": model_name,
                    "term": term,
                    "coef": coef,
                    "se": se,
                    "odds_ratio": float(np.exp(coef)),
                    "ci_lower": float(np.exp(coef - 1.96 * se)),
                    "ci_upper": float(np.exp(coef + 1.96 * se)),
                    "pvalue": result.pvalues[term],
                    "n": len(model_data),
                    "events": int(y.sum()),
                    "aic": aic,
                    "log_likelihood": llf,
                }
            )
        return {"ok": True, "rows": rows}
    except Exception as exc:
        return {"ok": False, "rows": [{"dataset_id": DATASET_ID, "model": model_name, "term": "MODEL_FAILED", "error": str(exc), "n": len(model_data), "events": int(y.sum())}]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run first-pass MIMIC screening baseline models.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/benchmarks/mimic_screening_baseline_models.tsv"))
    parser.add_argument("--sensitivity-out", type=Path, default=Path("results/sensitivity/daytime_vs_nocturnal_comparator.tsv"))
    args = parser.parse_args()

    data = read_matrix(args.matrix)
    data = data[(data["pre_landmark_delirium_positive"] == False)].copy()
    outcome = "incident_post_landmark_delirium"

    data["age_z"] = zscore(data["anchor_age"])
    data["icu_los_z"] = zscore(data["icu_los_hours"])
    data["night_all_z"] = zscore(data["event_all_night_22_06_count_48h"])
    data["day_all_z"] = zscore(data["event_all_day_06_22_count_48h"])
    data["total_all_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])
    data["braden_skin_records_z"] = zscore(data.get("nursing_skin_braden_pressure_n_records_48h", 0))
    data["rass_records_z"] = zscore(data.get("nursing_rass_sedation_n_records_48h", 0))
    data["mobility_records_z"] = zscore(data.get("nursing_mobility_turning_n_records_48h", 0))
    data["pain_records_z"] = zscore(data.get("nursing_pain_n_records_48h", 0))
    data["restraint_records_z"] = zscore(data.get("nursing_restraints_n_records_48h", 0))
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)

    models = {
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
    }

    rows = []
    for model_name, predictors in models.items():
        fit = fit_logit(data, outcome, predictors, model_name)
        rows.extend(fit["rows"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep="\t", index=False)

    comparator_models = {"care_intensity_total", "day_vs_night_counts", "night_fraction"}
    comparator_terms = {"total_all_z", "day_all_z", "night_all_z", "night_fraction_z"}
    sensitivity = out[(out["model"].isin(comparator_models)) & (out["term"].isin(comparator_terms))].copy()
    sensitivity["comparison_family"] = "daytime_nocturnal_total_documentation_intensity"
    sensitivity["interpretation_rule"] = (
        "Nocturnal specificity requires a night term or night fraction signal that is stronger than daytime "
        "and total documentation-intensity comparators."
    )
    args.sensitivity_out.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(args.sensitivity_out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.sensitivity_out}")
    print(f"Analysis cohort excluding pre-landmark delirium: n={len(data)}, events={int(data[outcome].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
