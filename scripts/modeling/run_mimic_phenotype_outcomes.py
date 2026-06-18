#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATASET_ID = "MIMIC-IV-2.2"


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


def fit_model(data: pd.DataFrame, model_name: str, predictors: list[str]) -> list[dict[str, object]]:
    outcome = "incident_post_landmark_delirium"
    model_data = data[[outcome] + predictors].dropna().copy()
    y = model_data[outcome].astype(int)
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    x = sm.add_constant(x, has_constant="add")
    result = sm.Logit(y, x).fit(disp=False, maxiter=200)
    rows = []
    for term in result.params.index:
        coef = result.params[term]
        se = result.bse[term]
        rows.append(
            {
                "claim_id": "dynamic_nursing_screening",
                "dataset_id": DATASET_ID,
                "outcome": "incident_post_landmark_delirium",
                "landmark_time": "48h",
                "phenotype_class": term,
                "reference_class": "low care intensity",
                "model": model_name,
                "effect_type": "odds_ratio",
                "effect": float(np.exp(coef)),
                "ci_lower": float(np.exp(coef - 1.96 * se)),
                "ci_upper": float(np.exp(coef + 1.96 * se)),
                "pvalue": result.pvalues[term],
                "n": len(model_data),
                "events": int(y.sum()),
                "covariate_set": ",".join(predictors),
                "aic": result.aic,
            }
        )
    return rows


def add_bh_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    p = pd.to_numeric(out["pvalue"], errors="coerce")
    valid = p.notna()
    out["fdr"] = np.nan
    if not valid.any():
        return out
    ranked = p[valid].sort_values()
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    out.loc[adjusted.index, "fdr"] = adjusted
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate incident delirium effects by MIMIC dynamic nursing phenotype.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/effect_sizes/claim_effects.tsv"))
    args = parser.parse_args()

    matrix = read_tsv(args.matrix)
    classes = read_tsv(args.classes)
    data = matrix.merge(classes[["stay_id", "assigned_class", "class_label", "posterior_probability"]], on="stay_id", how="inner")
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data["high_intensity_class"] = (data["class_label"].str.contains("high care intensity", na=False)).astype(int)
    data["age_z"] = zscore(data["anchor_age"])
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)
    data["total_event_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])

    rows = []
    rows.extend(fit_model(data, "unadjusted", ["high_intensity_class"]))
    rows.extend(fit_model(data, "demographic_adjusted", ["high_intensity_class", "age_z", "male"]))
    rows.extend(fit_model(data, "care_intensity_adjusted", ["high_intensity_class", "age_z", "male", "total_event_z", "night_fraction_z"]))
    out = add_bh_fdr(pd.DataFrame(rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)

    class_summary = (
        data.groupby(["assigned_class", "class_label"], dropna=False)
        .agg(n=("stay_id", "size"), events=("incident_post_landmark_delirium", "sum"), event_rate=("incident_post_landmark_delirium", "mean"))
        .reset_index()
    )
    class_summary["dataset_id"] = DATASET_ID
    class_summary.to_csv(Path("results/figures/figure3_outcome_effects_anchor.tsv"), sep="\t", index=False)
    print(f"Wrote {args.out}; n={len(data)}, events={int(data['incident_post_landmark_delirium'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
