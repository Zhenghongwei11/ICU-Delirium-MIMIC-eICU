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


def prepare_window(data: pd.DataFrame, window: int) -> pd.DataFrame:
    out = data[data["pre_landmark_delirium_positive"] == False].copy()
    out[OUTCOME] = out[OUTCOME].astype(int)
    out["age_z"] = zscore(out["anchor_age"])
    out["male"] = (out["gender"].astype(str).str.upper() == "M").astype(int)
    out["day_all_z"] = zscore(out[f"event_all_day_06_22_count_{window}h"])
    out["night_all_z"] = zscore(out[f"event_all_night_22_06_count_{window}h"])
    out["total_all_z"] = zscore(out[f"event_all_total_count_{window}h"])
    out["night_fraction_z"] = zscore(out[f"event_all_night_fraction_{window}h"])
    out["braden_skin_records_z"] = zscore(out.get(f"nursing_skin_braden_pressure_n_records_{window}h", 0))
    out["rass_records_z"] = zscore(out.get(f"nursing_rass_sedation_n_records_{window}h", 0))
    out["mobility_records_z"] = zscore(out.get(f"nursing_mobility_turning_n_records_{window}h", 0))
    out["pain_records_z"] = zscore(out.get(f"nursing_pain_n_records_{window}h", 0))
    out["restraint_records_z"] = zscore(out.get(f"nursing_restraints_n_records_{window}h", 0))
    return out


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
    }


def fit_model(data: pd.DataFrame, window: int, model: str, predictors: list[str]) -> list[dict[str, object]]:
    model_data = data[[OUTCOME] + predictors].dropna().copy()
    y = model_data[OUTCOME].astype(int)
    if y.nunique() < 2:
        return [
            {
                "dataset_id": DATASET_ID,
                "landmark_hours": window,
                "outcome": OUTCOME,
                "model": model,
                "term": "MODEL_NOT_ESTIMABLE",
                "odds_ratio": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "pvalue": np.nan,
                "auc_or_cstat": np.nan,
                "brier": np.nan,
                "aic": np.nan,
                "n": int(len(model_data)),
                "events": int(y.sum()),
                "status": "not_estimable_no_outcome_variation",
            }
        ]
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    x = sm.add_constant(x, has_constant="add")
    try:
        result = sm.Logit(y, x).fit(disp=False, maxiter=200)
    except Exception as exc:
        return [
            {
                "dataset_id": DATASET_ID,
                "landmark_hours": window,
                "outcome": OUTCOME,
                "model": model,
                "term": "MODEL_NOT_ESTIMABLE",
                "odds_ratio": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "pvalue": np.nan,
                "auc_or_cstat": np.nan,
                "brier": np.nan,
                "aic": np.nan,
                "n": int(len(model_data)),
                "events": int(y.sum()),
                "status": f"not_estimable_{type(exc).__name__}",
            }
        ]
    pred = np.asarray(result.predict(x), dtype=float)
    rows = []
    for term in result.params.index:
        coef = result.params[term]
        se = result.bse[term]
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "landmark_hours": window,
                "outcome": OUTCOME,
                "model": model,
                "term": term,
                "odds_ratio": float(np.exp(coef)),
                "ci_lower": float(np.exp(coef - 1.96 * se)),
                "ci_upper": float(np.exp(coef + 1.96 * se)),
                "pvalue": result.pvalues[term],
                "auc_or_cstat": float(roc_auc_score(y, pred)),
                "brier": float(brier_score_loss(y, pred)),
                "aic": float(result.aic),
                "n": int(len(model_data)),
                "events": int(y.sum()),
                "status": "ok",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare simple screening models across 24 h, 48 h, and 72 h landmarks.")
    parser.add_argument("--windows", type=int, nargs="+", default=[24, 48, 72])
    parser.add_argument("--matrix-dir", type=Path, default=Path("results/phenotypes"))
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity/landmark_window_comparison.tsv"))
    args = parser.parse_args()

    rows = []
    for window in args.windows:
        matrix = read_tsv(args.matrix_dir / f"mimic_feature_matrix_{window}h.tsv")
        data = prepare_window(matrix, window)
        for model, predictors in model_specs().items():
            rows.extend(fit_model(data, window, model, predictors))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
