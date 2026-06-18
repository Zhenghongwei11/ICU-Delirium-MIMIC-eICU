#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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


def zscore_from_train(train_values: pd.Series, values: pd.Series) -> pd.Series:
    train_x = pd.to_numeric(train_values, errors="coerce")
    x = pd.to_numeric(values, errors="coerce")
    std = train_x.std(ddof=0)
    if not std or np.isnan(std):
        return x * 0
    return (x - train_x.mean()) / std


def add_age_z(data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["age_z"] = zscore_from_train(reference["anchor_age"], data["anchor_age"])
    return out


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def fit_demographic(data: pd.DataFrame) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    y = data[OUTCOME].astype(int)
    x = sm.add_constant(data[["age_z", "male"]], has_constant="add")
    return sm.Logit(y, x).fit(disp=False, maxiter=200)


def fold_auc_rows(data: pd.DataFrame, folds: int) -> tuple[list[dict[str, object]], np.ndarray]:
    y = data[OUTCOME].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(data), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, y), start=1):
        train_raw = data.iloc[train_idx]
        test_raw = data.iloc[test_idx]
        train = add_age_z(train_raw, train_raw)
        test = add_age_z(test_raw, train_raw)
        result = fit_demographic(train)
        pred = result.predict(sm.add_constant(test[["age_z", "male"]], has_constant="add"))
        oof[test_idx] = pred
        rows.append(
            {
                "section": "Fold diagnostic",
                "metric": f"Fold {fold} demographics-only AUC",
                "estimate": fmt(roc_auc_score(y[test_idx], pred)),
                "interval_or_detail": f"events {int(y[test_idx].sum())}; n {len(test_idx)}",
                "n": len(test_idx),
                "events": int(y[test_idx].sum()),
                "interpretation": "Fold-level AUC varied around the null because age and sex carried little signal in this landmark-selected cohort.",
            }
        )
    return rows, oof


def age_band_rows(data: pd.DataFrame) -> list[dict[str, object]]:
    bins = [64, 69, 74, 79, 84, 89, 200]
    labels = ["65-69", "70-74", "75-79", "80-84", "85-89", "90+"]
    work = data.copy()
    work["age_band"] = pd.cut(work["anchor_age"], bins=bins, labels=labels)
    grouped = (
        work.groupby("age_band", observed=True)[OUTCOME]
        .agg(n="size", events="sum", event_rate="mean")
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                "section": "Age-band outcome gradient",
                "metric": f"Documented delirium rate, age {row['age_band']}",
                "estimate": f"{100 * float(row['event_rate']):.1f}%",
                "interval_or_detail": "",
                "n": int(row["n"]),
                "events": int(row["events"]),
                "interpretation": "Age-band event rates were not monotonic after 48-hour landmark selection and exclusion of pre-landmark documented delirium.",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the demographics-only AUC in the MIMIC landmark cohort.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--prediction", type=Path, default=Path("results/benchmarks/prediction_eval.tsv"))
    parser.add_argument("--prediction-ci", type=Path, default=Path("results/benchmarks/prediction_eval_ci.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/supplement_demographic_baseline_diagnostics.tsv"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data = read_tsv(args.matrix)
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data[OUTCOME] = data[OUTCOME].astype(int)
    data["age_z"] = zscore(data["anchor_age"])
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)

    prediction = read_tsv(args.prediction)
    prediction_ci = read_tsv(args.prediction_ci)
    apparent = prediction[
        (prediction["split_or_cohort"] == "MIMIC_derivation_apparent")
        & (prediction["model"] == "demographic_baseline")
    ].iloc[0]
    cv = prediction_ci[
        (prediction_ci["split_or_cohort"] == "MIMIC_internal_5fold_cv")
        & (prediction_ci["model"] == "demographic_baseline")
    ].iloc[0]

    result = fit_demographic(data)
    fold_rows, oof = fold_auc_rows(data, args.folds)
    y = data[OUTCOME].astype(int).to_numpy()
    rows: list[dict[str, object]] = [
        {
            "section": "Discrimination diagnostic",
            "metric": "Apparent demographics-only AUC",
            "estimate": fmt(float(apparent["auc_or_cstat"])),
            "interval_or_detail": "Derivation apparent performance, not external validation",
            "n": int(apparent["n"]),
            "events": int(apparent["events"]),
            "interpretation": "Age and sex showed only near-null apparent separation in the 48-hour landmark cohort.",
        },
        {
            "section": "Discrimination diagnostic",
            "metric": "Internal 5-fold CV demographics-only AUC",
            "estimate": fmt(float(cv["auc_or_cstat"])),
            "interval_or_detail": f"95% CI {fmt(float(cv['auc_ci_lower']))}-{fmt(float(cv['auc_ci_upper']))}",
            "n": int(cv["n"]),
            "events": int(cv["events"]),
            "interpretation": "The confidence interval crossed 0.5; the value below 0.5 reflects weak out-of-sample signal rather than evidence of a usable reverse predictor.",
        },
        {
            "section": "Coefficient diagnostic",
            "metric": "Age, per SD, demographics-only apparent model",
            "estimate": f"OR {np.exp(result.params['age_z']):.2f}",
            "interval_or_detail": f"p={result.pvalues['age_z']:.3f}",
            "n": len(data),
            "events": int(y.sum()),
            "interpretation": "The age coefficient was positive but small and statistically weak after landmark selection.",
        },
        {
            "section": "Coefficient diagnostic",
            "metric": "Male sex, demographics-only apparent model",
            "estimate": f"OR {np.exp(result.params['male']):.2f}",
            "interval_or_detail": f"p={result.pvalues['male']:.3f}",
            "n": len(data),
            "events": int(y.sum()),
            "interpretation": "Sex alone did not materially separate later documented delirium in this cohort.",
        },
        {
            "section": "Correlation diagnostic",
            "metric": "Pearson correlation between age and documented incident delirium",
            "estimate": fmt(float(np.corrcoef(data["anchor_age"], y)[0, 1])),
            "interval_or_detail": "",
            "n": len(data),
            "events": int(y.sum()),
            "interpretation": "The age-outcome association was minimal after selecting patients who reached the 48-hour landmark without pre-landmark documented delirium.",
        },
        {
            "section": "Directionality diagnostic",
            "metric": "AUC after reversing internal CV predictions",
            "estimate": fmt(roc_auc_score(y, 1 - oof)),
            "interval_or_detail": "",
            "n": len(data),
            "events": int(y.sum()),
            "interpretation": "The reversed-score AUC is the complement of the near-null CV AUC and does not indicate a prespecified reverse clinical predictor.",
        },
    ]
    rows.extend(fold_rows)
    rows.extend(age_band_rows(data))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
