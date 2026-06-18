#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DATASET_ID = "MIMIC-IV-2.2"


def clean_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t", **kwargs)


def pivot_nursing(nursing: pd.DataFrame, window: int) -> pd.DataFrame:
    subset = nursing[nursing["window_hours"] == window].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["variable_or_event_category"] = subset["variable_or_event_category"].map(clean_name)
    metrics = ["n_records", "n_numeric", "mean_valuenum", "min_valuenum", "max_valuenum", "first_hour", "last_hour"]
    pieces = []
    for metric in metrics:
        pivot = subset.pivot_table(
            index="patient_id_or_stay_id",
            columns="variable_or_event_category",
            values=metric,
            aggfunc="first",
        )
        pivot.columns = [f"nursing_{col}_{metric}_{window}h" for col in pivot.columns]
        pieces.append(pivot)
    out = pd.concat(pieces, axis=1).reset_index().rename(columns={"patient_id_or_stay_id": "stay_id"})
    return out


def pivot_nocturnal(nocturnal: pd.DataFrame, window: int) -> pd.DataFrame:
    subset = nocturnal[nocturnal["window_hours"] == window].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["variable_or_event_category"] = subset["variable_or_event_category"].map(clean_name)
    subset["day_night_period"] = subset["day_night_period"].map(clean_name)
    subset["feature"] = (
        "event_"
        + subset["variable_or_event_category"]
        + "_"
        + subset["day_night_period"]
        + "_count_"
        + str(window)
        + "h"
    )
    pivot = subset.pivot_table(index="patient_id_or_stay_id", columns="feature", values="value_or_count", aggfunc="sum", fill_value=0)
    out = pivot.reset_index().rename(columns={"patient_id_or_stay_id": "stay_id"})

    categories = sorted(subset["variable_or_event_category"].dropna().unique())
    for category in categories:
        night_col = f"event_{category}_night_22_06_count_{window}h"
        day_col = f"event_{category}_day_06_22_count_{window}h"
        total_col = f"event_{category}_total_count_{window}h"
        if night_col not in out.columns:
            out[night_col] = 0
        if day_col not in out.columns:
            out[day_col] = 0
        out[total_col] = out[night_col] + out[day_col]
        out[f"event_{category}_night_fraction_{window}h"] = out[night_col] / out[total_col].where(out[total_col] != 0)
    night_cols = [c for c in out.columns if c.endswith(f"_night_22_06_count_{window}h")]
    day_cols = [c for c in out.columns if c.endswith(f"_day_06_22_count_{window}h")]
    out[f"event_all_night_22_06_count_{window}h"] = out[night_cols].sum(axis=1) if night_cols else 0
    out[f"event_all_day_06_22_count_{window}h"] = out[day_cols].sum(axis=1) if day_cols else 0
    out[f"event_all_total_count_{window}h"] = out[f"event_all_night_22_06_count_{window}h"] + out[f"event_all_day_06_22_count_{window}h"]
    out[f"event_all_night_fraction_{window}h"] = out[f"event_all_night_22_06_count_{window}h"] / out[f"event_all_total_count_{window}h"].where(
        out[f"event_all_total_count_{window}h"] != 0
    )
    return out


def build_matrix(args: argparse.Namespace) -> pd.DataFrame:
    cohort = read_tsv(args.cohort)
    nursing = read_tsv(args.nursing)
    nocturnal = read_tsv(args.nocturnal)
    delirium = read_tsv(args.delirium)

    window = args.window_hours
    cohort = cohort[cohort[f"eligible_landmark_{window}h"]].copy()
    base_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "gender",
        "anchor_age",
        "race",
        "first_careunit",
        "admission_type",
        "admission_location",
        "icu_los_hours",
        "hospital_los_days",
        "hospital_expire_flag",
        "death_time_from_icu_admit_hours",
    ]
    matrix = cohort[base_cols].copy()
    matrix["dataset_id"] = DATASET_ID
    matrix["landmark_hours"] = window

    nursing_wide = pivot_nursing(nursing, window)
    nocturnal_wide = pivot_nocturnal(nocturnal, window)
    outcome = delirium[delirium["landmark_hours"] == window].copy()
    outcome = outcome.rename(columns={"stay_id": "stay_id"})

    for frame in [nursing_wide, nocturnal_wide, outcome]:
        if not frame.empty:
            matrix = matrix.merge(frame, on="stay_id", how="left")

    count_cols = [c for c in matrix.columns if c.endswith(f"_count_{window}h") or f"_n_records_{window}h" in c or f"_n_numeric_{window}h" in c]
    matrix[count_cols] = matrix[count_cols].fillna(0)
    bool_cols = ["pre_landmark_delirium_positive", "post_landmark_delirium_positive", "incident_post_landmark_delirium"]
    for col in bool_cols:
        if col in matrix.columns:
            matrix[col] = matrix[col].fillna(False).astype(bool)
    assessment_cols = ["pre_landmark_negative_assessments", "post_landmark_negative_assessments"]
    for col in assessment_cols:
        if col in matrix.columns:
            matrix[col] = matrix[col].fillna(0).astype(int)
    return matrix


def write_summaries(matrix: pd.DataFrame, out_dir: Path, window: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome_rows = []
    for col in ["pre_landmark_delirium_positive", "post_landmark_delirium_positive", "incident_post_landmark_delirium"]:
        if col in matrix.columns:
            outcome_rows.append({"dataset_id": DATASET_ID, "landmark_hours": window, "outcome_or_status": col, "n": int(matrix[col].sum()), "denominator": len(matrix)})
    pd.DataFrame(outcome_rows).to_csv(out_dir / f"mimic_outcome_summary_{window}h.tsv", sep="\t", index=False)

    feature_cols = [c for c in matrix.columns if c.startswith("nursing_") or c.startswith("event_")]
    rows = []
    for col in feature_cols:
        values = pd.to_numeric(matrix[col], errors="coerce")
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "landmark_hours": window,
                "feature": col,
                "nonmissing": int(values.notna().sum()),
                "nonzero": int((values.fillna(0) != 0).sum()),
                "mean": values.mean(),
                "median": values.median(),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
                "max": values.max(),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / f"mimic_feature_summary_{window}h.tsv", sep="\t", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MIMIC feature matrix for trajectory and baseline analyses.")
    parser.add_argument("--window-hours", type=int, default=48, choices=[24, 48, 72])
    parser.add_argument("--cohort", type=Path, default=Path("results/cohort/mimic_analysis_cohort.tsv"))
    parser.add_argument("--nursing", type=Path, default=Path("results/exposure_windows/nursing_stream_long.tsv"))
    parser.add_argument("--nocturnal", type=Path, default=Path("results/exposure_windows/nocturnal_events_long.tsv"))
    parser.add_argument("--delirium", type=Path, default=Path("results/cohort/mimic_delirium_landmark_status.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/phenotypes"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(args)
    out_path = args.out_dir / f"mimic_feature_matrix_{args.window_hours}h.tsv"
    matrix.to_csv(out_path, sep="\t", index=False)
    write_summaries(matrix, args.out_dir, args.window_hours)
    print(f"Wrote {out_path} with shape {matrix.shape[0]} x {matrix.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
