#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


DATASET_ID = "MIMIC-IV-2.2"
RANDOM_STATE = 20260608
WINDOWS = [24, 48]
CONCEPTS = [
    "skin_braden_pressure",
    "rass_sedation",
    "mobility_turning",
    "pain",
    "restraints",
    "vitals",
    "oral_suction_airway",
]


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def pivot_nursing_counts(nursing: pd.DataFrame) -> pd.DataFrame:
    subset = nursing[(nursing["window_hours"].isin(WINDOWS)) & (nursing["variable_or_event_category"].isin(CONCEPTS))].copy()
    pivot = subset.pivot_table(
        index="patient_id_or_stay_id",
        columns=["variable_or_event_category", "window_hours"],
        values="n_records",
        aggfunc="first",
        fill_value=0,
    )
    pivot.columns = [f"{concept}_records_{window}h" for concept, window in pivot.columns]
    out = pivot.reset_index().rename(columns={"patient_id_or_stay_id": "stay_id"})
    for concept in CONCEPTS:
        c24 = f"{concept}_records_24h"
        c48 = f"{concept}_records_48h"
        out[c24] = out.get(c24, 0)
        out[c48] = out.get(c48, 0)
        out[f"{concept}_records_24_48h_delta"] = (out[c48] - out[c24]).clip(lower=0)
    return out


def pivot_nocturnal_counts(nocturnal: pd.DataFrame) -> pd.DataFrame:
    subset = nocturnal[(nocturnal["window_hours"].isin(WINDOWS))].copy()
    total = (
        subset.groupby(["patient_id_or_stay_id", "window_hours", "day_night_period"], dropna=False)["value_or_count"]
        .sum()
        .reset_index()
    )
    pivot = total.pivot_table(
        index="patient_id_or_stay_id",
        columns=["day_night_period", "window_hours"],
        values="value_or_count",
        aggfunc="first",
        fill_value=0,
    )
    pivot.columns = [f"all_{period}_count_{window}h" for period, window in pivot.columns]
    out = pivot.reset_index().rename(columns={"patient_id_or_stay_id": "stay_id"})
    for window in WINDOWS:
        day = f"all_day_06_22_count_{window}h"
        night = f"all_night_22_06_count_{window}h"
        out[day] = out.get(day, 0)
        out[night] = out.get(night, 0)
        out[f"all_total_count_{window}h"] = out[day] + out[night]
        out[f"all_night_fraction_{window}h"] = out[night] / out[f"all_total_count_{window}h"].where(out[f"all_total_count_{window}h"] != 0)
    for period in ["day_06_22", "night_22_06", "total"]:
        c24 = f"all_{period}_count_24h"
        c48 = f"all_{period}_count_48h"
        out[f"all_{period}_count_24_48h_delta"] = (out[c48] - out[c24]).clip(lower=0)
    return out


def prepare_model_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    matrix = read_tsv(args.matrix)
    nursing = read_tsv(args.nursing)
    nocturnal = read_tsv(args.nocturnal)

    cohort = matrix[(matrix["pre_landmark_delirium_positive"] == False)].copy()
    nursing_wide = pivot_nursing_counts(nursing)
    nocturnal_wide = pivot_nocturnal_counts(nocturnal)
    data = cohort[["stay_id", "subject_id", "anchor_age", "gender", "incident_post_landmark_delirium"]].copy()
    data = data.merge(nursing_wide, on="stay_id", how="left").merge(nocturnal_wide, on="stay_id", how="left")

    raw_features = []
    for concept in CONCEPTS:
        raw_features.extend([f"{concept}_records_24h", f"{concept}_records_48h", f"{concept}_records_24_48h_delta"])
    raw_features.extend(
        [
            "all_day_06_22_count_24h",
            "all_day_06_22_count_48h",
            "all_day_06_22_count_24_48h_delta",
            "all_night_22_06_count_24h",
            "all_night_22_06_count_48h",
            "all_night_22_06_count_24_48h_delta",
            "all_total_count_24h",
            "all_total_count_48h",
            "all_total_count_24_48h_delta",
            "all_night_fraction_24h",
            "all_night_fraction_48h",
        ]
    )
    for col in raw_features:
        if col not in data.columns:
            data[col] = 0
    data[raw_features] = data[raw_features].fillna(0)

    model_features = []
    for col in raw_features:
        transformed = f"log1p_{col}" if "fraction" not in col else col
        if transformed != col:
            data[transformed] = np.log1p(pd.to_numeric(data[col], errors="coerce").fillna(0))
        else:
            data[transformed] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        model_features.append(transformed)
    return data, data[model_features], model_features


def label_classes(profile: pd.DataFrame) -> dict[int, str]:
    labels = {}
    for _, row in profile.iterrows():
        cls = int(row["assigned_class"])
        total = row.get("all_total_count_48h", 0)
        rass = row.get("rass_sedation_records_48h", 0)
        restraint = row.get("restraints_records_48h", 0)
        braden = row.get("skin_braden_pressure_records_48h", 0)
        night_frac = row.get("all_night_fraction_48h", 0)
        parts = []
        if total >= profile["all_total_count_48h"].quantile(0.67):
            parts.append("high care intensity")
        elif total <= profile["all_total_count_48h"].quantile(0.33):
            parts.append("low care intensity")
        else:
            parts.append("moderate care intensity")
        if rass >= profile["rass_sedation_records_48h"].quantile(0.67) or restraint >= profile["restraints_records_48h"].quantile(0.67):
            parts.append("sedation-restraint dense")
        if braden >= profile["skin_braden_pressure_records_48h"].quantile(0.67):
            parts.append("skin-risk documentation dense")
        if night_frac >= profile["all_night_fraction_48h"].quantile(0.67):
            parts.append("higher night fraction")
        labels[cls] = "; ".join(parts)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit first-pass dynamic nursing phenotypes in MIMIC.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--nursing", type=Path, default=Path("results/exposure_windows/nursing_stream_long.tsv"))
    parser.add_argument("--nocturnal", type=Path, default=Path("results/exposure_windows/nocturnal_events_long.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/phenotypes"))
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data, feature_frame, model_features = prepare_model_data(args)
    scaler = StandardScaler()
    x = scaler.fit_transform(feature_frame)

    fit_rows = []
    assignments = {}
    posteriors = {}
    for k in range(args.min_k, args.max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", n_init=10, random_state=RANDOM_STATE)
        gmm.fit(x)
        probs = gmm.predict_proba(x)
        labels = probs.argmax(axis=1)
        class_sizes = pd.Series(labels).value_counts(normalize=True)
        ave_pp = float(np.mean(probs.max(axis=1)))
        silhouette = float(silhouette_score(x, labels)) if k > 1 and len(set(labels)) > 1 else np.nan
        fit_rows.append(
            {
                "dataset_id": DATASET_ID,
                "model_family": "GaussianMixture_diag",
                "number_of_classes": k,
                "aic": gmm.aic(x),
                "bic": gmm.bic(x),
                "entropy_proxy_mean_max_posterior": ave_pp,
                "silhouette": silhouette,
                "min_class_proportion": float(class_sizes.min()),
                "selection_flag": False,
            }
        )
        assignments[k] = labels
        posteriors[k] = probs.max(axis=1)

    fit = pd.DataFrame(fit_rows)
    eligible = fit[fit["min_class_proportion"] >= 0.05].copy()
    selected_k = int((eligible if not eligible.empty else fit).sort_values(["bic", "number_of_classes"]).iloc[0]["number_of_classes"])
    fit.loc[fit["number_of_classes"] == selected_k, "selection_flag"] = True
    fit.to_csv(args.out_dir / "trajectory_model_fit.tsv", sep="\t", index=False)

    data["assigned_class"] = assignments[selected_k]
    data["posterior_probability"] = posteriors[selected_k]
    raw_profile_cols = [c for c in data.columns if c.endswith("_24h") or c.endswith("_48h") or c.endswith("_delta")]
    profile = (
        data.groupby("assigned_class", dropna=False)
        .agg(
            n=("stay_id", "size"),
            incident_delirium_events=("incident_post_landmark_delirium", "sum"),
            incident_delirium_rate=("incident_post_landmark_delirium", "mean"),
            **{col: (col, "mean") for col in raw_profile_cols},
        )
        .reset_index()
    )
    label_map = label_classes(profile)
    profile["class_label"] = profile["assigned_class"].map(label_map)
    profile["dataset_id"] = DATASET_ID
    profile.to_csv(args.out_dir / "trajectory_class_profiles.tsv", sep="\t", index=False)

    classes = data[["stay_id", "subject_id", "assigned_class", "posterior_probability", "incident_post_landmark_delirium"]].copy()
    classes["class_label"] = classes["assigned_class"].map(label_map)
    classes["dataset_id"] = DATASET_ID
    classes.to_csv(args.out_dir / "trajectory_classes.tsv", sep="\t", index=False)

    anchor = profile[
        [
            "dataset_id",
            "assigned_class",
            "class_label",
            "n",
            "incident_delirium_events",
            "incident_delirium_rate",
            "all_total_count_24h",
            "all_total_count_48h",
            "all_night_fraction_48h",
            "rass_sedation_records_48h",
            "restraints_records_48h",
            "skin_braden_pressure_records_48h",
        ]
    ].copy()
    anchor.to_csv(Path("results/figures/figure2_trajectory_anchor.tsv"), sep="\t", index=False)
    print(f"Selected k={selected_k}; wrote trajectory outputs for n={len(classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
