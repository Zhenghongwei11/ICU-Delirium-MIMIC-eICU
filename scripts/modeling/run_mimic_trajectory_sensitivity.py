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

from build_mimic_dynamic_phenotypes import DATASET_ID, RANDOM_STATE, prepare_model_data


SILHOUETTE_SAMPLE_SIZE = 2000


def feature_sets(features: list[str]) -> dict[str, list[str]]:
    nursing = [f for f in features if "_records_" in f]
    care_process = [f for f in features if f.startswith("log1p_all_") or f.startswith("all_night_fraction_")]
    return {
        "nursing_vulnerability_only": nursing,
        "care_process_day_night_only": care_process,
        "joint_nursing_care_process": features,
    }


def fit_gmm(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    model = GaussianMixture(n_components=k, covariance_type="diag", n_init=10, random_state=RANDOM_STATE)
    model.fit(x)
    probs = model.predict_proba(x)
    labels = probs.argmax(axis=1)
    return labels, probs.max(axis=1), float(model.aic(x)), float(model.bic(x)), float(np.mean(probs.max(axis=1)))


def fit_kmeans(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = model.fit_predict(x)
    # KMeans has no posterior probability; inverse-distance confidence is reported only as a descriptive proxy.
    distances = model.transform(x)
    nearest = distances.min(axis=1)
    confidence_proxy = 1 / (1 + nearest)
    inertia = float(model.inertia_)
    return labels, confidence_proxy, inertia, np.nan, float(np.mean(confidence_proxy))


def summarize_classes(
    data: pd.DataFrame,
    labels: np.ndarray,
    confidence: np.ndarray,
    feature_set: str,
    algorithm: str,
    k: int,
    raw_profile_cols: list[str],
) -> pd.DataFrame:
    work = data[["stay_id", "incident_post_landmark_delirium"] + raw_profile_cols].copy()
    work["assigned_class"] = labels
    work["assignment_confidence_proxy"] = confidence
    profile = (
        work.groupby("assigned_class", dropna=False)
        .agg(
            n=("stay_id", "size"),
            incident_delirium_events=("incident_post_landmark_delirium", "sum"),
            incident_delirium_rate=("incident_post_landmark_delirium", "mean"),
            mean_assignment_confidence_proxy=("assignment_confidence_proxy", "mean"),
            **{col: (col, "mean") for col in raw_profile_cols},
        )
        .reset_index()
    )
    profile["dataset_id"] = DATASET_ID
    profile["feature_set"] = feature_set
    profile["algorithm"] = algorithm
    profile["number_of_classes"] = k
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Run trajectory/cluster sensitivity models for MIMIC dynamic nursing phenotypes.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--nursing", type=Path, default=Path("results/exposure_windows/nursing_stream_long.tsv"))
    parser.add_argument("--nocturnal", type=Path, default=Path("results/exposure_windows/nocturnal_events_long.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/phenotypes"))
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=5)
    args = parser.parse_args()

    data, feature_frame, all_features = prepare_model_data(args)
    raw_profile_cols = [c for c in data.columns if c.endswith("_24h") or c.endswith("_48h") or c.endswith("_delta")]
    fit_rows = []
    profile_frames = []

    for set_name, selected_features in feature_sets(all_features).items():
        if not selected_features:
            continue
        scaler = StandardScaler()
        x = scaler.fit_transform(feature_frame[selected_features])
        for algorithm in ["GaussianMixture_diag", "KMeans"]:
            for k in range(args.min_k, args.max_k + 1):
                if algorithm == "GaussianMixture_diag":
                    labels, confidence, aic_or_inertia, bic, mean_confidence = fit_gmm(x, k)
                    aic = aic_or_inertia
                    inertia = np.nan
                else:
                    labels, confidence, inertia, bic, mean_confidence = fit_kmeans(x, k)
                    aic = np.nan
                class_props = pd.Series(labels).value_counts(normalize=True)
                silhouette = (
                    float(silhouette_score(x, labels, sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(labels)), random_state=RANDOM_STATE))
                    if len(set(labels)) > 1
                    else np.nan
                )
                fit_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "feature_set": set_name,
                        "algorithm": algorithm,
                        "number_of_classes": k,
                        "n_features": len(selected_features),
                        "aic": aic,
                        "bic": bic,
                        "inertia": inertia,
                        "silhouette": silhouette,
                        "mean_assignment_confidence_proxy": mean_confidence,
                        "min_class_proportion": float(class_props.min()),
                        "selection_eligible": bool(class_props.min() >= 0.05),
                    }
                )
                profile_frames.append(
                    summarize_classes(data, labels, confidence, set_name, algorithm, k, raw_profile_cols)
                )

    fit = pd.DataFrame(fit_rows)
    fit["selection_flag"] = False
    for (set_name, algorithm), group in fit.groupby(["feature_set", "algorithm"]):
        eligible = group[group["selection_eligible"]].copy()
        if eligible.empty:
            continue
        if algorithm == "GaussianMixture_diag":
            selected_idx = eligible.sort_values(["bic", "number_of_classes"]).index[0]
        else:
            selected_idx = eligible.sort_values(["silhouette", "number_of_classes"], ascending=[False, True]).index[0]
        fit.loc[selected_idx, "selection_flag"] = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit.to_csv(args.out_dir / "trajectory_sensitivity_model_fit.tsv", sep="\t", index=False)
    pd.concat(profile_frames, ignore_index=True).to_csv(
        args.out_dir / "trajectory_sensitivity_class_profiles.tsv", sep="\t", index=False
    )
    print(f"Wrote {args.out_dir / 'trajectory_sensitivity_model_fit.tsv'}")
    print(f"Wrote {args.out_dir / 'trajectory_sensitivity_class_profiles.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
