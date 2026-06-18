#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from build_mimic_dynamic_phenotypes import DATASET_ID, RANDOM_STATE, prepare_model_data


def high_intensity_label(data: pd.DataFrame, labels: np.ndarray) -> int:
    work = data[["stay_id"]].copy()
    work["label"] = labels
    profile = work.join(data[["all_total_count_48h", "rass_sedation_records_48h", "restraints_records_48h", "skin_braden_pressure_records_48h"]])
    scores = (
        profile.groupby("label")[["all_total_count_48h", "rass_sedation_records_48h", "restraints_records_48h", "skin_braden_pressure_records_48h"]]
        .mean()
        .sum(axis=1)
    )
    return int(scores.idxmax())


def fit_labels(x: np.ndarray, k: int, seed: int, n_init: int) -> tuple[np.ndarray, float, float, float]:
    model = GaussianMixture(n_components=k, covariance_type="diag", n_init=n_init, random_state=seed)
    model.fit(x)
    labels = model.predict(x)
    probs = model.predict_proba(x).max(axis=1)
    return labels, float(model.aic(x)), float(model.bic(x)), float(np.mean(probs))


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess GMM phenotype stability across random seeds and initialization counts.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--nursing", type=Path, default=Path("results/exposure_windows/nursing_stream_long.tsv"))
    parser.add_argument("--nocturnal", type=Path, default=Path("results/exposure_windows/nocturnal_events_long.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/phenotypes"))
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--stability-n-init", type=int, default=1)
    parser.add_argument("--final-n-init", type=int, default=100)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=5)
    args = parser.parse_args()

    data, feature_frame, _ = prepare_model_data(args)
    x = StandardScaler().fit_transform(feature_frame)

    reference_labels, reference_aic, reference_bic, reference_pp = fit_labels(x, 2, RANDOM_STATE, args.final_n_init)
    reference_high = high_intensity_label(data, reference_labels)
    reference_high_mask = reference_labels == reference_high

    seed_rows = []
    selection_rows = []
    for offset in range(args.seeds):
        seed = RANDOM_STATE + offset
        best = None
        for k in range(args.min_k, args.max_k + 1):
            labels, aic, bic, mean_pp = fit_labels(x, k, seed, args.stability_n_init)
            class_props = pd.Series(labels).value_counts(normalize=True)
            row = {
                "dataset_id": DATASET_ID,
                "seed": seed,
                "number_of_classes": k,
                "n_init": args.stability_n_init,
                "aic": aic,
                "bic": bic,
                "mean_max_posterior": mean_pp,
                "min_class_proportion": float(class_props.min()),
                "selection_eligible": bool(class_props.min() >= 0.05),
            }
            selection_rows.append(row)
            if row["selection_eligible"] and (best is None or (bic, k) < (best["bic"], best["number_of_classes"])):
                best = row
            if k == 2:
                high = high_intensity_label(data, labels)
                high_mask = labels == high
                seed_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "seed": seed,
                        "number_of_classes": k,
                        "n_init": args.stability_n_init,
                        "adjusted_rand_vs_reference": float(adjusted_rand_score(reference_labels, labels)),
                        "high_intensity_jaccard_vs_reference": float((high_mask & reference_high_mask).sum() / (high_mask | reference_high_mask).sum()),
                        "high_intensity_n": int(high_mask.sum()),
                        "lower_intensity_n": int((~high_mask).sum()),
                        "high_intensity_event_rate": float(data.loc[high_mask, "incident_post_landmark_delirium"].mean()),
                        "lower_intensity_event_rate": float(data.loc[~high_mask, "incident_post_landmark_delirium"].mean()),
                        "mean_max_posterior": mean_pp,
                        "min_class_proportion": float(class_props.min()),
                        "aic": aic,
                        "bic": bic,
                    }
                )
        if best is not None:
            selection_rows[-1]["seed_selected_k_by_eligible_bic"] = best["number_of_classes"]

    stability = pd.DataFrame(seed_rows)
    selection = pd.DataFrame(selection_rows)
    selected_k_by_seed = (
        selection[selection["selection_eligible"]]
        .sort_values(["seed", "bic", "number_of_classes"])
        .groupby("seed", as_index=False)
        .first()[["seed", "number_of_classes"]]
        .rename(columns={"number_of_classes": "selected_k_by_eligible_bic"})
    )
    selection = selection.merge(selected_k_by_seed, on="seed", how="left")

    summary = pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "reference_k": 2,
                "reference_n_init": args.final_n_init,
                "reference_aic": reference_aic,
                "reference_bic": reference_bic,
                "reference_mean_max_posterior": reference_pp,
                "n_seed_runs": len(stability),
                "median_adjusted_rand_vs_reference": stability["adjusted_rand_vs_reference"].median(),
                "min_adjusted_rand_vs_reference": stability["adjusted_rand_vs_reference"].min(),
                "median_high_intensity_jaccard_vs_reference": stability["high_intensity_jaccard_vs_reference"].median(),
                "min_high_intensity_jaccard_vs_reference": stability["high_intensity_jaccard_vs_reference"].min(),
                "median_high_intensity_n": stability["high_intensity_n"].median(),
                "min_high_intensity_n": stability["high_intensity_n"].min(),
                "max_high_intensity_n": stability["high_intensity_n"].max(),
                "selected_k_2_fraction": float((selected_k_by_seed["selected_k_by_eligible_bic"] == 2).mean()),
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stability.to_csv(args.out_dir / "gmm_seed_stability.tsv", sep="\t", index=False)
    selection.to_csv(args.out_dir / "gmm_seed_model_selection.tsv", sep="\t", index=False)
    summary.to_csv(args.out_dir / "gmm_stability_summary.tsv", sep="\t", index=False)

    row = summary.iloc[0]
    print(
        "GMM k=2 stability: "
        f"median ARI={row['median_adjusted_rand_vs_reference']:.3f}, "
        f"min ARI={row['min_adjusted_rand_vs_reference']:.3f}, "
        f"median high-class Jaccard={row['median_high_intensity_jaccard_vs_reference']:.3f}, "
        f"selected k=2 fraction={row['selected_k_2_fraction']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
