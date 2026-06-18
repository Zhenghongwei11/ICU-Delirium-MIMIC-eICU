#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def build_figure1(cohort_flow: pd.DataFrame, landmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = {
        "all_icu_stays": 1,
        "age_ge_65": 2,
        "first_icu_stay_per_subject": 3,
        "landmark_48h_eligible": 4,
    }
    for _, row in cohort_flow.iterrows():
        step = row["step_name"]
        if step in order:
            rows.append(
                {
                    "panel": "cohort_flow",
                    "display_order": order[step],
                    "dataset_id": row["dataset_id"],
                    "label": step,
                    "n": int(row["n_remaining"]),
                    "n_removed": int(row["n_removed"]),
                    "note": row["exclusion_reason"],
                }
            )
    for _, row in landmark.iterrows():
        rows.append(
            {
                "panel": "landmark_eligibility",
                "display_order": 10 + int(row["landmark_hours"]),
                "dataset_id": row["dataset_id"],
                "label": f"{int(row['landmark_hours'])}h landmark eligible",
                "n": int(row["n_landmark_eligible"]),
                "n_removed": int(row["n_icu_discharge_before_landmark"]),
                "note": f"death_before_landmark={int(row['n_death_before_landmark'])}",
            }
        )
    rows.append(
        {
            "panel": "study_timeline",
            "display_order": 100,
            "dataset_id": "MIMIC-IV-2.2",
            "label": "0-48h exposure window; post-48h incident delirium outcome window",
            "n": pd.NA,
            "n_removed": pd.NA,
            "note": "Primary landmark design uses only exposure data before 48h and outcomes after 48h.",
        }
    )
    return pd.DataFrame(rows)


def build_storyboard() -> pd.DataFrame:
    rows = [
        {
            "figure_id": "Figure 1",
            "working_title": "Cohort construction and landmark design",
            "claim_id": "C1",
            "primary_anchor_table": "results/figures/figure1_cohort_landmark_flow.tsv",
            "supporting_tables": "results/cohort/cohort_flow.tsv;results/cohort/landmark_eligibility.tsv",
            "planned_panels": "cohort flow; 24h/48h/72h landmark eligibility; 0-48h exposure and post-landmark outcome timeline",
            "message": "The study uses a leakage-safe 48h landmark design in older first ICU stays.",
        },
        {
            "figure_id": "Figure 2",
            "working_title": "Dynamic nursing vulnerability phenotypes",
            "claim_id": "C1",
            "primary_anchor_table": "results/figures/figure2_trajectory_anchor.tsv",
            "supporting_tables": "results/phenotypes/trajectory_model_fit.tsv;results/phenotypes/trajectory_class_profiles.tsv;results/phenotypes/trajectory_sensitivity_model_fit.tsv",
            "planned_panels": "class sizes; nursing documentation profiles; day/night care-process profiles; sensitivity model summary",
            "message": "A balanced two-class joint phenotype separates high care-intensity/sedation-restraint/skin-risk documentation from low care intensity.",
        },
        {
            "figure_id": "Figure 3",
            "working_title": "Post-landmark delirium risk by phenotype",
            "claim_id": "C2",
            "primary_anchor_table": "results/figures/figure3_outcome_effects_anchor.tsv",
            "supporting_tables": "results/effect_sizes/claim_effects.tsv",
            "planned_panels": "event rates by phenotype; odds ratios across adjustment sets",
            "message": "The high-intensity phenotype is associated with higher post-landmark incident delirium risk, attenuated after care-intensity adjustment.",
        },
        {
            "figure_id": "Figure 4",
            "working_title": "Screening performance and clinical utility checks",
            "claim_id": "C3",
            "primary_anchor_table": "results/figures/figure4_screening_utility_anchor.tsv",
            "supporting_tables": "results/benchmarks/prediction_eval_ci.tsv;results/clinical_utility/decision_curve.tsv;results/sensitivity/daytime_vs_nocturnal_comparator.tsv;results/sensitivity/landmark_window_comparison.tsv;results/sensitivity/subgroup_fairness_checks.tsv",
            "planned_panels": "AUC/Brier with bootstrap CI; decision curves; day/night/total comparator; 24h/48h/72h landmark sensitivity; subgroup checks",
            "message": "Nursing vulnerability density and joint density screens outperform demographic and simple event-burden baselines, but external validation is partial only.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build publication figure anchors and storyboard tables.")
    parser.add_argument("--cohort-flow", type=Path, default=Path("results/cohort/cohort_flow.tsv"))
    parser.add_argument("--landmark", type=Path, default=Path("results/cohort/landmark_eligibility.tsv"))
    parser.add_argument("--figure1-out", type=Path, default=Path("results/figures/figure1_cohort_landmark_flow.tsv"))
    parser.add_argument("--storyboard-out", type=Path, default=Path("docs/FIGURE_STORYBOARD.tsv"))
    args = parser.parse_args()

    figure1 = build_figure1(read_tsv(args.cohort_flow), read_tsv(args.landmark))
    args.figure1_out.parent.mkdir(parents=True, exist_ok=True)
    figure1.to_csv(args.figure1_out, sep="\t", index=False)

    storyboard = build_storyboard()
    args.storyboard_out.parent.mkdir(parents=True, exist_ok=True)
    storyboard.to_csv(args.storyboard_out, sep="\t", index=False)

    print(f"Wrote {args.figure1_out}")
    print(f"Wrote {args.storyboard_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
