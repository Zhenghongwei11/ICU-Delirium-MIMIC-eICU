#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASET_ID = "MIMIC-IV-2.2"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def summarize_group(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        n = len(sub)
        row.update(
            {
                "dataset_id": DATASET_ID,
                "n": n,
                "incident_delirium_events": int(sub["incident_post_landmark_delirium"].sum()),
                "incident_delirium_rate": float(sub["incident_post_landmark_delirium"].mean()),
                "post_landmark_positive_assessments_or_status": int(sub["post_landmark_delirium_positive"].sum()),
                "median_post_landmark_negative_assessments": float(sub["post_landmark_negative_assessments"].median()),
                "p25_post_landmark_negative_assessments": float(sub["post_landmark_negative_assessments"].quantile(0.25)),
                "p75_post_landmark_negative_assessments": float(sub["post_landmark_negative_assessments"].quantile(0.75)),
                "any_post_landmark_assessment": int(sub["any_post_landmark_assessment"].sum()),
                "any_post_landmark_assessment_rate": float(sub["any_post_landmark_assessment"].mean()),
                "no_post_landmark_assessment": int((~sub["any_post_landmark_assessment"]).sum()),
                "no_post_landmark_assessment_rate": float((~sub["any_post_landmark_assessment"]).mean()),
                "mean_post_landmark_negative_assessments": float(sub["post_landmark_negative_assessments"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build delirium outcome ascertainment and missing-assessment supplement.")
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/missingness/outcome_ascertainment_by_phenotype.tsv"))
    args = parser.parse_args()

    matrix = read_tsv(args.matrix)
    classes = read_tsv(args.classes)
    data = matrix.merge(classes[["stay_id", "class_label"]], on="stay_id", how="inner")
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data["phenotype_display"] = data["class_label"].map(
        lambda label: "High nursing documentation and care-process intensity"
        if "high care intensity" in str(label)
        else "Lower nursing documentation and care-process intensity"
    )
    for col in ["incident_post_landmark_delirium", "post_landmark_delirium_positive"]:
        data[col] = data[col].fillna(False).astype(bool)
    data["post_landmark_negative_assessments"] = pd.to_numeric(data["post_landmark_negative_assessments"], errors="coerce").fillna(0)
    data["any_post_landmark_assessment"] = data["post_landmark_delirium_positive"] | (data["post_landmark_negative_assessments"] > 0)
    data["outcome_event_status"] = data["incident_post_landmark_delirium"].map({True: "incident_delirium", False: "no_documented_incident_delirium"})
    data["assessment_status"] = data["any_post_landmark_assessment"].map({True: "any_post_landmark_assessment", False: "no_post_landmark_assessment"})

    parts = []
    parts.append(summarize_group(data, ["phenotype_display"]).assign(summary_level="phenotype"))
    parts.append(summarize_group(data, ["phenotype_display", "outcome_event_status"]).assign(summary_level="phenotype_by_outcome"))
    parts.append(summarize_group(data, ["phenotype_display", "assessment_status"]).assign(summary_level="phenotype_by_assessment_status"))
    out = pd.concat(parts, ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
