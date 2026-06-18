#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


CONCEPT_PATTERNS = [
    ("skin_braden_pressure", r"braden|pressure ulcer|pressure injury|skin risk|skin integrity|skin assessment|wound"),
    ("rass_sedation", r"rass|richmond|sedation|agitation"),
    ("cam_icu_delirium", r"cam-?icu|confusion assessment|delirium|acute change|inattention|disorganized|altered level"),
    ("pain", r"pain|cpot|behavioral pain|numeric pain|pain score"),
    ("mobility_turning", r"mobility|activity|ambulat|turn|reposition|position|bedrest|out of bed"),
    ("oral_suction_airway", r"oral care|mouth care|mouth|teeth|suction|airway suction|secretions"),
    ("nutrition", r"nutrition|enteral|tube feed|feeding|appetite|diet"),
    ("moisture_hygiene", r"moisture|incontinent|incontinence|stool|urine|diaper|hygiene|bath"),
    ("restraints", r"restraint|mitt|sitter"),
    ("orientation", r"orientation|oriented|person|place|time"),
    ("glucose_check", r"glucose|fingerstick|finger stick|blood sugar|fsbg"),
    ("vitals", r"heart rate|respiratory rate|blood pressure|temperature|spo2|oxygen saturation|vital"),
]


WINDOWS = [24, 48, 72]
DATASET_ID = "MIMIC-IV-2.2"


def classify(label: object) -> str | None:
    text = str(label).lower()
    for concept, pattern in CONCEPT_PATTERNS:
        if re.search(pattern, text):
            return concept
    return None


def read_required(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)


def append_window_groups(frames: list[pd.DataFrame], data: pd.DataFrame, group_cols: list[str], source_table: str) -> None:
    if data.empty:
        return
    for window in WINDOWS:
        subset = data[data["hours_from_icu_admit"].between(0, window, inclusive="both")]
        if subset.empty:
            continue
        grouped = (
            subset.groupby(group_cols, dropna=False)
            .agg(
                n_records=("stay_id", "size"),
                n_numeric=("valuenum", "count"),
                mean_valuenum=("valuenum", "mean"),
                min_valuenum=("valuenum", "min"),
                max_valuenum=("valuenum", "max"),
                first_hour=("hours_from_icu_admit", "min"),
                last_hour=("hours_from_icu_admit", "max"),
            )
            .reset_index()
        )
        grouped["dataset_id"] = DATASET_ID
        grouped["source_table"] = source_table
        grouped["exposure_window_flag"] = f"0-{window}h"
        grouped["window_hours"] = window
        grouped["icu_admit_time_relative"] = 0
        grouped["event_time_relative"] = window
        grouped["value_or_count"] = grouped["n_records"]
        frames.append(grouped)


def append_count_window_groups(frames: list[pd.DataFrame], data: pd.DataFrame, group_cols: list[str], source_table: str) -> None:
    if data.empty:
        return
    for window in WINDOWS:
        subset = data[data["hours_from_icu_admit"].between(0, window, inclusive="both")]
        if subset.empty:
            continue
        grouped = subset.groupby(group_cols, dropna=False).size().reset_index(name="value_or_count")
        grouped["dataset_id"] = DATASET_ID
        grouped["source_table"] = source_table
        grouped["exposure_window_flag"] = f"0-{window}h"
        grouped["window_hours"] = window
        grouped["icu_admit_time_relative"] = 0
        grouped["event_time_relative"] = window
        frames.append(grouped)


def aggregate_frames(frames: list[pd.DataFrame], group_cols: list[str], numeric_cols: Iterable[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    agg_spec = {}
    for col in numeric_cols:
        if col in {"first_hour", "min_valuenum"}:
            agg_spec[col] = (col, "min")
        elif col in {"last_hour", "max_valuenum"}:
            agg_spec[col] = (col, "max")
        elif col == "mean_valuenum":
            agg_spec[col] = (col, "mean")
        else:
            agg_spec[col] = (col, "sum")
    return data.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def build_cohort(root: Path, out_dir: Path) -> pd.DataFrame:
    patients = read_required(root / "hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "dod"], parse_dates=["dod"])
    admissions = read_required(
        root / "hosp/admissions.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "race",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    icustays = read_required(
        root / "icu/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )

    flow = []
    all_icu = len(icustays)
    flow.append({"dataset_id": DATASET_ID, "step_name": "all_icu_stays", "exclusion_reason": "none", "n_removed": 0, "n_remaining": all_icu})

    cohort = icustays.merge(patients, on="subject_id", how="inner").merge(admissions, on=["subject_id", "hadm_id"], how="left")
    older = cohort[cohort["anchor_age"] >= 65].copy()
    flow.append(
        {
            "dataset_id": DATASET_ID,
            "step_name": "age_ge_65",
            "exclusion_reason": "anchor_age_lt_65_or_missing",
            "n_removed": all_icu - len(older),
            "n_remaining": len(older),
        }
    )

    first = older.sort_values(["subject_id", "intime", "stay_id"]).drop_duplicates("subject_id", keep="first").copy()
    flow.append(
        {
            "dataset_id": DATASET_ID,
            "step_name": "first_icu_stay_per_subject",
            "exclusion_reason": "subsequent_icu_stay",
            "n_removed": len(older) - len(first),
            "n_remaining": len(first),
        }
    )

    first["icu_los_hours"] = (first["outtime"] - first["intime"]).dt.total_seconds() / 3600.0
    first["hospital_los_days"] = (first["dischtime"] - first["admittime"]).dt.total_seconds() / 86400.0
    first["death_time_from_icu_admit_hours"] = (first["deathtime"] - first["intime"]).dt.total_seconds() / 3600.0
    for window in WINDOWS:
        first[f"eligible_landmark_{window}h"] = first["icu_los_hours"] >= window
        flow.append(
            {
                "dataset_id": DATASET_ID,
                "step_name": f"landmark_{window}h_eligible",
                "exclusion_reason": f"icu_los_lt_{window}h",
                "n_removed": int((~first[f"eligible_landmark_{window}h"]).sum()),
                "n_remaining": int(first[f"eligible_landmark_{window}h"].sum()),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    first.to_csv(out_dir / "mimic_analysis_cohort.tsv", sep="\t", index=False)
    pd.DataFrame(flow).to_csv(out_dir / "cohort_flow.tsv", sep="\t", index=False)

    landmark_rows = []
    for window in WINDOWS:
        landmark_rows.append(
            {
                "dataset_id": DATASET_ID,
                "landmark_hours": window,
                "n_total_older_first_icu": len(first),
                "n_landmark_eligible": int(first[f"eligible_landmark_{window}h"].sum()),
                "n_icu_discharge_before_landmark": int((first["icu_los_hours"] < window).sum()),
                "n_death_before_landmark": int(((first["death_time_from_icu_admit_hours"].notna()) & (first["death_time_from_icu_admit_hours"] <= window)).sum()),
            }
        )
    pd.DataFrame(landmark_rows).to_csv(out_dir / "landmark_eligibility.tsv", sep="\t", index=False)
    return first


def build_chartevents_features(root: Path, cohort: pd.DataFrame, out_dirs: dict[str, Path], chunk_size: int, max_chunks: int | None) -> None:
    d_items = read_required(root / "icu/d_items.csv.gz")
    d_items["concept_group"] = d_items["label"].map(classify)
    candidates = d_items[d_items["concept_group"].notna()].copy()
    candidates.to_csv(out_dirs["docs"] / "MIMIC_IV_CANDIDATE_ITEMS.tsv", sep="\t", index=False)

    item_to_concept = candidates.set_index("itemid")["concept_group"].to_dict()
    item_to_label = candidates.set_index("itemid")["label"].to_dict()
    candidate_itemids = set(item_to_concept)
    stay_to_intime = cohort.set_index("stay_id")["intime"].to_dict()
    stay_to_subject = cohort.set_index("stay_id")["subject_id"].to_dict()
    eligible_stays = set(stay_to_intime)

    nursing_frames: list[pd.DataFrame] = []
    nocturnal_frames: list[pd.DataFrame] = []
    delirium_frames: list[pd.DataFrame] = []

    usecols = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valuenum"]
    for index, chunk in enumerate(pd.read_csv(root / "icu/chartevents.csv.gz", usecols=usecols, chunksize=chunk_size), start=1):
        if max_chunks is not None and index > max_chunks:
            break
        if index % 5 == 1:
            print(f"Scanning chartevents chunk {index}", flush=True)
        chunk = chunk[chunk["stay_id"].isin(eligible_stays) & chunk["itemid"].isin(candidate_itemids)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["intime"] = chunk["stay_id"].map(stay_to_intime)
        chunk["hours_from_icu_admit"] = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[chunk["hours_from_icu_admit"].between(0, 72, inclusive="both")].copy()
        if chunk.empty:
            continue
        chunk["variable_or_event_category"] = chunk["itemid"].map(item_to_concept)
        chunk["item_label"] = chunk["itemid"].map(item_to_label)
        chunk["patient_id_or_stay_id"] = chunk["stay_id"]
        chunk["subject_id"] = chunk["stay_id"].map(stay_to_subject)
        chunk["is_night_22_06"] = (chunk["charttime"].dt.hour >= 22) | (chunk["charttime"].dt.hour < 6)
        chunk["day_night_period"] = chunk["is_night_22_06"].map({True: "night_22_06", False: "day_06_22"})

        append_window_groups(
            nursing_frames,
            chunk,
            ["patient_id_or_stay_id", "subject_id", "variable_or_event_category"],
            "chartevents",
        )
        append_count_window_groups(
            nocturnal_frames,
            chunk,
            [
                "patient_id_or_stay_id",
                "subject_id",
                "variable_or_event_category",
                "day_night_period",
            ],
            "chartevents",
        )

        delirium = chunk[
            (chunk["variable_or_event_category"] == "cam_icu_delirium")
            & (chunk["item_label"].str.contains("delirium assessment", case=False, na=False))
        ].copy()
        if not delirium.empty:
            delirium["delirium_positive"] = delirium["value"].astype(str).str.contains("positive", case=False, na=False)
            delirium["delirium_negative"] = delirium["value"].astype(str).str.contains("negative", case=False, na=False)
            delirium_frames.append(delirium[["stay_id", "hours_from_icu_admit", "delirium_positive", "delirium_negative"]])

    nursing_group_cols = [
        "dataset_id",
        "patient_id_or_stay_id",
        "subject_id",
        "variable_or_event_category",
        "source_table",
        "exposure_window_flag",
        "window_hours",
    ]
    nursing = aggregate_frames(
        nursing_frames,
        nursing_group_cols,
        ["n_records", "n_numeric", "mean_valuenum", "min_valuenum", "max_valuenum", "first_hour", "last_hour", "value_or_count"],
    )
    nursing["icu_admit_time_relative"] = 0
    nursing["event_time_relative"] = nursing["window_hours"]
    nursing.to_csv(out_dirs["exposure"] / "nursing_stream_long.tsv", sep="\t", index=False)

    nocturnal_group_cols = [
        "dataset_id",
        "patient_id_or_stay_id",
        "subject_id",
        "variable_or_event_category",
        "day_night_period",
        "source_table",
        "exposure_window_flag",
        "window_hours",
    ]
    nocturnal = aggregate_frames(nocturnal_frames, nocturnal_group_cols, ["value_or_count"])
    nocturnal["icu_admit_time_relative"] = 0
    nocturnal["event_time_relative"] = nocturnal["window_hours"]

    if delirium_frames:
        delirium = pd.concat(delirium_frames, ignore_index=True)
        outcome_rows = []
        for window in WINDOWS:
            grouped = (
                delirium.groupby("stay_id")
                .agg(
                    pre_landmark_delirium_positive=("delirium_positive", lambda x: bool(x[delirium.loc[x.index, "hours_from_icu_admit"] <= window].any())),
                    post_landmark_delirium_positive=("delirium_positive", lambda x: bool(x[delirium.loc[x.index, "hours_from_icu_admit"] > window].any())),
                    pre_landmark_negative_assessments=("delirium_negative", lambda x: int(x[delirium.loc[x.index, "hours_from_icu_admit"] <= window].sum())),
                    post_landmark_negative_assessments=("delirium_negative", lambda x: int(x[delirium.loc[x.index, "hours_from_icu_admit"] > window].sum())),
                )
                .reset_index()
            )
            grouped["dataset_id"] = DATASET_ID
            grouped["landmark_hours"] = window
            grouped["incident_post_landmark_delirium"] = (~grouped["pre_landmark_delirium_positive"]) & grouped["post_landmark_delirium_positive"]
            outcome_rows.append(grouped)
        pd.concat(outcome_rows, ignore_index=True).to_csv(out_dirs["cohort"] / "mimic_delirium_landmark_status.tsv", sep="\t", index=False)

    nocturnal.to_csv(out_dirs["exposure"] / "nocturnal_events_long.tsv", sep="\t", index=False)
    write_density_outputs(nursing, nocturnal, cohort, out_dirs["missingness"])


def build_inputevents_nocturnal(root: Path, cohort: pd.DataFrame, exposure_dir: Path, chunk_size: int, max_chunks: int | None) -> None:
    path = root / "icu/inputevents.csv.gz"
    if not path.exists():
        return
    stay_to_intime = cohort.set_index("stay_id")["intime"].to_dict()
    stay_to_subject = cohort.set_index("stay_id")["subject_id"].to_dict()
    eligible_stays = set(stay_to_intime)
    frames: list[pd.DataFrame] = []
    usecols = ["subject_id", "stay_id", "starttime", "itemid", "ordercategoryname"]
    for index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunk_size), start=1):
        if max_chunks is not None and index > max_chunks:
            break
        if index % 5 == 1:
            print(f"Scanning inputevents chunk {index}", flush=True)
        chunk = chunk[chunk["stay_id"].isin(eligible_stays)].copy()
        if chunk.empty:
            continue
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["intime"] = chunk["stay_id"].map(stay_to_intime)
        chunk["hours_from_icu_admit"] = (chunk["starttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[chunk["hours_from_icu_admit"].between(0, 72, inclusive="both")].copy()
        if chunk.empty:
            continue
        chunk["patient_id_or_stay_id"] = chunk["stay_id"]
        chunk["subject_id"] = chunk["stay_id"].map(stay_to_subject)
        chunk["variable_or_event_category"] = "medication_infusion"
        chunk["is_night_22_06"] = (chunk["starttime"].dt.hour >= 22) | (chunk["starttime"].dt.hour < 6)
        chunk["day_night_period"] = chunk["is_night_22_06"].map({True: "night_22_06", False: "day_06_22"})
        append_count_window_groups(
            frames,
            chunk,
            [
                "patient_id_or_stay_id",
                "subject_id",
                "variable_or_event_category",
                "day_night_period",
            ],
            "inputevents",
        )
    if not frames:
        return
    infusion = aggregate_frames(
        frames,
        [
            "dataset_id",
            "patient_id_or_stay_id",
            "subject_id",
            "variable_or_event_category",
            "day_night_period",
            "source_table",
            "exposure_window_flag",
            "window_hours",
        ],
        ["value_or_count"],
    )
    infusion["icu_admit_time_relative"] = 0
    infusion["event_time_relative"] = infusion["window_hours"]
    out_path = exposure_dir / "nocturnal_events_long.tsv"
    if out_path.exists():
        existing = pd.read_csv(out_path, sep="\t")
        pd.concat([existing, infusion], ignore_index=True).to_csv(out_path, sep="\t", index=False)
    else:
        infusion.to_csv(out_path, sep="\t", index=False)


def write_density_outputs(nursing: pd.DataFrame, nocturnal: pd.DataFrame, cohort: pd.DataFrame, missingness_dir: Path) -> None:
    density = (
        nursing.groupby(["dataset_id", "variable_or_event_category", "source_table", "window_hours"], dropna=False)
        .agg(
            n_stays_with_records=("patient_id_or_stay_id", "nunique"),
            n_records=("n_records", "sum"),
            median_records_per_recorded_stay=("n_records", "median"),
            mean_records_per_recorded_stay=("n_records", "mean"),
        )
        .reset_index()
    )
    density.to_csv(missingness_dir / "repeated_measurement_density.tsv", sep="\t", index=False)

    cohort_size = len(cohort)
    missingness = density.copy()
    missingness["n_total_stays"] = cohort_size
    missingness["n_missing_any_record"] = cohort_size - missingness["n_stays_with_records"]
    missingness["missing_any_record_rate"] = missingness["n_missing_any_record"] / cohort_size
    missingness.to_csv(missingness_dir / "variable_missingness.tsv", sep="\t", index=False)

    if not nocturnal.empty:
        nocturnal_density = (
            nocturnal.groupby(["dataset_id", "variable_or_event_category", "source_table", "window_hours", "day_night_period"], dropna=False)
            .agg(n_stays=("patient_id_or_stay_id", "nunique"), n_events=("value_or_count", "sum"))
            .reset_index()
        )
        nocturnal_density.to_csv(missingness_dir / "daynight_event_density.tsv", sep="\t", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MIMIC-IV analysis-ready cohort and early nursing feature tables.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/mimiciv/2.2"))
    parser.add_argument("--cohort-dir", type=Path, default=Path("results/cohort"))
    parser.add_argument("--exposure-dir", type=Path, default=Path("results/exposure_windows"))
    parser.add_argument("--missingness-dir", type=Path, default=Path("results/missingness"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional smoke-test limit for chartevents/inputevents chunks.")
    args = parser.parse_args()

    for path in [args.cohort_dir, args.exposure_dir, args.missingness_dir, args.docs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    out_dirs = {
        "cohort": args.cohort_dir,
        "exposure": args.exposure_dir,
        "missingness": args.missingness_dir,
        "docs": args.docs_dir,
    }

    cohort = build_cohort(args.root, args.cohort_dir)
    print(f"Built MIMIC cohort: {len(cohort)} older first ICU stays.", flush=True)
    build_chartevents_features(args.root, cohort, out_dirs, args.chunk_size, args.max_chunks)
    build_inputevents_nocturnal(args.root, cohort, args.exposure_dir, args.chunk_size, args.max_chunks)

    summary = pd.DataFrame(
        [
            {"dataset_id": DATASET_ID, "metric": "older_first_icu_stays", "value": len(cohort)},
            {"dataset_id": DATASET_ID, "metric": "landmark_24h_eligible", "value": int(cohort["eligible_landmark_24h"].sum())},
            {"dataset_id": DATASET_ID, "metric": "landmark_48h_eligible", "value": int(cohort["eligible_landmark_48h"].sum())},
            {"dataset_id": DATASET_ID, "metric": "landmark_72h_eligible", "value": int(cohort["eligible_landmark_72h"].sum())},
        ]
    )
    summary.to_csv(Path("results/dataset_summary.tsv"), sep="\t", index=False)
    print("MIMIC analysis tables complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
