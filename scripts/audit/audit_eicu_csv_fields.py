#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def concept(text: str) -> str | None:
    text = str(text).lower()
    patterns = [
        ("skin_braden_pressure", r"braden|pressure ulcer|pressure injury|skin|wound"),
        ("rass_sedation", r"rass|richmond|sedation|agitation"),
        ("delirium_orientation", r"cam-?icu|confusion assessment|delirium|orientation|oriented|person|place|time"),
        ("pain", r"pain|cpot|behavioral pain|numeric pain"),
        ("mobility_turning", r"mobility|activity|ambulat|turn|reposition|position|bedrest|out of bed"),
        ("oral_suction_airway", r"oral care|mouth care|mouth|teeth|suction|secretions"),
        ("nutrition", r"nutrition|enteral|tube feed|feeding|diet"),
        ("moisture_hygiene", r"moisture|incontinent|incontinence|stool|urine|hygiene|bath"),
        ("restraints", r"restraint|mitt|sitter"),
        ("glucose_check", r"glucose|fingerstick|finger stick|blood sugar|fsbg"),
        ("vitals", r"heart rate|respiratory rate|blood pressure|temperature|spo2|oxygen saturation|vital"),
    ]
    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None


def read_required(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)


def age_numeric(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "> 89":
        return 90
    if text.isdigit():
        return float(text)
    return None


def audit_table(
    path: Path,
    source_table: str,
    offset_col: str,
    text_cols: list[str],
    chunk_size: int,
    older_ids: set[int],
    max_field_rows: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    density_rows = []
    sampled_rows = 0
    usecols = ["patientunitstayid", offset_col] + text_cols
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        chunk = chunk[chunk["patientunitstayid"].isin(older_ids)].copy()
        if chunk.empty:
            continue
        for col in text_cols:
            if col not in chunk.columns:
                chunk[col] = ""
        text_frame = chunk[text_cols].fillna("").astype(str)
        chunk["_text"] = text_frame[text_cols[0]]
        for col in text_cols[1:]:
            chunk["_text"] = chunk["_text"].str.cat(text_frame[col], sep=" ")
        chunk["concept_group"] = chunk["_text"].map(concept)
        chunk = chunk[chunk["concept_group"].notna()].copy()
        if chunk.empty:
            continue
        chunk[offset_col] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk = chunk[(chunk[offset_col] >= 0) & (chunk[offset_col] <= 72 * 60)]
        if chunk.empty:
            continue
        grouped = (
            chunk.groupby(["concept_group"], dropna=False)
            .agg(n_records_72h=("patientunitstayid", "size"), n_stays_with_any_record_72h=("patientunitstayid", "nunique"))
            .reset_index()
        )
        grouped["source_table"] = source_table
        grouped["n_records_0_24h"] = chunk[chunk[offset_col] <= 24 * 60].groupby("concept_group").size().reindex(grouped["concept_group"]).fillna(0).astype(int).values
        grouped["n_records_0_48h"] = chunk[chunk[offset_col] <= 48 * 60].groupby("concept_group").size().reindex(grouped["concept_group"]).fillna(0).astype(int).values
        grouped["n_records_0_72h"] = grouped["n_records_72h"]
        density_rows.append(grouped)

        if sampled_rows < max_field_rows:
            keep_cols = ["patientunitstayid", offset_col, "concept_group"] + text_cols
            take = min(max_field_rows - sampled_rows, len(chunk))
            rows.append(chunk[keep_cols].head(take).assign(source_table=source_table))
            sampled_rows += take
    fields = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    density = pd.concat(density_rows, ignore_index=True) if density_rows else pd.DataFrame()
    return fields, density


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit eICU-CRD CSV files for nursing-field feasibility.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/eicu-crd/2.0"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/missingness"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    patient = read_required(
        args.root / "patient.csv.gz",
        usecols=[
            "patientunitstayid",
            "patienthealthsystemstayid",
            "hospitalid",
            "wardid",
            "unittype",
            "unitvisitnumber",
            "age",
            "gender",
            "unitdischargeoffset",
        ],
    )
    patient["age_numeric"] = patient["age"].map(age_numeric)
    patient["icu_los_hours"] = pd.to_numeric(patient["unitdischargeoffset"], errors="coerce") / 60.0
    cohort = patient[(patient["age_numeric"] >= 65) & (patient["unitvisitnumber"] == 1)].copy()
    cohort.to_csv(args.out_dir / "eicu_older_first_unit_cohort.tsv", sep="\t", index=False)
    older_ids = set(cohort["patientunitstayid"].astype(int))

    specs = [
        ("nurseAssessment.csv.gz", "nurseassessment", "nurseassessoffset", ["celllabel", "cellattribute", "cellattributepath", "cellattributevalue"]),
        ("nurseCare.csv.gz", "nursecare", "nursecareoffset", ["celllabel", "cellattribute", "cellattributepath", "cellattributevalue"]),
        ("nurseCharting.csv.gz", "nursecharting", "nursingchartoffset", ["nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"]),
    ]

    all_fields = []
    all_density = []
    for filename, source_table, offset_col, text_cols in specs:
        path = args.root / filename
        print(f"Auditing {source_table} from {path}", flush=True)
        fields, density = audit_table(path, source_table, offset_col, text_cols, args.chunk_size, older_ids)
        if not fields.empty:
            fields.to_csv(args.docs_dir / f"EICU_CANDIDATE_NURSING_FIELDS_{source_table}.tsv", sep="\t", index=False)
        if not density.empty:
            density.to_csv(args.out_dir / f"eicu_variable_density_{source_table}.tsv", sep="\t", index=False)
        print(
            f"Finished {source_table}: {len(fields)} sampled rows, {len(density)} density rows",
            flush=True,
        )
        if not fields.empty:
            all_fields.append(fields)
        if not density.empty:
            all_density.append(density)

    fields_out = pd.concat(all_fields, ignore_index=True) if all_fields else pd.DataFrame()
    density_out = pd.concat(all_density, ignore_index=True) if all_density else pd.DataFrame()
    fields_out.to_csv(args.docs_dir / "EICU_CANDIDATE_NURSING_FIELDS.tsv", sep="\t", index=False)
    density_out.to_csv(args.out_dir / "eicu_variable_density.tsv", sep="\t", index=False)
    print("eICU CSV audit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
