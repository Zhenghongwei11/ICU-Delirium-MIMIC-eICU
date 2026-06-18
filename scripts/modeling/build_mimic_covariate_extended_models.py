#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATASET_ID = "MIMIC-IV-2.2"
LANDMARK_HOURS = 48
OUTCOME = "incident_post_landmark_delirium"

MED_CLASSES = {
    "sedative_exposure": {
        221668,  # Midazolam
        222168,  # Propofol
        225150,  # Dexmedetomidine
        226224,  # Propofol ingredient
        227210,  # Propofol intubation
        229420,  # Dexmedetomidine
    },
    "opioid_exposure": {
        221744,  # Fentanyl
        221833,  # Hydromorphone
        225154,  # Morphine
        225942,  # Fentanyl concentrate
        225972,  # Fentanyl push
    },
    "vasopressor_exposure": {
        221289,  # Epinephrine
        221653,  # Dobutamine
        221662,  # Dopamine
        221749,  # Phenylephrine
        221906,  # Norepinephrine
        221986,  # Milrinone
        222315,  # Vasopressin
        229617,  # Epinephrine
        229630,  # Phenylephrine
        229631,  # Phenylephrine old
        229632,  # Phenylephrine
        229789,  # Phenylephrine intubation
    },
}


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


def diagnosis_flags(diagnoses_path: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    diagnoses = pd.read_csv(diagnoses_path, usecols=["hadm_id", "icd_code", "icd_version"])
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(cohort["hadm_id"])].copy()
    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    def dementia(code: str) -> bool:
        return code.startswith(("F01", "F02", "F03", "G30", "290", "3310"))

    def sepsis(code: str) -> bool:
        return code.startswith(("A40", "A41", "R652", "038", "99591", "99592", "78552"))

    diagnoses["dementia_flag"] = diagnoses["icd_code"].map(dementia)
    diagnoses["sepsis_flag"] = diagnoses["icd_code"].map(sepsis)
    summary = (
        diagnoses.groupby("hadm_id", dropna=False)
        .agg(
            diagnosis_count=("icd_code", "nunique"),
            dementia_flag=("dementia_flag", "max"),
            sepsis_flag=("sepsis_flag", "max"),
        )
        .reset_index()
    )
    return summary


def medication_flags(inputevents_path: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    stay_windows = cohort[["stay_id", "intime"]].copy()
    stay_windows["intime"] = pd.to_datetime(stay_windows["intime"])
    stay_windows["landmark_end"] = stay_windows["intime"] + pd.to_timedelta(LANDMARK_HOURS, unit="h")
    stay_ids = set(stay_windows["stay_id"])
    med_itemids = set().union(*MED_CLASSES.values())
    pieces = []
    usecols = ["stay_id", "starttime", "itemid"]
    for chunk in pd.read_csv(inputevents_path, usecols=usecols, chunksize=500_000):
        chunk = chunk[chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(med_itemids)].copy()
        if chunk.empty:
            continue
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk = chunk.merge(stay_windows, on="stay_id", how="inner")
        chunk = chunk[(chunk["starttime"] >= chunk["intime"]) & (chunk["starttime"] <= chunk["landmark_end"])]
        if chunk.empty:
            continue
        for class_name, itemids in MED_CLASSES.items():
            chunk[class_name] = chunk["itemid"].isin(itemids)
        pieces.append(chunk[["stay_id", *MED_CLASSES.keys()]])

    if not pieces:
        out = cohort[["stay_id"]].copy()
        for class_name in MED_CLASSES:
            out[class_name] = False
        return out
    meds = pd.concat(pieces, ignore_index=True)
    summary = meds.groupby("stay_id", dropna=False).agg({class_name: "max" for class_name in MED_CLASSES}).reset_index()
    return summary


def build_covariates(args: argparse.Namespace) -> pd.DataFrame:
    cohort = read_tsv(args.cohort)
    cohort = cohort[cohort["eligible_landmark_48h"]].copy()
    dx = diagnosis_flags(args.diagnoses, cohort)
    meds = medication_flags(args.inputevents, cohort)

    cov = cohort[["subject_id", "hadm_id", "stay_id", "first_careunit", "admission_type"]].copy()
    cov = cov.merge(dx, on="hadm_id", how="left").merge(meds, on="stay_id", how="left")
    cov["diagnosis_count"] = cov["diagnosis_count"].fillna(0)
    for col in ["dementia_flag", "sepsis_flag", *MED_CLASSES.keys()]:
        cov[col] = cov[col].fillna(False).astype(bool)
    cov["emergency_admission"] = cov["admission_type"].astype(str).str.contains("EMER|URGENT|EW", case=False, na=False)
    cov["micu_unit"] = cov["first_careunit"].astype(str).str.contains("Medical Intensive Care", case=False, na=False)
    return cov


def fit_extended_model(matrix_path: Path, classes_path: Path, covariates: pd.DataFrame) -> pd.DataFrame:
    matrix = read_tsv(matrix_path)
    classes = read_tsv(classes_path)
    data = matrix.merge(classes[["stay_id", "class_label"]], on="stay_id", how="inner").merge(covariates, on=["subject_id", "hadm_id", "stay_id"], how="left")
    data = data[data["pre_landmark_delirium_positive"] == False].copy()
    data["high_intensity_class"] = data["class_label"].str.contains("high care intensity", na=False).astype(int)
    data["age_z"] = zscore(data["anchor_age"])
    data["male"] = (data["gender"].astype(str).str.upper() == "M").astype(int)
    data["total_event_z"] = zscore(data["event_all_total_count_48h"])
    data["night_fraction_z"] = zscore(data["event_all_night_fraction_48h"])
    data["diagnosis_count_z"] = zscore(data["diagnosis_count"])
    for col in ["dementia_flag", "sepsis_flag", "sedative_exposure", "opioid_exposure", "vasopressor_exposure", "emergency_admission", "micu_unit"]:
        data[col] = data[col].fillna(False).astype(int)

    predictors = [
        "high_intensity_class",
        "age_z",
        "male",
        "total_event_z",
        "night_fraction_z",
        "diagnosis_count_z",
        "dementia_flag",
        "sepsis_flag",
        "sedative_exposure",
        "opioid_exposure",
        "vasopressor_exposure",
        "emergency_admission",
        "micu_unit",
    ]
    model_data = data[[OUTCOME] + predictors].dropna().copy()
    y = model_data[OUTCOME].astype(int)
    x = model_data[predictors].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    x = sm.add_constant(x, has_constant="add")
    result = sm.Logit(y, x).fit(disp=False, maxiter=300)
    rows = []
    for term in result.params.index:
        coef = result.params[term]
        se = result.bse[term]
        rows.append(
            {
                "claim_id": "dynamic_nursing_screening",
                "dataset_id": DATASET_ID,
                "outcome": OUTCOME,
                "landmark_time": "48h",
                "model": "clinical_covariate_extended",
                "term": term,
                "effect_type": "odds_ratio",
                "effect": float(np.exp(coef)),
                "ci_lower": float(np.exp(coef - 1.96 * se)),
                "ci_upper": float(np.exp(coef + 1.96 * se)),
                "pvalue": float(result.pvalues[term]),
                "n": int(len(model_data)),
                "events": int(y.sum()),
                "covariate_set": ",".join(predictors),
                "aic": float(result.aic),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MIMIC 48h diagnosis/medication covariates and run extended phenotype outcome model.")
    parser.add_argument("--cohort", type=Path, default=Path("results/cohort/mimic_analysis_cohort.tsv"))
    parser.add_argument("--diagnoses", type=Path, default=Path("data/raw/mimiciv/2.2/hosp/diagnoses_icd.csv.gz"))
    parser.add_argument("--inputevents", type=Path, default=Path("data/raw/mimiciv/2.2/icu/inputevents.csv.gz"))
    parser.add_argument("--matrix", type=Path, default=Path("results/phenotypes/mimic_feature_matrix_48h.tsv"))
    parser.add_argument("--classes", type=Path, default=Path("results/phenotypes/trajectory_classes.tsv"))
    parser.add_argument("--covariates-out", type=Path, default=Path("results/covariates/mimic_clinical_covariates_48h.tsv"))
    parser.add_argument("--effects-out", type=Path, default=Path("results/effect_sizes/claim_effects_extended.tsv"))
    args = parser.parse_args()

    covariates = build_covariates(args)
    args.covariates_out.parent.mkdir(parents=True, exist_ok=True)
    covariates.to_csv(args.covariates_out, sep="\t", index=False)

    effects = fit_extended_model(args.matrix, args.classes, covariates)
    args.effects_out.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.effects_out, sep="\t", index=False)

    print(f"Wrote {args.covariates_out}")
    print(f"Wrote {args.effects_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
