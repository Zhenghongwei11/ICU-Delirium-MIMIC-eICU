#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASET_ID_MIMIC = "MIMIC-IV-2.2"
DATASET_ID_EICU = "eICU-CRD-2.0"

CONSTRUCT_MAP = {
    "skin_braden_pressure": {
        "mimic": ["skin_braden_pressure"],
        "eicu": ["skin_braden_pressure"],
        "tier": "partial_construct_validation",
        "reason": "Skin/Braden/pressure-related nursing documentation is present in both datasets, but exact item-level score equivalence is not yet locked.",
    },
    "rass_sedation": {
        "mimic": ["rass_sedation"],
        "eicu": ["rass_sedation"],
        "tier": "partial_construct_validation",
        "reason": "Sedation/arousal documentation is present in both datasets; coding should remain dataset-specific until final item harmonization.",
    },
    "delirium_orientation": {
        "mimic": ["cam_icu_delirium", "orientation"],
        "eicu": ["delirium_orientation"],
        "tier": "partial_construct_validation",
        "reason": "Delirium/orientation records exist in both datasets, but eICU endpoint timing is not yet harmonized for full incident-delirium validation.",
    },
    "mobility_turning": {
        "mimic": ["mobility_turning"],
        "eicu": ["mobility_turning"],
        "tier": "partial_construct_validation",
        "reason": "Mobility/turning care-process records are present in both datasets, with likely differences in charting semantics.",
    },
    "pain": {
        "mimic": ["pain"],
        "eicu": ["pain"],
        "tier": "partial_construct_validation",
        "reason": "Pain assessment/documentation density is present in both datasets.",
    },
    "vitals_monitoring": {
        "mimic": ["vitals"],
        "eicu": ["vitals"],
        "tier": "partial_construct_validation",
        "reason": "Vital-sign monitoring burden is present in both datasets and is best treated as a care-intensity comparator.",
    },
    "oral_suction_airway": {
        "mimic": ["oral_suction_airway"],
        "eicu": ["oral_suction_airway"],
        "tier": "partial_construct_validation",
        "reason": "Airway/oral/suction care-process records are present in both datasets, but item-level semantics need final harmonization.",
    },
    "nocturnal_clock_22_06": {
        "mimic": [],
        "eicu": [],
        "tier": "infeasible_for_full_external_validation",
        "reason": "MIMIC supports chart-clock 22:00-06:00 splits; current eICU CSV offsets are ICU-admission-relative and cannot reproduce local clock-night timing.",
    },
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def sum_mimic(mimic: pd.DataFrame, concepts: list[str]) -> dict[str, object]:
    subset = mimic[mimic["concept_group"].isin(concepts)].copy()
    if subset.empty:
        return {
            "mimic_records_0_24h": 0,
            "mimic_records_0_48h": 0,
            "mimic_records_0_72h": 0,
            "mimic_stays_with_any_record_72h": 0,
            "mimic_source_concepts": "",
        }
    return {
        "mimic_records_0_24h": int(subset["n_records_0_24h"].sum()),
        "mimic_records_0_48h": int(subset["n_records_0_48h"].sum()),
        "mimic_records_0_72h": int(subset["n_records_0_72h"].sum()),
        "mimic_stays_with_any_record_72h": int(subset["n_stays_with_any_record_72h"].max()),
        "mimic_source_concepts": ";".join(sorted(subset["concept_group"].unique())),
    }


def sum_eicu(eicu: pd.DataFrame, concepts: list[str]) -> dict[str, object]:
    subset = eicu[eicu["concept_group"].isin(concepts)].copy()
    if subset.empty:
        return {
            "eicu_records_0_24h": 0,
            "eicu_records_0_48h": 0,
            "eicu_records_0_72h": 0,
            "eicu_stay_count_contributions_72h": 0,
            "eicu_source_tables": "",
            "eicu_source_concepts": "",
        }
    grouped = subset.groupby("source_table", dropna=False).agg(
        n_records_0_24h=("n_records_0_24h", "sum"),
        n_records_0_48h=("n_records_0_48h", "sum"),
        n_records_0_72h=("n_records_0_72h", "sum"),
        n_stays_with_any_record_72h=("n_stays_with_any_record_72h", "sum"),
    )
    return {
        "eicu_records_0_24h": int(grouped["n_records_0_24h"].sum()),
        "eicu_records_0_48h": int(grouped["n_records_0_48h"].sum()),
        "eicu_records_0_72h": int(grouped["n_records_0_72h"].sum()),
        "eicu_stay_count_contributions_72h": int(grouped["n_stays_with_any_record_72h"].sum()),
        "eicu_source_tables": ";".join(sorted(subset["source_table"].dropna().astype(str).unique())),
        "eicu_source_concepts": ";".join(sorted(subset["concept_group"].unique())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MIMIC/eICU partial construct validation summary.")
    parser.add_argument("--mimic", type=Path, default=Path("results/missingness/mimic_variable_density.tsv"))
    parser.add_argument("--eicu", type=Path, default=Path("results/missingness/eicu_variable_density.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/replication/combined_summary.tsv"))
    args = parser.parse_args()

    mimic = read_tsv(args.mimic)
    eicu = read_tsv(args.eicu)

    rows = []
    for construct, spec in CONSTRUCT_MAP.items():
        row = {
            "construct": construct,
            "derivation_dataset": DATASET_ID_MIMIC,
            "replication_dataset": DATASET_ID_EICU,
            "validation_tier": spec["tier"],
            "interpretation": spec["reason"],
        }
        row.update(sum_mimic(mimic, spec["mimic"]))
        row.update(sum_eicu(eicu, spec["eicu"]))
        row["replication_status"] = (
            "supported_partial_construct"
            if row["eicu_records_0_48h"] > 0 and row["validation_tier"] == "partial_construct_validation"
            else row["validation_tier"]
        )
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
