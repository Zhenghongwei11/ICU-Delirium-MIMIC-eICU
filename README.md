# ICU Delirium Nursing Documentation Phenotypes

This repository contains the reproducibility package for a retrospective MIMIC-IV landmark cohort study of early nursing documentation and care-process phenotypes and subsequent documented delirium among older ICU patients.

## Study Summary

The primary analysis used MIMIC-IV v2.2. Adults aged 65 years or older during their first ICU stay were eligible. Nursing-relevant documentation and care-process features were extracted during the first 48 hours after ICU admission. Patients with documented delirium before the 48-hour landmark were excluded from the primary incident-delirium analysis. eICU-CRD v2.0 was used only to assess the availability of comparable nursing documentation constructs, not as endpoint-replicated external validation.

## Data Sources

Raw MIMIC-IV and eICU-CRD files are not included. Both datasets are available through PhysioNet to credentialed users who complete the required training and data use agreements:

- MIMIC-IV v2.2: https://physionet.org/content/mimiciv/2.2/
- eICU-CRD v2.0: https://physionet.org/content/eicu-crd/2.0/

Expected local paths are:

```text
data/raw/mimiciv/2.2/
  hosp/patients.csv.gz
  hosp/admissions.csv.gz
  hosp/diagnoses_icd.csv.gz
  icu/icustays.csv.gz
  icu/d_items.csv.gz
  icu/chartevents.csv.gz
  icu/inputevents.csv.gz

data/raw/eicu-crd/2.0/
  patient.csv.gz
  nurseAssessment.csv.gz
  nurseCare.csv.gz
  nurseCharting.csv.gz
```

## Environment

Python 3.13.7 was used for the submitted analyses.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduction Modes

Regenerate figures from the included aggregate tables:

```bash
bash scripts/reproduce_one_click.sh figures
```

Run the full local pipeline from credentialed raw PhysioNet CSV files:

```bash
export MIMIC_ROOT=/path/to/mimiciv/2.2
export EICU_ROOT=/path/to/eicu-crd/2.0
bash scripts/reproduce_one_click.sh full
```

The full pipeline scans MIMIC-IV `chartevents.csv.gz` and can take several hours depending on disk speed and CPU. It writes patient-level intermediate files locally under `results/`; these files are intentionally not prepackaged in this public repository.

## Included Outputs

The repository includes aggregate tables needed to verify reported numerical results and regenerate figures:

- `results/tables/`: main and supplementary table source files
- `results/figures/`: figure anchor tables
- `results/sensitivity/`: sensitivity-analysis outputs
- `results/benchmarks/`: internal screening-performance outputs
- `results/clinical_utility/`: decision-curve data
- `plots/manuscript/`: exported figure files

## Repository Policy

This public package excludes raw protected data, credentials, patient-level derived intermediates, manuscript submission files, cover letters, internal planning files, and local review materials.

## Citation

Please cite the GitHub release and the Zenodo version DOI once available.

Repository: https://github.com/Zhenghongwei11/ICU-Delirium-MIMIC-eICU
