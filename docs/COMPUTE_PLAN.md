# Compute Plan

The figure-only reproduction mode should run on a laptop because it uses aggregate TSV outputs included in this repository.

The full reproduction mode requires credentialed local copies of selected MIMIC-IV v2.2 and eICU-CRD v2.0 CSV files. The slowest step is scanning MIMIC-IV `icu/chartevents.csv.gz`. Runtime depends on storage and CPU; several hours is plausible on a laptop or workstation. Disk use depends on whether patient-level intermediate files are retained locally.

Recommended local configuration for full reproduction:

- Python 3.13-compatible environment
- At least 16 GB RAM
- Sufficient disk space for selected MIMIC-IV and eICU-CRD compressed CSV files and generated intermediate tables

Raw data and generated patient-level intermediate files should not be committed to GitHub.
