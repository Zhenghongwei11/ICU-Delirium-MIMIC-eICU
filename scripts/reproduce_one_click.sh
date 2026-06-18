#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-figures}"
MIMIC_ROOT="${MIMIC_ROOT:-data/raw/mimiciv/2.2}"
EICU_ROOT="${EICU_ROOT:-data/raw/eicu-crd/2.0}"

mkdir -p results plots/manuscript docs

if [[ "$MODE" == "figures" ]]; then
  python3 scripts/modeling/plot_ijns_figures.py --out-dir plots/manuscript
  cp plots/manuscript/figure1_ijns_study_design.pdf plots/manuscript/figure1_study_design.pdf
  cp plots/manuscript/figure1_ijns_study_design.png plots/manuscript/figure1_study_design.png
  cp plots/manuscript/figure2_ijns_phenotype_and_ascertainment.pdf plots/manuscript/figure2_phenotype_and_ascertainment.pdf
  cp plots/manuscript/figure2_ijns_phenotype_and_ascertainment.png plots/manuscript/figure2_phenotype_and_ascertainment.png
  cp plots/manuscript/figure3_ijns_effects_and_screening.pdf plots/manuscript/figure3_effects_and_screening.pdf
  cp plots/manuscript/figure3_ijns_effects_and_screening.png plots/manuscript/figure3_effects_and_screening.png
  rm -f plots/manuscript/figure1_ijns_study_design.* plots/manuscript/figure2_ijns_phenotype_and_ascertainment.* plots/manuscript/figure3_ijns_effects_and_screening.*
  echo "Figure regeneration from aggregate tables complete."
  exit 0
fi

if [[ "$MODE" != "full" ]]; then
  echo "Usage: bash scripts/reproduce_one_click.sh [figures|full]" >&2
  exit 2
fi

python3 scripts/extract/build_mimic_analysis_tables.py --root "$MIMIC_ROOT"
python3 scripts/audit/audit_eicu_csv_fields.py --root "$EICU_ROOT"

for window in 24 48 72; do
  python3 scripts/modeling/build_mimic_feature_matrix.py --window-hours "$window"
done

python3 scripts/modeling/build_mimic_dynamic_phenotypes.py
python3 scripts/modeling/run_mimic_trajectory_sensitivity.py
python3 scripts/modeling/run_mimic_phenotype_outcomes.py
python3 scripts/modeling/build_mimic_covariate_extended_models.py --diagnoses "$MIMIC_ROOT/hosp/diagnoses_icd.csv.gz" --inputevents "$MIMIC_ROOT/icu/inputevents.csv.gz"
python3 scripts/modeling/run_mimic_screening_baselines.py
python3 scripts/modeling/evaluate_mimic_prediction_models.py
python3 scripts/modeling/evaluate_prediction_bootstrap_ci.py
python3 scripts/modeling/run_mimic_landmark_sensitivity.py
python3 scripts/modeling/run_mimic_competing_and_secondary.py
python3 scripts/modeling/build_outcome_ascertainment_supplement.py
python3 scripts/modeling/run_mimic_ascertainment_bias_sensitivity.py
python3 scripts/modeling/run_mimic_subgroup_fairness_checks.py
python3 scripts/modeling/run_mimic_penalized_logistic_sensitivity.py
python3 scripts/modeling/run_mimic_severity_proxy_sensitivity.py
python3 scripts/modeling/run_mimic_gmm_stability.py --seeds 50 --final-n-init 100
python3 scripts/modeling/run_mimic_baseline_auc_diagnostics.py
python3 scripts/modeling/run_mimic_intervention_response_sensitivity.py
python3 scripts/modeling/build_replication_construct_summary.py
python3 scripts/modeling/build_publication_tables.py
python3 scripts/modeling/build_publication_anchors.py
python3 scripts/modeling/plot_ijns_figures.py --out-dir plots/manuscript
cp plots/manuscript/figure1_ijns_study_design.pdf plots/manuscript/figure1_study_design.pdf
cp plots/manuscript/figure1_ijns_study_design.png plots/manuscript/figure1_study_design.png
cp plots/manuscript/figure2_ijns_phenotype_and_ascertainment.pdf plots/manuscript/figure2_phenotype_and_ascertainment.pdf
cp plots/manuscript/figure2_ijns_phenotype_and_ascertainment.png plots/manuscript/figure2_phenotype_and_ascertainment.png
cp plots/manuscript/figure3_ijns_effects_and_screening.pdf plots/manuscript/figure3_effects_and_screening.pdf
cp plots/manuscript/figure3_ijns_effects_and_screening.png plots/manuscript/figure3_effects_and_screening.png
rm -f plots/manuscript/figure1_ijns_study_design.* plots/manuscript/figure2_ijns_phenotype_and_ascertainment.* plots/manuscript/figure3_ijns_effects_and_screening.*

echo "Full reproduction pipeline complete."
