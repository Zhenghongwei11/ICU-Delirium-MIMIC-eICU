# Statistical Decision Rules

The primary exposure window is the first 48 hours after ICU admission. Patients with documented delirium positivity before the 48-hour landmark are excluded from the primary incident-delirium analysis.

The primary phenotype is a two-class Gaussian mixture model of standardized early nursing documentation and care-process features. Candidate models with two to five classes are compared using fit, class size, assignment confidence, and clinical interpretability; very small classes are not used as the primary phenotype.

The primary association model is logistic regression for post-landmark documented incident delirium. Reported estimates are odds ratios with 95% confidence intervals.

Internal screening performance uses five-fold internal cross-validation, calibration, Brier score, and decision-curve analysis. These metrics are not presented as external validation or clinical deployment readiness.

Sensitivity analyses include landmark windows, death-related analyses, outcome ascertainment analyses, subgroup checks, severity-proxy adjustment, demographic-baseline diagnostics, GMM stability diagnostics, and intervention-response proxy models.

Values in manuscript tables and figures are rounded for presentation; source TSV files retain higher precision where applicable.
