#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def figure1(out_dir: Path) -> None:
    flow = read_tsv(Path("results/figures/figure1_cohort_landmark_flow.tsv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    cohort = flow[flow["panel"] == "cohort_flow"].sort_values("display_order")
    labels = ["All ICU stays", "Age >=65", "First ICU stay", "48h eligible"]
    y = np.arange(len(cohort))
    ax.barh(y, cohort["n"], color=PALETTE["blue"])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of stays")
    ax.set_title("Cohort construction")
    for i, n in enumerate(cohort["n"]):
        ax.text(n, i, f" {int(n):,}", va="center", fontsize=7)
    panel_label(ax, "A")

    ax = axes[1]
    lm = flow[flow["panel"] == "landmark_eligibility"].copy()
    lm["hours"] = lm["label"].str.extract(r"(\d+)").astype(int)
    ax.plot(lm["hours"], lm["n"], marker="o", color=PALETTE["orange"], linewidth=2)
    ax.set_xticks([24, 48, 72])
    ax.set_xlabel("Landmark (hours)")
    ax.set_ylabel("Eligible stays")
    ax.set_title("Landmark eligibility")
    for _, row in lm.iterrows():
        ax.text(row["hours"], row["n"], f"{int(row['n']):,}", ha="center", va="bottom", fontsize=7)
    panel_label(ax, "B")
    fig.suptitle("Older ICU cohort and landmark design", y=1.02, fontsize=10)
    save(fig, out_dir, "figure1_cohort_landmark_flow")


def figure2(out_dir: Path) -> None:
    prof = read_tsv(Path("results/figures/figure2_trajectory_anchor.tsv"))
    prof = prof.sort_values("assigned_class")
    labels = ["High documentation\nintensity" if "high" in x else "Lower documentation\nintensity" for x in prof["class_label"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.bar(labels, prof["incident_delirium_rate"] * 100, color=[PALETTE["red"], PALETTE["blue"]])
    ax.set_ylabel("Incident delirium (%)")
    ax.set_title("Post-landmark delirium")
    for i, row in enumerate(prof.itertuples()):
        ax.text(i, row.incident_delirium_rate * 100, f"n={int(row.n)}", ha="center", va="bottom", fontsize=7)
    panel_label(ax, "A")

    ax = axes[1]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, prof["rass_sedation_records_48h"], width=0.4, label="RASS/sedation", color=PALETTE["orange"])
    ax.bar(x + 0.2, prof["restraints_records_48h"], width=0.4, label="Restraints", color=PALETTE["purple"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean records, 0-48h")
    ax.set_title("Sedation and restraint density")
    ax.legend(frameon=False)
    panel_label(ax, "B")

    ax = axes[2]
    ax.bar(x - 0.2, prof["all_total_count_48h"], width=0.4, label="Total events", color=PALETTE["green"])
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, prof["all_night_fraction_48h"], marker="o", color=PALETTE["gray"], label="Night fraction")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean total events")
    ax2.set_ylabel("Night fraction")
    ax.set_title("Care-process burden")
    panel_label(ax, "C")
    fig.suptitle("Dynamic nursing vulnerability phenotypes", y=1.02, fontsize=10)
    save(fig, out_dir, "figure2_dynamic_phenotypes")


def figure3(out_dir: Path) -> None:
    effects = read_tsv(Path("results/effect_sizes/claim_effects.tsv"))
    effects = effects[effects["phenotype_class"] == "high_intensity_class"].copy()
    label_map = {
        "unadjusted": "Unadjusted",
        "demographic_adjusted": "Age/sex adjusted",
        "care_intensity_adjusted": "Care-intensity adjusted",
    }
    effects["label"] = effects["model"].map(label_map)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    y = np.arange(len(effects))
    ax.errorbar(
        effects["effect"],
        y,
        xerr=[effects["effect"] - effects["ci_lower"], effects["ci_upper"] - effects["effect"]],
        fmt="o",
        color=PALETTE["blue"],
        ecolor=PALETTE["gray"],
        capsize=3,
    )
    ax.axvline(1, color="#999999", linestyle="--", linewidth=1)
    ax.set_yticks(y, effects["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for incident delirium")
    ax.set_title("Phenotype association across adjustment sets")
    panel_label(ax, "A")
    save(fig, out_dir, "figure3_outcome_effects")


def figure4(out_dir: Path) -> None:
    pred_path = Path("results/benchmarks/prediction_eval_ci.tsv")
    pred = read_tsv(pred_path if pred_path.exists() else Path("results/benchmarks/prediction_eval.tsv"))
    pred = pred[pred["split_or_cohort"] == "MIMIC_internal_5fold_cv"].copy()
    keep = ["demographic_baseline", "care_intensity_total", "nursing_vulnerability_density", "joint_density_screen", "dynamic_phenotype_care_intensity_adjusted"]
    pred = pred[pred["model"].isin(keep)].copy()
    pred["model_label"] = pred["model"].map(
        {
            "demographic_baseline": "Demographic",
            "care_intensity_total": "Total events",
            "nursing_vulnerability_density": "Nursing density",
            "joint_density_screen": "Joint density",
            "dynamic_phenotype_care_intensity_adjusted": "Phenotype",
        }
    )
    dca = read_tsv(Path("results/clinical_utility/decision_curve.tsv"))
    dca = dca[(dca["split_or_cohort"] == "MIMIC_internal_5fold_cv") & (dca["model"].isin(["nursing_vulnerability_density", "joint_density_screen", "care_intensity_total", "treat_none", "treat_all"]))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    order = pred.sort_values("auc_or_cstat")["model_label"]
    pred_idx = pred.set_index("model_label").loc[order]
    if {"auc_ci_lower", "auc_ci_upper"}.issubset(pred_idx.columns):
        xerr = [
            pred_idx["auc_or_cstat"] - pred_idx["auc_ci_lower"],
            pred_idx["auc_ci_upper"] - pred_idx["auc_or_cstat"],
        ]
    else:
        xerr = None
    ax.barh(order, pred_idx["auc_or_cstat"], xerr=xerr, color=PALETTE["blue"], ecolor=PALETTE["gray"], capsize=2)
    ax.set_xlim(0.45, 0.8)
    ax.set_xlabel("5-fold C-statistic")
    ax.set_title("Internal discrimination")
    panel_label(ax, "A")

    ax = axes[1]
    label_map = {
        "nursing_vulnerability_density": "Nursing density",
        "joint_density_screen": "Joint density",
        "care_intensity_total": "Total events",
        "treat_none": "Treat none",
        "treat_all": "Treat all",
    }
    colors = [PALETTE["orange"], PALETTE["blue"], PALETTE["green"], "#333333", "#999999"]
    for color, (model, sub) in zip(colors, dca.groupby("model")):
        ax.plot(sub["threshold"], sub["net_benefit"], label=label_map.get(model, model), linewidth=1.5, color=color)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "B")
    save(fig, out_dir, "figure4_screening_utility")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot first-pass publication figures.")
    parser.add_argument("--out-dir", type=Path, default=Path("plots/publication"))
    args = parser.parse_args()
    style()
    figure1(args.out_dir)
    figure2(args.out_dir)
    figure3(args.out_dir)
    figure4(args.out_dir)
    print(f"Wrote publication figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
