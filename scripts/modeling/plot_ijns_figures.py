#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"
LIGHT_GRAY = "#D9D9D9"
VERY_LIGHT = "#F3F4F4"
DARK = "#222222"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")


def pct_ci(events: pd.Series, n: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    p = events / n
    se = np.sqrt(p * (1 - p) / n)
    return (p - 1.96 * se).clip(lower=0).to_numpy(), (p + 1.96 * se).clip(upper=1).to_numpy()


def incident_delirium_counts() -> tuple[int, int]:
    matrix_path = Path("results/phenotypes/mimic_feature_matrix_48h.tsv")
    if matrix_path.exists():
        matrix = read_tsv(matrix_path)
        incident = matrix[matrix["pre_landmark_delirium_positive"] == False].copy()
        return int(len(incident)), int(incident["incident_post_landmark_delirium"].sum())

    summary = read_tsv(Path("results/phenotypes/mimic_outcome_summary_48h.tsv"))
    pre = summary[summary["outcome_or_status"] == "pre_landmark_delirium_positive"].iloc[0]
    event = summary[summary["outcome_or_status"] == "incident_post_landmark_delirium"].iloc[0]
    primary_n = int(pre["denominator"]) - int(pre["n"])
    return primary_n, int(event["n"])


def figure1(out_dir: Path) -> None:
    flow = read_tsv(Path("results/figures/figure1_cohort_landmark_flow.tsv"))
    primary_n, incident_events = incident_delirium_counts()
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Cohort construction", loc="left", pad=6)
    cohort = flow[flow["panel"] == "cohort_flow"].sort_values("display_order").set_index("label")
    steps = [
        ("All ICU stays", int(cohort.loc["all_icu_stays", "n"])),
        ("Adults aged 65 years or older", int(cohort.loc["age_ge_65", "n"])),
        ("First ICU stay per patient", int(cohort.loc["first_icu_stay_per_subject", "n"])),
        ("48-hour landmark eligible", int(cohort.loc["landmark_48h_eligible", "n"])),
        ("Primary incident-delirium cohort", primary_n),
    ]
    y_positions = np.linspace(0.86, 0.16, len(steps))
    for i, ((label, n), y) in enumerate(zip(steps, y_positions)):
        face = "#EAF3F8" if i < len(steps) - 1 else "#F7EAE3"
        edge = BLUE if i < len(steps) - 1 else ORANGE
        ax.add_patch(Rectangle((0.08, y - 0.055), 0.72, 0.11, facecolor=face, edgecolor=edge, linewidth=0.9))
        ax.text(0.11, y + 0.018, label, ha="left", va="center", fontsize=7.2, color=DARK)
        ax.text(0.77, y - 0.018, f"n={n:,}", ha="right", va="center", fontsize=7.2, color=DARK)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((0.44, y - 0.062), (0.44, y_positions[i + 1] + 0.062), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=GRAY))
    pre_landmark_excluded = int(cohort.loc["landmark_48h_eligible", "n"]) - primary_n
    ax.text(
        0.84,
        y_positions[-2] - 0.02,
        f"Exclude\npre-landmark\ndocumented\ndelirium\nn={pre_landmark_excluded:,}",
        ha="left",
        va="center",
        fontsize=6.4,
        color=GRAY,
    )
    ax.text(0.08, 0.04, f"Post-landmark documented delirium events: {incident_events:,}", ha="left", va="center", fontsize=7.0, color=DARK)
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Landmark analysis design", loc="left", pad=6)
    y0 = 0.38
    ax.add_patch(FancyArrowPatch((0.08, y0), (0.94, y0), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=DARK))
    ticks = [(0.08, "0 h\nICU admission"), (0.50, "48 h\nlandmark"), (0.94, "Post-landmark\nfollow-up")]
    for x, text in ticks:
        ax.plot([x, x], [y0 - 0.04, y0 + 0.04], color=DARK, linewidth=1)
        ax.text(x, y0 - 0.10, text, ha="center", va="top", fontsize=6.8)
    ax.add_patch(Rectangle((0.10, 0.57), 0.38, 0.18, facecolor="#EAF3F8", edgecolor=BLUE, linewidth=0.9))
    ax.text(0.29, 0.66, "Early nursing\ndocumentation\nphenotype", ha="center", va="center", fontsize=6.6, linespacing=1.0)
    ax.add_patch(Rectangle((0.55, 0.57), 0.37, 0.18, facecolor="#F7EAE3", edgecolor=ORANGE, linewidth=0.9))
    ax.text(0.735, 0.66, "Post-landmark\ndocumented delirium", ha="center", va="center", fontsize=6.9)
    ax.plot([0.10, 0.48], [0.52, 0.52], color=BLUE, linewidth=3.0, solid_capstyle="butt")
    ax.plot([0.55, 0.92], [0.52, 0.52], color=ORANGE, linewidth=3.0, solid_capstyle="butt")
    ax.text(0.29, 0.48, "Exposure window", ha="center", va="top", fontsize=6.6, color=BLUE)
    ax.text(0.735, 0.48, "Outcome window", ha="center", va="top", fontsize=6.6, color=ORANGE)
    ax.text(0.50, 0.86, "Early nursing documentation precedes outcome ascertainment", ha="center", va="center", fontsize=6.8, color=GRAY)
    panel(ax, "B")
    save(fig, out_dir, "figure1_ijns_study_design")


def figure2(out_dir: Path) -> None:
    prof = read_tsv(Path("results/figures/figure2_trajectory_anchor.tsv")).sort_values("assigned_class")
    ascertain = read_tsv(Path("results/tables/supplement_outcome_ascertainment.tsv"))
    ascertain = ascertain[ascertain["summary_level"] == "phenotype"].copy()
    fig = plt.figure(figsize=(7.2, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.34, 1.0], wspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    domains = [
        ("Total care-process events", "all_total_count_48h", "per stay"),
        ("Skin/Braden/pressure records", "skin_braden_pressure_records_48h", "per stay"),
        ("RASS/sedation records", "rass_sedation_records_48h", "per stay"),
        ("Restraint records", "restraints_records_48h", "per stay"),
        ("Night-event fraction", "all_night_fraction_48h", "%"),
    ]
    high = prof.iloc[0]
    low = prof.iloc[1]
    values = np.array([[low[col], high[col]] for _name, col, _unit in domains], dtype=float)
    row_min = values.min(axis=1, keepdims=True)
    row_range = np.maximum(values.max(axis=1, keepdims=True) - row_min, 1e-9)
    scaled = (values - row_min) / row_range
    cmap = colors.LinearSegmentedColormap.from_list("density", ["#F5F5F5", "#D7E9F3", BLUE])
    ax.imshow(scaled, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["Lower intensity", "High intensity"])
    ax.set_yticks(np.arange(len(domains)), [d[0] for d in domains])
    ax.tick_params(length=0)
    ax.set_title("Early nursing documentation profile", loc="left", pad=7)
    for i, (_name, _col, unit) in enumerate(domains):
        for j, val in enumerate(values[i]):
            text = f"{val*100:.1f}%" if unit == "%" else f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.8, color=DARK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(0.5, -0.22, "Restraint records were concentrated in the high-intensity phenotype.", transform=ax.transAxes, ha="center", va="top", fontsize=6.4, color=GRAY)
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Documented delirium and post-landmark assessment", loc="left", pad=7)
    high_assessment = ascertain[ascertain["phenotype_display"].str.startswith("High ")].iloc[0]
    low_assessment = ascertain[ascertain["phenotype_display"].str.startswith("Lower ")].iloc[0]
    outcome_rows = [
        ("Documented delirium", high["incident_delirium_rate"] * 100, low["incident_delirium_rate"] * 100, "event rate"),
        (
            "Any assessment",
            high_assessment["any_post_landmark_assessment_rate"] * 100,
            low_assessment["any_post_landmark_assessment_rate"] * 100,
            "assessment rate",
        ),
    ]
    y = np.arange(len(outcome_rows))[::-1]
    for yi, (label, high_val, low_val, _note) in zip(y, outcome_rows):
        ax.plot([low_val, high_val], [yi, yi], color=LIGHT_GRAY, lw=1.8, zorder=1)
        ax.scatter(low_val, yi, color=BLUE, s=34, label="Lower intensity" if yi == y[0] else None, zorder=3)
        ax.scatter(high_val, yi, color=ORANGE, s=34, label="High intensity" if yi == y[0] else None, zorder=3)
        ax.text(low_val, yi - 0.18, f"{low_val:.1f}%", ha="center", va="top", fontsize=6.5, color=BLUE)
        ax.text(high_val, yi + 0.18, f"{high_val:.1f}%", ha="center", va="bottom", fontsize=6.5, color=ORANGE)
    ax.set_yticks(y, [row[0] for row in outcome_rows])
    ax.set_xlabel("Percentage of patients")
    ax.set_xlim(0, 55)
    ax.set_ylim(-0.55, 1.55)
    ax.grid(axis="x", color="#EFEFEF", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", ncol=1, handletextpad=0.4)
    ax.text(0.02, -0.32, "Assessment was incomplete, so the endpoint remains documented delirium.", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=GRAY)
    panel(ax, "B")
    save(fig, out_dir, "figure2_ijns_phenotype_and_ascertainment")


def figure3(out_dir: Path) -> None:
    effects = read_tsv(Path("results/tables/supplement_outcome_ascertainment_bias_models.tsv"))
    primary = read_tsv(Path("results/tables/table3_effect_estimates.tsv"))
    rows = []
    model_labels = {
        "unadjusted": "Unadjusted",
        "demographic_adjusted": "Age/sex adjusted",
        "care_intensity_adjusted": "Care-intensity adj.",
        "clinical_covariate_extended": "Clinical extended",
    }
    for model, label in model_labels.items():
        sub = primary[(primary["model"] == model) & (primary["term"].fillna(primary["phenotype_class"]) == "high_intensity_class")]
        if sub.empty:
            sub = primary[(primary["model"] == model) & (primary["phenotype_class"] == "high_intensity_class")]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append((label, r["effect"], r["ci_lower"], r["ci_upper"]))
    sens_labels = {
        "restricted_to_any_post_landmark_assessment": "Assessed only",
        "adjusted_for_post_landmark_assessment_count": "Assessment-count adj.",
        "ipw_restricted_to_assessed": "IPW assessed",
    }
    for analysis, label in sens_labels.items():
        r = effects[effects["analysis"] == analysis].iloc[0]
        rows.append((label, r["effect"], r["ci_lower"], r["ci_upper"]))
    forest = pd.DataFrame(rows, columns=["label", "or", "lo", "hi"])

    pred = read_tsv(Path("results/tables/table5_screening_performance_with_ci.tsv"))
    pred = pred[pred["split_or_cohort"] == "MIMIC_internal_5fold_cv"].copy()
    keep = {
        "demographic_baseline": "Demographics only",
        "care_intensity_total": "Total care-process events",
        "dynamic_phenotype_care_intensity_adjusted": "Phenotype plus care intensity",
        "nursing_vulnerability_density": "Nursing documentation model",
        "joint_density_screen": "Nursing plus care-process model",
    }
    pred = pred[pred["model"].isin(keep)].copy()
    pred["label"] = pred["model"].map(keep)
    pred = pred.sort_values("auc_or_cstat")

    fig = plt.figure(figsize=(7.2, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.62)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(forest))[::-1]
    ax.axvline(1, color=LIGHT_GRAY, linewidth=1)
    ax.errorbar(forest["or"], y, xerr=[forest["or"] - forest["lo"], forest["hi"] - forest["or"]], fmt="o", color=BLUE, ecolor=GRAY, capsize=2, markersize=4)
    ax.set_yticks(y, forest["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for documented delirium")
    ax.set_title("Primary and ascertainment sensitivity models", loc="left", pad=7)
    ax.set_xlim(0.8, 3.6)
    ax.xaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3]))
    ax.set_xticklabels(["1.0", "1.5", "2.0", "3.0"])
    ax.grid(axis="x", color="#EFEFEF", linewidth=0.8)
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(len(pred))
    ax.hlines(y, pred["auc_ci_lower"], pred["auc_ci_upper"], color=GRAY, linewidth=1.2)
    ax.scatter(pred["auc_or_cstat"], y, color=GREEN, s=28, zorder=3)
    ax.errorbar(
        pred["auc_or_cstat"],
        y,
        xerr=[pred["auc_or_cstat"] - pred["auc_ci_lower"], pred["auc_ci_upper"] - pred["auc_or_cstat"]],
        fmt="none",
        ecolor=GRAY,
        capsize=2,
        linewidth=0.8,
    )
    ax.set_yticks(y, pred["label"])
    ax.set_xlim(0.45, 0.80)
    ax.set_xlabel("Internal 5-fold C-statistic")
    ax.set_title("Internal screening performance", loc="left", pad=7)
    ax.grid(axis="x", color="#EFEFEF", linewidth=0.7)
    for yi, r in zip(y, pred.itertuples()):
        ax.text(0.795, yi, f"{r.auc_or_cstat:.3f}", ha="right", va="center", fontsize=6.4, color=DARK)
    panel(ax, "B")
    save(fig, out_dir, "figure3_ijns_effects_and_screening")


def supplementary_decision_curve(out_dir: Path) -> None:
    dca = read_tsv(Path("results/clinical_utility/decision_curve.tsv"))
    dca = dca[(dca["split_or_cohort"] == "MIMIC_internal_5fold_cv") & (dca["model"].isin(["nursing_vulnerability_density", "joint_density_screen", "care_intensity_total", "treat_none", "treat_all"]))]
    label_map = {
        "nursing_vulnerability_density": "Nursing documentation model",
        "joint_density_screen": "Nursing plus care-process model",
        "care_intensity_total": "Total care-process events",
        "treat_none": "Treat none",
        "treat_all": "Treat all",
    }
    color_map = {
        "nursing_vulnerability_density": GREEN,
        "joint_density_screen": BLUE,
        "care_intensity_total": ORANGE,
        "treat_none": "#9E9E9E",
        "treat_all": DARK,
    }
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for model, sub in dca.groupby("model"):
        linewidth = 1.35 if model not in {"treat_none", "treat_all"} else 1.0
        ax.plot(sub["threshold"], sub["net_benefit"], label=label_map[model], color=color_map[model], linewidth=linewidth)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Internal decision-curve analysis", loc="left", pad=5)
    ax.text(0.01, 0.96, "MIMIC-IV internal 5-fold cross-validation", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=GRAY)
    ax.set_ylim(-0.02, 0.055)
    ax.grid(axis="y", color="#EFEFEF", linewidth=0.7)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    save(fig, out_dir, "supplementary_figure_s1_decision_curve")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot IJNS-style manuscript figures.")
    parser.add_argument("--out-dir", type=Path, default=Path("plots/ijns"))
    args = parser.parse_args()
    style()
    figure1(args.out_dir)
    figure2(args.out_dir)
    figure3(args.out_dir)
    supplementary_decision_curve(args.out_dir)
    print(f"Wrote IJNS-style figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
