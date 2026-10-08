"""
Analysis pipeline for Experiment 1.

Reads all raw per-trial CSVs from results/raw/ and produces, into
results/summary/:

    condition_summary.csv     -- per-condition means/SDs across seeds
    run_summary.csv           -- per (condition, seed) summary row
    statistical_tests.csv     -- paired Wilcoxon tests between every pair
                                  of conditions, for each key metric
    early_late_tests.csv      -- paired Wilcoxon test (early vs late) of
                                  rally length and ace rate, within each
                                  condition, across seeds
    learning_curve.csv        -- block-averaged rally length per trial
                                  block, per condition (mean + SD across
                                  seeds) -- feeds Figure 1
    learning_trends.csv       -- per (condition, seed) linear-regression
                                  slope of rally length vs. trial block
    threshold_results.csv     -- per (condition, seed) trial index at
                                  which the block-average rally length
                                  first reaches config.LEARNING_THRESHOLD
                                  (NaN if never reached)

Run this after run_experiment.py. Run make_figures.py afterwards to
generate the plots.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config

RAW_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")

EARLY_FRACTION = 0.25   # rallies ending in the first 25% of trials
LATE_FRACTION = 0.75    # rallies ending in the last 25% of trials


def load_all_raw():
    frames = []
    for condition in config.CONDITIONS:
        for seed in config.SEEDS:
            path = RAW_DIR / f"{condition}_seed{seed}.csv"
            df = pd.read_csv(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_run_summary(df):
    """One row per (condition, seed): rally-level and weight-change
    summary statistics."""

    rows = []
    n_trials = config.TRIALS
    early_cutoff = n_trials * EARLY_FRACTION
    late_cutoff = n_trials * LATE_FRACTION

    for (condition, seed), g in df.groupby(["condition", "seed"]):
        rallies = g[g["rally_ended"] == 1]

        early = rallies[rallies["trial"] <= early_cutoff]
        late = rallies[rallies["trial"] > late_cutoff]

        row = {
            "condition": condition,
            "seed": seed,
            "n_rallies": len(rallies),
            "mean_rally_length": rallies["completed_rally_length"].mean(),
            "mean_ace_rate": rallies["ace"].mean(),
            "mean_long_rally_rate": rallies["long_rally"].mean(),
            "total_abs_weight_change": g["mean_abs_weight_change"].sum(),
            "early_rally_length": early["completed_rally_length"].mean(),
            "late_rally_length": late["completed_rally_length"].mean(),
            "early_ace_rate": early["ace"].mean(),
            "late_ace_rate": late["ace"].mean(),
        }
        row["rally_improvement"] = (
            row["late_rally_length"] - row["early_rally_length"]
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["condition", "seed"])


def condition_level_summary(run_summary):
    """Aggregate the per-run summary across seeds, one row per
    condition."""

    metrics = [
        "mean_rally_length", "mean_ace_rate", "mean_long_rally_rate",
        "total_abs_weight_change", "early_rally_length", "late_rally_length",
        "rally_improvement", "early_ace_rate", "late_ace_rate",
    ]

    rows = []
    for condition, g in run_summary.groupby("condition"):
        row = {"condition": condition, "n_seeds": len(g)}
        for m in metrics:
            row[f"mean_{m}"] = g[m].mean()
            row[f"std_{m}"] = g[m].std(ddof=1)
        rows.append(row)

    order = {c: i for i, c in enumerate(config.CONDITIONS)}
    result = pd.DataFrame(rows)
    result["_order"] = result["condition"].map(order)
    return result.sort_values("_order").drop(columns="_order")


def paired_condition_tests(run_summary):
    """Wilcoxon signed-rank test between every pair of conditions, for
    each key metric, matched by seed (valid because network
    initialisation and ball sequence are shared across conditions for a
    given seed -- see experiment_runner.py)."""

    metrics = [
        "mean_rally_length", "mean_ace_rate", "mean_long_rally_rate",
        "rally_improvement",
    ]

    rows = []
    for metric in metrics:
        pivot = run_summary.pivot(index="seed", columns="condition",
                                   values=metric)
        for c1, c2 in combinations(config.CONDITIONS, 2):
            x, y = pivot[c1], pivot[c2]
            if (x == y).all():
                stat, p = np.nan, np.nan
                note = "identical across all seeds"
            else:
                try:
                    stat, p = stats.wilcoxon(x, y)
                    note = ""
                except ValueError as e:
                    stat, p = np.nan, np.nan
                    note = str(e)
            rows.append({
                "metric": metric,
                "condition_a": c1,
                "condition_b": c2,
                "mean_a": x.mean(),
                "mean_b": y.mean(),
                "wilcoxon_stat": stat,
                "p_value": p,
                "note": note,
            })

    return pd.DataFrame(rows)


def early_late_tests(run_summary):
    """Within each condition, paired Wilcoxon test of early vs. late
    rally length and ace rate across seeds."""

    rows = []
    for condition, g in run_summary.groupby("condition"):
        for metric_pair, label in [
            (("early_rally_length", "late_rally_length"), "rally_length"),
            (("early_ace_rate", "late_ace_rate"), "ace_rate"),
        ]:
            early_col, late_col = metric_pair
            x, y = g[early_col], g[late_col]
            if (x == y).all():
                stat, p = np.nan, np.nan
                note = "identical across all seeds"
            else:
                try:
                    stat, p = stats.wilcoxon(x, y)
                    note = ""
                except ValueError as e:
                    stat, p = np.nan, np.nan
                    note = str(e)
            rows.append({
                "condition": condition,
                "measure": label,
                "mean_early": x.mean(),
                "mean_late": y.mean(),
                "wilcoxon_stat": stat,
                "p_value": p,
                "note": note,
            })

    return pd.DataFrame(rows)


def learning_curve(df):
    """Block-averaged rally length per trial block, per condition (mean
    and SD across seeds). Blocks are defined over the trial index at
    which each rally ended."""

    rallies = df[df["rally_ended"] == 1].copy()
    rallies["block"] = ((rallies["trial"] - 1) // config.BLOCK_SIZE)

    per_seed_block = (
        rallies.groupby(["condition", "seed", "block"])[
            "completed_rally_length"
        ]
        .mean()
        .reset_index()
    )

    agg = (
        per_seed_block.groupby(["condition", "block"])[
            "completed_rally_length"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["trial_block_start"] = agg["block"] * config.BLOCK_SIZE + 1
    agg["trial_block_end"] = agg["trial_block_start"] + config.BLOCK_SIZE - 1
    agg = agg.rename(columns={
        "mean": "mean_rally_length",
        "std": "std_rally_length",
        "count": "n_seeds_with_data",
    })

    order = {c: i for i, c in enumerate(config.CONDITIONS)}
    agg["_order"] = agg["condition"].map(order)
    agg = agg.sort_values(["_order", "block"]).drop(columns="_order")
    return agg


def learning_trends(learning_curve_df):
    """Per-condition linear regression slope of mean rally length vs.
    trial block (using the already block-averaged, cross-seed data),
    quantifying the direction/strength of any trend over training."""

    rows = []
    for condition, g in learning_curve_df.groupby("condition"):
        g = g.dropna(subset=["mean_rally_length"])
        if len(g) < 2:
            rows.append({"condition": condition, "slope": np.nan,
                          "intercept": np.nan, "r_value": np.nan,
                          "p_value": np.nan, "n_blocks": len(g)})
            continue
        slope, intercept, r, p, se = stats.linregress(
            g["block"], g["mean_rally_length"]
        )
        rows.append({
            "condition": condition,
            "slope_per_block": slope,
            "intercept": intercept,
            "r_value": r,
            "p_value": p,
            "n_blocks": len(g),
        })

    order = {c: i for i, c in enumerate(config.CONDITIONS)}
    result = pd.DataFrame(rows)
    result["_order"] = result["condition"].map(order)
    return result.sort_values("_order").drop(columns="_order")


def threshold_results(df):
    """Per (condition, seed): the trial index at which the block-average
    rally length first reaches config.LEARNING_THRESHOLD. NaN if the
    threshold is never reached within the run."""

    rallies = df[df["rally_ended"] == 1].copy()
    rallies["block"] = ((rallies["trial"] - 1) // config.BLOCK_SIZE)

    rows = []
    for (condition, seed), g in rallies.groupby(["condition", "seed"]):
        block_means = (
            g.groupby("block")["completed_rally_length"].mean().sort_index()
        )
        reached = block_means[block_means >= config.LEARNING_THRESHOLD]
        if len(reached) > 0:
            first_block = reached.index[0]
            trial_reached = int(first_block * config.BLOCK_SIZE
                                 + config.BLOCK_SIZE)
        else:
            trial_reached = np.nan
        rows.append({
            "condition": condition,
            "seed": seed,
            "threshold": config.LEARNING_THRESHOLD,
            "trial_reached": trial_reached,
            "reached": trial_reached is not np.nan and not pd.isna(trial_reached),
        })

    order = {c: i for i, c in enumerate(config.CONDITIONS)}
    result = pd.DataFrame(rows)
    result["_order"] = result["condition"].map(order)
    return result.sort_values(["_order", "seed"]).drop(columns="_order")


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df = load_all_raw()
    print(f"  {len(df)} trial records loaded.")

    print("Computing per-run summary...")
    run_summary = per_run_summary(df)
    run_summary.to_csv(SUMMARY_DIR / "run_summary.csv", index=False)

    print("Computing condition-level summary...")
    cond_summary = condition_level_summary(run_summary)
    cond_summary.to_csv(SUMMARY_DIR / "condition_summary.csv", index=False)

    print("Running paired condition tests...")
    cond_tests = paired_condition_tests(run_summary)
    cond_tests.to_csv(SUMMARY_DIR / "statistical_tests.csv", index=False)

    print("Running early-vs-late tests...")
    el_tests = early_late_tests(run_summary)
    el_tests.to_csv(SUMMARY_DIR / "early_late_tests.csv", index=False)

    print("Computing learning curves...")
    curve = learning_curve(df)
    curve.to_csv(SUMMARY_DIR / "learning_curve.csv", index=False)

    print("Computing learning trends...")
    trends = learning_trends(curve)
    trends.to_csv(SUMMARY_DIR / "learning_trends.csv", index=False)

    print("Computing trials-to-threshold...")
    thresh = threshold_results(df)
    thresh.to_csv(SUMMARY_DIR / "threshold_results.csv", index=False)

    print(f"\nAll summary files written to {SUMMARY_DIR}/")
    print("\n=== condition_summary.csv ===")
    print(cond_summary.to_string(index=False))


if __name__ == "__main__":
    main()
