"""
Generates figures from results/summary/ data into results/figures/.

Run after analysis.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

SUMMARY_DIR = Path("results/summary")
FIG_DIR = Path("results/figures")

COLORS = {
    "stimulus": "#1f77b4",
    "no_feedback": "#ff7f0e",
    "frozen": "#2ca02c",
}


def fig1_learning_curves():
    curve = pd.read_csv(SUMMARY_DIR / "learning_curve.csv")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for condition in config.CONDITIONS:
        g = curve[curve["condition"] == condition].sort_values("block")
        x = g["trial_block_end"]
        mean = g["mean_rally_length"]
        std = g["std_rally_length"].fillna(0)
        ax.plot(x, mean, label=condition, color=COLORS[condition], lw=1.6)
        ax.fill_between(x, mean - std, mean + std,
                         color=COLORS[condition], alpha=0.15)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean rally length (block-averaged, \u00b1 1 SD across seeds)")
    ax.set_title("Experiment 1: Rally-Length Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_learning_curves.png", dpi=200)
    plt.close(fig)


def fig2_weight_change():
    """Cumulative absolute weight change over trials, per condition
    (averaged across seeds)."""

    frames = []
    for condition in config.CONDITIONS:
        for seed in config.SEEDS:
            df = pd.read_csv(f"results/raw/{condition}_seed{seed}.csv")
            df["cumulative_abs_weight_change"] = (
                df["mean_abs_weight_change"].cumsum()
            )
            frames.append(df[["trial", "condition", "seed",
                               "cumulative_abs_weight_change"]])
    all_df = pd.concat(frames, ignore_index=True)

    agg = (
        all_df.groupby(["condition", "trial"])["cumulative_abs_weight_change"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for condition in config.CONDITIONS:
        g = agg[agg["condition"] == condition].sort_values("trial")
        ax.plot(g["trial"], g["cumulative_abs_weight_change"],
                label=condition, color=COLORS[condition], lw=1.6)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative mean |weight change| (averaged across seeds)")
    ax.set_title("Experiment 1: Synaptic Plasticity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_weight_change.png", dpi=200)
    plt.close(fig)


def fig3_ace_rate_curve():
    """Block-averaged ace rate over trials, per condition."""

    frames = []
    for condition in config.CONDITIONS:
        for seed in config.SEEDS:
            df = pd.read_csv(f"results/raw/{condition}_seed{seed}.csv")
            rallies = df[df["rally_ended"] == 1].copy()
            rallies["block"] = (rallies["trial"] - 1) // config.BLOCK_SIZE
            block_ace = (
                rallies.groupby("block")["ace"].mean().reset_index()
            )
            block_ace["condition"] = condition
            block_ace["seed"] = seed
            frames.append(block_ace)
    all_df = pd.concat(frames, ignore_index=True)

    agg = (
        all_df.groupby(["condition", "block"])["ace"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["trial_block_end"] = (agg["block"] + 1) * config.BLOCK_SIZE

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for condition in config.CONDITIONS:
        g = agg[agg["condition"] == condition].sort_values("block")
        std = g["std"].fillna(0)
        ax.plot(g["trial_block_end"], g["mean"], label=condition,
                color=COLORS[condition], lw=1.6)
        ax.fill_between(g["trial_block_end"], g["mean"] - std,
                         g["mean"] + std, color=COLORS[condition], alpha=0.15)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Ace rate (block-averaged, \u00b1 1 SD across seeds)")
    ax.set_title("Experiment 1: Ace Rate Over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_ace_rate.png", dpi=200)
    plt.close(fig)


def fig4_rally_distribution():
    """Distribution of completed rally lengths per condition (all seeds
    pooled) -- shows the shape of the behavioural measure, not just its
    mean."""

    frames = []
    for condition in config.CONDITIONS:
        for seed in config.SEEDS:
            df = pd.read_csv(f"results/raw/{condition}_seed{seed}.csv")
            rallies = df[df["rally_ended"] == 1]
            frames.append(rallies[["condition", "completed_rally_length"]])
    all_df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    max_len = int(all_df["completed_rally_length"].max())
    bins = np.arange(-0.5, max_len + 1.5, 1)

    for condition in config.CONDITIONS:
        vals = all_df.loc[all_df["condition"] == condition,
                           "completed_rally_length"]
        ax.hist(vals, bins=bins, density=True, histtype="step",
                linewidth=1.8, label=condition, color=COLORS[condition])

    ax.set_xlabel("Completed rally length (hits before a miss)")
    ax.set_ylabel("Proportion of rallies")
    ax.set_title("Experiment 1: Rally-Length Distribution (all seeds pooled)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_rally_distribution.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating fig1_learning_curves.png ...")
    fig1_learning_curves()
    print("Generating fig2_weight_change.png ...")
    fig2_weight_change()
    print("Generating fig3_ace_rate.png ...")
    fig3_ace_rate_curve()
    print("Generating fig4_rally_distribution.png ...")
    fig4_rally_distribution()
    print(f"\nAll figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
