"""
ANN baseline for the minimal SNN Pong experiment.

This is a separate baseline. It does not modify config.py, network.py,
learning.py, pong.py, or any original SNN results.

Model:
  20-dimensional one-hot state (same channel mapping as the SNN)
  -> 32-unit tanh hidden layer
  -> 2-action softmax policy
  -> online REINFORCE update with a running reward baseline

The epsilon-greedy mixture is included in the action probabilities. Its
policy-gradient correction is used so exploratory actions are handled
consistently with the stated behaviour policy.
"""

import csv
from pathlib import Path

import numpy as np
import config
from pong import PongEnvironment
from encoding import relative_to_channel


# Keep ANN settings explicit and separate from the SNN configuration.
ANN_SEEDS = list(config.SEEDS)
ANN_TRIALS = config.TRIALS
HIDDEN_SIZE = 32
ANN_LEARNING_RATE = 0.01
ANN_EXPLORATION_RATE = config.EXPLORATION_RATE
REWARD_BASELINE_RATE = 0.01
ANN_WEIGHT_INIT_STD = 0.1
BLOCK_SIZE = config.BLOCK_SIZE
OUTPUT_DIR = Path("results_ann")


class ANNPolicy:
    """Small one-hidden-layer neural policy trained with REINFORCE."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, ANN_WEIGHT_INIT_STD,
                             (config.INPUT_SIZE, HIDDEN_SIZE))
        self.b1 = np.zeros(HIDDEN_SIZE, dtype=float)
        self.w2 = rng.normal(0.0, ANN_WEIGHT_INIT_STD,
                             (HIDDEN_SIZE, config.OUTPUT_SIZE))
        self.b2 = np.zeros(config.OUTPUT_SIZE, dtype=float)
        self.reward_baseline = 0.0

    @staticmethod
    def state_vector(ball, paddle):
        """Use the same relative-position-to-channel mapping as the SNN."""
        channel = relative_to_channel(ball, paddle)
        x = np.zeros(config.INPUT_SIZE, dtype=float)
        x[channel] = 1.0
        return x

    def probabilities(self, x):
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return hidden, probs

    def select_action(self, x, rng):
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        eps = ANN_EXPLORATION_RATE
        behaviour_probs = (1.0 - eps) * probs + eps / config.OUTPUT_SIZE
        action = int(rng.choice(config.OUTPUT_SIZE, p=behaviour_probs))
        return action, hidden, probs, behaviour_probs

    def update(self, x, action, reward, hidden, probs, behaviour_probs):
        # Advantage uses the running average reward as a variance-reduction baseline.
        advantage = float(reward - self.reward_baseline)

        # Gradient of log of the epsilon-mixture behaviour policy:
        # d log q(a|s) = ((1-eps) * pi(a|s) / q(a|s)) * d log pi(a|s)
        one_hot = np.zeros(config.OUTPUT_SIZE, dtype=float)
        one_hot[action] = 1.0
        dlogits = one_hot - probs
        eps = ANN_EXPLORATION_RATE
        correction = ((1.0 - eps) * probs[action] /
                      max(behaviour_probs[action], 1e-12))
        dlogits *= correction

        grad_w2 = np.outer(hidden, dlogits)
        grad_b2 = dlogits
        dhidden = (self.w2 @ dlogits) * (1.0 - hidden ** 2)
        grad_w1 = np.outer(x, dhidden)
        grad_b1 = dhidden

        lr = ANN_LEARNING_RATE * advantage
        self.w2 += lr * grad_w2
        self.b2 += lr * grad_b2
        self.w1 += lr * grad_w1
        self.b1 += lr * grad_b1

        # Update reward baseline after computing this trial's advantage.
        self.reward_baseline += REWARD_BASELINE_RATE * (
            float(reward) - self.reward_baseline
        )


def run_seed(seed, trials=ANN_TRIALS):
    # Separate streams make environment randomness independent from action sampling.
    env = PongEnvironment(seed=seed)
    action_rng = np.random.default_rng(seed * 1000 + 101)
    policy = ANNPolicy(seed=seed)

    rally_length = 0
    records = []
    completed_rallies = []

    for trial in range(1, trials + 1):
        ball = env.ball
        paddle_before = env.paddle
        x = policy.state_vector(ball, paddle_before)
        action, hidden, probs, behaviour_probs = policy.select_action(x, action_rng)

        outcome = env.step(action)
        reward = float(outcome["natural_reward"])
        policy.update(x, action, reward, hidden, probs, behaviour_probs)

        if outcome["hit"]:
            rally_length += 1
            rally_ended = 0
            completed_length = ""
            zero_hit = ""
            long_rally = ""
            env.serve()
        else:
            rally_ended = 1
            completed_length = rally_length
            zero_hit = int(rally_length == 0)
            long_rally = int(rally_length >= config.LONG_RALLY_MIN_HITS)
            completed_rallies.append(rally_length)
            rally_length = 0
            env.start_new_rally()

        records.append({
            "seed": seed,
            "trial": trial,
            "ball": ball,
            "paddle_before": paddle_before,
            "action": action,
            "hit": int(outcome["hit"]),
            "reward": reward,
            "rally_ended": rally_ended,
            "completed_rally_length": completed_length,
            "zero_hit_rally": zero_hit,
            "long_rally": long_rally,
        })

    hit_rate = float(np.mean([r["hit"] for r in records])) if records else float("nan")
    mean_rally = float(np.mean(completed_rallies)) if completed_rallies else float("nan")
    zero_hit_rate = (float(np.mean([x == 0 for x in completed_rallies]))
                     if completed_rallies else float("nan"))
    long_rally_rate = (float(np.mean([x >= config.LONG_RALLY_MIN_HITS
                                      for x in completed_rallies]))
                       if completed_rallies else float("nan"))

    # Per-block rally-length means use rallies that end within that block.
    block_rows = []
    for start in range(1, trials + 1, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE - 1, trials)
        lengths = [
            int(r["completed_rally_length"])
            for r in records
            if start <= r["trial"] <= end and r["rally_ended"] == 1
        ]
        block_hit_rate = float(np.mean([
            r["hit"] for r in records if start <= r["trial"] <= end
        ]))
        block_rows.append({
            "seed": seed,
            "block_start": start,
            "block_end": end,
            "mean_completed_rally_length": (
                float(np.mean(lengths)) if lengths else ""
            ),
            "hit_rate": block_hit_rate,
            "completed_rallies": len(lengths),
        })

    summary = {
        "seed": seed,
        "trials": trials,
        "mean_rally_length": mean_rally,
        "hit_rate": hit_rate,
        "zero_hit_rally_rate": zero_hit_rate,
        "long_rally_rate": long_rally_rate,
        "completed_rallies": len(completed_rallies),
        "final_reward_baseline": policy.reward_baseline,
        "parameter_count": (
            config.INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE
            + HIDDEN_SIZE * config.OUTPUT_SIZE + config.OUTPUT_SIZE
        ),
    }
    return records, block_rows, summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    all_blocks = []
    summaries = []

    for seed in ANN_SEEDS:
        print(f"Running ANN baseline: seed={seed}, trials={ANN_TRIALS}")
        records, blocks, summary = run_seed(seed)
        all_records.extend(records)
        all_blocks.extend(blocks)
        summaries.append(summary)
        print(
            f"  mean rally={summary['mean_rally_length']:.4f}, "
            f"hit rate={summary['hit_rate']:.4f}, "
            f"zero-hit rate={summary['zero_hit_rally_rate']:.4f}"
        )

    write_csv(OUTPUT_DIR / "ann_trial_results.csv", all_records)
    write_csv(OUTPUT_DIR / "ann_learning_curve.csv", all_blocks)
    write_csv(OUTPUT_DIR / "ann_seed_summary.csv", summaries)

    # Aggregate across seeds, preserving per-seed values in the summary file.
    metric_names = [
        "mean_rally_length", "hit_rate", "zero_hit_rally_rate",
        "long_rally_rate", "completed_rallies"
    ]
    aggregate = []
    for metric in metric_names:
        values = np.asarray([s[metric] for s in summaries], dtype=float)
        aggregate.append({
            "metric": metric,
            "mean_across_seeds": float(np.nanmean(values)),
            "sd_across_seeds": float(np.nanstd(values, ddof=1)),
            "n_seeds": int(np.sum(np.isfinite(values))),
        })
    write_csv(OUTPUT_DIR / "ann_condition_summary.csv", aggregate)

    print("\nSaved ANN outputs in:", OUTPUT_DIR.resolve())
    print("Files: ann_trial_results.csv, ann_learning_curve.csv,")
    print("       ann_seed_summary.csv, ann_condition_summary.csv")


if __name__ == "__main__":
    main()
