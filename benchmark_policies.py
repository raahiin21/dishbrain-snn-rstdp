"""
Reference policies for the same Pong environment: a random 50/50 policy and
an oracle that always steps toward the ball. Both are run over the same
seeds and trial count as the main experiment, and give the floor and ceiling
that the SNN conditions are compared against.

Run:  python3 benchmark_policies.py   ->  results/summary/policy_benchmarks.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config
from pong import PongEnvironment


def run_policy(policy, seed, trials=config.TRIALS):
    env = PongEnvironment(seed=seed)
    rng = np.random.default_rng(seed * 1000 + 9)  # only used for the policy's own choices

    rally, completed, hits = 0, [], 0
    for _ in range(trials):
        if policy == "random":
            action = int(rng.integers(0, 2))
        else:  # oracle: step toward the ball; if already aligned, pick a side
            if env.ball > env.paddle:
                action = 1
            elif env.ball < env.paddle:
                action = 0
            else:
                action = int(rng.integers(0, 2))

        outcome = env.step(action)
        if outcome["hit"]:
            hits += 1
            rally += 1
            env.serve()
        else:
            completed.append(rally)
            rally = 0
            env.start_new_rally()

    return {
        "policy": policy,
        "seed": seed,
        "mean_rally_length": float(np.mean(completed)),
        "hit_rate": hits / trials,
    }


def main():
    rows = [run_policy(p, s) for p in ("random", "oracle") for s in config.SEEDS]
    df = pd.DataFrame(rows)
    summary = df.groupby("policy")[["mean_rally_length", "hit_rate"]].mean().reset_index()
    print(summary.to_string(index=False))

    out = Path("results/summary")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "policy_benchmarks.csv", index=False)


if __name__ == "__main__":
    main()
