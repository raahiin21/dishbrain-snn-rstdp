"""
Runs the full Experiment 1: config.CONDITIONS x config.SEEDS, each for
config.TRIALS trials, and saves raw per-trial CSVs to results/raw/.

This is the only script that needs to be run to (re)generate all raw
data. Run analysis.py afterwards to produce summary tables, statistical
tests, and figures.
"""

import csv
import time
from pathlib import Path

import config
from experiment_runner import run_condition

RAW_DIR = Path("results/raw")


def save_records(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    total_runs = len(config.CONDITIONS) * len(config.SEEDS)
    done = 0

    for condition in config.CONDITIONS:
        for seed in config.SEEDS:
            t0 = time.time()
            records = run_condition(condition, seed, trials=config.TRIALS)
            out_path = RAW_DIR / f"{condition}_seed{seed}.csv"
            save_records(records, out_path)
            done += 1
            elapsed = time.time() - t0
            print(f"[{done}/{total_runs}] {condition:12s} seed={seed:3d}  "
                  f"({elapsed:.1f}s)  -> {out_path}")

    print(f"\nAll runs complete in {time.time() - start:.1f}s total.")


if __name__ == "__main__":
    main()
