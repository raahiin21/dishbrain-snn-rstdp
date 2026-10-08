"""
Runs one (condition, seed) experiment of config.TRIALS trials, and records
both trial-level and rally-level data.

Random-number design (important, and documented for the paper's Methods):
    - A single "core" RNG, seeded directly from `seed`, controls the
      network's initial weights AND the environment's ball-position
      sequence. This is shared identically across all three conditions
      for a given seed -- i.e. stimulus/no_feedback/frozen at seed=42 all
      start from the same initial weights and see the same sequence of
      served ball positions, which is what makes the paired statistical
      comparisons (Wilcoxon signed-rank, matched by seed) valid.
    - A separate "exploration" RNG, seeded from a distinct derived value
      per (seed, condition), governs only the random tie-break / explore
      decisions in action selection. This is deliberately NOT shared
      across conditions.

      This matters specifically for no_feedback vs. frozen: both
      conditions produce exactly zero net weight change (no_feedback
      because its reward is always 0; frozen because the update is
      skipped entirely), so if exploration randomness were also shared,
      the two conditions would be forced into bit-for-bit identical
      trial sequences for every seed -- which is what happened in the
      previous version of this experiment (confirmed by inspecting the
      raw per-seed CSVs: no_feedback and frozen were identical to full
      floating-point precision). That is a mechanical artifact of shared
      randomness, not a finding about the conditions themselves. Giving
      exploration its own independent stream lets no_feedback and frozen
      each be a genuine (if similar) sample from "no learning occurs"
      space, which is what a reader would expect two nominally distinct
      control conditions to be.
"""

import numpy as np
import config
from encoding import relative_to_channel, generate_spikes
from network import SpikingNetwork
from learning import RewardModulatedSTDP
from pong import PongEnvironment


def _exploration_seed(seed, condition):
    """Derive an independent seed for the exploration RNG stream, unique
    to this (seed, condition) pair."""

    condition_id = {"stimulus": 1, "no_feedback": 2, "frozen": 3}[condition]
    return seed * 1000 + condition_id


def choose_action(output_spikes, explore_rng):
    """LEFT/RIGHT decided by whichever output neuron spiked more; ties
    (including 0-0) broken uniformly at random. A small independent
    exploration rate occasionally overrides this with a random action."""

    if explore_rng.random() < config.EXPLORATION_RATE:
        return int(explore_rng.integers(0, 2))

    counts = output_spikes.sum(axis=0)
    best = np.flatnonzero(counts == counts.max())
    if len(best) > 1:
        return int(explore_rng.choice(best))
    return int(best[0])


def run_condition(condition, seed, trials=None):
    """Run `trials` attempts for the given condition/seed and return a
    list of per-trial result dicts."""

    if trials is None:
        trials = config.TRIALS
    assert condition in config.CONDITIONS

    core_rng_seed = seed
    network = SpikingNetwork(seed=core_rng_seed)
    environment = PongEnvironment(seed=core_rng_seed)
    learner = RewardModulatedSTDP()
    explore_rng = np.random.default_rng(_exploration_seed(seed, condition))

    rally_length = 0          # hits accumulated in the current rally
    records = []

    for trial_index in range(1, trials + 1):

        channel = relative_to_channel(environment.ball, environment.paddle)
        input_spikes = generate_spikes(channel)

        network.reset_neurons()
        learner.reset()

        output_spikes = network.run(input_spikes)
        action = choose_action(output_spikes, explore_rng)

        outcome = environment.step(action)

        # Build the eligibility trace from this trial's spike activity.
        for t in range(config.SIMULATION_STEPS):
            learner.step(input_spikes[t], output_spikes[t])

        # Decide which reward this condition actually uses, and whether
        # the weight update is applied at all.
        if condition == "stimulus":
            reward_used = outcome["natural_reward"]
            weight_change = learner.apply_reward(network.weights, reward_used)
        elif condition == "no_feedback":
            reward_used = config.REWARD_NONE
            weight_change = learner.apply_reward(network.weights, reward_used)
        else:  # frozen
            reward_used = outcome["natural_reward"]  # logged, not applied
            weight_change = np.zeros_like(network.weights)

        if outcome["hit"]:
            rally_length += 1
            rally_ended = False
            completed_rally_length = None
            environment.serve()   # re-serve within the same rally
        else:
            rally_ended = True
            completed_rally_length = rally_length
            rally_length = 0
            environment.start_new_rally()

        records.append({
            "trial": trial_index,
            "condition": condition,
            "seed": seed,
            "ball": outcome["ball"],
            "paddle": outcome["paddle"],
            "action": action,
            "hit": int(outcome["hit"]),
            "natural_reward": outcome["natural_reward"],
            "reward_used": reward_used,
            "rally_ended": int(rally_ended),
            "completed_rally_length": (
                completed_rally_length if rally_ended else np.nan
            ),
            "ace": (
                int(completed_rally_length == 0) if rally_ended else np.nan
            ),
            "long_rally": (
                int(completed_rally_length >= config.LONG_RALLY_MIN_HITS)
                if rally_ended else np.nan
            ),
            "mean_abs_weight_change": float(np.mean(np.abs(weight_change))),
            "mean_weight": float(np.mean(network.weights)),
        })

    return records
