"""
Reward-modulated STDP (R-STDP) with an eligibility trace, applied directly
to the input(20) -> output(2) weight matrix (no hidden layer).

Mechanism (three-factor learning rule):
    1. Presynaptic (input channel) and postsynaptic (output neuron) spike
       traces decay exponentially (tau_pre = tau_post = 20 ms) and are
       incremented by spikes at each timestep.
    2. A pairwise STDP term (potentiation - depression) is computed at
       every timestep from these traces and accumulated into a slowly
       decaying eligibility trace (tau_eligibility = 500 ms). This is
       what lets the trace "remember" recent spike-timing coincidences
       for long enough to be linked to a reward that arrives after the
       trial ends.
    3. At the end of the trial, a reward signal gates the eligibility
       trace into an actual weight change: dW = learning_rate * reward *
       eligibility.

This module implements the mechanism only. Which reward value is passed
to `apply_reward`, and whether it is applied to the weights at all, is
decided per-condition by experiment_runner.py:
    - stimulus:     natural reward (+1 hit / -1 miss), weights updated.
    - no_feedback:  reward forced to 0.0 every trial, weights updated
                    (with zero effect, since reward = 0).
    - frozen:       natural reward is computed and logged for reference,
                    but apply_reward() is never called, so weights never
                    change. Skipping the call (rather than passing
                    reward=0) is a deliberate, documented distinction --
                    see README / paper Methods.
"""

import numpy as np
import config


class RewardModulatedSTDP:

    def __init__(self):
        self.pre_trace = np.zeros(config.INPUT_SIZE)
        self.post_trace = np.zeros(config.OUTPUT_SIZE)
        self.eligibility = np.zeros((config.INPUT_SIZE, config.OUTPUT_SIZE))

    def reset(self):
        """Clear traces at the start of each trial. Eligibility does NOT
        automatically persist across trials in this implementation --
        each trial's reward is credited only to spike-timing coincidences
        that occurred within that same trial's 200 ms window."""

        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.eligibility.fill(0.0)

    def step(self, pre_spikes, post_spikes):
        """Update STDP traces and accumulate the eligibility trace for
        one timestep. pre_spikes: shape (INPUT_SIZE,), post_spikes:
        shape (OUTPUT_SIZE,)."""

        pre_decay = np.exp(-config.DT / config.TAU_PRE)
        post_decay = np.exp(-config.DT / config.TAU_POST)
        eligibility_decay = np.exp(-config.DT / config.TAU_ELIGIBILITY)

        self.pre_trace *= pre_decay
        self.post_trace *= post_decay

        self.pre_trace += pre_spikes
        self.post_trace += post_spikes

        potentiation = config.A_PLUS * np.outer(self.pre_trace, post_spikes)
        depression = config.A_MINUS * np.outer(pre_spikes, self.post_trace)

        self.eligibility *= eligibility_decay
        self.eligibility += potentiation - depression

    def apply_reward(self, weights, reward):
        """Gate the accumulated eligibility trace by the reward signal
        and apply the resulting weight change in place. Returns the
        weight change array (useful for logging total synaptic
        modification)."""

        weight_change = config.LEARNING_RATE * reward * self.eligibility
        weights += weight_change
        np.clip(weights, config.WEIGHT_MIN, config.WEIGHT_MAX, out=weights)
        return weight_change
