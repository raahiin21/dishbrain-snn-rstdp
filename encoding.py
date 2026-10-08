"""
Position encoding: converts the relative ball-paddle position into a spike
train across the 20 input channels.

Design (matches the final Experiment 1 description):
    - The environment has 9 discrete positions (0..8) for both ball and
      paddle.
    - The *relative* position (ball - paddle) is what actually drives the
      network, ranging from -8 to +8 (17 possible values).
    - This is mapped onto one of the 20 input channels via a simple
      offset: channel = relative_position + 9, so channel indices 1..17
      are used and channels 0, 18, 19 are always silent (unused margin).
    - Exactly one input channel is active per trial. That channel fires a
      constant spike train (1 at every timestep) for the full 200 ms
      observation window -- there is no rate or population coding here,
      matching the simplified final implementation (not the earlier,
      abandoned Gaussian population-coding design).
"""

import numpy as np
import config


def relative_to_channel(ball_position, paddle_position):
    """Map (ball, paddle) discrete positions to a single active input
    channel index in [0, config.INPUT_SIZE)."""

    relative = ball_position - paddle_position
    channel = relative + (config.N_POSITIONS)  # offset so index is >= 0
    channel = int(np.clip(channel, 0, config.INPUT_SIZE - 1))
    return channel


def generate_spikes(channel):
    """Create the input spike train for the trial: the selected channel
    fires at every timestep of the observation window; all others are
    silent throughout.

    Returns an array of shape (SIMULATION_STEPS, INPUT_SIZE).
    """

    spikes = np.zeros((config.SIMULATION_STEPS, config.INPUT_SIZE), dtype=int)
    spikes[:, channel] = 1
    return spikes
