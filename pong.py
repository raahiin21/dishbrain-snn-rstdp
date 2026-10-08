"""
Simplified 1D Pong environment with 9 discrete positions and rally-based
play (matches the final Experiment 1 description).

Rules:
    - Positions are integers 0..8. The paddle starts at position 4
      (centre) at the start of every rally.
    - The ball position is drawn uniformly at random at the start of a
      rally, and again after every successful hit ("re-serve").
    - Each step, the agent chooses LEFT (0) or RIGHT (1); the paddle moves
      exactly one position in that direction (clipped to [0, 8]).
    - A "hit" occurs if the paddle's new position is within HIT_TOLERANCE
      of the ball's position. A hit increments the rally length and
      re-serves the ball (new random position, same rally continues).
      A miss ends the rally.
    - Reward is condition-dependent (see learning.py / experiment_runner.py):
      the environment itself always reports the "natural" hit/miss reward
      (+1 / -1); how that reward is actually used (or zeroed, or ignored)
      is decided by the calling condition, not by this environment.
"""

import numpy as np
import config

HIT_TOLERANCE = 1  # paddle catches the ball if within this many positions


class PongEnvironment:
    """One persistent environment per (condition, seed) run. Call
    `serve()` to start a rally (or re-serve after a hit), and `step()`
    once per action within the rally."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.paddle = config.PADDLE_START
        self.ball = None
        self.serve()

    def serve(self):
        """(Re-)serve the ball at a new random position. Does not move
        or reset the paddle -- the paddle keeps its current position
        between serves within the same rally, and only resets to centre
        when a new rally begins after a miss (see step())."""

        self.ball = int(self.rng.integers(0, config.N_POSITIONS))
        return self.ball, self.paddle

    def start_new_rally(self):
        """Called after a miss: reset paddle to centre and serve a fresh
        ball."""

        self.paddle = config.PADDLE_START
        return self.serve()

    def step(self, action):
        """Apply one action (0 = LEFT, 1 = RIGHT), move the paddle by one
        position, and evaluate hit/miss against the current ball
        position.

        Returns a dict with the ball/paddle positions, whether this was a
        hit, and the "natural" reward for this outcome (+1 hit / -1
        miss). Conditions that should not use this reward information
        (e.g. no_feedback) substitute their own reward value when calling
        the learning rule -- this environment does not know about
        experimental conditions.
        """

        if action == 0:      # LEFT
            self.paddle -= 1
        elif action == 1:    # RIGHT
            self.paddle += 1

        self.paddle = int(np.clip(self.paddle, 0, config.N_POSITIONS - 1))

        hit = abs(self.ball - self.paddle) <= HIT_TOLERANCE
        natural_reward = config.REWARD_HIT if hit else config.REWARD_MISS

        return {
            "ball": self.ball,
            "paddle": self.paddle,
            "hit": bool(hit),
            "natural_reward": natural_reward,
        }
