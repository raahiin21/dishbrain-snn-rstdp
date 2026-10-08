"""
The full network for Experiment 1:

    20 spike-input channels --(plastic weights, R-STDP)--> 2 LIF output
    neurons (LEFT, RIGHT)

There is deliberately NO hidden layer, no recurrent connections, and every
input channel connects to both output neurons (full connectivity,
weight matrix shape (20, 2)). This matches the final architecture
description exactly.
"""

import numpy as np
import config
from neuron import LIFNeuron


class SpikingNetwork:
    """Minimal feed-forward SNN used throughout Experiment 1."""

    def __init__(self, seed=None):
        rng = np.random.default_rng(seed)

        self.input_size = config.INPUT_SIZE
        self.output_size = config.OUTPUT_SIZE

        # Single plastic weight matrix: input channels -> output neurons.
        self.weights = rng.normal(
            0.0, config.WEIGHT_INIT_STD,
            (self.input_size, self.output_size)
        )
        np.clip(self.weights, config.WEIGHT_MIN, config.WEIGHT_MAX,
                out=self.weights)

        self.output_neurons = [LIFNeuron() for _ in range(self.output_size)]

    def reset_neurons(self):
        """Reset output neuron membrane state at the start of each trial.
        Weights are NOT reset here -- they persist across trials."""

        for neuron in self.output_neurons:
            neuron.reset()

    def run(self, input_spikes):
        """Run the output LIF layer for the full observation window.

        input_spikes: array of shape (SIMULATION_STEPS, INPUT_SIZE)

        Returns output_spikes, an array of shape
        (SIMULATION_STEPS, OUTPUT_SIZE) with a 1 at each timestep a given
        output neuron fires.
        """

        steps = input_spikes.shape[0]
        output_spikes = np.zeros((steps, self.output_size), dtype=int)

        for t in range(steps):
            # Raw synaptic drive: (1, input_size) @ (input_size, output_size)
            current = (input_spikes[t] @ self.weights) * config.SYNAPTIC_GAIN

            for j in range(self.output_size):
                output_spikes[t, j] = self.output_neurons[j].step(current[j])

        return output_spikes
