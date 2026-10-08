"""
Leaky Integrate-and-Fire (LIF) neuron -- used ONLY for the 2 output neurons.

The 20 input channels are encoded spike trains (see encoding.py); they are
not LIF units themselves. This distinction matters for the paper: do not
describe the inputs as LIF neurons.
"""

import config


class LIFNeuron:
    """A single LIF neuron in millivolt units, with an absolute refractory
    period, matching the parameters recorded in config.py:

        tau_mem = 20 ms, V_rest = V_reset = -65 mV, V_threshold = -50 mV,
        refractory period = 2 ms, dt = 1 ms.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.v = config.V_REST
        self.refractory_remaining = 0.0

    def step(self, input_current):
        """Advance the neuron by one timestep (config.DT ms).

        input_current is in the same units as (V_threshold - V_rest), i.e.
        an already-scaled synaptic current contribution for this step.

        Returns 1 if the neuron fires this step, else 0.
        """

        if self.refractory_remaining > 0.0:
            self.refractory_remaining -= config.DT
            # Membrane is clamped at reset during the refractory period.
            self.v = config.V_RESET
            return 0

        # Leaky integration: membrane potential decays toward V_rest and
        # integrates the incoming synaptic current.
        dv = ((config.V_REST - self.v) / config.TAU_MEM) * config.DT
        self.v += dv + input_current

        if self.v >= config.V_THRESHOLD:
            self.v = config.V_RESET
            self.refractory_remaining = config.REFRACTORY_MS
            return 1

        return 0
