"""
Configuration for Experiment 1: minimal SNN vs. DishBrain benchmark.

All values here match the "final Experiment 1" specification:
- 20 input channels, 2 LIF output neurons, no hidden layer
- Reward-modulated STDP (R-STDP) with a 500 ms eligibility trace
- 9-position 1D Pong environment with rally-based play
- 3 conditions x 10 seeds x 3000 trials

Nothing here should need to change between runs unless you deliberately
want to explore a different configuration (e.g. for a supplementary
sweep) -- for the paper's main result, treat this file as the single
source of truth for all reported parameters and hyperparameter tables.
"""

# ---------------------------------------------------------------------------
# Experiment scale
# ---------------------------------------------------------------------------
TRIALS = 3000                       # trials per (condition, seed) run
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]   # 10 seeds
CONDITIONS = ["stimulus", "no_feedback", "frozen"]

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
INPUT_SIZE = 20                     # spike-input channels (encoded, not LIF)
OUTPUT_SIZE = 2                     # LIF output neurons: 0 = LEFT, 1 = RIGHT
# No hidden layer: input channels connect directly to both output neurons.

# ---------------------------------------------------------------------------
# LIF neuron parameters (output neurons only -- inputs are spike trains, not
# LIF units themselves)
# ---------------------------------------------------------------------------
TAU_MEM = 20.0          # membrane time constant, ms
V_REST = -65.0          # resting / reset potential, mV
V_THRESHOLD = -50.0     # firing threshold, mV
V_RESET = -65.0         # post-spike reset potential, mV
REFRACTORY_MS = 2.0     # absolute refractory period, ms
DT = 1.0                # simulation timestep, ms
TRIAL_WINDOW_MS = 200.0 # duration of the observation window per trial, ms
SIMULATION_STEPS = int(TRIAL_WINDOW_MS / DT)   # = 200 steps per trial

# Synaptic (input current) time constant -- filters the raw spike input
# from the 20 input channels into a smoother synaptic current driving the
# output LIF neurons.
TAU_SYN = 5.0           # ms

# Synaptic gain: scales the raw (spike x weight) dot product into an
# input current for the output LIF neurons. This value was calibrated
# empirically (see calibration note in README) so that the plastic
# weight range [-1, 1] produces a graded, learnable firing-rate response
# in the output neurons: weight <= 0 -> silent, weight ~0.5 -> partial
# firing, weight = 1.0 -> near-maximal firing given the 2 ms refractory
# period. Without this gain, LIF output neurons driven directly by
# weight-scaled current remain silent across almost the entire weight
# range, which would prevent R-STDP from having any postsynaptic spikes
# to learn from.
SYNAPTIC_GAIN = 10.0

# ---------------------------------------------------------------------------
# STDP / eligibility-trace / reward parameters (R-STDP)
# ---------------------------------------------------------------------------
TAU_PRE = 20.0          # presynaptic STDP trace time constant, ms
TAU_POST = 20.0         # postsynaptic STDP trace time constant, ms
TAU_ELIGIBILITY = 500.0 # eligibility trace time constant, ms (final value)

A_PLUS = 0.01           # potentiation amplitude
A_MINUS = 0.012         # depression amplitude (slight depression bias)

LEARNING_RATE = 0.01    # scales reward-gated weight update

WEIGHT_MIN = -1.0
WEIGHT_MAX = 1.0
WEIGHT_INIT_STD = 0.1   # initial weight standard deviation (input -> output)

# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------
REWARD_HIT = 1.0
REWARD_MISS = -1.0
REWARD_NONE = 0.0       # used by the no_feedback condition

# ---------------------------------------------------------------------------
# Pong environment
# ---------------------------------------------------------------------------
N_POSITIONS = 9                     # discrete horizontal positions: 0..8
PADDLE_START = 4                    # centre position
# Relative position (ball - paddle) ranges over -8..+8 (17 values).
# This is mapped into the 20 input channels as (relative + 9), leaving a
# small unused margin (channels 0 and 18, 19) -- see encoding.py.

# ---------------------------------------------------------------------------
# Action-selection exploration
# ---------------------------------------------------------------------------
EXPLORATION_RATE = 0.10   # probability of a random action instead of the
                          # SNN's own spike-count decision

# ---------------------------------------------------------------------------
# Analysis / figure settings
# ---------------------------------------------------------------------------
BLOCK_SIZE = 50              # trials per block for learning-curve smoothing
LEARNING_THRESHOLD = 0.90    # rally length considered "learned" for the
                             # trials-to-threshold analysis
LONG_RALLY_MIN_HITS = 3      # rallies of this length or more are "long"
