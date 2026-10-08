# Neurons in a Dish vs. Neurons in Code

A small spiking neural network that tries to learn a simplified Pong game, built to check whether a basic reward-modulated STDP rule is enough to reproduce the feedback-dependent learning reported in the DishBrain experiment (Kagan et al., 2022, *Neuron*).

This is the code and data behind our paper, *From Neurons in a Dish to Neurons in Code: Benchmarking Biological Learning Claims Against a Minimal Spiking Model* (Rahin Khamkar, Hayat Azad, Sadaf Malik, Arshiya Shaikh, Ayaan Surve; Anjuman Islam Technical Campus).

**Short version:** it didn't reproduce. The synaptic weights changed, but the network played no better than a random player.

## The question

DishBrain reported that cultured neurons playing Pong improved their rally length only when they got closed-loop feedback. We wanted to know if a minimal, well-understood learning rule shows the same pattern: better with feedback, no better without it. If it does, a simple local rule may explain the effect. If it doesn't, that rules out one candidate mechanism. Either outcome is informative, so we ran it.

## What's in the model

- **Task:** 1-D Pong with 9 positions. The ball is served at a random position and re-served after every hit, so a rally keeps going. A miss ends the rally and the paddle resets to the centre.
- **Input:** the ball-minus-paddle offset (-8 to +8) maps to a single active channel out of 20, which fires on every timestep of a 200 ms window. There is no rate or population coding.
- **Network:** 20 inputs connect straight to 2 leaky integrate-and-fire output neurons (LEFT and RIGHT). There is no hidden layer. Whichever neuron spikes more decides the move, and ties are broken randomly.
- **Learning:** reward-modulated STDP. Pre/post spike timing builds up an eligibility trace (τ = 500 ms), and a global reward then turns that trace into a weight change: `Δw = lr × reward × eligibility`. Reward is +1 for a hit, -1 for a miss, with no baseline subtracted.
- **Conditions:**
  - `stimulus`: real reward, weights update.
  - `no_feedback`: reward forced to 0, weights update (so nothing changes).
  - `frozen`: reward is computed and logged, but the weight update is never applied.
- **Scale:** 10 seeds (42–51), 3,000 trials per run. For a given seed, all three conditions share the same initial weights and ball sequence, so the Wilcoxon tests are paired. Each condition gets its own exploration RNG, so the two no-learning controls aren't forced into identical runs.

All parameters live in `config.py`.

## Results

| Condition | Mean rally length | Ace rate | Cumulative \|Δw\| |
|---|---|---|---|
| stimulus | 0.487 | 0.680 | 0.270 |
| no_feedback | 0.485 | 0.679 | 0 |
| frozen | 0.493 | 0.676 | 0 |

Every paired Wilcoxon test between conditions gives p > 0.1 (most are above 0.4), and none of the conditions shows a significant early-vs-late change. Full tables are in `results/summary/`.

The weights are clearly moving under `stimulus` (the blue line in `fig2_weight_change.png`), but that doesn't show up in behaviour.

To make sure this wasn't a floor or ceiling problem, we ran two reference policies in the same environment (`benchmark_policies.py`):

| Policy | Mean rally length | Hit rate |
|---|---|---|
| Random 50/50 | ≈ 0.49 | ≈ 33% |
| Oracle (always steps toward the ball) | ≈ 1.2 | ≈ 55% |

All three SNN conditions sit at the random level, and the oracle shows there was room to do better.

## Why we think it didn't learn

Frémaux, Sprekeler & Gerstner (2010) showed that R-STDP only learns reliably when the reward signal is centred, meaning its prediction error averages to zero. Our raw +1/-1 reward isn't centred, and the weight drift we see is consistent with that bias rather than with the network picking up a useful policy. This is our interpretation, and we didn't test it here. The obvious follow-up is subtracting a running reward baseline before gating the eligibility trace. Other things we left out (hidden layers, inhibition, spontaneous activity, criticality) could also matter, and this experiment can't tell them apart.

## Details that aren't obvious from the paper

- **A "trial" is one volley, not one rally.** Each trial is one decision by the network. 3,000 trials means 3,000 paddle moves, which end up being many short rallies. Rally length is measured over completed rallies.
- **A hit means the paddle lands within ±1 of the ball** (`HIT_TOLERANCE` in `pong.py`).
- **10% of actions are random** (`EXPLORATION_RATE` in `config.py`), on top of the spike-count decision.
- **`SYNAPTIC_GAIN = 10`** was calibrated by hand. With weights in [-1, 1] and no gain, the output neurons barely fire, and R-STDP has nothing to learn from without postsynaptic spikes.
- Initial weights are drawn from N(0, 0.1²) and clipped to [-1, 1].
- Input channels 0, 18 and 19 are never used. The offset range only reaches channels 1–17.

## Running it

```
pip install -r requirements.txt
python3 run_experiment.py        # all 30 runs, writes results/raw/ (a few minutes)
python3 analysis.py              # tables and stats -> results/summary/
python3 make_figures.py          # figures -> results/figures/
python3 benchmark_policies.py    # random and oracle reference policies
```

Everything is seeded. Re-running `run_experiment.py` gives the same raw CSVs that are already in `results/raw/`.

## Layout

```
config.py              all parameters
neuron.py              LIF neuron
encoding.py            offset -> input channel -> spike train
network.py             20 -> 2 network
pong.py                the environment
learning.py            R-STDP with eligibility trace
experiment_runner.py   one (condition, seed) run
run_experiment.py      all conditions x seeds
analysis.py            summaries and statistical tests
make_figures.py        the four figures
benchmark_policies.py  random / oracle reference policies
results/raw/           per-trial logs, one CSV per run
results/summary/       aggregated tables and tests
results/figures/       figures used in the paper
```

## License

MIT, see `LICENSE`.
