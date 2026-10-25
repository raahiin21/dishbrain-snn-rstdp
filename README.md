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

To contextualize this performance, we evaluated benchmark policies in the same environment:

| Policy / Architecture | Mean rally length | Hit rate | Zero-hit (ace) rate |
|---|---|---|---|
| SNN (stimulus, R-STDP) | 0.487 ± 0.016 | ≈ 32.8% | 68.0% |
| SNN (no-feedback control) | 0.485 ± 0.018 | ≈ 32.7% | 67.9% |
| Random 50/50 | ≈ 0.49 | ≈ 33.0% | ≈ 67.0% |
| ANN Baseline (REINFORCE) | **0.844 ± 0.025** | **45.8%** | **55.0%** |
| Oracle (steps toward ball) | ≈ 1.20 | ≈ 55.0% | ≈ 45.0% |

While all three SNN conditions stay pinned at the random baseline, the ANN baseline demonstrates that an active policy gradient agent easily learns in this environment and climbs well toward the oracle.

## Revision Experiment: ANN Baseline (`revision/ann-baseline`)

To directly evaluate the hypothesis that task learnability or reward formulation limits learning, we introduced a classic Artificial Neural Network (ANN) baseline on the branch `revision/ann-baseline`.

### Architecture & Setup
- **Input:** Identical 20-dimensional one-hot state mapping as the SNN (`relative_to_channel`).
- **Network:** Feedforward network with a 32-unit `tanh` hidden layer (`20 -> 32 -> 2`) predicting softmax action probabilities (LEFT vs. RIGHT).
- **Learning Rule:** Online REINFORCE policy gradient with an exponential running reward baseline ($\alpha = 0.01$) to center advantage estimates.
- **Exploration:** 10% $\epsilon$-greedy exploration with full policy-gradient probability correction.
- **Evaluation:** Run across the identical 10 random seeds (42–51) for 3,000 trials.

### Findings
- The ANN baseline achieved a mean rally length of **0.844 ± 0.025** and a **45.8%** hit rate, compared to 0.487 and 32.8% for the SNN.
- This confirms the simplified Pong environment provides a clear, learnable gradient when paired with credit assignment and a centered baseline, underscoring that the SNN's failure stems from the minimal R-STDP mechanism rather than task insolubility.

## Why we think the SNN didn't learn

Frémaux, Sprekeler & Gerstner (2010) showed that R-STDP only learns reliably when the reward signal is centred, meaning its prediction error averages to zero. Our raw +1/-1 reward in the SNN isn't centred, and the weight drift we see is consistent with that bias rather than with the network picking up a useful policy. The successful learning in our ANN baseline with a running baseline reinforces this interpretation. Other architectural elements omitted from the minimal SNN (hidden layers, lateral inhibition, spontaneous activity, criticality) may also be essential.

## Details that aren't obvious from the paper

- **A "trial" is one volley, not one rally.** Each trial is one decision by the network. 3,000 trials means 3,000 paddle moves, which end up being many short rallies. Rally length is measured over completed rallies.
- **A hit means the paddle lands within ±1 of the ball** (`HIT_TOLERANCE` in `pong.py`).
- **10% of actions are random** (`EXPLORATION_RATE` in `config.py`), on top of the model decision.
- **`SYNAPTIC_GAIN = 10`** was calibrated by hand. With weights in [-1, 1] and no gain, the output neurons barely fire, and R-STDP has nothing to learn from without postsynaptic spikes.
- Initial weights are drawn from N(0, 0.1²) and clipped to [-1, 1].
- Input channels 0, 18 and 19 are never used. The offset range only reaches channels 1–17.

## Running it

```bash
pip install -r requirements.txt

# Original SNN Experiment
python3 run_experiment.py        # all 30 SNN runs -> results/raw/ (a few minutes)
python3 analysis.py              # statistical analysis -> results/summary/
python3 make_figures.py          # figures -> results/figures/
python3 benchmark_policies.py    # random and oracle reference policies

# ANN Baseline Experiment (branch: revision/ann-baseline)
python3 ann_baseline.py          # 10 seeds, 3000 trials -> results_ann/
```

Everything is seeded and deterministic across runs.

## Layout

```
config.py              all parameters for the SNN experiment
neuron.py              LIF neuron implementation
encoding.py            offset -> input channel -> spike train
network.py             20 -> 2 spiking network
pong.py                1D Pong environment
learning.py            R-STDP with eligibility trace
experiment_runner.py   one (condition, seed) SNN run
run_experiment.py      all SNN conditions x seeds
analysis.py            summaries and statistical tests
make_figures.py        the four experiment figures
benchmark_policies.py  random / oracle reference policies
ann_baseline.py        ANN REINFORCE baseline (revision/ann-baseline)
results/raw/           per-trial SNN logs, one CSV per run
results/summary/       aggregated tables and statistical tests
results/figures/       figures used in the paper
results_ann/           ANN baseline logs, seed summaries, and learning curves
```

## License

MIT, see `LICENSE`.

