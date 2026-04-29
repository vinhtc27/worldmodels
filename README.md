# World Models

PyTorch implementation of [World Models](https://arxiv.org/abs/1803.10122) — Ha & Schmidhuber, 2018.

---

## Overview

World Models asks a fundamental question: *can an agent learn to act entirely inside its own imagination?*

The paper proposes that intelligent agents build an internal model of their environment — a **world model** — and use it to make decisions. The architecture separates perception, memory, and control into three independently trained components:

- **V (Vision)** — a Variational Autoencoder that compresses raw pixel observations into a compact latent representation `z`
- **M (Memory)** — an MDN-RNN that learns the dynamics of the environment: given the current state and action, what comes next?
- **C (Controller)** — a minimal linear controller that maps the agent's internal state `(z, h)` directly to actions, trained with evolutionary search (CMA-ES)

The key insight is that the controller is intentionally kept simple — all the "understanding" of the world lives in V and M. This separation makes the system modular, interpretable, and scalable.

---

## How It Works

### The Three Components

#### V — Variational Autoencoder (Vision)

Each raw frame (96×96 RGB) is compressed into a 32-dimensional latent vector `z` using a convolutional VAE. The encoder learns to discard irrelevant visual detail and retain only what matters. The decoder can reconstruct the frame from `z`, which can be used to visualize what the model "sees."

#### M — MDN-RNN (Memory)

A recurrent neural network with a Mixture Density Network head models the distribution over future latent states: `p(z_{t+1} | z_t, a_t, h_t)`. The hidden state `h` serves as the agent's memory — it summarizes everything that has happened so far. Together, `z` and `h` form the agent's complete internal representation of the world.

#### C — Controller

A single linear layer mapping `[z, h] → action`. Despite its simplicity, it can drive effectively because V and M have already extracted all the relevant information. Training uses CMA-ES, a gradient-free evolutionary algorithm — no backpropagation through the environment is needed.

### Training Pipeline

Training happens in three sequential stages:

```text
1. Collect  →  random rollouts from the environment (frames + actions)
2. Train V  →  VAE learns to encode/decode frames into latent z
3. Train M  →  MDN-RNN learns environment dynamics from encoded sequences
4. Train C  →  CMA-ES evolves the controller (dream mode by default)
```

**Dream mode (default):** The controller is evaluated entirely inside the learned world model — no real environment needed. A small reward model `R(z, h, a) → r` is trained automatically on the encoded rollouts, then CMA-ES evolves the controller by rolling out V+M+R in latent space. This is ~50× faster than real-env evaluation.

**Real-env mode (`--real-env`):** The original approach — run full CarRacing episodes, encode each frame via V, maintain hidden state via M, and use ground-truth rewards. Slower but uses no learned reward approximation.

### Dreaming

Once V and M are trained, the agent can "dream" — hallucinating future frames by rolling out the MDN-RNN in latent space, bypassing the real environment entirely. In extended experiments, the paper trains the controller *inside the dream* and transfers the resulting policy to the real environment.

---

## Controller Training Modes

The controller can be trained in two fundamentally different ways. The choice affects what components are active during CMA-ES evaluation and where the reward signal comes from.

### Real-env mode (`--real-env`)

```text
  ┌─────────────────── one step ────────────────────────────┐
  │                                                         │
  │  ┌──────────────┐  frame   ┌───────┐   z (32d)          │
  │  │  Real Env    │─────────▶│  VAE  │──────────┐         │
  │  │  (Box2D)     │          └───────┘           ▼        │
  │  │              │                    ┌─────────────────┐│
  │  │  env.step(a) │◀──── action ───────│   Controller    ││
  │  │              │                    │    z ⊕ h → a    ││
  │  └──────┬───────┘                    └────────▲────────┘│
  │         │                                     │         │
  │         │ reward ✓             ┌───────┐   h  │         │
  │         │ (ground truth)       │  RNN  │──────┘         │
  │         │                      └───────┘                │
  └─────────┴───────────────────────────────────────────────┘
```

At every step the real Box2D simulator produces a pixel frame. The VAE encodes it to `z`, the RNN maintains hidden state `h` across steps, the controller maps `(z, h) → action`, and the environment returns a ground-truth reward. The bottleneck is the physics simulator — each episode is seconds of wall time.

**Pros:** Ground-truth reward; controller cannot exploit model inaccuracies.
**Cons:** 1000 steps × frame_skip=4 × n_episodes × pop_size evaluations per generation — very slow.

### Dream mode (default)

```text
  ┌─────────────────── one step ────────────────────────────┐
  │                                                         │
  │   z_t ──▶ ┌─────────────────┐                           │
  │           │   Controller    │─── action ───┐            │
  │   h_t ──▶ │    z ⊕ h → a    │              │            │
  │           └─────────────────┘              │            │
  │                                            ▼            │
  │                                    ┌─────────────────┐  │
  │                                    │  Reward Model   │  │
  │                                    │  (z, h, a) → r̂  │  │
  │                                    └─────────────────┘  │
  │                                            │            │
  │                               ┌──────────────────────┐  │
  │                               │    RNN  .sample()    │  │
  │                               │ (z, a, h) → z', h'   │  │
  │                               └────────────┬─────────┘  │
  │                                            │            │
  │               z_{t+1}, h_{t+1} ◀───────────┘(next step) │
  └─────────────────────────────────────────────────────────┘
```

No real environment. No VAE. The RNN replaces the physics engine — given `(z_t, a_t, h_t)` it samples the next latent `z_{t+1}` from its learned MDN mixture. The reward model predicts `r̂` from the agent's internal state. Everything is pure tensor ops on CPU.

**Pros:** ~50× faster — no physics, no rendering, no VAE inference per step.
**Cons:** Reward is approximate; controller can exploit model imperfections. RNN temperature τ=1.15 adds stochasticity to the dream, making exploitation harder.

### Architectural summary

| | Real-env | Dream |
| --- | --- | --- |
| Real environment | ✓ (Box2D physics) | ✗ |
| VAE (frame → z) | ✓ per step | ✗ (z from RNN) |
| RNN (hidden state h) | ✓ memory only | ✓ dynamics + memory |
| Reward model | ✗ | ✓ R(z,h,a)→r̂ |
| Reward source | ground truth | learned approximation |
| Speed per generation | slow (real episodes) | fast (tensor rollouts) |
| Risk | none | model exploitation |

### Why the paper's dream mode doesn't directly apply to CarRacing

The original paper implements dream training only for VizDoom (DoomTakeCover), not CarRacing. The reason is subtle: **the dream world needs a reward signal**, and the two tasks provide it very differently.

In VizDoom the reward is simply survival time — the agent gets +1 for every frame it stays alive. The paper modifies the MDN-RNN to also predict a binary `done` signal (did the agent die this frame?). With that, the dream world has everything it needs:

> *"M model here will also predict whether the agent dies in the next frame (as a binary event done_t), in addition to the next frame z_t."*

So the full VizDoom dream loop is: `RNN predicts (z_next, done)` — no reward model needed, just count steps until `done = True`.

**CarRacing is harder.** The reward is continuous (`+1000/N` per new tile crossed, `−0.1` per frame), depends on track position, and has no clean binary signal the RNN can easily predict. The paper sidesteps this entirely by keeping CarRacing in the real environment.

**Our solution — a learned reward model `R(z, h, a) → r̂`:** We train a small MLP on the encoded rollouts to approximate the per-step reward from the agent's internal state. This lets us bring dream training to CarRacing at the cost of reward approximation error. The same temperature regularisation the paper uses for VizDoom applies here too:

> *"agents that perform well in higher temperature settings generally perform better in the normal setting. In fact, increasing τ helps prevent our controller from taking advantage of the imperfections of our world model."*

The reward model is an extension beyond the paper. Use `--real-env` if you want to match the paper's exact CarRacing setup.

---

## Environment

**CarRacing-v3** (Gymnasium) — a top-down racing environment with continuous control.

| Property | Value |
| --- | --- |
| Observation | 96×96 RGB frames (downsampled to 64×64 for training) |
| Action space | Continuous: `[steer (−1,1), gas (0,1), brake (0,1)]` |
| Reward | +1000/N per track tile visited, −0.1 per frame |
| Episode length | ~1000 steps |
| Solved threshold | 900+ averaged over 100 consecutive episodes |

The environment is simulated with Box2D physics. A `frame_skip` of 4 is used — each controller decision is held for 4 physics steps — matching standard practice and giving the car enough time to respond to steering.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Project Structure

```text
world/
├── config/config.py        # All hyperparameters in one place
├── models/
│   ├── vae.py              # V model
│   ├── mdn_rnn.py          # M model
│   ├── controller.py       # C model
│   └── reward_model.py     # R model — (z, h, a) → r, used in dream mode
├── data/                   # Rollout collection and datasets
├── training/               # Training loops for V, M, C, R
├── evaluation/             # Run the agent in the real environment
├── visualization/          # 6 interactive visualization panels
├── utils/                  # Shared helpers (checkpoint load/save)
├── checkpoint/             # Saved model weights
├── log/                    # Training histories
├── research/               # Paper-scale run outputs (auto-created)
└── main.py                 # CLI entry point
```

All hyperparameters (architecture, training, CMA-ES) are in `config/config.py`. CLI flags override config values for a single run without editing the file.

---

## Usage

### Quick start (~10–20 min, collect + train VAE)

```bash
python main.py quick
python main.py quick --panel rollout_replay
python main.py quick --panel vae_reconstruction
```

`quick` also uses the biased collection preset by default (200 rollouts × 1000 steps), so it matches the stronger quick-run data collection behavior.

### Full end-to-end, watch agent play live (~30–60 min)

```bash
python main.py quick --full
```

Runs the stronger preset (200 biased rollouts × 1000 steps → VAE 10 epochs → then the remaining steps in `quick --full`) and opens a live game window at the end. Controller trains in dream mode by default — no real env needed for CMA-ES.

### Skip steps already completed

```bash
python main.py quick --full --skip-collect
python main.py quick --full --skip-collect --skip-vae
python main.py quick --full --skip-collect --skip-vae --skip-rnn --skip-ctrl
```

### Full pipeline (production scale)

```bash
# Run all steps with defaults
python main.py all

# Step by step with custom settings
python main.py collect --n-rollouts 500
python main.py train-vae --epochs 20
python main.py train-rnn --epochs 30
python main.py train-ctrl --generations 100 --pop-size 16 --n-workers 4   # dream mode (default)
python main.py train-ctrl --generations 100 --pop-size 16 --real-env      # real environment
```

### Evaluate

```bash
# Watch the agent play live (ESC or ✕ to close)
python main.py eval --render --episodes 3 --window-size 600 600

# Benchmark (matches paper's evaluation protocol)
python main.py eval --episodes 100
```

### Paper-scale reproduction

```bash
# Pure random policy — matches the paper's collection method
make research-random

# Biased policy — our variant
make research-bias

# Custom output directory
RESEARCH_DIR=my_run make research-random
```

`biased` collection means the rollout policy samples **higher gas, lower brake, and holds the same action for 8 control steps** before resampling. In practice this helps the car keep moving and cover more of the track, so the dataset has fewer idle / spinning / stuck trajectories than pure random collection.

Runs: 10k rollouts → VAE 1 epoch → RNN 20 epochs → CMA-ES 1800 gens × pop 64 × 16 eval → benchmark 100 episodes + save all visualizations. Outputs land in `research/` by default.

### Visualize

```bash
python main.py viz --panel vae_reconstruction   # real vs reconstructed frames
python main.py viz --panel rollout_replay       # scrub through a rollout with slider
python main.py viz --panel latent_space         # PCA of z vectors colored by reward
python main.py viz --panel latent_walk          # interpolate between two frames in z-space
python main.py viz --panel rnn_dream            # MDN-RNN hallucinating future frames
python main.py viz --panel training_curves      # loss curves for V and M

# Save output
python main.py viz --panel rnn_dream --save dream.gif
```

---

## Benchmark

The paper reports results on CarRacing averaged over 100 consecutive episodes:

| Method | Score |
| --- | --- |
| DQN | 343 ± 18 |
| A3C (continuous) | 591 ± 45 |
| GA (evolution) | 753 ± 40 |
| World Models (real env) | **906 ± 21** |
| World Models (dream env) | 906 ± 21 |
| Human | ~900 |

The environment is considered solved at 900+. To reproduce, run `eval --episodes 100` — the mean reward is the leaderboard score.

Realistic expectations for this implementation:

| Training scale | Approximate score |
| --- | --- |
| Quick (`quick --full`, biased, 200 rollouts, 50 gens × pop16 × 4 eval) | 100 – 500 |
| Default (config defaults, random collection, dream) | 400 – 650 |
| Extended (100+ gens, 500+ rollouts, dream) | 650 – 850 |
| Paper (1800 gens, 10k rollouts, pop 64 × 16 eval) | **906 ± 21** |

The controller is the main bottleneck — more CMA-ES generations with a larger population raises the score most. More rollouts improve the quality of V and M, giving the controller a better representation to work with.

---

## Key Concepts

**Why keep the controller so simple?**
The paper argues that the complexity of understanding the environment should be captured by V and M, not by the controller. A linear controller with ~900 parameters trained by CMA-ES can outperform deep RL agents because it operates on a rich, well-structured internal representation rather than raw pixels.

**Why CMA-ES instead of gradient descent?**
The controller's objective (cumulative reward) is non-differentiable with respect to the environment. CMA-ES is a natural fit — it treats the parameter search as a black-box optimization problem and works well in low-dimensional spaces (the controller has only ~900 parameters).

**What is the MDN-RNN predicting?**
It models `p(z_{t+1} | z_t, a_t, h_t)` as a mixture of Gaussians. The mixture allows the model to represent multimodal futures — for example, a car approaching a fork can go either left or right. A single Gaussian would be forced to predict the average, which corresponds to driving into the wall.

**Frame skip**
Each controller decision is applied for 4 consecutive physics steps. This is standard in continuous control — it gives the car enough time to respond to inputs (at near-zero speed, single-frame steering has no effect) and reduces the temporal frequency the RNN needs to model.

---

## Related Work

- [Dream to Control (Dreamer)](https://arxiv.org/abs/1912.01603) — Hafner et al., 2020. Learns a world model and trains the policy entirely inside it using backpropagation through the model.
- [DreamerV2](https://arxiv.org/abs/2010.02193) — Uses discrete latents (categorical VAE) for improved stability.
- [DreamerV3](https://arxiv.org/abs/2301.04104) — A single general algorithm that achieves human-level performance across many domains.
- [PlaNet](https://arxiv.org/abs/1811.04551) — Hafner et al., 2019. Planning with a learned latent dynamics model using CEM.
- [RSSM](https://arxiv.org/abs/1811.04551) — Recurrent State Space Model, the dynamics backbone used in Dreamer.
- [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl) — Higgins et al., 2017. A generalization of VAE that learns disentangled representations by weighting the KL term.

---

## Reference

Ha, D. & Schmidhuber, J. (2018). **World Models**. *arXiv:1803.10122*

```bibtex
@article{ha2018world,
  title   = {World Models},
  author  = {Ha, David and Schmidhuber, J{\"u}rgen},
  journal = {arXiv preprint arXiv:1803.10122},
  year    = {2018}
}
```
