# World Models

PyTorch implementation of [World Models](https://arxiv.org/abs/1803.10122) — Ha & Schmidhuber, 2018.

---

## Overview

World Models asks a fundamental question: *can an agent learn to act entirely inside its own imagination?*

The paper proposes that intelligent agents build an internal model of their environment — a **world model** — and use it to make decisions. The architecture separates perception, memory, and control into three independently trained components:

- **V (Vision)** — a Variational Autoencoder that compresses raw pixel observations into a compact latent representation `z`
- **M (Memory)** — an MDN-RNN that learns the dynamics of the environment: given the current state and action, what comes next?
- **C (Controller)** — a minimal linear controller that maps the agent's internal state `(z, h)` directly to actions, trained with evolutionary search (CMA-ES)

See the [**How It Works**](#how-it-works) section below for detailed explanations of the V, M, and C components and the differences between dream and real-env controller training.

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

All hyperparameters (architecture, training, CMA-ES) are in `config/config.py`.

CLI flags override config values for a single run without editing the file.

---

## Usage

### Quick start (collect + train VAE)

```bash
python main.py quick
python main.py quick --panel rollout_replay
python main.py quick --panel vae_reconstruction
```

`quick` also uses the biased collection preset by default (200 rollouts × 1000 steps), so it matches the stronger quick-run data collection behavior.

### Full end-to-end, watch agent play live

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

# Biased policy — our custom collection method
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
| World Models | **906 ± 21** |
| Human | ~900 |

The environment is considered solved at 900+.

To reproduce, run `eval --episodes 100` — the mean reward is the leaderboard score.

## This implementation's results

| Training scale | Dream-env score | Real-env score |
| --- | --- | --- |
| Quick (200 biased rollouts, 50 gens × pop16 × 4 eval) | Pending benchmark | Pending benchmark |
| Paper (10k random rollouts, 1800 gens × pop64 × 16 eval) | Pending benchmark | Pending benchmark |

Fill these values from:

```bash
python main.py eval --episodes 100 --controller-mode dream
python main.py eval --episodes 100 --controller-mode real
```

---

## How It Works

### The Three Components

#### V — Variational Autoencoder (Vision)

Each raw frame (96×96 RGB) is compressed into a 32-dimensional latent vector `z` using a convolutional VAE. The encoder learns to discard irrelevant visual detail and retain only what matters. The decoder can reconstruct the frame from `z`, which can be used to visualize what the model "sees".

#### M — MDN-RNN (Memory)

A Recurrent Neural Network with a Mixture Density Network head models the distribution over future latent states: `p(z_{t+1} | z_t, a_t, h_t)`. The hidden state `h` serves as the agent's memory — it summarizes everything that has happened so far. Together, `z` and `h` form the agent's complete internal representation of the world.

#### C — Controller

A single linear layer mapping `[z, h] → action`. Despite its simplicity, it can drive effectively because V and M have already extracted all the relevant information. Training uses CMA-ES, a gradient-free evolutionary algorithm — no backpropagation through the environment is needed.

### Training Pipeline

Training happens in three sequential stages:

```text
1. Collect  →  rollout data from the environment (frames + actions)
2. Train V  →  VAE learns to encode/decode frames into latent z
3. Train M  →  MDN-RNN learns environment dynamics from encoded sequences
4. Train C  →  CMA-ES evolves the controller (dream mode by default)
```

**Dream mode (default):** The controller is evaluated entirely inside the learned world model — no real environment needed during controller training. A small reward model `R(z, h, a) → r` is trained automatically on encoded rollouts (if missing), then CMA-ES evolves the controller by rolling out **C+M+R** in latent space. This is ~50× faster than real-env evaluation.

**Real-env mode (`--real-env`):** The original approach — run full CarRacing episodes, encode each frame via V, maintain hidden state via M, and use ground-truth rewards. Slower but uses no learned reward approximation.

### Mathematical Formulation (Paper + This Implementation)

Let observation be $x_t$, latent be $z_t$, action be $a_t$, and recurrent hidden state be $h_t$.

The following equations formalize each component of the architecture and the training objectives used in this implementation.

**VAE (V model):**

$$
q_\phi(z_t \mid x_t), \quad p_\theta(x_t \mid z_t)
$$

- $q_\phi(z_t \mid x_t)$ is the encoder distribution that maps an observation $x_t$ to a latent code $z_t$.

- $p_\theta(x_t \mid z_t)$ is the decoder distribution that reconstructs the observation from the latent.

The VAE is trained by maximizing the Evidence Lower BOund (ELBO):
$$
\mathcal{L}_{\mathrm{VAE}}
=
\mathbb{E}_{q_\phi(z_t\mid x_t)}\big[\log p_\theta(x_t\mid z_t)\big]
- \beta\, D_{\mathrm{KL}}\!\big(q_\phi(z_t\mid x_t)\,\|\,\mathcal{N}(0,I)\big)
$$

- This repo uses $\beta=1.0$ with KL tolerance (free-bits style stabilization) in training.

- The first term is the reconstruction term: it rewards the decoder for matching the input frame.

- The second term is the KL divergence term: it regularizes the latent space by pushing the posterior toward a standard normal prior.

**MDN-RNN (M model):**

$$
h_{t+1} = f_{\mathrm{LSTM}}\big(h_t, [z_t, a_t]\big)
$$

- $f_{\mathrm{LSTM}}$ is the recurrent update. It combines the previous hidden state $h_t$ with the current latent $z_t$ and action $a_t$, then produces the next hidden state $h_{t+1}$.

$$
p(z_{t+1}\mid z_t,a_t,h_t)
=
\sum_{k=1}^{K} \pi_{t,k}\,\mathcal{N}\!\big(z_{t+1};\mu_{t,k},\sigma_{t,k}^2 I\big)
$$

- This is the MDN output. $K$ is the number of Gaussian components, $\pi_{t,k}$ are the mixture weights, $\mu_{t,k}$ are the component means, and $\sigma_{t,k}$ are the component standard deviations.
- The RNN is trained by negative log-likelihood of $z_{t+1}$ under this mixture.

**Controller (C model):**

$$
a_t = \tanh\big(W[z_t;h_t] + b\big)
$$

- This is the controller. The vector $[z_t; h_t]$ concatenates the latent and hidden state, $W$ is the learned weight matrix, and $b$ is the bias.
- $\tanh$ keeps each action head in $[-1,1]$; CarRacing then clips gas and brake to $[0,1]$.

**CMA-ES objective:**

$$
  \theta^* = \arg\max_{\theta}\; \mathbb{E}_{\tau\sim p_\theta(\tau)}\Big[\sum_{t=0}^{T-1} r_t\Big]
$$

- $\theta$ is a candidate controller parameter vector and $\theta^*$ is the optimal parameter vector found by CMA-ES.
- $\tau$ denotes a sampled trajectory, so $p_\theta(\tau)$ is the trajectory distribution induced by controller parameters $\theta$.
- The sum $\sum_{t=0}^{T-1} r_t$ is the return over one rollout.

In practice CMA-ES approximates this expectation by evaluating many rollouts per candidate $\theta$; the MDN-RNN sampling temperature $T_{\mathrm{MDN}}$ affects the trajectory distribution $p_\theta(\tau)$ but is not a direct input to the reward model.

**Dream-mode reward extension (CarRacing in this repo):**

The original paper did not use dream-mode controller training for CarRacing. In this repo, dream mode adds a learned reward model:

$$
\hat r_t = R_\psi(z_t,h_t,a_t)
$$

- $R_\psi$ is the learned reward model. It takes the current latent $z_t$, hidden state $h_t$, and action $a_t$, then predicts the per-step reward $\hat r_t$.
- The MDN-RNN still samples future latents with temperature $T_{\mathrm{MDN}}$.

### Dreaming

Once V and M are trained, the agent can "dream" — rolling out latent futures in model space, bypassing the real environment. In the paper, dream training is demonstrated for VizDoom; in this repo we extend the idea to CarRacing by adding a learned reward model so the controller can be evaluated in latent space.

Concretely, the reward model is a small MLP that maps `(z_t, h_t, a_t) -> r_t`. Training uses encoded rollouts: we run the frozen MDN-RNN forward to compute the hidden state *before* each step (`h_prev`), pair `(z, h_prev, a)` with the recorded per-step reward, and optimize mean-squared error (MSE) with Adam. The training script batches samples (default batch size 512), holds out a small validation split, and saves the best checkpoint. Once trained, `R(z,h,a)` supplies per-step rewards during dream rollouts while the MDN-RNN samples next latents (with configurable temperature $T_{\mathrm{MDN}}$).

---

## Controller Training Modes

The controller can be trained in two fundamentally different ways. The choice affects what components are active during CMA-ES evaluation and where the reward signal comes from.

### Real-env mode (`--real-env`)

```text
  ┌─────────────────── one step ────────────────────────────┐
  │                                                         │
  │  ┌──────────────┐  frame   ┌───────┐   z (32d)          │
  │  │  Real Env    │─────────▶│  VAE  │──────────┐         │
  │  │  (Box2D)     │          └───────┘          ▼         │
  │  │              │                    ┌─────────────────┐│
  │  │  env.step(a) │◀──── action ───────│   Controller    ││
  │  │              │                    │    z ⊕ h → a    ││
  │  └──────┬───────┘                    └─────────────────┘│
  │         │                                     ▲         │
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
  │                               │      RNN.sample()    │  │
  │                               │ (z, a, h) → z', h'   │  │
  │                               └────────────┬─────────┘  │
  │                                            │            │
  │               z_{t+1}, h_{t+1} ◀───────────┘(next step) │
  └─────────────────────────────────────────────────────────┘
```

No real environment. No VAE. The RNN replaces the physics engine — given `(z_t, a_t, h_t)` it samples the next latent `z_{t+1}` from its learned MDN mixture. The reward model predicts `r̂` from the agent's internal state. Everything is pure tensor ops on CPU.

**Pros:** ~50× faster — no physics, no rendering, no VAE inference per step.

**Cons:** Reward is approximate; controller can exploit model imperfections. RNN temperature $T_{\mathrm{MDN}}=1.15$ adds stochasticity to the dream, making exploitation harder.

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

### Why the paper's Dream mode doesn't directly apply to CarRacing

The original paper implements dream training only for VizDoom (DoomTakeCover), not CarRacing. The reason is subtle: **the dream world needs a reward signal**, and the two tasks provide it very differently.

In VizDoom the reward is simply survival time — the agent gets +1 for every frame it stays alive. The paper modifies the MDN-RNN to also predict a binary `done` signal (did the agent die this frame?). With that, the dream world has everything it needs:

The paper also has the MDN-RNN predict a binary `done_t` signal (whether the agent dies this frame) alongside the next latent `z_{t+1}`, providing a natural termination signal for dreams.

So the full VizDoom dream loop is: `RNN predicts (z_next, done)` — no reward model needed, just count steps until `done = True`.

**CarRacing is harder.** The reward is continuous (`+1000/N` per new tile crossed, `−0.1` per frame), depends on track position, and has no clean binary signal the RNN can easily predict. The paper sidesteps this entirely by keeping CarRacing in the real environment.

**Our solution — a learned reward model `R(z, h, a) → r̂`:** We train a small MLP on the encoded rollouts to approximate the per-step reward from the agent's internal state. This lets us bring dream training to CarRacing at the cost of reward approximation error. The paper also observes that using a higher MDN-RNN sampling temperature $T_{\mathrm{MDN}}$ during dream rollouts can reduce exploitation of model errors and often yields controllers that generalize better to normal settings.

The reward model is an extension beyond the paper. Use `--real-env` if you want to match the paper's exact CarRacing setup.

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
