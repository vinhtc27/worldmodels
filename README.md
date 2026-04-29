# World Models

PyTorch implementation of [World Models](https://arxiv.org/abs/1803.10122) — Ha & Schmidhuber, 2018.

---

## Overview

World Models asks a fundamental question: *can an agent learn to act entirely inside its own imagination?*

The paper proposes that intelligent agents build an internal model of their environment — a **world model** — and use it to make decisions. The architecture separates perception, memory, and control into three independently trained components:

- **V (Vision)** — a Variational Autoencoder that compresses raw pixel observations into a compact latent representation `z`
- **M (Memory)** — an MDN-RNN that learns the dynamics of the environment: given the current state and action, what comes next?
- **C (Controller)** — a minimal linear controller that maps the agent's internal state `(z, h)` directly to actions, trained with evolutionary search (CMA-ES)

See the [**How It Works**](#how-it-works) section below for detailed explanations of the V, M, and C components.

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
│   └── controller.py       # C model
├── data/                   # Rollout collection and datasets
├── training/               # Training loops for V, M, C
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

Runs the stronger preset (200 biased rollouts × 1000 steps → VAE 10 epochs → RNN 20 epochs → CMA-ES 50 gens × pop 16 × 4 eval) and opens a live game window at the end.

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
python main.py train-ctrl --generations 100 --pop-size 16 --n-workers 4
python main.py train-ctrl --generations 100 --pop-size 16 --resume   # continue from checkpoint
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

| Training scale | Score |
| --- | --- |
| Quick (200 biased rollouts, 50 gens × pop16 × 4 eval) | Pending benchmark |
| Paper (10k random rollouts, 1800 gens × pop64 × 16 eval) | Pending benchmark |

Fill these values from:

```bash
python main.py eval --episodes 100
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
4. Train C  →  CMA-ES evolves the controller in the real CarRacing environment
```

Each CMA-ES candidate is evaluated by running full CarRacing episodes: the VAE encodes each frame to `z`, the RNN maintains hidden state `h`, the controller maps `(z, h) → action`, and the environment returns ground-truth rewards.

### Hallucinating Future Trajectories

Once V and M are trained, the agent can *dream* — generating plausible futures entirely in latent space, without ever touching the real environment. This is the world model's internal simulation: the RNN predicts what it *expects* to see next, and the VAE decoder renders those predictions back into pixel space.

**How a dream sequence is generated:**

```text
seed frame x_0
    │
    ▼
[VAE encoder] → z_0          (compress to 32-d latent)
    │
    └─▶ h_0 = zeros          (RNN starts with empty memory)

for each step t:
    (z_t, a_t, h_t) ──▶ [MDN-RNN] ──▶ π, μ, σ    (mixture of K Gaussians over z_{t+1})
                                            │
                                            ▼
                                    sample z_{t+1} ~ MDN(π, μ, σ, τ)
                                            │
                                            ▼
                                    [VAE decoder] → rendered frame
```

The action $a_t$ during dreaming is drawn uniformly at random — the agent imagines what would happen under a random policy. The RNN hidden state `h_t` accumulates across steps, so the dream is temporally coherent: early decisions affect what is imagined later.

**Temperature τ and the quality of imagination:**

The MDN does not output a single predicted $z_{t+1}$ — it outputs a mixture of $K=5$ Gaussians. Sampling from this mixture is controlled by temperature $\tau$:

$$\tilde{\pi}_k \propto \pi_k^{1/\tau}, \qquad \tilde{\sigma}_k = \tau \cdot \sigma_k$$

| τ | Effect |
| --- | --- |
| τ → 0 | Deterministic: always picks the dominant Gaussian mode. Dreams are sharp but repetitive. |
| τ = 1 | Standard sampling. Dreams are realistic but varied. |
| τ > 1 | Flattens the mixture: more uncertainty, more creative / chaotic dreams. |
| τ = 1.15 | Default in this repo — slight stochasticity, matches paper's choice. |

At low temperature the world model replays the most likely future. At high temperature it explores unlikely branches — the agent's imagination becomes unstable, blurring and distorting as compounding prediction errors accumulate over steps.

**What the visualization shows:**

The `rnn_dream` panel seeds the RNN with one real frame from a recorded rollout, then runs the dream loop for `--n-steps` steps (default 200). Left panel shows the real seed frame; right panel shows the hallucinated sequence with an interactive step slider. You can regenerate with a different random action sequence to see alternative futures from the same starting point.

```bash
python main.py viz --panel rnn_dream                        # interactive
python main.py viz --panel rnn_dream --n-steps 300          # longer dream
python main.py viz --panel rnn_dream --temperature 0.5      # sharper
python main.py viz --panel rnn_dream --save dream.gif       # export GIF
```

**What good dreams look like:** After sufficient RNN training, the hallucinated frames should show recognizable track geometry — road curves, grass borders, the car — that evolves plausibly over time. Blurry or incoherent dreams indicate the RNN hasn't converged or the VAE latent space is poorly structured. The dream degrades gracefully over time as prediction errors compound, which is expected behaviour.

### Mathematical Formulation

Let observation be $x_t$, latent be $z_t$, action be $a_t$, and recurrent hidden state be $h_t$.

**VAE (V model):**

$$q_\phi(z_t \mid x_t), \qquad p_\theta(x_t \mid z_t)$$

- $q_\phi(z_t \mid x_t)$ is the encoder distribution that maps an observation $x_t$ to a latent code $z_t$.
- $p_\theta(x_t \mid z_t)$ is the decoder distribution that reconstructs the observation from the latent.

The VAE is trained by maximizing the Evidence Lower BOund (ELBO):

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z_t \mid x_t)}\left[\log p_\theta(x_t \mid z_t)\right] - \beta \, D_{\text{KL}}\left(q_\phi(z_t \mid x_t) \;\middle\|\; \mathcal{N}(0, I)\right)$$

- This repo uses $\beta = 1.0$ with KL tolerance (free-bits style stabilization) in training.
- The first term is the reconstruction term: it rewards the decoder for matching the input frame.
- The second term is the KL divergence term: it regularizes the latent space by pushing the posterior toward a standard normal prior.

**MDN-RNN (M model):**

$$h_{t+1} = f_{\text{LSTM}}\left(h_t,\, [z_t,\, a_t]\right)$$

- $f_{\text{LSTM}}$ is the recurrent update. It combines the previous hidden state $h_t$ with the current latent $z_t$ and action $a_t$, then produces the next hidden state $h_{t+1}$.

$$p(z_{t+1} \mid z_t, a_t, h_t) = \sum_{k=1}^{K} \pi_{t,k} \, \mathcal{N}\!\left(z_{t+1};\, \mu_{t,k},\, \sigma_{t,k}^2 I\right)$$

- This is the MDN output. $K$ is the number of Gaussian components, $\pi_{t,k}$ are the mixture weights, $\mu_{t,k}$ are the component means, and $\sigma_{t,k}$ are the component standard deviations.
- The RNN is trained by negative log-likelihood of $z_{t+1}$ under this mixture.

**Controller (C model):**

$$a_t = W_c \left[z_t;\, h_t\right] + b_c$$

- This is the controller as defined in the paper. The vector $[z_t;\, h_t]$ concatenates the latent and hidden state, $W_c$ is the weight matrix, and $b_c$ is the bias.
- In this implementation, a component-wise $\tanh$ is applied to keep each action head in $[-1, 1]$; CarRacing then clips gas and brake to $[0, 1]$. This is an implementation detail not in the original paper's formula.

**CMA-ES objective:**

$$\theta^* = \arg\max_{\theta} \; \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} r_t\right]$$

- $\theta$ is a candidate controller parameter vector and $\theta^*$ is the optimal parameter vector found by CMA-ES.
- $\tau$ denotes a sampled trajectory, so $p_\theta(\tau)$ is the trajectory distribution induced by controller parameters $\theta$.
- The sum $\sum_{t=0}^{T-1} r_t$ is the cumulative return over one rollout.

In practice CMA-ES approximates this expectation by evaluating many rollouts per candidate $\theta$.

---

## Related Work

- [Dream to Control (Dreamer)](https://arxiv.org/abs/1912.01603) — Hafner et al., 2020. Learns a world model and trains the policy entirely inside it using backpropagation through the model.
- [DreamerV2](https://arxiv.org/abs/2010.02193) — Uses discrete latents (categorical VAE) for improved stability.
- [DreamerV3](https://arxiv.org/abs/2301.04104) — A single general algorithm that achieves human-level performance across many domains.
- [PlaNet](https://arxiv.org/abs/1811.04551) — Hafner et al., 2019. Planning with a learned latent dynamics model using CEM.
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
