# CLAUDE.md — Repo Context

## What this repo is

PyTorch implementation of World Models (Ha & Schmidhuber, 2018) applied to CarRacing-v3.
Three independently trained components: VAE (vision) → MDN-RNN (memory) → Controller (CMA-ES).

## Environment

- Python 3.9, venv at `.venv/` — always use `.venv/bin/python`, never system python
- Run commands: `source .venv/bin/activate` or prefix with `.venv/bin/python`
- Key packages: torch 2.8, gymnasium 1.1 (CarRacing-v3), cma, pygame 2.5.2 (NOT 2.6.x — macOS crash), rich, matplotlib (MacOSX backend on macOS, NOT TkAgg)

## Project layout

```text
config/config.py      — all hyperparameters, edit here first before touching model code
models/               — vae.py, mdn_rnn.py, controller.py
data/                 — rollout_generator.py (collection), dataset.py (loaders)
training/             — train_vae.py, train_rnn.py, train_controller.py
evaluation/           — evaluate.py (run agent, custom pygame window)
visualization/        — visualize.py (6 interactive matplotlib panels)
main.py               — single CLI entry point for everything
Makefile              — run `make help` to see all commands
```

## Architecture defaults (config/config.py)

| Component | Key params |
| --- | --- |
| VAE | latent_dim=32, enc_channels=[32,64,128,256], img_size=64 |
| MDN-RNN | hidden_size=256, n_gaussians=5, num_layers=1 |
| Controller | input=latent_dim+hidden_size=288, output=3, n_params=867 |
| CMA-ES | pop_size=16, n_generations=50, sigma0=0.1, n_workers=4 |
| Env | frame_skip=4, max_steps=1000, img_size=64 |

## Training pipeline dependencies

```text
collect  →  train-vae  →  (auto-encodes rollouts)  →  train-rnn  →  train-ctrl
```

- train-vae automatically encodes all rollouts to z after finishing
- train-rnn reads `*_encoded.npz` files, fails if VAE hasn't run yet
- train-ctrl evaluates in real env, does NOT use rollout data

## Known gotchas

1. **Controller outputs must all use tanh, not sigmoid for gas/brake.**
   Symptom: front wheels visibly steer right but the car goes straight.
   Root cause: CarRacing's Box2D physics gives each tire a fixed friction budget. Any brake input —
   even brake=0.176 — consumes that budget for longitudinal force, leaving zero lateral grip for
   steering. Verified: `STEER=0.675 GAS=0.758 BRAKE=0.0` → car turns; `STEER=0.675 GAS=0.758 BRAKE=0.176` → car goes straight.
   Why sigmoid causes this: sigmoid(0)=0.5, so random CMA-ES init immediately produces gas≈0.5 and
   brake≈0.5 on every step. The controller never learns to release the brake because pushing brake to
   near 0 requires very negative weights, which 5–50 generations never discovers.
   Fix: use tanh for all 3 outputs (as in the paper). tanh(0)=0 so init is gas=0, brake=0 — neutral
   start. Negative tanh outputs are clipped to 0 by the env. CMA-ES only needs to learn to push gas
   positive, which it finds quickly. Changing this activation requires retraining from scratch.
