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
utils/                — helpers.py (checkpoint load/save, MetricLogger)
main.py               — single CLI entry point for everything
Makefile              — run `make help` to see all commands
checkpoint/           — saved model weights (.pt files)
log/                  — training metric histories (JSON, written after every epoch/gen)
data/rollouts/        — collected rollout .npz files
research/             — paper-scale run outputs (override with RESEARCH_DIR=path)
```

## Architecture defaults (config/config.py)

| Component | Key params |
| --- | --- |
| VAE | latent_dim=32, enc_channels=[32,64,128,256], img_size=64, batch_size=256, kl_weight=1.0, kl_tolerance=0.5 |
| MDN-RNN | hidden_size=256, n_gaussians=5, num_layers=1, batch_size=128, grad_clip=1.0, temperature=1.15 |
| Controller | input=latent_dim+hidden_size=288, output=3, n_params=867 |
| CMA-ES | pop_size=16, n_generations=50, sigma0=0.1, n_eval_episodes=4, n_workers=cpu_count(), save_interval=5 |
| Env | frame_skip=4, max_steps=1000, img_size=64, collection_mode="random", n_workers=cpu_count() |

## Training pipeline dependencies

```text
collect  →  train-vae  →  (auto-encodes rollouts)  →  train-rnn  →  train-ctrl
```

- train-vae automatically encodes all rollouts to z after finishing
- train-rnn reads `*_encoded.npz` files, fails if VAE hasn't run yet
- train-ctrl evaluates each CMA-ES candidate in the real CarRacing environment (ground-truth rewards)

## Quick run params (all consistent)

| Command | Rollouts | VAE epochs | RNN epochs | Ctrl gens | Pop size |
| --- | --- | --- | --- | --- | --- |
| `make quick` | 200 | 10 | — | — | — |
| `make full` (`quick --full`) | 200 | 10 | 20 | 50 | 16 |
| `make quick-collect` | 200 | — | — | — | — |
| `make quick-vae` | — | 10 | — | — | — |
| `make quick-rnn` | — | — | 20 | — | — |
| `make quick-ctrl` | — | — | — | 50 | 16 |
| `make quick-ctrl-resume` | — | — | — | 50 | 16 |

- `make quick` and `make quick-collect` both use the default rollout length (`max_steps=1000`) with biased collection so the standalone commands match the stronger `quick --full` preset.

## Research (paper-scale) pipeline

```bash
make research          # alias for research-random (paper method)
make research-random   # pure random policy: 10k rollouts | VAE 1ep | RNN 20ep | CMA-ES 1800gen×pop64×16eval
make research-bias     # biased policy (our custom): same scale, high-gas collection
RESEARCH_DIR=mydir make research   # outputs go to mydir/ instead of research/
```

All research outputs (checkpoints, logs, viz PNGs/GIFs) land under `research/` by default. Override with `RESEARCH_DIR=path`.

## CLI features

- `--base-dir DIR`: global flag on `main.py` redirecting ALL outputs (data, checkpoint, log) under `DIR`. Created automatically. Used by `make research`.
- `--collection-mode random|biased`: override per-run on `collect`. "random" = pure iid (paper default); "biased" = hold 8 steps, high gas.
- `--skip-collect/--skip-vae/--skip-rnn/--skip-ctrl`: skip steps on `all` and `quick --full` commands to reuse existing checkpoints/data.
- `quick --full` uses biased collection by default and a larger budget (200 rollouts, VAE 10 epochs, RNN 20 epochs, CMA-ES 50 gens × pop 16 × 4 eval).
- `--resume` on `train-ctrl`: resumes from most recent periodic gen checkpoint (`controller_gen_NNN.pt`), falls back to `controller_best.pt`. CMA-ES restarts with best params as x0 — not true resume of covariance state.
- Rollout collection is **incremental**: existing `rollout_NNNNN.npz` files are detected and skipped; only missing indices are collected.

## Controller checkpoints

Two checkpoint types saved during training:

- `checkpoint/controller_best.pt` — saved whenever a new best reward is found
- `checkpoint/controller_gen_NNN.pt` — saved every `save_interval=5` generations, always holds the global best params at that point

Resume prefers the highest-numbered `controller_gen_*.pt` (most recent gen) over `controller_best.pt`.

## Frame normalization

Frames from the environment are uint8 [0, 255]. **Always normalize to float32 [0, 1] before passing to the VAE:**

```python
x = torch.from_numpy((frame.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0)
```

This applies in `evaluate.py`, `train_controller.py`, and anywhere `preprocess_frame()` output feeds into a model. The visualization helper `_frame_to_tensor()` in `visualize.py` is the canonical reference.

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

2. **MPS (Apple Silicon) workers run on CPU.**
   CMA-ES uses `ProcessPoolExecutor` — spawned worker processes cannot inherit the Metal GPU context
   from the parent. Workers silently fall back to CPU. This is a macOS/Metal limitation, not a bug.
   The main process uses MPS for VAE/RNN loading; evaluation workers use CPU. Expected behaviour.
