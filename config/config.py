"""
Central configuration for World Models.
Edit any section to adjust architecture or training hyperparameters.
All components read from a single Config instance so values stay in sync.
"""
from dataclasses import dataclass, field
from typing import Optional


# ─── Environment ────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    name: str       = "CarRacing-v3"
    img_size: int   = 64     # resize frames to img_size × img_size before encoding
    frame_skip: int = 4      # repeat each action N physics steps (paper uses 4)
                             # reduces decision frequency, speeds up training
    max_steps: int  = 1000   # episode cutoff (1000 steps × frame_skip=4 = 4000 frames)
    n_rollouts: int = 200    # random rollouts to collect for VAE/RNN training
    # Render window size (used during eval --render)
    window_width:  int = 500
    window_height: int = 500
    render_mode: str = "rgb_array"  # default for gym.make; override as needed


# ─── VAE (Vision Model) ──────────────────────────────────────────────────────

@dataclass
class VAEConfig:
    latent_dim:    int   = 32    # dimension of z — paper uses 32
    img_channels:  int   = 3     # RGB
    # Encoder conv channels — each doubles filters, halves spatial dims
    enc_channels:  list  = field(default_factory=lambda: [32, 64, 128, 256])
    # Training
    batch_size:    int   = 64
    lr:            float = 1e-4
    epochs:        int   = 10
    kl_weight:     float = 1.0   # β in β-VAE (β=1 → standard VAE)
    kl_tolerance:  float = 0.5   # free bits: minimum KL per latent dimension,
                                 # prevents posterior collapse where encoder ignores input
    save_interval: int   = 5     # save periodic checkpoint every N epochs


# ─── MDN-RNN (Memory Model) ──────────────────────────────────────────────────

@dataclass
class RNNConfig:
    hidden_size:   int   = 256   # LSTM hidden state size — the "memory" capacity
    num_layers:    int   = 1     # stacked LSTM layers (paper uses 1)
    n_gaussians:   int   = 5     # MDN mixture components K — more = richer distributions
                                 # but slower and harder to train
    latent_dim:    int   = 32    # must match VAEConfig.latent_dim
    action_dim:    int   = 3     # CarRacing: [steer, gas, brake]
    # Training
    batch_size:    int   = 32
    seq_len:       int   = 32    # BPTT truncation length
    lr:            float = 1e-3
    epochs:        int   = 20
    grad_clip:     float = 1.0   # gradient clipping threshold for BPTT stability
    temperature:   float = 1.15  # sampling temperature for dream/viz (>1 = more random)
    save_interval: int   = 5


# ─── Controller ──────────────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    latent_dim:       int   = 32    # must match VAEConfig.latent_dim
    hidden_size:      int   = 256   # must match RNNConfig.hidden_size
    action_dim:       int   = 3     # must match RNNConfig.action_dim
    # CMA-ES
    pop_size:         int   = 16    # population size (paper uses 64; ≥4 required)
    n_generations:    int   = 50    # generations to run (paper uses 200+)
    sigma0:           float = 0.1   # initial search radius in parameter space
    n_eval_episodes:  int   = 4     # episodes per candidate (more = less noisy fitness)
    n_workers:        int   = 4     # parallel worker processes (set 1 to disable)


# ─── Paths ───────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    data_dir:               str = "data/rollouts"
    checkpoint_dir:         str = "checkpoints"
    log_dir:                str = "logs"
    vae_checkpoint:         str = "checkpoints/vae_best.pt"
    rnn_checkpoint:         str = "checkpoints/rnn_best.pt"
    controller_checkpoint:  str = "checkpoints/controller_best.pt"


# ─── Master Config ────────────────────────────────────────────────────────────

@dataclass
class Config:
    env:        EnvConfig        = field(default_factory=EnvConfig)
    vae:        VAEConfig        = field(default_factory=VAEConfig)
    rnn:        RNNConfig        = field(default_factory=RNNConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    paths:      PathConfig       = field(default_factory=PathConfig)
    seed:       int              = 42
    device:     str              = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    def get_device(self):
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device


# Singleton default config — import and mutate as needed
cfg = Config()
