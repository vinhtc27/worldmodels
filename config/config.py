"""
Central configuration for World Models.
Edit any section to adjust architecture or training hyperparameters.
"""
from dataclasses import dataclass, field
from typing import Optional


# ─── Environment ────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    name: str = "CarRacing-v3"
    img_size: int = 64               # resize frames to img_size x img_size
    frame_skip: int = 4              # repeat each action N times (paper uses 4)
    max_steps: int = 1000            # max steps per episode
    n_rollouts: int = 200            # random rollouts to collect for VAE/RNN training
    render_mode: Optional[str] = None  # "human" or None
    # Render window (used during eval --render)
    window_width: int = 500
    window_height: int = 500


# ─── VAE (Vision Model) ──────────────────────────────────────────────────────

@dataclass
class VAEConfig:
    latent_dim: int = 32             # size of z
    img_channels: int = 3            # RGB
    # Encoder conv channels
    enc_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    # Training
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 10
    kl_weight: float = 1.0           # β in β-VAE
    kl_tolerance: float = 0.5        # free bits per dimension
    save_interval: int = 5           # save checkpoint every N epochs


# ─── MDN-RNN (Memory Model) ──────────────────────────────────────────────────

@dataclass
class RNNConfig:
    hidden_size: int = 256           # LSTM hidden size
    num_layers: int = 1              # LSTM layers
    n_gaussians: int = 5             # MDN mixture components
    latent_dim: int = 32             # must match VAEConfig.latent_dim
    action_dim: int = 3              # CarRacing: [steer, gas, brake]
    # Training
    batch_size: int = 32
    seq_len: int = 32                # BPTT sequence length
    lr: float = 1e-3
    epochs: int = 20
    grad_clip: float = 1.0
    temperature: float = 1.15        # sampling temperature
    save_interval: int = 5


# ─── Controller ──────────────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    latent_dim: int = 32             # must match VAEConfig.latent_dim
    hidden_size: int = 256           # must match RNNConfig.hidden_size
    action_dim: int = 3              # must match RNNConfig.action_dim
    # CMA-ES
    pop_size: int = 16               # population size (≥4)
    n_generations: int = 50          # CMA-ES generations
    sigma0: float = 0.1              # initial step size
    n_eval_episodes: int = 4         # episodes per individual evaluation
    n_workers: int = 4               # parallel workers (set 1 to disable)
    # Dream (hallucinate in latent space instead of real env)
    dream: bool = False
    dream_steps: int = 999
    dream_temperature: float = 1.15


# ─── Paths ───────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    data_dir: str = "data/rollouts"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    vae_checkpoint: str = "checkpoints/vae_best.pt"
    rnn_checkpoint: str = "checkpoints/rnn_best.pt"
    controller_checkpoint: str = "checkpoints/controller_best.pt"


# ─── Master Config ────────────────────────────────────────────────────────────

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    seed: int = 42
    device: str = "auto"             # "auto" | "cpu" | "cuda" | "mps"

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
