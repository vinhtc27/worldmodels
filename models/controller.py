"""
C Model — Linear Controller (Ha & Schmidhuber 2018, Section 2.3)

Goal: map the current world state (z_t, h_t) → action a_t.

Why so simple (just one linear layer)?
  The VAE and MDN-RNN already do the heavy lifting of understanding the world.
  The controller only needs to learn WHAT TO DO given a good representation,
  not how to interpret raw pixels. A linear layer is enough and has far
  fewer parameters — making gradient-free optimisation (CMA-ES) tractable.

Why CMA-ES instead of gradient descent?
  The reward signal from the environment is not differentiable — we can't
  backprop through a physics simulator and a controller jointly. CMA-ES
  treats the controller as a black box, evaluating full episodes and
  evolving a population of weight vectors toward higher reward.

Action outputs — all through tanh:
  steer ∈ [-1, 1]: tanh output used directly
  gas   ∈ [-1, 1]: tanh output, env clips negative values to 0
  brake ∈ [-1, 1]: tanh output, env clips negative values to 0

  IMPORTANT: do not use sigmoid for gas/brake. sigmoid(0) = 0.5 means
  random initialisation produces simultaneous gas=0.5 + brake=0.5.
  In CarRacing's Box2D physics, any brake input consumes lateral grip,
  preventing steering. tanh(0) = 0 gives a neutral init — CMA-ES only
  needs to learn to push gas positive.
"""
import numpy as np
import torch
import torch.nn as nn


class Controller(nn.Module):
    """
    Linear controller: a = tanh(W · [z ; h] + b)

    Parameters are flattened into a 1-D numpy vector for CMA-ES.
    CMA-ES calls set_params() to inject candidate weights, runs an episode,
    and uses the total reward as fitness.
    """
    def __init__(self, ctrl_cfg):
        super().__init__()
        in_dim       = ctrl_cfg.latent_dim + ctrl_cfg.hidden_size
        out_dim      = ctrl_cfg.action_dim
        self.fc      = nn.Linear(in_dim, out_dim)
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.n_params = (in_dim + 1) * out_dim  # weights + bias

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]   — VAE latent (current frame encoding)
            h: [B, hidden_size]  — LSTM hidden state (memory of past)
        Returns:
            action: [B, action_dim]  — all outputs through tanh ∈ [-1, 1]
        """
        x = torch.cat([z, h], dim=-1)
        return torch.tanh(self.fc(x))

    # ── CMA-ES parameter interface ────────────────────────────────────────────

    def get_params(self) -> np.ndarray:
        """Flatten weights + bias into a 1-D numpy vector for CMA-ES."""
        w = self.fc.weight.data.cpu().numpy().flatten()
        b = self.fc.bias.data.cpu().numpy()
        return np.concatenate([w, b])

    def set_params(self, params: np.ndarray):
        """Load a flat numpy vector back into the linear layer."""
        device = self.fc.weight.device
        w_size = self.out_dim * self.in_dim
        w = params[:w_size].reshape(self.out_dim, self.in_dim)
        b = params[w_size:]
        self.fc.weight.data = torch.FloatTensor(w).to(device)
        self.fc.bias.data   = torch.FloatTensor(b).to(device)

