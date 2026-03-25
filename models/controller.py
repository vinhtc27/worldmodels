"""
C Model — Linear Controller
Maps (z_t ⊕ h_t) → action_t using a single linear layer.
Trained with CMA-ES (no gradient).
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class Controller(nn.Module):
    """
    Simple linear controller: a = tanh(W @ [z; h] + b)

    Weight vector is flattened for CMA-ES parameter passing.
    """
    def __init__(self, ctrl_cfg):
        super().__init__()
        in_dim = ctrl_cfg.latent_dim + ctrl_cfg.hidden_size
        out_dim = ctrl_cfg.action_dim
        self.fc = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_params = (in_dim + 1) * out_dim

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]
            h: [B, hidden_size]  — LSTM hidden state (first layer)
        Returns:
            action: [B, action_dim]
              steer ∈ [-1, 1]  (tanh)
              gas   ∈ [ 0, 1]  (sigmoid)
              brake ∈ [ 0, 1]  (sigmoid)
        """
        x = torch.cat([z, h], dim=-1)
        out = self.fc(x)
        steer = torch.tanh(out[..., 0:1])
        gas   = torch.sigmoid(out[..., 1:2])
        brake = torch.sigmoid(out[..., 2:3])
        return torch.cat([steer, gas, brake], dim=-1)

    # ── CMA-ES parameter interface ────────────────────────────────────────────

    def get_params(self) -> np.ndarray:
        """Return flat numpy parameter vector."""
        w = self.fc.weight.data.cpu().numpy().flatten()
        b = self.fc.bias.data.cpu().numpy()
        return np.concatenate([w, b])

    def set_params(self, params: np.ndarray):
        """Set weights from flat numpy vector."""
        device = self.fc.weight.device
        w_size = self.out_dim * self.in_dim
        w = params[:w_size].reshape(self.out_dim, self.in_dim)
        b = params[w_size:]
        self.fc.weight.data = torch.FloatTensor(w).to(device)
        self.fc.bias.data = torch.FloatTensor(b).to(device)

    @staticmethod
    def params_from_solution(solution: np.ndarray) -> np.ndarray:
        return solution


class ActionClip(nn.Module):
    """Optional post-processing: clip CarRacing actions to valid ranges."""
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # steer ∈ [-1, 1], gas ∈ [0, 1], brake ∈ [0, 1]
        steer = a[..., 0:1].clamp(-1, 1)
        gas   = a[..., 1:2].clamp(0, 1)
        brake = a[..., 2:3].clamp(0, 1)
        return torch.cat([steer, gas, brake], dim=-1)
