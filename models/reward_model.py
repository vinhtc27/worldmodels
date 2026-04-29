"""
Reward Model R: (z_t, h_t, a_t) → r_t

Used during dream-mode controller training to provide a reward signal
without running the real environment. Trained on encoded rollout data
where h_t is the RNN hidden state before processing step t.
"""
import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, action_dim: int):
        super().__init__()
        in_dim = latent_dim + hidden_size + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]
            h: [B, hidden_size]
            a: [B, action_dim]
        Returns:
            r: [B] predicted per-step reward
        """
        return self.net(torch.cat([z, h, a], dim=-1)).squeeze(-1)
