"""
M Model — MDN-RNN (Mixed Density Network + LSTM)
Models p(z_{t+1} | z_t, a_t, h_t) as a mixture of Gaussians.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MDNHead(nn.Module):
    """Outputs mixture parameters from an input feature vector."""
    def __init__(self, in_dim: int, out_dim: int, n_gaussians: int):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, n_gaussians * (2 * out_dim + 1))

    def forward(self, h: torch.Tensor):
        # h: [..., in_dim]
        out = self.fc(h)
        pi_logits = out[..., : self.n_gaussians]
        mu = out[..., self.n_gaussians : self.n_gaussians * (1 + self.out_dim)]
        sigma_raw = out[..., self.n_gaussians * (1 + self.out_dim) :]

        mu = mu.view(*mu.shape[:-1], self.n_gaussians, self.out_dim)
        sigma = torch.exp(sigma_raw.view(*sigma_raw.shape[:-1], self.n_gaussians, self.out_dim))
        log_pi = F.log_softmax(pi_logits, dim=-1)   # [... , K]
        return log_pi, mu, sigma


class MDNRNN(nn.Module):
    """
    MDN-RNN (M model).

    Usage:
        rnn = MDNRNN(cfg.rnn)
        # Training step:
        log_pi, mu, sigma, h_out = rnn(z_seq, a_seq, h_init)
        loss = rnn.mdn_loss(z_next_seq, log_pi, mu, sigma)

        # Single-step forward (inference):
        log_pi, mu, sigma, h = rnn.forward_step(z_t, a_t, h)
        z_next = rnn.sample(log_pi, mu, sigma, temperature)
    """
    def __init__(self, rnn_cfg):
        super().__init__()
        self.latent_dim = rnn_cfg.latent_dim
        self.hidden_size = rnn_cfg.hidden_size
        self.n_gaussians = rnn_cfg.n_gaussians
        self.num_layers = rnn_cfg.num_layers
        self.temperature = rnn_cfg.temperature

        input_dim = rnn_cfg.latent_dim + rnn_cfg.action_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_cfg.hidden_size,
            num_layers=rnn_cfg.num_layers,
            batch_first=True,
        )
        self.mdn = MDNHead(rnn_cfg.hidden_size, rnn_cfg.latent_dim, rnn_cfg.n_gaussians)

    def initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def forward(
        self,
        z: torch.Tensor,      # [B, T, latent_dim]
        a: torch.Tensor,      # [B, T, action_dim]
        state: Optional[Tuple] = None,
    ):
        x = torch.cat([z, a], dim=-1)          # [B, T, latent+action]
        lstm_out, state_out = self.lstm(x, state)  # [B, T, H]
        log_pi, mu, sigma = self.mdn(lstm_out)
        return log_pi, mu, sigma, state_out

    def forward_step(
        self,
        z: torch.Tensor,       # [B, latent_dim]
        a: torch.Tensor,       # [B, action_dim]
        state: Optional[Tuple] = None,
    ):
        return self.forward(z.unsqueeze(1), a.unsqueeze(1), state)

    # ── Loss ─────────────────────────────────────────────────────────────────

    def mdn_loss(
        self,
        z_next: torch.Tensor,  # [B, T, latent_dim]
        log_pi: torch.Tensor,  # [B, T, K]
        mu: torch.Tensor,      # [B, T, K, latent_dim]
        sigma: torch.Tensor,   # [B, T, K, latent_dim]
    ) -> torch.Tensor:
        # Expand target for mixture
        z_exp = z_next.unsqueeze(-2)  # [B, T, 1, D]
        # Log-prob under each Gaussian component
        log_p = (
            -0.5 * ((z_exp - mu) / sigma).pow(2)
            - sigma.log()
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)  # [B, T, K]
        # Log-sum-exp over mixture
        log_prob = torch.logsumexp(log_pi + log_p, dim=-1)  # [B, T]
        return -log_prob.mean()

    # ── Sampling ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        log_pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample z_next from the MDN output."""
        # log_pi: [B, 1, K], mu/sigma: [B, 1, K, D]
        log_pi = log_pi.squeeze(1)  # [B, K]
        mu = mu.squeeze(1)          # [B, K, D]
        sigma = sigma.squeeze(1)    # [B, K, D]

        # Select mixture component
        pi = torch.exp(log_pi / temperature)
        pi /= pi.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(pi, 1).squeeze(-1)  # [B]

        # Gather selected component
        B, K, D = mu.shape
        idx_exp = idx.view(B, 1, 1).expand(B, 1, D)
        mu_sel = mu.gather(1, idx_exp).squeeze(1)       # [B, D]
        sigma_sel = sigma.gather(1, idx_exp).squeeze(1) # [B, D]

        return mu_sel + sigma_sel * torch.randn_like(mu_sel) * temperature
