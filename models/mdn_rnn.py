"""
M Model — MDN-RNN (Ha & Schmidhuber 2018, Section 2.2)

Goal: model the distribution of the next latent state given the current one
and the action taken — p(z_{t+1} | z_t, a_t, h_t).

Why a Mixture Density Network instead of just predicting z_next directly?
  The future is multimodal: at a junction the car could go left OR right.
  A single Gaussian would predict the average (straight into the wall).
  A mixture of K Gaussians can represent multiple distinct futures.

Architecture:
  Input at each step: [z_t ; a_t]  (latent + action concatenated)
  LSTM maintains hidden state h_t across time (the "memory")
  MDN head on top of LSTM output predicts K Gaussian components:
    π  — mixture weights (which future is most likely)
    μ  — means of each Gaussian (where z_next is centred)
    σ  — standard deviations (how uncertain each prediction is)

Temperature (τ) at sampling time:
  τ < 1 → sharper / more deterministic dreams (picks the dominant mode)
  τ > 1 → more random / creative dreams (spreads probability mass)
  τ = 1 → standard sampling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MDNHead(nn.Module):
    """
    Projects LSTM hidden state → mixture parameters (log_π, μ, σ).

    Output layout from a single linear layer, then split:
      [ π_logits (K) | μ (K·D) | σ_raw (K·D) ]

    log_π: log-softmax over K components for numerical stability
    μ:     reshaped to [... , K, D]
    σ:     exp(σ_raw) to ensure positivity
    """
    def __init__(self, in_dim: int, out_dim: int, n_gaussians: int):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.out_dim     = out_dim
        self.fc          = nn.Linear(in_dim, n_gaussians * (2 * out_dim + 1))

    def forward(self, h: torch.Tensor):
        out       = self.fc(h)
        pi_logits = out[..., : self.n_gaussians]
        mu        = out[..., self.n_gaussians : self.n_gaussians * (1 + self.out_dim)]
        sigma_raw = out[..., self.n_gaussians * (1 + self.out_dim) :]

        mu      = mu.view(*mu.shape[:-1], self.n_gaussians, self.out_dim)
        sigma   = torch.exp(sigma_raw.view(*sigma_raw.shape[:-1], self.n_gaussians, self.out_dim))
        log_pi  = F.log_softmax(pi_logits, dim=-1)  # log-space for numerical stability
        return log_pi, mu, sigma


class MDNRNN(nn.Module):
    """
    MDN-RNN (M model).

    Usage:
        rnn = MDNRNN(cfg.rnn)

        # Training — process a full sequence at once:
        log_pi, mu, sigma, h_out = rnn(z_seq, a_seq, h_init)
        loss = rnn.mdn_loss(z_next_seq, log_pi, mu, sigma)

        # Inference — one step at a time (evaluation / dreaming):
        log_pi, mu, sigma, h = rnn.forward_step(z_t, a_t, h)
        z_next = rnn.sample(log_pi, mu, sigma, temperature=1.0)
    """
    def __init__(self, rnn_cfg):
        super().__init__()
        self.latent_dim  = rnn_cfg.latent_dim
        self.hidden_size = rnn_cfg.hidden_size
        self.n_gaussians = rnn_cfg.n_gaussians
        self.num_layers  = rnn_cfg.num_layers
        self.temperature = rnn_cfg.temperature

        # LSTM input: z and a concatenated at each timestep
        input_dim  = rnn_cfg.latent_dim + rnn_cfg.action_dim
        self.lstm  = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_cfg.hidden_size,
            num_layers=rnn_cfg.num_layers,
            batch_first=True,
        )
        self.mdn = MDNHead(rnn_cfg.hidden_size, rnn_cfg.latent_dim, rnn_cfg.n_gaussians)

    def initial_state(self, batch_size: int, device):
        """Zero-initialise (h, c) — used at the start of each episode."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def forward(
        self,
        z: torch.Tensor,      # [B, T, latent_dim]
        a: torch.Tensor,      # [B, T, action_dim]
        state: Optional[Tuple] = None,
    ):
        """Process a full sequence. Returns MDN params and updated LSTM state."""
        x = torch.cat([z, a], dim=-1)          # [B, T, latent+action]
        lstm_out, state_out = self.lstm(x, state)
        log_pi, mu, sigma = self.mdn(lstm_out)
        return log_pi, mu, sigma, state_out

    def forward_step(
        self,
        z: torch.Tensor,       # [B, latent_dim]
        a: torch.Tensor,       # [B, action_dim]
        state: Optional[Tuple] = None,
    ):
        """Single-step forward — adds a time dimension then delegates to forward."""
        return self.forward(z.unsqueeze(1), a.unsqueeze(1), state)

    # ── Loss ─────────────────────────────────────────────────────────────────

    def mdn_loss(
        self,
        z_next: torch.Tensor,  # [B, T, latent_dim]   ground-truth next latent
        log_pi: torch.Tensor,  # [B, T, K]             log mixture weights
        mu: torch.Tensor,      # [B, T, K, latent_dim] component means
        sigma: torch.Tensor,   # [B, T, K, latent_dim] component std devs
    ) -> torch.Tensor:
        """
        Negative log-likelihood of z_next under the predicted mixture.

        For each timestep and each mixture component k:
          log p_k(z) = Σ_d log N(z_d; μ_kd, σ_kd)   (independent Gaussians per dim)

        Then marginalise over components:
          log p(z) = log Σ_k π_k · p_k(z)
                   = logsumexp_k (log π_k + log p_k(z))

        Using logsumexp instead of direct sum avoids numerical underflow.
        """
        z_exp  = z_next.unsqueeze(-2)  # [B, T, 1, D] — broadcast over K
        log_p  = (
            -0.5 * ((z_exp - mu) / sigma).pow(2)
            - sigma.log()
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)                  # sum over D → [B, T, K]
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
        use_mean: bool = False,
    ) -> torch.Tensor:
        """
        Sample z_next from the predicted mixture.

        Temperature τ scales the log-weights before softmax:
          τ < 1 → concentrates mass on the highest-weight component (less random)
          τ > 1 → flattens the distribution (more exploratory dreams)

        use_mean=True → return the mean of the argmax component with no added
        Gaussian noise. Produces sharper decoded frames for visualization because
        stochastic noise in z-space compounds over many dream steps.
        """
        log_pi = log_pi.squeeze(1)  # [B, K]
        mu     = mu.squeeze(1)      # [B, K, D]
        sigma  = sigma.squeeze(1)   # [B, K, D]

        B, K, D = mu.shape

        if use_mean:
            idx = torch.argmax(log_pi, dim=-1)  # [B]
        else:
            pi  = torch.exp(log_pi / temperature)
            pi /= pi.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(pi, 1).squeeze(-1)  # [B]

        idx_exp   = idx.view(B, 1, 1).expand(B, 1, D)
        mu_sel    = mu.gather(1, idx_exp).squeeze(1)     # [B, D]
        sigma_sel = sigma.gather(1, idx_exp).squeeze(1)  # [B, D]

        if use_mean:
            return mu_sel

        noise = torch.randn_like(mu_sel)
        return mu_sel + sigma_sel * noise
