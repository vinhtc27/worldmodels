"""
V Model — Variational Autoencoder (Ha & Schmidhuber 2018, Section 2.1)

Goal: compress each 64×64×3 frame into a compact latent vector z ∈ R^latent_dim
so the controller never has to reason about raw pixels.

Training objective — ELBO (Evidence Lower BOund):
    ELBO = E[log p(x|z)]  −  β · KL(q(z|x) ‖ p(z))
           ─────────────     ──────────────────────
           reconstruction     regularisation
           (how well we         (keep z close to
            reconstruct x)       standard normal)

The KL term acts as a bottleneck: it forces the encoder to compress
meaningfully rather than just memorising every pixel.

β-VAE: scaling KL by β > 1 encourages more disentangled representations.
       Here β = kl_weight (default 1.0, i.e. standard VAE).

Free bits (kl_tolerance): clamp KL per dimension from below so the model
cannot "cheat" by setting all KL to 0 (posterior collapse).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    Maps image x → (μ, log σ²) — the parameters of q(z|x).

    Architecture: 4 stride-2 conv layers halve spatial dims each time:
        64×64 → 32×32 → 16×16 → 8×8 → 4×4
    Then flatten → two linear heads for μ and log σ².
    """
    def __init__(self, img_channels: int, enc_channels: list, latent_dim: int):
        super().__init__()
        layers = []
        in_ch = img_channels
        for out_ch in enc_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        # After 4× stride-2 convs on 64×64 input: spatial size = 64 / 2^4 = 4
        self.flat_dim = enc_channels[-1] * 4 * 4
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    Maps latent z → reconstructed image x̂.

    Mirror of the encoder: linear projection back to 4×4 feature map,
    then 4 stride-2 transposed convolutions upsample back to 64×64.
    Final Sigmoid squashes output to [0, 1] to match normalised input.
    """
    def __init__(self, img_channels: int, enc_channels: list, latent_dim: int):
        super().__init__()
        rev = list(reversed(enc_channels))
        self.flat_dim  = enc_channels[-1] * 4 * 4
        self.fc        = nn.Linear(latent_dim, self.flat_dim)
        self.first_ch  = enc_channels[-1]

        layers = []
        in_ch = self.first_ch
        for out_ch in rev[1:]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers += [
            nn.ConvTranspose2d(in_ch, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.first_ch, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """
    Variational Autoencoder (V model).

    Usage:
        vae = VAE(cfg.vae)
        mu, logvar = vae.encode(obs)          # obs: [B, C, H, W] in [0,1]
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        loss, recon_l, kl_l = vae.loss(obs, recon, mu, logvar)
    """
    def __init__(self, vae_cfg):
        super().__init__()
        self.latent_dim   = vae_cfg.latent_dim
        self.encoder      = Encoder(vae_cfg.img_channels, vae_cfg.enc_channels, vae_cfg.latent_dim)
        self.decoder      = Decoder(vae_cfg.img_channels, vae_cfg.enc_channels, vae_cfg.latent_dim)
        self.kl_weight    = vae_cfg.kl_weight     # β in β-VAE
        self.kl_tolerance = vae_cfg.kl_tolerance  # free bits per latent dimension

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ·ε,  ε ~ N(0, I)

        Sampling z directly would block gradient flow. Instead we sample
        ε from a fixed distribution and scale/shift it — making z a
        deterministic function of (μ, σ, ε) so gradients flow through μ and σ.
        At eval time we return μ directly (no noise).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO loss = reconstruction loss + β · KL loss

        Reconstruction: MSE summed over pixels, averaged over batch.
        KL per dimension: -0.5 · (1 + log σ² - μ² - σ²)
            = KL(N(μ,σ²) ‖ N(0,1)) in closed form.
        Free bits: clamp KL per dim from below at kl_tolerance so the model
            must use at least that much capacity per dimension, preventing
            posterior collapse (where the encoder ignores x entirely).
        """
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = kl_per_dim.clamp(min=self.kl_tolerance)
        kl_loss    = kl_per_dim.sum(dim=1).mean()
        total      = recon_loss + self.kl_weight * kl_loss
        return total, recon_loss, kl_loss

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return deterministic μ (no sampling) for inference.
        Assumes model is already in eval mode and torch.no_grad() is active.
        Using μ directly (rather than a sample) gives stable, reproducible
        latents — important for the controller to see consistent inputs.
        """
        mu, _ = self.encode(x)
        return mu
