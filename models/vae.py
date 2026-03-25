"""
V Model — Variational Autoencoder
Compresses 64×64×3 frames → latent z ∈ R^latent_dim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
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
        # After 4× stride-2 convs on 64×64: 64 → 32 → 16 → 8 → 4
        self.flat_dim = enc_channels[-1] * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, img_channels: int, enc_channels: list, latent_dim: int):
        super().__init__()
        rev = list(reversed(enc_channels))
        self.flat_dim = enc_channels[-1] * 4 * 4
        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.first_ch = enc_channels[-1]

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
        self.latent_dim = vae_cfg.latent_dim
        self.encoder = Encoder(vae_cfg.img_channels, vae_cfg.enc_channels, vae_cfg.latent_dim)
        self.decoder = Decoder(vae_cfg.img_channels, vae_cfg.enc_channels, vae_cfg.latent_dim)
        self.kl_weight = vae_cfg.kl_weight
        self.kl_tolerance = vae_cfg.kl_tolerance

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        # KL with free bits (per-dimension clamp)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = kl_per_dim.clamp(min=self.kl_tolerance)
        kl_loss = kl_per_dim.sum(dim=1).mean()
        total = recon_loss + self.kl_weight * kl_loss
        return total, recon_loss, kl_loss

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic mu (no sampling) — used at test time."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
