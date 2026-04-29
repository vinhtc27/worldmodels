"""
Train reward model R: (z_t, h_t, a_t) → r_t on encoded rollout data.

For each rollout the frozen RNN is run forward once to produce h_t at every step.
h_t is the hidden state *before* processing step t (matching dream-rollout alignment).
MSE loss against the stored per-step rewards.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from models import MDNRNN, RewardModel
from utils import set_seed, save_checkpoint, load_checkpoint

console = Console()


def _encoded_paths(cfg, tag: str = "train"):
    p = Path(cfg.paths.data_dir) / tag
    return sorted(p.glob("*_encoded.npz"))


def train_reward_model(cfg, rnn: MDNRNN = None) -> RewardModel:
    set_seed(cfg.seed)
    device = cfg.get_device()
    console.print(f"[bold]Training Reward Model[/] on [cyan]{device}[/]")

    if rnn is None:
        ckpt = load_checkpoint(cfg.paths.rnn_checkpoint, device)
        rnn = MDNRNN(cfg.rnn).to(device).eval()
        rnn.load_state_dict(ckpt["model"])
    else:
        rnn = rnn.to(device).eval()

    paths = _encoded_paths(cfg)
    if not paths:
        raise FileNotFoundError(
            f"No encoded rollouts in {cfg.paths.data_dir}/train — run train-vae first."
        )
    console.print(f"  Processing {len(paths)} encoded rollouts...")

    z_all, h_all, a_all, r_all = [], [], [], []

    with torch.no_grad():
        for p in paths:
            d     = np.load(p)
            z_seq = torch.from_numpy(d["z"].astype(np.float32)).to(device)        # [T, latent_dim]
            a_seq = torch.from_numpy(d["actions"].astype(np.float32)).to(device)  # [T, action_dim]
            r_seq = torch.from_numpy(d["rewards"].astype(np.float32))             # [T]

            # One LSTM forward pass over the full sequence
            x = torch.cat([z_seq.unsqueeze(0), a_seq.unsqueeze(0)], dim=-1)  # [1, T, z+a]
            h0 = rnn.initial_state(1, device)
            h_out, _ = rnn.lstm(x, h0)  # [1, T, hidden_size]
            h_out = h_out.squeeze(0)     # [T, hidden_size]

            # h_prev[t] = hidden state BEFORE step t; h_out[t] is AFTER step t
            h_zero = torch.zeros(1, cfg.rnn.hidden_size, device=device)
            h_prev = torch.cat([h_zero, h_out[:-1]], dim=0)  # [T, hidden_size]

            z_all.append(z_seq.cpu())
            h_all.append(h_prev.cpu())
            a_all.append(a_seq.cpu())
            r_all.append(r_seq)

    Z = torch.cat(z_all)
    H = torch.cat(h_all)
    A = torch.cat(a_all)
    R = torch.cat(r_all)
    console.print(f"  Collected {len(Z):,} reward samples  (r range [{R.min():.2f}, {R.max():.2f}])")

    dataset   = TensorDataset(Z, H, A, R)
    val_n     = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n])
    train_dl  = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_dl    = DataLoader(val_ds,   batch_size=512)

    model     = RewardModel(cfg.rnn.latent_dim, cfg.rnn.hidden_size, cfg.rnn.action_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val  = float("inf")
    # Use configured epochs from controller config if available, else default to 25
    n_epochs = getattr(cfg.controller, "reward_model_epochs", 25)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]RewardModel"),
        BarColumn(),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("train", total=n_epochs, status="")

        for epoch in range(n_epochs):
            model.train()
            for z_b, h_b, a_b, r_b in train_dl:
                z_b, h_b, a_b, r_b = z_b.to(device), h_b.to(device), a_b.to(device), r_b.to(device)
                loss = nn.functional.mse_loss(model(z_b, h_b, a_b), r_b)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    nn.functional.mse_loss(
                        model(z_b.to(device), h_b.to(device), a_b.to(device)),
                        r_b.to(device),
                    ).item() * len(z_b)
                    for z_b, h_b, a_b, r_b in val_dl
                ) / len(val_ds)

            progress.update(task, advance=1,
                            status=f"epoch {epoch+1}/{n_epochs}  val_mse={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint({"model": model.state_dict()}, cfg.paths.reward_model_checkpoint)

    console.print(f"[green]✓ Reward model saved  (best val_mse={best_val:.4f})")
    ckpt = load_checkpoint(cfg.paths.reward_model_checkpoint, device)
    model.load_state_dict(ckpt["model"])
    return model.eval()
