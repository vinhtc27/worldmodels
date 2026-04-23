"""
Train the MDN-RNN (Memory Model) on pre-encoded latent sequences.

Requires VAE training to have completed first — train_vae.py automatically
encodes all rollouts to *_encoded.npz files at the end of training.

Training objective: minimise the negative log-likelihood of the true
next latent z_{t+1} under the MDN-RNN's predicted mixture distribution.

At each step the RNN receives [z_t ; a_t] and must predict the distribution
of z_{t+1}. Backpropagation Through Time (BPTT) is truncated to seq_len
steps to keep memory bounded.
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from models import MDNRNN
from data import LatentSequenceDataset
from utils import set_seed, save_checkpoint, load_checkpoint, MetricLogger, print_model_summary

console = Console()


def get_encoded_paths(cfg, tag: str = "train"):
    p = Path(cfg.paths.data_dir) / tag
    return sorted(p.glob("*_encoded.npz"))


def train_rnn(cfg, resume: bool = False):
    set_seed(cfg.seed)
    device = cfg.get_device()
    console.print(f"[bold]Training MDN-RNN[/] on [cyan]{device}[/]")

    # ── Data ─────────────────────────────────────────────────────────────────
    paths = [str(p) for p in get_encoded_paths(cfg, "train")]
    if not paths:
        console.print("[red]No encoded rollouts found. Train VAE first (it encodes rollouts).")
        return None

    dataset  = LatentSequenceDataset(paths, seq_len=cfg.rnn.seq_len)
    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    num_workers = min((os.cpu_count() or 4) // 2, 4)  # data loading is not the bottleneck; more workers waste RAM and contend with training
    train_loader = DataLoader(train_ds, batch_size=cfg.rnn.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.rnn.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)

    console.print(f"  Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    rnn       = MDNRNN(cfg.rnn).to(device)
    print_model_summary("MDN-RNN", rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=cfg.rnn.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.rnn.epochs)

    start_epoch = 1
    best_val    = float("inf")
    if resume and Path(cfg.paths.rnn_checkpoint).exists():
        ckpt        = load_checkpoint(cfg.paths.rnn_checkpoint, device)
        rnn.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", best_val)
        console.print(f"[yellow]Resumed from epoch {start_epoch - 1}")

    logger = MetricLogger("rnn", cfg.paths.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.rnn.epochs + 1):
        rnn.train()
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Epoch {epoch}/{cfg.rnn.epochs}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("train", total=len(train_loader))
            for z_seq, a_seq in train_loader:
                z_seq = z_seq.to(device)  # [B, T+1, D]
                a_seq = a_seq.to(device)  # [B, T+1, A]

                # Next-step prediction: feed z_t and a_t, predict z_{t+1}
                z_in   = z_seq[:, :-1, :]  # [B, T, D] — inputs
                z_next = z_seq[:, 1:,  :]  # [B, T, D] — targets
                a_in   = a_seq[:, :-1, :]  # [B, T, A]

                log_pi, mu, sigma, _ = rnn(z_in, a_in)
                loss = rnn.mdn_loss(z_next, log_pi, mu, sigma)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents exploding gradients in BPTT
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), cfg.rnn.grad_clip)
                optimizer.step()

                logger.update(loss=loss)
                progress.advance(task)

        # ── Validation ────────────────────────────────────────────────────────
        rnn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_seq, a_seq in val_loader:
                z_seq  = z_seq.to(device)
                a_seq  = a_seq.to(device)
                z_in   = z_seq[:, :-1, :]
                z_next = z_seq[:, 1:,  :]
                a_in   = a_seq[:, :-1, :]
                log_pi, mu, sigma, _ = rnn(z_in, a_in)
                val_loss += rnn.mdn_loss(z_next, log_pi, mu, sigma).item()
        val_loss /= len(val_loader)
        logger.update(val_loss=val_loss)

        row = logger.print_epoch(epoch, cfg.rnn.epochs)
        scheduler.step()

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = {
            "epoch":     epoch,
            "model":     rnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val":  best_val,
        }
        if val_loss < best_val:
            best_val       = val_loss
            ckpt["best_val"] = best_val
            save_checkpoint(ckpt, cfg.paths.rnn_checkpoint)
            console.print(f"  [green]✓ Best RNN saved (val_loss={val_loss:.4f})")

        if epoch % cfg.rnn.save_interval == 0:
            save_checkpoint(ckpt, f"{cfg.paths.checkpoint_dir}/rnn_epoch_{epoch:03d}.pt")

    logger.save()
    console.print("[bold green]MDN-RNN training complete.")
    return rnn
