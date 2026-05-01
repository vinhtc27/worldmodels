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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from models import MDNRNN
from data import LatentSequenceDataset
from utils import set_seed, save_checkpoint, load_checkpoint, unwrap_state_dict, MetricLogger, print_model_summary

console = Console()


def get_encoded_paths(cfg, tag: str = "train"):
    p = Path(cfg.paths.data_dir) / tag
    return sorted(p.glob("*_encoded.npz"))


def train_rnn(cfg, resume: bool = False):
    set_seed(cfg.seed)
    device = cfg.get_device()
    console.print(f"[bold]Training MDN-RNN[/] on [cyan]{device}[/]")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
        torch.backends.cudnn.benchmark = True
    use_amp   = device in ("cuda", "mps")
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    scaler    = torch.amp.GradScaler("cuda") if device == "cuda" else None

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
    num_workers = min(os.cpu_count() or 4, 8)
    # Only pin memory when using CUDA; MPS doesn't support pinned memory and CPU
    # workers can be heavier on some macOS setups. Keep behavior consistent with train_vae.
    pin_memory = True if device == "cuda" else False
    train_loader = DataLoader(train_ds, batch_size=cfg.rnn.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0) if pin_memory else False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.rnn.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0) if pin_memory else False)

    console.print(f"  Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    rnn = MDNRNN(cfg.rnn).to(device)
    if device == "cuda":
        rnn = torch.compile(rnn)
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
    n_epochs = cfg.rnn.epochs

    # ── Training loop ─────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]MDN-RNN"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("epochs", total=n_epochs - start_epoch + 1, status=f"Epoch {start_epoch}/{n_epochs}")
        batch_task = progress.add_task("batches", total=len(train_loader), status="")

        for epoch in range(start_epoch, n_epochs + 1):
            progress.update(epoch_task, status=f"Epoch {epoch}/{n_epochs}")
            progress.reset(batch_task, total=len(train_loader))
            rnn.train()
            for z_seq, a_seq in train_loader:
                z_seq = z_seq.to(device, non_blocking=True)  # [B, T+1, D]
                a_seq = a_seq.to(device, non_blocking=True)  # [B, T+1, A]

                z_in   = z_seq[:, :-1, :]
                z_next = z_seq[:, 1:,  :]
                a_in   = a_seq[:, :-1, :]

                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    log_pi, mu, sigma, _ = rnn(z_in, a_in)
                    loss = rnn.mdn_loss(z_next, log_pi, mu, sigma)

                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(rnn.parameters(), cfg.rnn.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(rnn.parameters(), cfg.rnn.grad_clip)
                    optimizer.step()

                logger.update(loss=loss)
                progress.advance(batch_task)

            # ── Validation ────────────────────────────────────────────────────
            rnn.eval()
            val_loss = 0.0
            with torch.no_grad():
                for z_seq, a_seq in val_loader:
                    z_seq  = z_seq.to(device, non_blocking=True)
                    a_seq  = a_seq.to(device, non_blocking=True)
                    z_in   = z_seq[:, :-1, :]
                    z_next = z_seq[:, 1:,  :]
                    a_in   = a_seq[:, :-1, :]
                    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                        log_pi, mu, sigma, _ = rnn(z_in, a_in)
                        val_loss += rnn.mdn_loss(z_next, log_pi, mu, sigma).item()
            val_loss /= len(val_loader)
            logger.update(val_loss=val_loss)

            logger.print_epoch(epoch, n_epochs)
            scheduler.step()

            # ── Checkpoint ────────────────────────────────────────────────────
            ckpt = {
                "epoch":     epoch,
                "model":     unwrap_state_dict(rnn),
                "optimizer": optimizer.state_dict(),
                "best_val":  best_val,
            }
            if val_loss < best_val:
                best_val         = val_loss
                ckpt["best_val"] = best_val
                save_checkpoint(ckpt, cfg.paths.rnn_checkpoint)
                console.print(f"  [green]✓ Best RNN saved (val_loss={val_loss:.4f})")

            if epoch % cfg.rnn.save_interval == 0:
                save_checkpoint(ckpt, f"{cfg.paths.checkpoint_dir}/rnn_epoch_{epoch:03d}.pt")

            progress.update(epoch_task, status=f"Epoch {epoch}/{n_epochs} — val_loss={val_loss:.4f}")
            progress.advance(epoch_task)

    logger.save()
    console.print("[bold green]MDN-RNN training complete.")
    return rnn
