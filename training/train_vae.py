"""
Train the VAE (Vision Model).
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from models import VAE
from data import FrameDataset, get_rollout_paths
from utils import set_seed, save_checkpoint, load_checkpoint, MetricLogger, print_model_summary

console = Console()


def encode_and_save_rollouts(vae: VAE, cfg, tag: str = "train"):
    """
    After VAE training: encode all rollouts to latent z and save alongside originals.
    This speeds up MDN-RNN training significantly.
    """
    from data import get_rollout_paths
    import numpy as np

    device = cfg.get_device()
    vae = vae.to(device).eval()
    use_amp  = device in ("cuda", "mps")
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    paths = get_rollout_paths(cfg, tag)
    console.print(f"[cyan]Encoding {len(paths)} rollouts to latent space...")

    for p in paths:
        d = np.load(p)
        obs = d["obs"]  # [T, H, W, C]
        T = len(obs)
        z_list = []
        batch_size = 512
        for start in range(0, T, batch_size):
            batch = obs[start : start + batch_size]
            x = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                mu, _ = vae.encode(x)
            z_list.append(mu.float().cpu().numpy())  # store as float32
        z = np.concatenate(z_list, axis=0)
        out_path = Path(p).parent / (Path(p).stem + "_encoded.npz")
        np.savez(out_path, z=z, actions=d["actions"], rewards=d["rewards"], dones=d["dones"])

    console.print(f"[green]Encoded rollouts saved.")


def train_vae(cfg, resume: bool = False):
    set_seed(cfg.seed)
    device = cfg.get_device()
    console.print(f"[bold]Training VAE[/] on [cyan]{device}[/]")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
        torch.backends.cudnn.benchmark = True  # auto-tune conv algorithms for fixed 64×64 input
    use_amp  = device in ("cuda", "mps")
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    scaler   = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ── Data ─────────────────────────────────────────────────────────────────
    paths = get_rollout_paths(cfg, "train")
    if not paths:
        console.print("[red]No rollouts found. Run data collection first.")
        return None

    dataset = FrameDataset(paths)
    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    num_workers = min(os.cpu_count() or 4, 8)
    train_loader = DataLoader(train_ds, batch_size=cfg.vae.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.vae.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)

    # ── Model ─────────────────────────────────────────────────────────────────
    vae = VAE(cfg.vae).to(device)
    if device == "cuda":
        vae = torch.compile(vae)  # Triton kernel fusion; first epoch slower while compiling
    print_model_summary("VAE", vae)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.vae.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.vae.epochs)

    start_epoch = 1
    best_val = float("inf")
    if resume and Path(cfg.paths.vae_checkpoint).exists():
        ckpt = load_checkpoint(cfg.paths.vae_checkpoint, device)
        vae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        console.print(f"[yellow]Resumed from epoch {start_epoch - 1}")

    logger = MetricLogger("vae", cfg.paths.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.vae.epochs + 1):
        vae.train()
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Epoch {epoch}/{cfg.vae.epochs}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("train", total=len(train_loader))
            for x in train_loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    recon, mu, logvar, _ = vae(x)
                    loss, recon_l, kl_l = vae.loss(x, recon, mu, logvar)
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                logger.update(loss=loss, recon=recon_l, kl=kl_l)
                progress.advance(task)

        # ── Validation ────────────────────────────────────────────────────────
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    recon, mu, logvar, _ = vae(x)
                    l, _, _ = vae.loss(x, recon, mu, logvar)
                val_loss += l.item()
        val_loss /= len(val_loader)
        logger.update(val_loss=val_loss)

        row = logger.print_epoch(epoch, cfg.vae.epochs)
        scheduler.step()

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model": vae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
        }
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            save_checkpoint(ckpt, cfg.paths.vae_checkpoint)
            console.print(f"  [green]✓ Best VAE saved (val_loss={val_loss:.4f})")

        if epoch % cfg.vae.save_interval == 0:
            save_checkpoint(ckpt, f"{cfg.paths.checkpoint_dir}/vae_epoch_{epoch:03d}.pt")

    logger.save()
    console.print("[bold green]VAE training complete.")

    # Encode rollouts for RNN training
    encode_and_save_rollouts(vae, cfg, "train")
    return vae
