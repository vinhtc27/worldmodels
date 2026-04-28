"""
Interactive visualizations for World Models.

Panels available:
  1. vae_reconstruction  — compare real vs reconstructed frames
  2. latent_space        — 2-D PCA/UMAP of z across rollout
  3. rnn_dream           — hallucinate frames inside the world model (no real env)
  4. training_curves     — plot VAE + RNN loss histories
  5. rollout_replay      — step through a recorded rollout with z/h overlay
  6. latent_walk         — interpolate between two random latent vectors
"""
import sys
import numpy as np
import torch
import matplotlib
# Use native macOS backend (no Tk/SDL dependency); fall back to Agg if headless
if sys.platform == "darwin":
    matplotlib.use("MacOSX")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider, Button
from pathlib import Path
from typing import Optional

from models import VAE, MDNRNN, Controller
from data import preprocess_frame, get_rollout_paths
from utils import load_checkpoint
from rich.console import Console

console = Console()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_all(cfg, device):
    vae  = VAE(cfg.vae).to(device).eval()
    rnn  = MDNRNN(cfg.rnn).to(device).eval()
    ctrl = Controller(cfg.controller).to(device).eval()

    for path, model, label in [
        (cfg.paths.vae_checkpoint,        vae,  "VAE"),
        (cfg.paths.rnn_checkpoint,        rnn,  "MDN-RNN"),
        (cfg.paths.controller_checkpoint, ctrl, "Controller"),
    ]:
        if Path(path).exists():
            model.load_state_dict(load_checkpoint(path, device)["model"])
            console.print(f"  [green]{label} loaded.")
        else:
            console.print(f"  [yellow][WARN] {label} checkpoint not found — using random weights.")
    return vae, rnn, ctrl


def _frame_to_tensor(frame: np.ndarray, size: int, device: str) -> torch.Tensor:
    f = preprocess_frame(frame, size).astype(np.float32) / 255.0
    return torch.from_numpy(f.transpose(2, 0, 1)).unsqueeze(0).to(device)


# ── 1. VAE Reconstruction ─────────────────────────────────────────────────────

def vae_reconstruction(cfg, n_samples: int = 8, save_path: Optional[str] = None):
    """Side-by-side real vs reconstructed frames with latent z histogram."""
    device = cfg.get_device()
    vae = VAE(cfg.vae).to(device).eval()

    if Path(cfg.paths.vae_checkpoint).exists():
        vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, device)["model"])

    paths = get_rollout_paths(cfg, "train")
    if not paths:
        console.print("No rollouts found.")
        return

    # Sample random frames from a random rollout
    d = np.load(paths[np.random.randint(len(paths))])
    obs = d["obs"]
    idxs = np.random.choice(len(obs), n_samples, replace=False)
    frames = obs[idxs]  # [N, H, W, C]

    x = torch.from_numpy((frames.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        recon, mu, logvar, z = vae(x)

    fig = plt.figure(figsize=(n_samples * 1.8 + 1, 5), constrained_layout=True)
    fig.suptitle("VAE Reconstruction  (top: original | bottom: reconstructed)", fontsize=11)
    gs = gridspec.GridSpec(3, n_samples, figure=fig, hspace=0.05)

    for i in range(n_samples):
        # Original
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[i])
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Original", fontsize=8)

        # Reconstruction
        ax2 = fig.add_subplot(gs[1, i])
        rec = recon[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ax2.imshow(rec)
        ax2.axis("off")
        if i == 0:
            ax2.set_ylabel("Recon", fontsize=8)

    # Latent histogram
    ax_hist = fig.add_subplot(gs[2, :])
    z_np = mu.cpu().numpy().flatten()
    ax_hist.hist(z_np, bins=50, color="steelblue", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax_hist.set_title(f"Latent μ distribution  (dim={cfg.vae.latent_dim})", fontsize=9)
    ax_hist.set_xlabel("μ value")
    ax_hist.set_ylabel("count")

    if save_path:
        fig.savefig(save_path, dpi=150)
        console.print(f"Saved → {save_path}")
    plt.show()


# ── 2. Latent Space (PCA) ─────────────────────────────────────────────────────

def latent_space_pca(cfg, n_rollouts: int = 5, save_path: Optional[str] = None):
    """Color-coded PCA scatter of z vectors over time."""
    from sklearn.decomposition import PCA  # soft dependency

    device = cfg.get_device()
    vae = VAE(cfg.vae).to(device).eval()
    if Path(cfg.paths.vae_checkpoint).exists():
        vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, device)["model"])

    paths = get_rollout_paths(cfg, "train")[:n_rollouts]
    all_z, all_t, all_r = [], [], []

    for p in paths:
        d = np.load(p)
        obs = d["obs"]
        rewards = d["rewards"]
        x = torch.from_numpy((obs.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(x)
        all_z.append(mu.cpu().numpy())
        all_t.append(np.arange(len(obs)))
        all_r.append(rewards)

    Z = np.concatenate(all_z)
    T = np.concatenate(all_t)
    R = np.concatenate(all_r)

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)
    var = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Latent Space PCA", fontsize=12)

    sc1 = axes[0].scatter(Z2[:, 0], Z2[:, 1], c=T, cmap="plasma", s=3, alpha=0.6)
    axes[0].set_title(f"Colored by timestep\nPC1 {var[0]:.1f}%  PC2 {var[1]:.1f}%")
    plt.colorbar(sc1, ax=axes[0], label="step")

    sc2 = axes[1].scatter(Z2[:, 0], Z2[:, 1], c=R, cmap="RdYlGn", s=3, alpha=0.6)
    axes[1].set_title("Colored by reward")
    plt.colorbar(sc2, ax=axes[1], label="reward")

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ── 3. RNN Dream (hallucination) ──────────────────────────────────────────────

def rnn_dream(cfg, n_steps: int = 200, temperature: float = 1.0,
              save_gif: Optional[str] = None):
    """
    Let the world model dream: seed with a real frame, then hallucinate
    future frames using the MDN-RNN and the decoder.
    Interactive: slider controls temperature in real-time.
    """
    device = cfg.get_device()
    vae = VAE(cfg.vae).to(device).eval()
    rnn = MDNRNN(cfg.rnn).to(device).eval()

    for path, model in [(cfg.paths.vae_checkpoint, vae), (cfg.paths.rnn_checkpoint, rnn)]:
        if Path(path).exists():
            model.load_state_dict(load_checkpoint(path, device)["model"])

    # Seed from a random real frame
    paths = get_rollout_paths(cfg, "train")
    d = np.load(paths[np.random.randint(len(paths))])
    seed_frame = d["obs"][0]
    x0 = torch.from_numpy((seed_frame.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        z = vae.get_latent(x0)

    # Pre-generate dream frames
    def generate_dream(temp):
        frames_dream = []
        z_cur = z.clone()
        h = rnn.initial_state(1, device)
        for _ in range(n_steps):
            with torch.no_grad():
                img = vae.decode(z_cur).squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                frames_dream.append(img)
                # Random action
                a = torch.FloatTensor(1, cfg.rnn.action_dim).uniform_(-1, 1).to(device)
                log_pi, mu_mix, sigma_mix, h = rnn.forward_step(z_cur, a, h)
                z_cur = rnn.sample(log_pi, mu_mix, sigma_mix, temperature=temp)
        return frames_dream

    dream_frames = generate_dream(temperature)

    # ── Interactive plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)
    fig.suptitle("MDN-RNN Dream  (world model hallucination)", fontsize=11)

    axes[0].set_title("Seed frame (real)")
    axes[0].imshow(seed_frame)
    axes[0].axis("off")

    axes[1].set_title("Dream frame")
    im = axes[1].imshow(dream_frames[0])
    axes[1].axis("off")

    ax_slider = plt.axes([0.15, 0.05, 0.55, 0.03])
    ax_btn    = plt.axes([0.78, 0.04, 0.1, 0.05])
    slider    = Slider(ax_slider, "Step", 0, n_steps - 1, valinit=0, valstep=1)
    btn       = Button(ax_btn, "Regen")

    state = {"frames": dream_frames}

    def update_frame(val):
        idx = int(slider.val)
        im.set_data(state["frames"][idx])
        fig.canvas.draw()

    def regen(event):
        temp = temperature  # could add a temp slider too
        state["frames"] = generate_dream(temp)
        update_frame(slider.val)

    slider.on_changed(update_frame)
    btn.on_clicked(regen)

    if save_gif:
        console.print(f"Saving dream GIF to {save_gif}…")
        writer = PillowWriter(fps=20)
        anim = FuncAnimation(fig, lambda i: im.set_data(dream_frames[i]),
                             frames=n_steps, interval=50)
        anim.save(save_gif, writer=writer)
        console.print(f"Saved → {save_gif}")

    plt.show()


# ── 4. Training Curves ────────────────────────────────────────────────────────

def training_curves(cfg, save_path: Optional[str] = None):
    """Plot VAE + RNN training loss histories from saved JSON logs."""
    import json

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=12)

    # Log filenames match MetricLogger("vae"/"rnn", ...) → vae_history.json / rnn_history.json
    for ax, name, log_name, metrics in [
        (axes[0], "VAE",     "vae", ["loss", "recon", "kl", "val_loss"]),
        (axes[1], "MDN-RNN", "rnn", ["loss", "val_loss"]),
    ]:
        path = Path(cfg.paths.log_dir) / f"{log_name}_history.json"
        if not path.exists():
            ax.text(0.5, 0.5, "No log found", ha="center", va="center")
            ax.set_title(name)
            continue
        with open(path) as f:
            history = json.load(f)
        for m in metrics:
            if m in history:
                ax.plot(history[m], label=m)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ── 5. Rollout Replay with Overlaid Latents ───────────────────────────────────

def rollout_replay(cfg, rollout_idx: int = 0, save_gif: Optional[str] = None):
    """
    Step through a recorded rollout with:
      - top-left: original frame
      - top-right: VAE reconstruction
      - bottom-left: latent z heatmap (timestep vs dimension)
      - bottom-right: reward curve
    Interactive slider to scrub through time.
    """
    device = cfg.get_device()
    vae = VAE(cfg.vae).to(device).eval()
    if Path(cfg.paths.vae_checkpoint).exists():
        vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, device)["model"])

    paths = get_rollout_paths(cfg, "train")
    if not paths:
        console.print("No rollouts found.")
        return
    d = np.load(paths[rollout_idx % len(paths)])
    obs     = d["obs"]      # [T, H, W, C]
    rewards = d["rewards"]  # [T]
    T = len(obs)

    # Encode all frames
    x_all = torch.from_numpy((obs.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        z_all, logvar_all = vae.encode(x_all)
        recon_all = vae.decode(z_all)
    z_np    = z_all.cpu().numpy()        # [T, D]
    recon_np = recon_all.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)  # [T, H, W, C]

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f"Rollout Replay  (rollout #{rollout_idx})", fontsize=11)

    ax_orig  = fig.add_subplot(gs[0, 0])
    ax_recon = fig.add_subplot(gs[0, 1])
    ax_z     = fig.add_subplot(gs[0, 2])
    ax_rew   = fig.add_subplot(gs[1, :])
    plt.subplots_adjust(bottom=0.15)

    ax_orig.set_title("Original");    ax_orig.axis("off")
    ax_recon.set_title("Recon");      ax_recon.axis("off")
    ax_z.set_title("Latent z (heatmap)"); ax_z.axis("off")

    im_orig  = ax_orig.imshow(obs[0])
    im_recon = ax_recon.imshow(recon_np[0])
    im_z     = ax_z.imshow(z_np.T, aspect="auto", cmap="RdBu_r",
                            vmin=z_np.min(), vmax=z_np.max())
    plt.colorbar(im_z, ax=ax_z, fraction=0.04)

    # Reward curve
    ax_rew.plot(rewards, color="steelblue", alpha=0.8)
    ax_rew.set_xlabel("Step")
    ax_rew.set_ylabel("Reward")
    ax_rew.set_title("Reward over episode")
    ax_rew.grid(alpha=0.3)
    vline = ax_rew.axvline(x=0, color="red", linestyle="--", alpha=0.7)

    # Slider
    ax_slider = plt.axes([0.15, 0.03, 0.7, 0.025])
    slider = Slider(ax_slider, "Step", 0, T - 1, valinit=0, valstep=1)

    def update(val):
        t = int(slider.val)
        im_orig.set_data(obs[t])
        im_recon.set_data(recon_np[t])
        # Highlight current column in z heatmap
        im_z.set_data(z_np.T)
        vline.set_xdata([t, t])
        fig.canvas.draw()

    slider.on_changed(update)

    if save_gif:
        console.print(f"Saving rollout GIF to {save_gif}…")
        writer = PillowWriter(fps=15)
        def anim_func(i):
            slider.set_val(i)
        anim = FuncAnimation(fig, anim_func, frames=T, interval=66)
        anim.save(save_gif, writer=writer)
        console.print(f"Saved → {save_gif}")

    plt.show()


# ── 6. Latent Walk ────────────────────────────────────────────────────────────

def latent_walk(cfg, n_steps: int = 60, save_gif: Optional[str] = None):
    """
    Spherical interpolation between two random latent vectors.
    Shows decoded images as we walk through latent space.
    """
    device = cfg.get_device()
    vae = VAE(cfg.vae).to(device).eval()
    if Path(cfg.paths.vae_checkpoint).exists():
        vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, device)["model"])

    # Sample two random z vectors from a real rollout
    paths = get_rollout_paths(cfg, "train")
    d = np.load(paths[np.random.randint(len(paths))])
    obs = d["obs"]
    rng = np.random.default_rng()
    i1, i2 = rng.choice(len(obs), 2, replace=False)
    x1 = torch.from_numpy((obs[i1].astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)
    x2 = torch.from_numpy((obs[i2].astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        z1 = vae.get_latent(x1)
        z2 = vae.get_latent(x2)

    # Slerp interpolation (norm-preserving: slerp on unit sphere, then rescale)
    def slerp(z_a, z_b, t):
        norm_a = z_a.norm()
        norm_b = z_b.norm()
        norm_t = (1 - t) * norm_a + t * norm_b
        z_a_n = z_a / norm_a
        z_b_n = z_b / norm_b
        omega = torch.acos((z_a_n * z_b_n).sum().clamp(-1, 1))
        if omega.abs() < 1e-4:
            return norm_t * ((1 - t) * z_a_n + t * z_b_n)
        z_unit = (torch.sin((1 - t) * omega) / torch.sin(omega)) * z_a_n + \
                 (torch.sin(t * omega) / torch.sin(omega)) * z_b_n
        return norm_t * z_unit

    frames_walk = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        z_interp = slerp(z1, z2, t)
        with torch.no_grad():
            img = vae.decode(z_interp).squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        frames_walk.append(img)

    # ── Interactive ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    plt.subplots_adjust(bottom=0.2)
    fig.suptitle("Latent Space Walk  (slerp interpolation)", fontsize=11)

    axes[0].imshow(obs[i1]); axes[0].set_title("Start"); axes[0].axis("off")
    im = axes[1].imshow(frames_walk[0]); axes[1].set_title("Interpolated"); axes[1].axis("off")
    axes[2].imshow(obs[i2]); axes[2].set_title("End"); axes[2].axis("off")

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, "t", 0, n_steps - 1, valinit=0, valstep=1)

    def update(val):
        im.set_data(frames_walk[int(slider.val)])
        fig.canvas.draw()

    slider.on_changed(update)

    if save_gif:
        console.print(f"Saving latent walk GIF to {save_gif}…")
        writer = PillowWriter(fps=20)
        anim = FuncAnimation(fig, lambda i: im.set_data(frames_walk[i]),
                             frames=n_steps, interval=50)
        anim.save(save_gif, writer=writer)
        console.print(f"Saved → {save_gif}")

    plt.show()
