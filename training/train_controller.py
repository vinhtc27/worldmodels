"""
Train the Controller (C model) with CMA-ES.
Each candidate parameter set is evaluated by running episodes in the real env
using (VAE + MDN-RNN hidden state) as the representation.
"""
import os
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import cma
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table

from models import VAE, MDNRNN, Controller
from data import preprocess_frame
from utils import set_seed, save_checkpoint, load_checkpoint, MetricLogger

console = Console()


# ── Per-individual evaluation (runs in worker process) ───────────────────────

def _evaluate_params(args):
    """
    Evaluate a single set of controller params.
    Runs in a subprocess — no rich/tqdm output.
    """
    (params, vae_state, rnn_state, ctrl_state_proto,
     cfg_env, cfg_vae, cfg_rnn, cfg_ctrl, seed) = args

    import torch, numpy as np, gymnasium as gym
    from models import VAE, MDNRNN, Controller
    from data import preprocess_frame

    device = "cpu"

    # Rebuild models
    vae = VAE(cfg_vae).to(device).eval()
    vae.load_state_dict(vae_state)

    rnn = MDNRNN(cfg_rnn).to(device).eval()
    rnn.load_state_dict(rnn_state)

    ctrl = Controller(cfg_ctrl).to(device).eval()
    ctrl.set_params(params)

    env = gym.make(cfg_env.name)
    rng = np.random.default_rng(seed)
    total_reward = 0.0

    for _ in range(cfg_ctrl.n_eval_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        h_state = rnn.initial_state(1, device)
        ep_reward = 0.0
        for _ in range(cfg_env.max_steps):
            frame = preprocess_frame(obs, cfg_env.img_size)
            x = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0)  # [1,C,H,W]
            with torch.no_grad():
                z = vae.get_latent(x)
                h_vec = h_state[0][-1]  # [1, H]
                action = ctrl(z, h_vec).squeeze(0).numpy()
                # Advance RNN hidden state
                a_t = torch.from_numpy(action).unsqueeze(0)
                _, _, _, h_state = rnn.forward_step(z, a_t, h_state)
            step_reward = 0.0
            for _ in range(cfg_env.frame_skip):
                obs, reward, term, trunc, _ = env.step(action)
                step_reward += reward
                if term or trunc:
                    break
            ep_reward += step_reward
            if term or trunc:
                break
        total_reward += ep_reward

    env.close()
    return total_reward / cfg_ctrl.n_eval_episodes


# ── Main training function ────────────────────────────────────────────────────

def train_controller(cfg, vae: VAE = None, rnn: MDNRNN = None, resume: bool = False):
    set_seed(cfg.seed)
    device = cfg.get_device()
    console.print(f"[bold]Training Controller (CMA-ES)[/] on [cyan]{device}[/]")

    # ── Load models ───────────────────────────────────────────────────────────
    if vae is None:
        ckpt = load_checkpoint(cfg.paths.vae_checkpoint, device)
        vae = VAE(cfg.vae).to(device).eval()
        vae.load_state_dict(ckpt["model"])
        console.print("[green]VAE loaded.")

    if rnn is None:
        ckpt = load_checkpoint(cfg.paths.rnn_checkpoint, device)
        rnn = MDNRNN(cfg.rnn).to(device).eval()
        rnn.load_state_dict(ckpt["model"])
        console.print("[green]MDN-RNN loaded.")

    ctrl = Controller(cfg.controller).to(device)
    n_params = ctrl.n_params
    console.print(f"  Controller params: [cyan]{n_params}[/]")

    # ── CMA-ES init ───────────────────────────────────────────────────────────
    start_gen = 0
    best_reward = -float("inf")
    x0 = ctrl.get_params()

    if resume and Path(cfg.paths.controller_checkpoint).exists():
        ckpt = load_checkpoint(cfg.paths.controller_checkpoint, device)
        ctrl.load_state_dict(ckpt["model"])
        x0 = ctrl.get_params()
        start_gen = ckpt.get("generation", 0)
        best_reward = ckpt.get("best_reward", best_reward)
        console.print(f"[yellow]Resumed from generation {start_gen}")

    es = cma.CMAEvolutionStrategy(
        x0,
        cfg.controller.sigma0,
        {
            "popsize": cfg.controller.pop_size,
            "seed": cfg.seed,
            "verbose": -9,  # suppress CMA stdout
        },
    )

    logger = MetricLogger("controller", cfg.paths.log_dir)
    vae_state = {k: v.cpu() for k, v in vae.state_dict().items()}
    rnn_state  = {k: v.cpu() for k, v in rnn.state_dict().items()}

    # ── Evolution loop ────────────────────────────────────────────────────────
    for gen in range(start_gen, cfg.controller.n_generations):
        solutions = es.ask()

        # Build worker args
        args_list = [
            (
                sol,
                vae_state, rnn_state, None,
                cfg.env, cfg.vae, cfg.rnn, cfg.controller,
                cfg.seed + gen * cfg.controller.pop_size + i,
            )
            for i, sol in enumerate(solutions)
        ]

        # Evaluate population
        rewards = np.zeros(len(solutions))
        n_workers = min(cfg.controller.n_workers, len(solutions))

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Gen {gen+1}/{cfg.controller.n_generations}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("eval", total=len(solutions))
            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(_evaluate_params, a): i for i, a in enumerate(args_list)}
                    for fut in as_completed(futures):
                        i = futures[fut]
                        rewards[i] = fut.result()
                        progress.advance(task)
            else:
                for i, a in enumerate(args_list):
                    rewards[i] = _evaluate_params(a)
                    progress.advance(task)

        # CMA-ES minimizes — negate rewards
        es.tell(solutions, (-rewards).tolist())

        mean_r = rewards.mean()
        max_r  = rewards.max()
        logger.update(mean_reward=mean_r, max_reward=max_r)
        row = logger.print_epoch(gen + 1, cfg.controller.n_generations)

        # ── Save best ─────────────────────────────────────────────────────────
        best_idx = np.argmax(rewards)
        if max_r > best_reward:
            best_reward = max_r
            ctrl.set_params(solutions[best_idx])
            save_checkpoint(
                {
                    "generation": gen + 1,
                    "model": ctrl.state_dict(),
                    "params": solutions[best_idx],
                    "best_reward": best_reward,
                },
                cfg.paths.controller_checkpoint,
            )
            console.print(f"  [green]✓ Best controller saved (reward={best_reward:.2f})")

        if es.stop():
            console.print("[yellow]CMA-ES converged early.")
            break

    logger.save()
    console.print(f"[bold green]Controller training complete. Best reward: {best_reward:.2f}")
    return ctrl
