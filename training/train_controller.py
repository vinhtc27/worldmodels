"""
Train the Controller (C model) with CMA-ES.

Why CMA-ES?
  Covariance Matrix Adaptation Evolution Strategy is a gradient-free
  optimiser. We need it because reward is not differentiable — we can't
  backprop through the environment physics and the full VAE+RNN pipeline
  jointly. CMA-ES evaluates the controller as a black box by running full
  episodes and evolving a population of weight vectors.

How it works:
  1. CMA-ES maintains a distribution N(μ, C) over weight vectors.
  2. Each generation it samples pop_size candidates from this distribution.
  3. Each candidate is evaluated: run n_eval_episodes in the real env,
     record total reward.
  4. The best candidates update μ and C — the distribution shifts toward
     high-reward regions and its shape adapts to the fitness landscape.
  5. Repeat for n_generations.

CMA-ES minimises by convention, so we negate rewards before passing to tell().
"""
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Evaluate a single set of controller params in a subprocess.
    Each worker rebuilds the models from state dicts (can't pickle nn.Module).
    Returns average reward over n_eval_episodes.
    """
    params, vae_state, rnn_state, cfg_env, cfg_vae, cfg_rnn, cfg_ctrl, seed = args

    import torch
    import numpy as np
    import gymnasium as gym
    from models import VAE, MDNRNN, Controller
    from data import preprocess_frame

    device = "cpu"

    vae = VAE(cfg_vae).to(device).eval()
    vae.load_state_dict(vae_state)

    rnn = MDNRNN(cfg_rnn).to(device).eval()
    rnn.load_state_dict(rnn_state)

    ctrl = Controller(cfg_ctrl).to(device).eval()
    ctrl.set_params(params)

    env = gym.make(cfg_env.name, max_episode_steps=cfg_env.max_steps * cfg_env.frame_skip)
    rng = np.random.default_rng(seed)
    total_reward = 0.0

    for _ in range(cfg_ctrl.n_eval_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        h_state  = rnn.initial_state(1, device)
        ep_reward = 0.0

        for _ in range(cfg_env.max_steps):
            frame = preprocess_frame(obs, cfg_env.img_size)
            x = torch.from_numpy((frame.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                z      = vae.get_latent(x)
                h_vec  = h_state[0][-1]
                action = ctrl(z, h_vec).squeeze(0).cpu().numpy()
                _, _, _, h_state = rnn.forward_step(
                    z, torch.from_numpy(action).unsqueeze(0).to(device), h_state
                )

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
        vae  = VAE(cfg.vae).to(device).eval()
        vae.load_state_dict(ckpt["model"])
        console.print("[green]VAE loaded.")

    if rnn is None:
        ckpt = load_checkpoint(cfg.paths.rnn_checkpoint, device)
        rnn  = MDNRNN(cfg.rnn).to(device).eval()
        rnn.load_state_dict(ckpt["model"])
        console.print("[green]MDN-RNN loaded.")

    ctrl     = Controller(cfg.controller).to(device)
    n_params = ctrl.n_params
    console.print(f"  Controller params: [cyan]{n_params}[/]")

    # ── CMA-ES init ───────────────────────────────────────────────────────────
    start_gen    = 0
    best_reward  = -float("inf")
    x0           = ctrl.get_params()  # random init from Controller.__init__

    if resume and Path(cfg.paths.controller_checkpoint).exists():
        ckpt        = load_checkpoint(cfg.paths.controller_checkpoint, device)
        ctrl.load_state_dict(ckpt["model"])
        x0          = ctrl.get_params()
        start_gen   = ckpt.get("generation", 0)
        best_reward = ckpt.get("best_reward", best_reward)
        console.print(f"[yellow]Resumed from generation {start_gen}")

    es = cma.CMAEvolutionStrategy(
        x0,
        cfg.controller.sigma0,
        {
            "popsize": cfg.controller.pop_size,
            "seed":    cfg.seed,
            "verbose": -9,  # suppress CMA stdout
        },
    )

    logger    = MetricLogger("controller", cfg.paths.log_dir)
    # Serialise model states once — workers receive them via pickling
    vae_state = {k: v.cpu() for k, v in vae.state_dict().items()}
    rnn_state = {k: v.cpu() for k, v in rnn.state_dict().items()}

    # ── Evolution loop ────────────────────────────────────────────────────────
    n_gens = cfg.controller.n_generations
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]CMA-ES"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        gen_task = progress.add_task("gens", total=n_gens - start_gen, status=f"Gen {start_gen+1}/{n_gens}")

        for gen in range(start_gen, n_gens):
            progress.update(gen_task, status=f"Gen {gen+1}/{n_gens} — evaluating {cfg.controller.pop_size} candidates")
            solutions = es.ask()

            args_list = [
                (
                    sol,
                    vae_state, rnn_state,
                    cfg.env, cfg.vae, cfg.rnn, cfg.controller,
                    cfg.seed + gen * cfg.controller.pop_size + i,
                )
                for i, sol in enumerate(solutions)
            ]

            rewards   = np.zeros(len(solutions))
            n_workers = min(cfg.controller.n_workers, len(solutions))

            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(_evaluate_params, a): i for i, a in enumerate(args_list)}
                    for fut in as_completed(futures):
                        i = futures[fut]
                        rewards[i] = fut.result()
            else:
                for i, a in enumerate(args_list):
                    rewards[i] = _evaluate_params(a)

            # CMA-ES minimises — negate rewards to turn maximisation into minimisation
            es.tell(solutions, (-rewards).tolist())

            mean_r = rewards.mean()
            max_r  = rewards.max()
            logger.update(mean_reward=mean_r, max_reward=max_r)
            logger.print_epoch(gen + 1, n_gens)

            # ── Save best ─────────────────────────────────────────────────────
            best_idx = np.argmax(rewards)
            if max_r > best_reward:
                best_reward = max_r
                ctrl.set_params(solutions[best_idx])
                save_checkpoint(
                    {
                        "generation":  gen + 1,
                        "model":       ctrl.state_dict(),
                        "params":      solutions[best_idx],
                        "best_reward": best_reward,
                    },
                    cfg.paths.controller_checkpoint,
                )
                console.print(f"  [green]✓ Best controller saved (reward={best_reward:.2f})")

            progress.update(gen_task, status=f"Gen {gen+1}/{n_gens} — best={best_reward:.1f}")
            progress.advance(gen_task)

            if es.stop():
                console.print("[yellow]CMA-ES converged early.")
                break

    logger.save()
    console.print(f"[bold green]Controller training complete. Best reward: {best_reward:.2f}")
    return ctrl
