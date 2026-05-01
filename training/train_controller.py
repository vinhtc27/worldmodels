"""
Train the Controller (C model) with CMA-ES.

Evaluates each candidate by running full CarRacing episodes: encode frames
via VAE, maintain RNN hidden state across steps, accumulate real rewards.

CMA-ES minimises by convention — we negate rewards before tell().
"""
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cma
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from models import VAE, MDNRNN, Controller
from utils import set_seed, save_checkpoint, load_checkpoint, MetricLogger

console = Console()

# ── Worker-process globals (populated by _worker_init once per process) ───────
_W_VAE      = None
_W_RNN      = None
_W_CTRL     = None
_W_CFG_ENV  = None
_W_CFG_CTRL = None


def _worker_init(vae_state, rnn_state, cfg_env, cfg_vae, cfg_rnn, cfg_ctrl):
    """Run once per worker process to load models into process-local globals."""
    global _W_VAE, _W_RNN, _W_CTRL, _W_CFG_ENV, _W_CFG_CTRL
    import torch
    from models import VAE, MDNRNN, Controller

    _W_VAE = VAE(cfg_vae).eval()
    _W_VAE.load_state_dict(vae_state)

    _W_RNN = MDNRNN(cfg_rnn).eval()
    _W_RNN.load_state_dict(rnn_state)

    _W_CTRL     = Controller(cfg_ctrl).eval()
    _W_CFG_ENV  = cfg_env
    _W_CFG_CTRL = cfg_ctrl


def _evaluate_params_real(args):
    """
    Evaluate one controller candidate using the models already loaded in this
    worker process (via _worker_init). Only params and seed are passed per call.
    """
    params, seed = args

    import torch
    import numpy as np
    import gymnasium as gym
    from data import preprocess_frame

    device = "cpu"
    _W_CTRL.set_params(params)

    cfg_env  = _W_CFG_ENV
    cfg_ctrl = _W_CFG_CTRL

    env = gym.make(cfg_env.name, max_episode_steps=cfg_env.max_steps * cfg_env.frame_skip)
    rng = np.random.default_rng(seed)
    total_reward = 0.0

    for _ in range(cfg_ctrl.n_eval_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        h_state   = _W_RNN.initial_state(1, device)
        ep_reward = 0.0

        for _ in range(cfg_env.max_steps):
            frame = preprocess_frame(obs, cfg_env.img_size)
            x = torch.from_numpy(
                (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                z      = _W_VAE.get_latent(x)
                h_vec  = h_state[0][-1]
                action = _W_CTRL(z, h_vec).squeeze(0).cpu().numpy()
                _, _, _, h_state = _W_RNN.forward_step(
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


def train_controller(cfg, vae: VAE = None, rnn: MDNRNN = None, resume: bool = False):
    set_seed(cfg.seed)
    device = cfg.get_device()

    console.print(f"[bold]Training Controller (CMA-ES)[/] on [cyan]{device}[/]")

    if device == "mps" and cfg.controller.n_workers > 1:
        console.print(
            "[yellow]MPS detected but worker processes cannot use MPS (macOS multiprocessing limitation). "
            "CMA-ES evaluation runs on CPU across all workers — this is expected behaviour.[/]"
        )

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

    start_gen   = 0
    best_reward = -float("inf")
    best_params = None
    x0          = ctrl.get_params()
    ckpt_path   = cfg.paths.controller_checkpoint

    if resume:
        gen_ckpts = sorted(Path(cfg.paths.checkpoint_dir).glob("controller_gen_*.pt"))
        resume_path = (
            gen_ckpts[-1] if gen_ckpts
            else Path(ckpt_path) if Path(ckpt_path).exists()
            else None
        )
        if resume_path:
            ckpt        = load_checkpoint(str(resume_path), device)
            ctrl.load_state_dict(ckpt["model"])
            x0          = ctrl.get_params()
            start_gen   = ckpt.get("generation", 0)
            best_reward = ckpt.get("best_reward", best_reward)
            console.print(
                f"[yellow]Resumed from generation {start_gen} "
                f"(best_reward={best_reward:.2f}, source={resume_path.name})"
            )

    es = cma.CMAEvolutionStrategy(
        x0,
        cfg.controller.sigma0,
        {"popsize": cfg.controller.pop_size, "seed": cfg.seed, "verbose": -9},
    )

    logger    = MetricLogger("controller", cfg.paths.log_dir)
    vae_state = {k: v.cpu() for k, v in vae.state_dict().items()}
    rnn_state = {k: v.cpu() for k, v in rnn.state_dict().items()}

    n_gens   = cfg.controller.n_generations
    n_workers = min(cfg.controller.n_workers, cfg.controller.pop_size)

    def _run_generation(executor, solutions, gen):
        args_list = [
            (sol, cfg.seed + gen * cfg.controller.pop_size + i)
            for i, sol in enumerate(solutions)
        ]
        rewards = np.zeros(len(solutions))
        if executor is not None:
            futures = {executor.submit(_evaluate_params_real, a): i for i, a in enumerate(args_list)}
            for fut in as_completed(futures):
                rewards[futures[fut]] = fut.result()
        else:
            # Single-worker path: initialise globals once if needed
            if _W_VAE is None:
                _worker_init(vae_state, rnn_state, cfg.env, cfg.vae, cfg.rnn, cfg.controller)
            for i, a in enumerate(args_list):
                rewards[i] = _evaluate_params_real(a)
        return rewards

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
        gen_task = progress.add_task(
            "gens", total=n_gens - start_gen,
            status=f"Gen {start_gen+1}/{n_gens}"
        )

        # Create pool once outside the loop — workers load models via initializer
        executor_ctx = (
            ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_worker_init,
                initargs=(vae_state, rnn_state, cfg.env, cfg.vae, cfg.rnn, cfg.controller),
            )
            if n_workers > 1
            else None
        )

        try:
            for gen in range(start_gen, n_gens):
                progress.update(
                    gen_task,
                    status=f"Gen {gen+1}/{n_gens} — evaluating {cfg.controller.pop_size} candidates"
                )
                solutions = es.ask()
                rewards   = _run_generation(executor_ctx, solutions, gen)

                es.tell(solutions, (-rewards).tolist())

                mean_r = rewards.mean()
                max_r  = rewards.max()
                logger.update(mean_reward=mean_r, max_reward=max_r)
                logger.print_epoch(gen + 1, n_gens)

                gen_best_idx = np.argmax(rewards)
                if max_r > best_reward:
                    best_reward = max_r
                    best_params = solutions[gen_best_idx]
                    ctrl.set_params(best_params)
                    save_checkpoint(
                        {
                            "generation":  gen + 1,
                            "model":       ctrl.state_dict(),
                            "params":      best_params,
                            "best_reward": best_reward,
                        },
                        ckpt_path,
                    )
                    console.print(f"  [green]✓ Best controller saved (reward={best_reward:.2f})")

                if (gen + 1) % cfg.controller.save_interval == 0 and best_params is not None:
                    ctrl.set_params(best_params)
                    save_checkpoint(
                        {
                            "generation":  gen + 1,
                            "model":       ctrl.state_dict(),
                            "params":      best_params,
                            "best_reward": best_reward,
                        },
                        f"{cfg.paths.checkpoint_dir}/controller_gen_{gen+1:03d}.pt",
                    )

                progress.update(
                    gen_task,
                    status=f"Gen {gen+1}/{n_gens} — best={best_reward:.1f}",
                )
                progress.advance(gen_task)

                if es.stop():
                    console.print("[yellow]CMA-ES converged early.")
                    break
        finally:
            if executor_ctx is not None:
                executor_ctx.shutdown(wait=False)

    logger.save()
    console.print(f"[bold green]Controller training complete. Best reward: {best_reward:.2f}")
    return ctrl
