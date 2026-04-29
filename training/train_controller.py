"""
Train the Controller (C model) with CMA-ES.

Two evaluation modes (set cfg.controller.dream_mode):

  dream (default, fast):
    Controller runs entirely in latent space — no real environment.
    Each step: ctrl(z, h) → action → RewardModel(z, h, action) → r_hat
                                    → RNN.sample_next_z → z_next
    ~50× faster than real-env mode. Requires a trained RewardModel checkpoint.
    If the checkpoint is missing, the reward model is trained automatically.

  real-env (--real-env flag):
    Original approach: run full CarRacing episodes, encode frames via VAE,
    use RNN for hidden state. Accurate but slow.

CMA-ES minimises by convention — we negate rewards before tell().
"""
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cma
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from models import VAE, MDNRNN, Controller, RewardModel
from utils import set_seed, save_checkpoint, load_checkpoint, MetricLogger

console = Console()


# ── Dream evaluation (runs in worker process) ─────────────────────────────────

def _evaluate_params_dream(args):
    """
    Evaluate one controller candidate entirely in latent space (no real env).
    Each step: ctrl(z,h) → action → reward_model(z,h,a) + rnn.sample_next_z
    """
    params, rnn_state, reward_state, cfg_rnn, cfg_ctrl, seed = args

    import torch
    import numpy as np
    from models import MDNRNN, Controller, RewardModel

    device = "cpu"
    # Make per-candidate evaluation reproducible by seeding RNGs with the provided seed
    import random as _random
    torch.manual_seed(int(seed))
    try:
        import numpy as _np
        _np.random.seed(int(seed))
    except Exception:
        pass
    _random.seed(int(seed))

    rnn = MDNRNN(cfg_rnn).to(device).eval()
    rnn.load_state_dict(rnn_state)

    ctrl = Controller(cfg_ctrl).to(device).eval()
    ctrl.set_params(params)

    reward_model = RewardModel(
        cfg_rnn.latent_dim, cfg_rnn.hidden_size, cfg_rnn.action_dim
    ).to(device).eval()
    reward_model.load_state_dict(reward_state)

    total_reward = 0.0

    for _ in range(cfg_ctrl.n_eval_episodes):
        z       = torch.zeros(1, cfg_rnn.latent_dim, device=device)
        h_state = rnn.initial_state(1, device)
        ep_reward = 0.0

        for _ in range(cfg_ctrl.dream_max_steps):
            h_vec = h_state[0][-1]  # [1, hidden_size] — state before this step
            with torch.no_grad():
                action = ctrl(z, h_vec)                                          # [1, action_dim]
                r_hat  = reward_model(z, h_vec, action).item()
                log_pi, mu, sigma, h_state = rnn.forward_step(z, action, h_state)
                z = rnn.sample(log_pi, mu, sigma, temperature=cfg_ctrl.dream_temperature)
            ep_reward += r_hat

        total_reward += ep_reward

    return total_reward / cfg_ctrl.n_eval_episodes


# ── Real-env evaluation (runs in worker process) ──────────────────────────────

def _evaluate_params_real(args):
    """
    Evaluate one controller candidate in the real CarRacing environment.
    Encodes each frame via VAE, maintains RNN hidden state across steps.
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
        h_state   = rnn.initial_state(1, device)
        ep_reward = 0.0

        for _ in range(cfg_env.max_steps):
            frame = preprocess_frame(obs, cfg_env.img_size)
            x = torch.from_numpy(
                (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)
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
    device   = cfg.get_device()
    dream    = cfg.controller.dream_mode

    console.print(
        f"[bold]Training Controller (CMA-ES)[/] on [cyan]{device}[/]  "
        f"mode=[cyan]{'dream' if dream else 'real-env'}[/]"
    )

    # ── Load VAE (real-env mode only) ─────────────────────────────────────────
    if not dream:
        if vae is None:
            ckpt = load_checkpoint(cfg.paths.vae_checkpoint, device)
            vae  = VAE(cfg.vae).to(device).eval()
            vae.load_state_dict(ckpt["model"])
            console.print("[green]VAE loaded.")

    # ── Load RNN ──────────────────────────────────────────────────────────────
    if rnn is None:
        ckpt = load_checkpoint(cfg.paths.rnn_checkpoint, device)
        rnn  = MDNRNN(cfg.rnn).to(device).eval()
        rnn.load_state_dict(ckpt["model"])
        console.print("[green]MDN-RNN loaded.")

    # ── Load / train reward model (dream mode only) ───────────────────────────
    reward_model = None
    if dream:
        rp = Path(cfg.paths.reward_model_checkpoint)
        if rp.exists():
            ckpt = load_checkpoint(str(rp), device)
            reward_model = RewardModel(
                cfg.rnn.latent_dim, cfg.rnn.hidden_size, cfg.rnn.action_dim
            ).to(device).eval()
            reward_model.load_state_dict(ckpt["model"])
            console.print("[green]Reward model loaded.")
        else:
            console.print("[yellow]Reward model checkpoint not found — training now...")
            from .train_reward_model import train_reward_model
            reward_model = train_reward_model(cfg, rnn=rnn)

    ctrl     = Controller(cfg.controller).to(device)
    n_params = ctrl.n_params
    console.print(f"  Controller params: [cyan]{n_params}[/]")

    # ── CMA-ES init ───────────────────────────────────────────────────────────
    start_gen   = 0
    best_reward = -float("inf")
    x0          = ctrl.get_params()

    ckpt_path = (
        cfg.paths.controller_dream_checkpoint if dream
        else cfg.paths.controller_real_checkpoint
    )

    if resume and Path(ckpt_path).exists():
        ckpt        = load_checkpoint(ckpt_path, device)
        ctrl.load_state_dict(ckpt["model"])
        x0          = ctrl.get_params()
        start_gen   = ckpt.get("generation", 0)
        best_reward = ckpt.get("best_reward", best_reward)
        console.print(f"[yellow]Resumed from generation {start_gen}")

    es = cma.CMAEvolutionStrategy(
        x0,
        cfg.controller.sigma0,
        {"popsize": cfg.controller.pop_size, "seed": cfg.seed, "verbose": -9},
    )

    logger    = MetricLogger("controller", cfg.paths.log_dir)
    rnn_state = {k: v.cpu() for k, v in rnn.state_dict().items()}
    if not dream:
        vae_state = {k: v.cpu() for k, v in vae.state_dict().items()}
    else:
        reward_state = {k: v.cpu() for k, v in reward_model.state_dict().items()}

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
        gen_task = progress.add_task(
            "gens", total=n_gens - start_gen,
            status=f"Gen {start_gen+1}/{n_gens}"
        )

        for gen in range(start_gen, n_gens):
            progress.update(
                gen_task,
                status=f"Gen {gen+1}/{n_gens} — evaluating {cfg.controller.pop_size} candidates"
            )
            solutions = es.ask()
            n_workers = min(cfg.controller.n_workers, len(solutions))

            if dream:
                args_list = [
                    (sol, rnn_state, reward_state, cfg.rnn, cfg.controller,
                     cfg.seed + gen * cfg.controller.pop_size + i)
                    for i, sol in enumerate(solutions)
                ]
                worker_fn = _evaluate_params_dream
            else:
                args_list = [
                    (sol, vae_state, rnn_state,
                     cfg.env, cfg.vae, cfg.rnn, cfg.controller,
                     cfg.seed + gen * cfg.controller.pop_size + i)
                    for i, sol in enumerate(solutions)
                ]
                worker_fn = _evaluate_params_real

            rewards = np.zeros(len(solutions))
            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(worker_fn, a): i for i, a in enumerate(args_list)}
                    for fut in as_completed(futures):
                        rewards[futures[fut]] = fut.result()
            else:
                for i, a in enumerate(args_list):
                    rewards[i] = worker_fn(a)

            es.tell(solutions, (-rewards).tolist())

            mean_r = rewards.mean()
            max_r  = rewards.max()
            logger.update(mean_reward=mean_r, max_reward=max_r)
            logger.print_epoch(gen + 1, n_gens)

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
                        "dream_mode":  dream,
                    },
                    ckpt_path,
                )
                console.print(f"  [green]✓ Best controller saved (reward={best_reward:.2f})")

            progress.update(
                gen_task,
                status=f"Gen {gen+1}/{n_gens} — best={best_reward:.1f}",
            )
            progress.advance(gen_task)

            if es.stop():
                console.print("[yellow]CMA-ES converged early.")
                break

    logger.save()
    console.print(f"[bold green]Controller training complete. Best reward: {best_reward:.2f}")
    return ctrl
