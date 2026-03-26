"""
Evaluate the full World Models pipeline.
Runs the controller in the real environment and reports metrics.
"""
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from typing import List, Dict, Optional

from rich.console import Console
from rich.table import Table

from models import VAE, MDNRNN, Controller
from data import preprocess_frame
from utils import load_checkpoint

console = Console()


def run_episode(
    vae: VAE,
    rnn: MDNRNN,
    ctrl: Controller,
    cfg,
    device: str,
    render: bool = False,
    seed: Optional[int] = None,
    debug_action: Optional[List] = None,
) -> Dict:
    """Run a single episode, optionally with a pygame display window.

    debug_action: if set, overrides the controller with a fixed action every step.
        Values are passed from the Makefile via `make debug STEER=.. GAS=.. BRAKE=..`.
        Use this to verify the gym responds correctly, independent of training.
    """
    env = gym.make(
        cfg.env.name,
        render_mode="rgb_array" if render else None,
        max_episode_steps=cfg.env.max_steps * cfg.env.frame_skip,
    )
    obs, _ = env.reset(seed=seed)

    h_state = rnn.initial_state(1, device)
    total_reward = 0.0
    z_traj, h_traj, rewards = [], [], []
    step = 0
    running = True
    term = trunc = False

    if render:
        import pygame
        W, H = cfg.env.window_width, cfg.env.window_height
        pygame.init()
        screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        pygame.display.set_caption("World Models — Agent  [ESC / ✕ to stop]")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 14)

    print(f"max_steps = {cfg.env.max_steps}")

    try:
        for step in range(cfg.env.max_steps):
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    if event.type == pygame.VIDEORESIZE:
                        W, H = event.w, event.h
                        screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
                if not running:
                    break

            # ── Model step ───────────────────────────────────────────────────
            frame = preprocess_frame(obs, cfg.env.img_size)
            x = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                z = vae.get_latent(x)
                h_vec = h_state[0][-1]
                if debug_action is not None:
                    action = np.array(debug_action, dtype=np.float32)
                else:
                    action = ctrl(z, h_vec).squeeze(0).cpu().numpy()
                _, _, _, h_state = rnn.forward_step(
                    z, torch.from_numpy(action).unsqueeze(0).to(device), h_state
                )

            z_np = z.squeeze(0).cpu().numpy()
            h_np = h_vec.squeeze(0).cpu().numpy()
            z_traj.append(z_np)
            h_traj.append(h_np)

            # Frame skip: hold action for N physics steps
            step_reward = 0.0
            for _ in range(cfg.env.frame_skip):
                obs, reward, term, trunc, _ = env.step(action)
                step_reward += reward
                if term or trunc:
                    break
            total_reward += step_reward
            rewards.append(step_reward)

            if render:
                # rgb_array is already car-centered (96×96 crop)
                raw = env.render()  # [96, 96, 3] uint8
                surf = pygame.surfarray.make_surface(raw.transpose(1, 0, 2))
                surf = pygame.transform.smoothscale(surf, (W, H))
                screen.blit(surf, (0, 0))

                # HUD — color reflects magnitude: green=high, red=low
                # Clamp to [0,1] before RGB calc — tanh outputs can be negative
                steer_col = (255, 255, 0) if abs(action[0]) > 0.5 else (200, 200, 200)
                gas_v     = max(0.0, float(action[1]))
                brake_v   = max(0.0, float(action[2]))
                gas_col   = (int(255*(1-gas_v)), int(255*gas_v), 0)
                brake_col = (int(255*brake_v), int(255*(1-brake_v)), 0)

                hud = [
                    (f"Step:   {step:4d}",           (255, 255, 255)),
                    (f"Reward: {total_reward:6.1f}", (255, 255, 255)),
                    (f"Steer:  {action[0]:+.2f}",   steer_col),
                    (f"Gas:    {action[1]:.2f}",     gas_col),
                    (f"Brake:  {action[2]:.2f}",     brake_col),
                ]
                for i, (line, color) in enumerate(hud):
                    txt = font.render(line, True, color)
                    backing = pygame.Surface((txt.get_width() + 6, txt.get_height() + 2))
                    backing.set_alpha(140)
                    backing.fill((0, 0, 0))
                    screen.blit(backing, (6, 6 + i * 18))
                    screen.blit(txt, (9, 7 + i * 18))

                pygame.display.flip()
                clock.tick(30)   # cap at 30 fps for readability

                print(
                    f"step={step:4d} | "
                    f"steer={action[0]:+.3f}  gas={action[1]:.3f}  brake={action[2]:.3f} | "
                    f"reward={step_reward:+.2f} | "
                    f"|z|={np.linalg.norm(z_np):.2f}  std={z_np.std():.2f} | "
                    f"|h|={np.linalg.norm(h_np):.2f}  std={h_np.std():.2f}"
                )

            if term or trunc:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if render:
            pygame.quit()
        env.close()

    return {
        "reward": total_reward,
        "length": step + 1,
        "rewards": np.array(rewards) if rewards else np.array([0.0]),
        "z_traj": np.array(z_traj),
        "h_traj": np.array(h_traj),
        "frames": [],
    }


def evaluate(cfg, n_episodes: int = 10, render: bool = False, seed: Optional[int] = None,
             debug_action: Optional[List] = None):
    """Load saved models and evaluate for n_episodes. Print summary table."""
    device = cfg.get_device()

    for path, label in [
        (cfg.paths.vae_checkpoint, "VAE"),
        (cfg.paths.rnn_checkpoint, "MDN-RNN"),
        (cfg.paths.controller_checkpoint, "Controller"),
    ]:
        if not Path(path).exists():
            console.print(f"[red]Missing checkpoint: {path} ({label})")
            return None

    vae = VAE(cfg.vae).to(device).eval()
    vae.load_state_dict(load_checkpoint(cfg.paths.vae_checkpoint, device)["model"])

    rnn = MDNRNN(cfg.rnn).to(device).eval()
    rnn.load_state_dict(load_checkpoint(cfg.paths.rnn_checkpoint, device)["model"])

    ctrl = Controller(cfg.controller).to(device).eval()
    ctrl.load_state_dict(load_checkpoint(cfg.paths.controller_checkpoint, device)["model"])
    console.print("[green]All models loaded.")

    if render:
        msg = f"[cyan]Opening game window ({cfg.env.window_width}×{cfg.env.window_height})  [dim]— ESC / ✕ to stop[/]"
        if debug_action is not None:
            msg += f"\n[yellow]DEBUG MODE — fixed action: steer={debug_action[0]:+.1f}  gas={debug_action[1]:.1f}  brake={debug_action[2]:.1f}[/]"
        console.print(msg)

    results = []
    try:
        for i in range(n_episodes):
            console.print(f"  Episode {i+1}/{n_episodes}")
            ep_seed = (seed or 0) + i
            r = run_episode(vae, rnn, ctrl, cfg, device, render=render, seed=ep_seed,
                            debug_action=debug_action)
            results.append(r)
            console.print(f"  → reward: [green]{r['reward']:.1f}[/]  steps: {r['length']}")
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted — showing results so far.")

    if not results:
        return []

    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]

    table = Table(title="[bold]Evaluation Results[/]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="green")
    for k, v in {
        "Episodes":    len(results),
        "Mean Reward": f"{np.mean(rewards):.2f}",
        "Std  Reward": f"{np.std(rewards):.2f}",
        "Max  Reward": f"{np.max(rewards):.2f}",
        "Min  Reward": f"{np.min(rewards):.2f}",
        "Mean Length": f"{np.mean(lengths):.1f}",
    }.items():
        table.add_row(k, str(v))
    console.print(table)

    return results
