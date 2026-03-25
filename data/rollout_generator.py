"""
Collect random rollouts from the environment and save as numpy arrays.
Each rollout: obs [T, H, W, C], actions [T, A], rewards [T], dones [T]
"""
import os
import numpy as np
import gymnasium as gym
from PIL import Image
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)

console = Console()


def preprocess_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize frame to [size×size] and return as float32 in [0,1]."""
    img = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


class _CarRacingPolicy:
    """
    Simple biased-random policy for CarRacing that actually drives.
    Holds steering/gas for several steps so the car moves meaningfully.
    """
    def __init__(self, rng: np.random.Generator, repeat: int = 8):
        self.rng = rng
        self.repeat = repeat          # hold each action for N steps
        self._action = np.array([0.0, 0.5, 0.0])
        self._countdown = 0

    def __call__(self) -> np.ndarray:
        if self._countdown == 0:
            steer = float(self.rng.uniform(-1, 1))
            gas   = float(self.rng.uniform(0.5, 1.0))   # always press gas
            brake = float(self.rng.uniform(0.0, 0.1))   # rarely brake
            self._action = np.array([steer, gas, brake], dtype=np.float32)
            self._countdown = self.repeat
        self._countdown -= 1
        return self._action.copy()


def collect_rollouts(cfg, n_rollouts: Optional[int] = None, tag: str = "train"):
    """
    Collect rollouts using a biased-random driving policy and save to disk.

    Returns: list of rollout file paths
    """
    n_rollouts = n_rollouts or cfg.env.n_rollouts
    save_dir = Path(cfg.paths.data_dir) / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(cfg.env.name, render_mode=cfg.env.render_mode)
    rng = np.random.default_rng(seed=None)

    paths = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Collecting rollouts"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("rollouts", total=n_rollouts)

        for i in range(n_rollouts):
            obs_list, act_list, rew_list, done_list = [], [], [], []
            obs, _ = env.reset()
            policy = _CarRacingPolicy(rng)
            for _ in range(cfg.env.max_steps):
                action = policy()
                # Frame skip: repeat action, accumulate reward, use last obs
                total_reward = 0.0
                done = False
                for _ in range(cfg.env.frame_skip):
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    if done:
                        break

                obs_list.append(preprocess_frame(obs, cfg.env.img_size))
                act_list.append(action)
                rew_list.append(total_reward)
                done_list.append(done)

                obs = next_obs
                if done:
                    break

            path = save_dir / f"rollout_{i:05d}.npz"
            np.savez_compressed(
                path,
                obs=np.array(obs_list, dtype=np.float32),
                actions=np.array(act_list, dtype=np.float32),
                rewards=np.array(rew_list, dtype=np.float32),
                dones=np.array(done_list, dtype=bool),
            )
            paths.append(str(path))
            progress.advance(task)

    env.close()
    console.print(f"[green]Saved {n_rollouts} rollouts → {save_dir}")
    return paths


def get_rollout_paths(cfg, tag: str = "train"):
    p = Path(cfg.paths.data_dir) / tag
    # Exclude *_encoded.npz files — those have z/actions, not obs
    paths = sorted(x for x in p.glob("*.npz") if not x.stem.endswith("_encoded"))
    return [str(x) for x in paths]
