"""
Collect random rollouts from the environment and save as numpy arrays.
Each rollout: obs [T, H, W, C], actions [T, A], rewards [T], dones [T]
"""
import numpy as np
import gymnasium as gym
from PIL import Image
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)

console = Console()


def preprocess_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize frame to [size×size] and return as uint8."""
    return np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR), dtype=np.uint8)


class _CarRacingPolicy:
    """
    Random policy for CarRacing.

    mode="biased": holds actions for `repeat` steps, biases gas high and brake low
                   so the car actually moves — good for quick runs.
    mode="random": pure iid sampling each step across full action space (paper method).
    """
    def __init__(self, rng: np.random.Generator, mode: str = "biased", repeat: int = 8):
        self.rng = rng
        self.mode = mode
        self.repeat = repeat
        self._action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        self._countdown = 0

    def __call__(self) -> np.ndarray:
        if self.mode == "random":
            return np.array([
                float(self.rng.uniform(-1, 1)),
                float(self.rng.uniform(0, 1)),
                float(self.rng.uniform(0, 1)),
            ], dtype=np.float32)
        # biased
        if self._countdown == 0:
            self._action = np.array([
                float(self.rng.uniform(-1, 1)),
                float(self.rng.uniform(0.5, 1.0)),
                float(self.rng.uniform(0.0, 0.1)),
            ], dtype=np.float32)
            self._countdown = self.repeat
        self._countdown -= 1
        return self._action.copy()


def _collect_one(args):
    """Worker: collect and save one rollout. Self-contained for subprocess pickling."""
    idx, env_name, render_mode, max_steps, frame_skip, img_size, save_dir, seed, collection_mode = args

    import numpy as np
    import gymnasium as gym
    from PIL import Image
    from pathlib import Path

    def _preprocess(frame, size):
        return np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR), dtype=np.uint8)

    env = gym.make(env_name, render_mode=render_mode,
                   max_episode_steps=max_steps * frame_skip)
    rng = np.random.default_rng(seed)

    repeat = 8
    cur_action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    countdown = 0

    obs_list, act_list, rew_list, done_list = [], [], [], []
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))

    for _ in range(max_steps):
        if collection_mode == "random":
            action = np.array([
                float(rng.uniform(-1, 1)),
                float(rng.uniform(0, 1)),
                float(rng.uniform(0, 1)),
            ], dtype=np.float32)
        else:
            if countdown == 0:
                cur_action = np.array([
                    float(rng.uniform(-1, 1)),
                    float(rng.uniform(0.5, 1.0)),
                    float(rng.uniform(0.0, 0.1)),
                ], dtype=np.float32)
                countdown = repeat
            countdown -= 1
            action = cur_action.copy()

        total_reward = 0.0
        done = False
        for _ in range(frame_skip):
            next_obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            done = term or trunc
            if done:
                break

        obs_list.append(_preprocess(obs, img_size))
        act_list.append(action)
        rew_list.append(total_reward)
        done_list.append(done)
        obs = next_obs
        if done:
            break

    path = Path(save_dir) / f"rollout_{idx:05d}.npz"
    np.savez(
        path,
        obs=np.array(obs_list, dtype=np.uint8),
        actions=np.array(act_list, dtype=np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        dones=np.array(done_list, dtype=bool),
    )
    env.close()
    return str(path)


def collect_rollouts(cfg, n_rollouts: Optional[int] = None, tag: str = "train"):
    """
    Collect rollouts using cfg.env.collection_mode policy ("biased" or "random") and save to disk.
    Uses n_workers parallel processes when cfg.env.n_workers > 1.

    Returns: list of rollout file paths
    """
    n_rollouts = n_rollouts or cfg.env.n_rollouts
    save_dir   = Path(cfg.paths.data_dir) / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine which indices are missing
    existing_indices = {
        int(p.stem.split("_")[1])
        for p in save_dir.glob("rollout_?????.npz")
        if not p.stem.endswith("_encoded")
    }
    missing = [i for i in range(n_rollouts) if i not in existing_indices]

    console.print(f"[cyan]Collection mode: [bold]{cfg.env.collection_mode}[/bold] ({'pure iid, paper method' if cfg.env.collection_mode == 'random' else 'biased gas/brake, held 8 steps'})[/]")

    if not missing:
        console.print(f"[green]Already have {n_rollouts} rollouts in {save_dir} — skipping collection.")
        return [str(save_dir / f"rollout_{i:05d}.npz") for i in range(n_rollouts)]

    if len(missing) < n_rollouts:
        sample = missing[:5]
        more = f"…+{len(missing)-5}" if len(missing) > 5 else ""
        console.print(
            f"[yellow]Found {n_rollouts - len(missing)}/{n_rollouts} rollouts. "
            f"Collecting {len(missing)} missing (indices: {sample}{more})[/]"
        )

    n_workers  = min(getattr(cfg.env, "n_workers", 1), len(missing))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Collecting rollouts"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("rollouts", total=len(missing))

        if n_workers <= 1:
            # Sequential — no spawn overhead, best for small runs
            env = gym.make(cfg.env.name, render_mode=cfg.env.render_mode,
                           max_episode_steps=cfg.env.max_steps * cfg.env.frame_skip)
            rng = np.random.default_rng(seed=None)
            paths = []

            for i in missing:
                obs_list, act_list, rew_list, done_list = [], [], [], []
                obs, _ = env.reset()
                policy = _CarRacingPolicy(rng, mode=cfg.env.collection_mode)
                for _ in range(cfg.env.max_steps):
                    action = policy()
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
                np.savez(
                    path,
                    obs=np.array(obs_list, dtype=np.uint8),
                    actions=np.array(act_list, dtype=np.float32),
                    rewards=np.array(rew_list, dtype=np.float32),
                    dones=np.array(done_list, dtype=bool),
                )
                paths.append(str(path))
                progress.advance(task)

            env.close()

        else:
            # Parallel — each worker owns its own env, good for large runs
            args_list = [
                (i, cfg.env.name, cfg.env.render_mode, cfg.env.max_steps,
                 cfg.env.frame_skip, cfg.env.img_size, str(save_dir), i, cfg.env.collection_mode)
                for i in missing
            ]
            paths = [None] * len(missing)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_collect_one, a): idx for idx, a in enumerate(args_list)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    paths[idx] = fut.result()
                    progress.advance(task)
            paths = [p for p in paths if p is not None]

    console.print(
        f"[green]Saved {len(missing)} new rollouts → {save_dir}"
        + (f"  (workers={n_workers})" if n_workers > 1 else "")
        + f"  (total: {n_rollouts})"
    )
    return [str(save_dir / f"rollout_{i:05d}.npz") for i in range(n_rollouts)]


def get_rollout_paths(cfg, tag: str = "train"):
    p = Path(cfg.paths.data_dir) / tag
    # Exclude *_encoded.npz files — those have z/actions, not obs
    paths = sorted(x for x in p.glob("*.npz") if not x.stem.endswith("_encoded"))
    return [str(x) for x in paths]
