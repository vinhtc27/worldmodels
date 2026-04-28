"""
PyTorch Dataset wrappers for VAE and RNN training.

Two paths for RNN training:
  1. LatentSequenceDataset (recommended) — reads pre-encoded *_encoded.npz files
     produced by encode_and_save_rollouts after VAE training. VAE inference
     runs once up front, so each RNN training step is much faster.

  2. SequenceDataset — reads raw rollout .npz files and stores raw frames.
     The VAE would need to encode on the fly during RNN training, which is
     slow. Included for completeness but not used in the standard pipeline.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List


def _build_frame_cache(paths: List[str], out_path: Path) -> None:
    """
    Consolidate all rollout obs arrays into a single memory-mapped .npy file.
    Two streaming passes — peak RAM is one rollout's obs at a time (~12 MB).
    The resulting file supports O(1) random frame access via mmap.
    Regenerated automatically if any source rollout is newer than the cache.
    """
    from rich.console import Console
    console = Console()

    # Pass 1: total count + shape (one array in RAM at a time)
    total, shape = 0, None
    for p in paths:
        obs = np.load(p)["obs"]
        total += len(obs)
        if shape is None:
            shape = obs.shape[1:]

    console.print(f"[cyan]Building frame cache: {total:,} frames → {out_path.name} ...")

    # Pass 2: stream each obs into the mmap file
    out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32,
                                    shape=(total,) + shape)
    offset = 0
    for p in paths:
        obs = np.load(p)["obs"]
        n = len(obs)
        out[offset : offset + n] = obs.astype(np.float32) / 255.0
        offset += n
        del obs
    del out  # flush to disk
    console.print(f"[green]Frame cache ready ({total:,} frames).")


class FrameDataset(Dataset):
    """
    Flat dataset of individual frames for VAE training.

    On first use (or when rollouts are newer than the cache), all obs arrays
    are consolidated into data/rollouts/train/all_obs.npy — a single
    memory-mapped file. Subsequent runs skip this step and mmap the file
    directly, so RAM usage stays near zero regardless of dataset size.
    """
    def __init__(self, rollout_paths: List[str]):
        paths = list(rollout_paths)
        if not paths:
            self._obs = np.empty((0, 64, 64, 3), dtype=np.float32)
            return

        out_path = Path(paths[0]).parent / "all_obs.npy"
        newest_rollout = max(Path(p).stat().st_mtime for p in paths)
        if not out_path.exists() or out_path.stat().st_mtime < newest_rollout:
            _build_frame_cache(paths, out_path)

        self._obs = np.load(str(out_path), mmap_mode="r")

    def __len__(self):
        return len(self._obs)

    def __getitem__(self, idx):
        # .copy() required: mmap arrays are read-only, torch.from_numpy needs writable
        frame = self._obs[idx].transpose(2, 0, 1).copy()
        return torch.from_numpy(frame)


class SequenceDataset(Dataset):
    """
    Windowed (obs, action) sequences from raw rollouts for RNN training.

    NOTE: this class is not used in the standard pipeline — prefer
    LatentSequenceDataset which reads pre-encoded z sequences and is
    significantly faster because VAE inference runs once up front.
    """
    def __init__(self, rollout_paths: List[str], seq_len: int):
        self.seq_len = seq_len
        self.windows = []

        for p in rollout_paths:
            d    = np.load(p)
            obs  = d["obs"]      # [T, H, W, C]
            acts = d["actions"]  # [T, A]
            T    = len(acts)
            # Stride by seq_len//2 for 50% overlap between windows
            for start in range(0, T - seq_len - 1, seq_len // 2):
                end = start + seq_len + 1
                if end > T:
                    break
                self.windows.append((obs[start:end], acts[start:end]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        obs, acts = self.windows[idx]
        obs = obs.transpose(0, 3, 1, 2)  # [T+1, H, W, C] → [T+1, C, H, W]
        return (
            torch.from_numpy(obs.astype(np.float32) / 255.0),
            torch.from_numpy(acts.astype(np.float32)),
        )


class LatentSequenceDataset(Dataset):
    """
    Windowed (z, action) sequences from pre-encoded rollouts.

    Reads *_encoded.npz files produced by encode_and_save_rollouts.
    Each item is a (seq_len+1) window:
      z_seq   [T+1, latent_dim]  — latent vectors
      act_seq [T+1, action_dim]  — actions taken

    During RNN training the loader slices these as:
      z_in   = z_seq[:T]    (input)
      z_next = z_seq[1:]    (target — what the RNN must predict)
      a_in   = act_seq[:T]  (action that caused the transition)
    """
    def __init__(self, encoded_paths: List[str], seq_len: int):
        self.seq_len = seq_len
        self.windows = []

        for p in encoded_paths:
            d    = np.load(p)
            z    = d["z"]        # [T, latent_dim]
            acts = d["actions"]  # [T, action_dim]
            T    = len(acts)
            for start in range(0, T - seq_len - 1, seq_len // 2):
                end = start + seq_len + 1
                if end > T:
                    break
                self.windows.append((z[start:end], acts[start:end]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        z, acts = self.windows[idx]
        return (
            torch.from_numpy(z.astype(np.float32)),
            torch.from_numpy(acts.astype(np.float32)),
        )
