"""
PyTorch Dataset wrappers for VAE and RNN training.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List


class FrameDataset(Dataset):
    """Flat dataset of individual frames for VAE training."""
    def __init__(self, rollout_paths: List[str]):
        self.frames = []
        for p in rollout_paths:
            d = np.load(p)
            self.frames.append(d["obs"])  # [T, H, W, C]
        self.frames = np.concatenate(self.frames, axis=0)  # [N, H, W, C]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Convert HWC → CHW, already float32 [0,1]
        frame = self.frames[idx].transpose(2, 0, 1)
        return torch.from_numpy(frame)


class SequenceDataset(Dataset):
    """
    Dataset of (obs_seq, action_seq) windows for RNN training.
    Each item: obs [T+1, C, H, W], actions [T, A]
    The VAE encodes obs → z on the fly during training, so we store raw frames.
    Or: pre-encode and store z to speed up RNN training (recommended).
    """
    def __init__(self, rollout_paths: List[str], seq_len: int, preencoded: bool = False):
        self.seq_len = seq_len
        self.preencoded = preencoded
        self.windows = []

        for p in rollout_paths:
            d = np.load(p)
            obs = d["obs"]    # [T, H, W, C]  or  [T, latent] if preencoded
            acts = d["actions"]  # [T, A]
            T = len(acts)
            for start in range(0, T - seq_len - 1, seq_len // 2):
                end = start + seq_len + 1
                if end > T:
                    break
                o_window = obs[start:end]    # [seq+1, ...]
                a_window = acts[start:end]   # [seq+1, A]
                self.windows.append((o_window, a_window))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        obs, acts = self.windows[idx]
        if not self.preencoded:
            # HWC → CHW
            obs = obs.transpose(0, 3, 1, 2)
        obs_t = torch.from_numpy(obs.astype(np.float32))
        acts_t = torch.from_numpy(acts.astype(np.float32))
        return obs_t, acts_t


class LatentSequenceDataset(Dataset):
    """
    Dataset where frames are pre-encoded to latent vectors.
    Much faster RNN training — encode rollouts once, then train RNN on z's.
    """
    def __init__(self, encoded_paths: List[str], seq_len: int):
        self.seq_len = seq_len
        self.windows = []

        for p in encoded_paths:
            d = np.load(p)
            z_seq = d["z"]       # [T, latent_dim]
            acts = d["actions"]  # [T, A]
            T = len(acts)
            for start in range(0, T - seq_len - 1, seq_len // 2):
                end = start + seq_len + 1
                if end > T:
                    break
                self.windows.append((z_seq[start:end], acts[start:end]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        z, acts = self.windows[idx]
        return (
            torch.from_numpy(z.astype(np.float32)),
            torch.from_numpy(acts.astype(np.float32)),
        )
