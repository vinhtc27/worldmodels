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


class FrameDataset(Dataset):
    """
    Flat dataset of individual frames for VAE training.
    Loads all rollouts into memory, converts HWC → CHW.
    """
    def __init__(self, rollout_paths: List[str]):
        self.frames = []
        for p in rollout_paths:
            d = np.load(p)
            self.frames.append(d["obs"])  # [T, H, W, C]
        self.frames = np.concatenate(self.frames, axis=0)  # [N, H, W, C]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Convert HWC → CHW; frames are already float32 in [0, 1]
        frame = self.frames[idx].transpose(2, 0, 1)
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
            torch.from_numpy(obs.astype(np.float32)),
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
