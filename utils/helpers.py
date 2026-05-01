"""
Shared utilities: checkpoint I/O, logging, seeding.
"""
import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def set_seed(seed: int):
    """Seed all RNGs for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _strip_orig_mod(sd: Dict[str, Any]) -> Dict[str, Any]:
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}


def unwrap_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    """Return state_dict without torch.compile's _orig_mod. prefix."""
    sd = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = _strip_orig_mod(sd)
    return sd


def save_checkpoint(state: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" in ckpt and any(k.startswith("_orig_mod.") for k in ckpt["model"]):
        ckpt["model"] = _strip_orig_mod(ckpt["model"])
    return ckpt


def save_json(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class MetricLogger:
    """
    Accumulates per-batch metrics, commits epoch averages, and prints a rich table.

    Usage:
        logger = MetricLogger("vae", cfg.paths.log_dir)
        for batch in loader:
            ...
            logger.update(loss=loss.item(), kl=kl.item())
        logger.print_epoch(epoch, total_epochs)  # commits + prints
        logger.save()                             # writes history to JSON
    """
    def __init__(self, name: str, log_dir: str):
        self.name      = name
        self.log_dir   = log_dir
        self.history:   Dict[str, list] = {}
        self._step_buf: Dict[str, list] = {}

    def update(self, **kwargs):
        """Buffer per-batch values — averaged at commit time."""
        for k, v in kwargs.items():
            # If v is a tensor (possibly requires_grad), detach and move to CPU
            if isinstance(v, torch.Tensor):
                v_val = v.detach().cpu().item()
            else:
                v_val = float(v)
            self._step_buf.setdefault(k, []).append(v_val)

    def commit(self, epoch: int):
        """Average buffered values and append to history. Returns the epoch row."""
        row = {"epoch": epoch}
        for k, vals in self._step_buf.items():
            avg = sum(vals) / len(vals)
            self.history.setdefault(k, []).append(avg)
            row[k] = avg
        self._step_buf = {}
        return row

    def print_epoch(self, epoch: int, total: int):
        row   = self.commit(epoch)
        table = Table(title=f"[bold]{self.name}[/] — Epoch {epoch}/{total}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value",  style="green")
        for k, v in row.items():
            if k != "epoch":
                table.add_row(k, f"{v:.4f}")
        console.print(table)
        self.save()
        return row

    def save(self):
        """Write full metric history to JSON for later plotting."""
        path = os.path.join(self.log_dir, f"{self.name}_history.json")
        save_json(self.history, path)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(name: str, model: torch.nn.Module):
    n = count_parameters(model)
    console.print(Panel(
        f"[bold cyan]{name}[/]\n"
        f"Parameters: [green]{n:,}[/]\n"
        f"Architecture:\n{model}",
        title="Model Summary",
    ))
