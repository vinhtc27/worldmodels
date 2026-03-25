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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def save_json(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class MetricLogger:
    """Accumulates metrics per epoch and prints a rich table."""
    def __init__(self, name: str, log_dir: str):
        self.name = name
        self.log_dir = log_dir
        self.history: Dict[str, list] = {}
        self._step_buf: Dict[str, list] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._step_buf.setdefault(k, []).append(float(v))

    def commit(self, epoch: int):
        row = {"epoch": epoch}
        for k, vals in self._step_buf.items():
            avg = sum(vals) / len(vals)
            self.history.setdefault(k, []).append(avg)
            row[k] = avg
        self._step_buf = {}
        return row

    def print_epoch(self, epoch: int, total: int):
        row = self.commit(epoch)
        table = Table(title=f"[bold]{self.name}[/] — Epoch {epoch}/{total}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in row.items():
            if k != "epoch":
                table.add_row(k, f"{v:.4f}")
        console.print(table)
        return row

    def save(self):
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
