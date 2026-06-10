from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.ntt.tcn.utils import resolve_path, sanitize_paths_for_json


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    best_val_loss: float,
    parameter_count: int,
    receptive_field: int,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": sanitize_paths_for_json(config),
        "best_val_loss": best_val_loss,
        "parameter_count": parameter_count,
        "receptive_field": receptive_field,
    }
    if training_config is not None:
        payload["training_config"] = sanitize_paths_for_json(training_config)
    return payload


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    best_val_loss: float,
    parameter_count: int,
    receptive_field: int,
    training_config: dict[str, Any] | None = None,
) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            best_val_loss=best_val_loss,
            parameter_count=parameter_count,
            receptive_field=receptive_field,
            training_config=training_config,
        ),
        resolved,
    )


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    try:
        return torch.load(resolved, map_location=map_location)
    except TypeError:
        return torch.load(resolved, map_location=map_location, weights_only=False)
