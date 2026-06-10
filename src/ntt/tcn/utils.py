from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def created_at() -> str:
    return datetime.now(UTC).isoformat()


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relative_to_repo(path: str | Path) -> str:
    resolved = resolve_path(path)
    try:
        return resolved.resolve().relative_to(repository_root().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
