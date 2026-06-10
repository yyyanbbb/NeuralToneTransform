from __future__ import annotations

from pathlib import Path


def model_size_bytes(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return path.stat().st_size
