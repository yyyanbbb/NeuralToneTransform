from __future__ import annotations

from pathlib import Path


def ensure_chunk_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "data" / "chunks" / split).mkdir(parents=True, exist_ok=True)
