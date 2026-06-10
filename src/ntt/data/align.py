from __future__ import annotations

from pathlib import Path


def require_audio_pair(input_path: Path, output_path: Path) -> tuple[Path, Path]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio not found: {input_path}")
    if not output_path.is_file():
        raise FileNotFoundError(f"Output audio not found: {output_path}")
    return input_path, output_path
