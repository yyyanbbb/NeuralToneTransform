from __future__ import annotations

from pathlib import Path

import soundfile as sf


def audio_info(path: Path) -> sf.SoundFile:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return sf.info(path)
