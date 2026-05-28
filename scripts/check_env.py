from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# Import smoke checks
import librosa  # noqa: F401
import pandas  # noqa: F401
import scipy  # noqa: F401
import tensorboard  # noqa: F401
import torchaudio  # noqa: F401


class TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def package_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def run_check() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports"
    raw_dir = project_root / "data" / "raw"
    logs_dir = project_root / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sr = 48_000
    duration_s = 1.0
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    wav_path = raw_dir / "smoke_test_tone.wav"
    sf.write(wav_path, x, sr, subtype="PCM_24")

    y, y_sr = sf.read(wav_path, dtype="float32")
    if y_sr != sr:
        raise RuntimeError(f"Sample rate mismatch: expected {sr}, got {y_sr}")

    fig = plt.figure(figsize=(10, 3))
    plt.plot(y[:2000])
    plt.title("Smoke Test Waveform (first 2000 samples)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    fig.tight_layout()
    plot_path = report_dir / "waveform_smoke.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("=== Step1 Environment Check ===")
    print(f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"python executable: {sys.executable}")
    print(f"project root: {project_root}")
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"torchaudio version: {package_version('torchaudio')}")
    print(f"librosa version: {package_version('librosa')}")
    print(f"matplotlib version: {package_version('matplotlib')}")
    print(f"tensorboard version: {package_version('tensorboard')}")
    print(f"soundfile version: {package_version('soundfile')}")
    print(f"scipy version: {package_version('scipy')}")
    print(f"numpy version: {package_version('numpy')}")
    print(f"pandas version: {package_version('pandas')}")
    print(f"wav read/write OK: {wav_path}")
    print(f"waveform plot saved: {plot_path}")
    print("all imports OK")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "step1_env_check.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = TeeStream(sys.stdout, log_file)
        tee_stderr = TeeStream(sys.stderr, log_file)
        try:
            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                run_check()
                print(f"log file: {log_path}")
        except Exception as exc:
            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                print(f"ERROR: {exc}")
            raise SystemExit(1) from exc
