from __future__ import annotations

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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports"
    raw_dir = project_root / "data" / "raw"
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"wav read/write OK: {wav_path}")
    print(f"waveform plot saved: {plot_path}")
    print("all imports OK")


if __name__ == "__main__":
    main()
