from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from nam.models import init_from_nam


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_nam_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"NAM model not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def read_mono_audio(path: str | Path) -> tuple[np.ndarray, int]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Input WAV not found: {resolved}")
    data, sample_rate = sf.read(resolved, always_2d=True, dtype="float32")
    if data.shape[1] != 1:
        raise ValueError(f"NAM offline inference expects mono WAV input, got {data.shape[1]} channels")
    if not np.isfinite(data).all():
        raise ValueError(f"Input WAV contains NaN or Inf: {resolved}")
    return data[:, 0], int(sample_rate)


def run_model_in_chunks(model, samples: np.ndarray, *, block_size: int = 65_536) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    model.eval()
    receptive_field = int(getattr(model, "receptive_field", 1))
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(samples), block_size):
            end = min(start + block_size, len(samples))
            chunk = samples[start:end]
            if start == 0:
                tensor = torch.as_tensor(chunk, dtype=torch.float32)
                y = model(tensor, pad_start=True)
            else:
                context_start = max(0, start - receptive_field + 1)
                context = samples[context_start:start]
                tensor = torch.as_tensor(np.concatenate([context, chunk]), dtype=torch.float32)
                y = model(tensor, pad_start=False)
            y_np = y.detach().cpu().numpy().astype(np.float32, copy=False)
            if y_np.ndim != 1:
                raise ValueError(f"Expected mono model output, got shape {y_np.shape}")
            outputs.append(y_np[-len(chunk) :])
    if not outputs:
        return np.zeros((0,), dtype=np.float32)
    output = np.concatenate(outputs).astype(np.float32, copy=False)
    if len(output) != len(samples):
        raise RuntimeError(f"Inference length mismatch: input={len(samples)}, output={len(output)}")
    if not np.isfinite(output).all():
        raise ValueError("Model output contains NaN or Inf")
    return output


def write_prediction(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sf.write(resolved, samples, sample_rate)
    return resolved


def load_model_from_export(model_export: dict[str, Any]):
    return init_from_nam(model_export)


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"), **metadata}
    resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return resolved
