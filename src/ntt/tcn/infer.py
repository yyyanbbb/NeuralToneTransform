from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import soundfile as sf
import torch

from src.ntt.tcn.checkpoint import load_checkpoint
from src.ntt.tcn.model import GatedTCN
from src.ntt.tcn.utils import created_at, relative_to_repo, resolve_path, select_device, write_json


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Input audio not found: {resolved}")
    data, sample_rate = sf.read(resolved, always_2d=True, dtype="float32")
    if not np.isfinite(data).all():
        raise ValueError(f"Input audio contains NaN or Inf: {resolved}")
    return data, int(sample_rate)


@torch.inference_mode()
def chunked_predict(model: GatedTCN, audio: np.ndarray, *, device: torch.device, chunk_size: int) -> np.ndarray:
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shape (samples, channels), got {audio.shape}")
    if audio.shape[1] != model.in_channels:
        raise ValueError(f"Expected {model.in_channels} input channel(s), got {audio.shape[1]}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    context = max(model.receptive_field() - 1, 0)
    outputs: list[np.ndarray] = []
    total_samples = audio.shape[0]
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        context_start = max(0, start - context)
        chunk = audio[context_start:end]
        dry = torch.as_tensor(chunk.T, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(dry).squeeze(0).detach().cpu().numpy().T
        drop = start - context_start
        outputs.append(pred[drop : drop + (end - start)])
    if not outputs:
        return np.empty((0, model.out_channels), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def infer(
    *,
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    device_name: str,
    chunk_size: int,
) -> dict[str, Any]:
    device = select_device(device_name)
    print(f"using device: {device.type}")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint does not contain model config under key 'config'")
    model = GatedTCN.from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    audio, sample_rate = load_audio(input_path)
    prediction = chunked_predict(model, audio, device=device, chunk_size=chunk_size)
    if prediction.shape[0] != audio.shape[0]:
        raise RuntimeError(f"Prediction length {prediction.shape[0]} does not match input length {audio.shape[0]}")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or Inf")

    resolved_output = resolve_path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(resolved_output, prediction, sample_rate, subtype="FLOAT")

    metadata = {
        "checkpoint_path": relative_to_repo(checkpoint_path),
        "input_path": relative_to_repo(input_path),
        "output_path": relative_to_repo(output_path),
        "sample_rate": sample_rate,
        "input_num_samples": int(audio.shape[0]),
        "output_num_samples": int(prediction.shape[0]),
        "device": device.type,
        "model_name": model_config.get("model_name", "GatedTCN"),
        "parameter_count": int(checkpoint.get("parameter_count", model.count_parameters())),
        "receptive_field": int(checkpoint.get("receptive_field", model.receptive_field())),
        "chunk_size": int(chunk_size),
        "created_at": created_at(),
    }
    write_json(resolved_output.parent / "inference_metadata.json", metadata)
    print(f"prediction audio: {relative_to_repo(resolved_output)}")
    print(f"inference metadata: {relative_to_repo(resolved_output.parent / 'inference_metadata.json')}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chunked inference with a trained Gated TCN checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--chunk-size", type=int, default=262_144)
    args = parser.parse_args()

    try:
        infer(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            output_path=args.output,
            device_name=args.device,
            chunk_size=args.chunk_size,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: TCN inference failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
