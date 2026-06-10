from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import soundfile as sf
import torch

from src.ntt.tcn.checkpoint import load_checkpoint
from src.ntt.tcn.model import GatedTCN
from src.ntt.tcn.utils import created_at, resolve_path, sanitize_paths_for_json, select_device, to_repo_relative, write_json


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Input audio not found: {resolved}")
    data, sample_rate = sf.read(resolved, always_2d=True, dtype="float32")
    if not np.isfinite(data).all():
        raise ValueError(f"Input audio contains NaN or Inf: {resolved}")
    return data, int(sample_rate)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_chunks(
    model: GatedTCN,
    audio: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
    warmup_chunks: int,
) -> dict[str, float | int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shape (samples, channels), got {audio.shape}")

    context = max(model.receptive_field() - 1, 0)
    total_samples = audio.shape[0]
    starts = list(range(0, total_samples, chunk_size))
    if not starts:
        raise ValueError("Input audio is empty")

    for start in starts[: max(0, warmup_chunks)]:
        end = min(start + chunk_size, total_samples)
        context_start = max(0, start - context)
        chunk = audio[context_start:end]
        dry = torch.as_tensor(chunk.T, dtype=torch.float32, device=device).unsqueeze(0)
        _ = model(dry)
    sync_if_cuda(device)

    chunk_latencies: list[float] = []
    measured_samples = 0
    wall_start = time.perf_counter()
    for start in starts:
        end = min(start + chunk_size, total_samples)
        context_start = max(0, start - context)
        chunk = audio[context_start:end]
        dry = torch.as_tensor(chunk.T, dtype=torch.float32, device=device).unsqueeze(0)
        sync_if_cuda(device)
        chunk_start = time.perf_counter()
        _ = model(dry)
        sync_if_cuda(device)
        chunk_latencies.append(time.perf_counter() - chunk_start)
        measured_samples += end - start
    wall_seconds = time.perf_counter() - wall_start

    return {
        "inference_time_seconds": wall_seconds,
        "samples_per_second": measured_samples / wall_seconds if wall_seconds > 0 else float("inf"),
        "average_chunk_latency_ms": 1000.0 * sum(chunk_latencies) / len(chunk_latencies),
        "num_chunks": len(chunk_latencies),
        "chunk_size": chunk_size,
        "warmup_chunks": warmup_chunks,
    }


def benchmark(
    *,
    checkpoint_path: str | Path,
    input_path: str | Path,
    device_name: str,
    chunk_size: int,
    warmup_chunks: int,
    output_path: str | Path | None = None,
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
    timing = benchmark_chunks(
        model,
        audio,
        device=device,
        chunk_size=chunk_size,
        warmup_chunks=warmup_chunks,
    )
    audio_duration = audio.shape[0] / sample_rate
    output = {
        "model_name": model_config.get("model_name", "GatedTCN"),
        "checkpoint_path": checkpoint_path,
        "input_path": input_path,
        "device": device.type,
        "sample_rate": sample_rate,
        "input_num_samples": int(audio.shape[0]),
        "audio_duration_seconds": audio_duration,
        "inference_time_seconds": timing["inference_time_seconds"],
        "real_time_factor": float(timing["inference_time_seconds"]) / audio_duration,
        "samples_per_second": timing["samples_per_second"],
        "average_chunk_latency_ms": timing["average_chunk_latency_ms"],
        "num_chunks": timing["num_chunks"],
        "chunk_size": timing["chunk_size"],
        "warmup_chunks": timing["warmup_chunks"],
        "parameter_count": int(checkpoint.get("parameter_count", model.count_parameters())),
        "receptive_field": int(checkpoint.get("receptive_field", model.receptive_field())),
        "created_at": created_at(),
    }

    resolved_output = resolve_path(output_path) if output_path is not None else resolve_path(checkpoint_path).parent.parent / "benchmark.json"
    payload = write_benchmark_json(resolved_output, output)
    print(f"benchmark metadata: {to_repo_relative(resolved_output)}")
    print(f"RTF: {payload['real_time_factor']:.6g}")
    return payload


def write_benchmark_json(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_paths_for_json(payload)
    write_json(path, sanitized)
    return sanitized


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Custom Gated TCN inference speed.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument("--warmup-chunks", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    try:
        benchmark(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            device_name=args.device,
            chunk_size=args.chunk_size,
            warmup_chunks=args.warmup_chunks,
            output_path=args.output,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: TCN benchmark failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
