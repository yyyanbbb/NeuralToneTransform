from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from nam_offline import load_model_from_export, load_nam_json, read_mono_audio, relative, resolve_path, run_model_in_chunks
from run_a2_inference import select_submodel


def select_model_export(export: dict[str, Any], variant: str) -> tuple[dict[str, Any], str]:
    if variant == "a1":
        return export, export.get("architecture", "unknown")
    if variant == "a2-lite":
        model_export = select_submodel(export, 3)
        return model_export, f"{export.get('architecture', 'unknown')} submodel channels=3"
    if variant == "a2-full":
        model_export = select_submodel(export, 8)
        return model_export, f"{export.get('architecture', 'unknown')} submodel channels=8"
    raise ValueError("variant must be one of: a1, a2-lite, a2-full")


def benchmark_model(
    *,
    model_path: str | Path,
    input_path: str | Path,
    variant: str,
    block_size: int,
) -> dict[str, Any]:
    export = load_nam_json(model_path)
    model_export, architecture = select_model_export(export, variant)
    samples, sample_rate = read_mono_audio(input_path)
    model = load_model_from_export(model_export)

    start = time.perf_counter()
    prediction = run_model_in_chunks(model, samples, block_size=block_size)
    inference_time = time.perf_counter() - start
    audio_duration = len(samples) / sample_rate
    num_chunks = int(np.ceil(len(samples) / block_size)) if block_size else 0

    return {
        "model_path": relative(resolve_path(model_path)),
        "input_path": relative(resolve_path(input_path)),
        "variant": variant,
        "model_architecture": architecture,
        "sample_rate": sample_rate,
        "input_num_samples": int(len(samples)),
        "output_num_samples": int(len(prediction)),
        "audio_duration_seconds": audio_duration,
        "inference_time_seconds": inference_time,
        "real_time_factor": inference_time / audio_duration,
        "samples_per_second": len(samples) / inference_time if inference_time > 0 else float("inf"),
        "average_chunk_latency_ms": (inference_time / num_chunks * 1000.0) if num_chunks else 0.0,
        "block_size": block_size,
        "num_chunks": num_chunks,
        "created_at": datetime.now(UTC).isoformat(),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NAM offline inference RTF.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--variant", required=True, choices=["a1", "a2-lite", "a2-full"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--block-size", type=int, default=65_536)
    args = parser.parse_args()

    try:
        payload = benchmark_model(
            model_path=args.model,
            input_path=args.input,
            variant=args.variant,
            block_size=args.block_size,
        )
        write_json(args.out, payload)
    except Exception as exc:
        raise SystemExit(f"ERROR: NAM benchmark failed: {exc}") from exc

    print(f"benchmark metadata: {relative(resolve_path(args.out))}")
    print(f"RTF: {payload['real_time_factor']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
