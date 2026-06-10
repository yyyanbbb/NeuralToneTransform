from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from nam_offline import (
    load_model_from_export,
    load_nam_json,
    read_mono_audio,
    relative,
    resolve_path,
    run_model_in_chunks,
    write_metadata,
    write_prediction,
)


def submodel_channels(submodel: dict[str, Any]) -> int | None:
    try:
        layers = submodel["model"]["config"]["layers"]
        return int(layers[0]["channels"])
    except Exception:
        return None


def select_submodel(container: dict[str, Any], channels: int) -> dict[str, Any]:
    if container.get("architecture") != "SlimmableContainer":
        raise ValueError(f"Expected SlimmableContainer, got {container.get('architecture')!r}")
    submodels = container.get("config", {}).get("submodels", [])
    for submodel in submodels:
        if submodel_channels(submodel) == channels:
            return submodel["model"]
    found = [submodel_channels(submodel) for submodel in submodels]
    raise ValueError(f"No A2 submodel with {channels} channels found; detected {found}")


def run_one(container: dict[str, Any], channels: int, samples: np.ndarray, sample_rate: int, output: str, block_size: int) -> dict[str, Any]:
    model_export = select_submodel(container, channels)
    model = load_model_from_export(model_export)
    prediction = run_model_in_chunks(model, samples, block_size=block_size)
    output_path = write_prediction(output, prediction, sample_rate)
    return {
        "output_path": relative(output_path),
        "output_num_samples": int(len(prediction)),
        "output_peak": float(np.max(np.abs(prediction))) if prediction.size else 0.0,
        "architecture": model_export.get("architecture", "unknown"),
        "channels": channels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline inference for A2 SlimmableContainer submodels.")
    parser.add_argument("--model", default="outputs/a2_baseline/model.nam")
    parser.add_argument("--input", default="data/aligned/aligned_dry.wav")
    parser.add_argument("--lite-output", default="outputs/a2_baseline/a2_lite_prediction.wav")
    parser.add_argument("--full-output", default="outputs/a2_baseline/a2_full_prediction.wav")
    parser.add_argument("--block-size", type=int, default=65_536)
    args = parser.parse_args()

    metadata_path = resolve_path("outputs/a2_baseline/inference_metadata.json")
    try:
        container = load_nam_json(args.model)
        samples, sample_rate = read_mono_audio(args.input)
        submodels = container.get("config", {}).get("submodels", [])
        detected_submodels = [
            {"channels": submodel_channels(submodel), "max_value": submodel.get("max_value")}
            for submodel in submodels
        ]
        lite = run_one(container, 3, samples, sample_rate, args.lite_output, args.block_size)
        full = run_one(container, 8, samples, sample_rate, args.full_output, args.block_size)
        metadata = {
            "inference_status": "completed",
            "model_path": relative(resolve_path(args.model)),
            "input_path": relative(resolve_path(args.input)),
            "lite_output_path": lite["output_path"],
            "full_output_path": full["output_path"],
            "sample_rate": sample_rate,
            "input_num_samples": int(len(samples)),
            "lite_output_num_samples": lite["output_num_samples"],
            "full_output_num_samples": full["output_num_samples"],
            "model_architecture": container.get("architecture", "unknown"),
            "detected_submodels": detected_submodels,
            "lite": lite,
            "full": full,
        }
        write_metadata(metadata_path, metadata)
        print(f"A2 Lite inference complete: {lite['output_path']}")
        print(f"A2 Full inference complete: {full['output_path']}")
        return 0
    except Exception as exc:
        metadata = {
            "inference_status": "failed",
            "model_path": args.model,
            "input_path": args.input,
            "lite_output_path": args.lite_output,
            "full_output_path": args.full_output,
            "error_summary": str(exc),
            "pending_reason": "A2 inference is pending until the SlimmableContainer submodels can be loaded and evaluated with the local NAM API.",
        }
        write_metadata(metadata_path, metadata)
        print(f"ERROR: A2 inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
