from __future__ import annotations

import argparse
import sys

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


PENDING_REASON = (
    "A1 inference is pending because neural-amp-modeler 0.12.2 does not expose "
    "a verified offline inference API in this environment."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline inference for the A1 NAM model.")
    parser.add_argument("--model", default="outputs/a1_baseline/model.nam")
    parser.add_argument("--input", default="data/aligned/aligned_dry.wav")
    parser.add_argument("--output", default="outputs/a1_baseline/prediction.wav")
    parser.add_argument("--block-size", type=int, default=65_536)
    args = parser.parse_args()

    metadata_path = resolve_path("outputs/a1_baseline/inference_metadata.json")
    try:
        model_export = load_nam_json(args.model)
        samples, sample_rate = read_mono_audio(args.input)
        model = load_model_from_export(model_export)
        prediction = run_model_in_chunks(model, samples, block_size=args.block_size)
        output_path = write_prediction(args.output, prediction, sample_rate)
        metadata = {
            "inference_status": "completed",
            "model_path": relative(resolve_path(args.model)),
            "input_path": relative(resolve_path(args.input)),
            "output_path": relative(output_path),
            "sample_rate": sample_rate,
            "input_num_samples": int(len(samples)),
            "output_num_samples": int(len(prediction)),
            "model_architecture": model_export.get("architecture", "unknown"),
            "output_peak": float(np.max(np.abs(prediction))) if prediction.size else 0.0,
        }
        write_metadata(metadata_path, metadata)
        print(f"A1 inference complete: {relative(output_path)}")
        return 0
    except Exception as exc:
        metadata = {
            "inference_status": "failed",
            "model_path": args.model,
            "input_path": args.input,
            "output_path": args.output,
            "error_summary": str(exc),
            "pending_reason": PENDING_REASON,
        }
        write_metadata(metadata_path, metadata)
        print(f"ERROR: A1 inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
