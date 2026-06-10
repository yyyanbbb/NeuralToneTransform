from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import soundfile as sf

from src.ntt.tcn.checkpoint import load_checkpoint
from src.ntt.tcn.utils import relative_to_repo, resolve_path


def check_file(path: Path, label: str, results: list[tuple[str, bool, str]]) -> bool:
    ok = path.is_file()
    results.append((label, ok, relative_to_repo(path) if ok else f"missing: {relative_to_repo(path)}"))
    return ok


def verify(output_dir: str | Path) -> bool:
    resolved_output = resolve_path(output_dir)
    results: list[tuple[str, bool, str]] = []
    config_dir = resolve_path("configs/tcn_gated")
    for config_name in (
        "small.json",
        "medium.json",
        "large.json",
        "training.json",
        "training_formal_small.json",
        "training_formal_medium.json",
        "training_formal_large.json",
        "training_smoke_small.json",
        "training_smoke_medium.json",
        "training_smoke_large.json",
    ):
        check_file(config_dir / config_name, f"config {config_name}", results)

    checkpoint_path = resolved_output / "checkpoints" / "best.pt"
    last_checkpoint_path = resolved_output / "checkpoints" / "last.pt"
    prediction_path = resolved_output / "prediction.wav"
    metrics_path = resolved_output / "training_metrics.json"
    metadata_path = resolved_output / "inference_metadata.json"
    has_best = check_file(checkpoint_path, "best checkpoint", results)
    check_file(last_checkpoint_path, "last checkpoint", results)
    has_prediction = check_file(prediction_path, "prediction.wav", results)
    has_metrics = check_file(metrics_path, "training_metrics.json", results)
    has_metadata = check_file(metadata_path, "inference_metadata.json", results)

    reference_path = resolve_path("data/aligned/aligned_dry.wav")
    if has_prediction and reference_path.is_file():
        pred_info = sf.info(prediction_path)
        ref_info = sf.info(reference_path)
        results.append(
            (
                "prediction sample rate matches dry input",
                pred_info.samplerate == ref_info.samplerate,
                f"prediction={pred_info.samplerate}, dry={ref_info.samplerate}",
            )
        )
        length_ok = pred_info.frames > 0 and abs(pred_info.frames - ref_info.frames) <= max(1, ref_info.frames // 100)
        results.append(
            (
                "prediction length is reasonable",
                length_ok,
                f"prediction={pred_info.frames}, dry={ref_info.frames}",
            )
        )
        data, _ = sf.read(prediction_path, always_2d=True, dtype="float32")
        finite = bool(np.isfinite(data).all())
        results.append(("prediction contains no NaN/Inf", finite, f"samples={data.shape[0]}"))
    elif has_prediction:
        results.append(("dry reference exists", False, f"missing: {relative_to_repo(reference_path)}"))

    parameter_count = None
    receptive_field = None
    if has_metadata:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parameter_count = metadata.get("parameter_count")
        receptive_field = metadata.get("receptive_field")
    if has_metrics:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        parameter_count = parameter_count or metrics.get("parameter_count")
        receptive_field = receptive_field or metrics.get("receptive_field")
    if has_best:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        parameter_count = parameter_count or checkpoint.get("parameter_count")
        receptive_field = receptive_field or checkpoint.get("receptive_field")
    results.append(("parameter_count recorded", isinstance(parameter_count, int) and parameter_count > 0, str(parameter_count)))
    results.append(("receptive_field recorded", isinstance(receptive_field, int) and receptive_field > 0, str(receptive_field)))

    for label, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {label}: {detail}")
    passed = all(ok for _, ok, _ in results)
    print(f"{'PASS' if passed else 'FAIL'} summary: {sum(1 for _, ok, _ in results if ok)}/{len(results)} checks passed")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Custom Gated TCN training and inference outputs.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    return 0 if verify(args.output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
