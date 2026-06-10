from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class Result:
    passed: bool
    message: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def inspect_prediction(label: str, path: Path, reference_rate: int) -> list[Result]:
    results: list[Result] = []
    results.append(Result(path.is_file(), f"{label} exists: {path.relative_to(repository_root()).as_posix()}"))
    if not path.is_file():
        results.append(Result(False, f"{label} missing; inference may be pending"))
        return results
    results.append(Result(path.stat().st_size > 0, f"{label} file size is greater than 0"))
    try:
        data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        results.append(Result(sample_rate == reference_rate, f"{label} sample rate matches reference ({sample_rate})"))
        results.append(Result(data.shape[0] > 0 and data.shape[1] > 0, f"{label} shape is reasonable: {data.shape}"))
        results.append(Result(np.isfinite(data).all(), f"{label} has no NaN or Inf"))
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        results.append(Result(True, f"{label} peak amplitude: {peak:.6f}"))
    except Exception as exc:
        results.append(Result(False, f"{label} inspection failed: {exc}"))
    return results


def main() -> int:
    root = repository_root()
    reference_path = root / "data" / "aligned" / "aligned_dry.wav"
    results = [Result(reference_path.is_file(), "reference input exists: data/aligned/aligned_dry.wav")]
    if reference_path.is_file():
        _, reference_rate = sf.read(reference_path, always_2d=True, dtype="float32")
        info = sf.info(reference_path)
        reference_rate = int(info.samplerate)
    else:
        reference_rate = 0

    targets = [
        ("A1 prediction", root / "outputs" / "a1_baseline" / "prediction.wav"),
        ("A2 Lite prediction", root / "outputs" / "a2_baseline" / "a2_lite_prediction.wav"),
        ("A2 Full prediction", root / "outputs" / "a2_baseline" / "a2_full_prediction.wav"),
    ]
    for label, path in targets:
        results.extend(inspect_prediction(label, path, reference_rate))

    pass_count = sum(1 for result in results if result.passed)
    fail_count = len(results) - pass_count
    print("=== Inference Output Verification ===")
    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.message}")
    print("")
    print("=== Summary ===")
    print(f"pass_count: {pass_count}")
    print(f"fail_count: {fail_count}")
    print(f"OVERALL: {'PASS' if fail_count == 0 else 'FAIL'}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
