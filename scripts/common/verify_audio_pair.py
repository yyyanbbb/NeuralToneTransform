from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class AudioInfo:
    path: Path
    exists: bool
    sample_rate: int | None = None
    channels: int | None = None
    frames: int | None = None
    duration: float | None = None
    subtype: str | None = None
    dtype: str | None = None
    peak: float | None = None
    clipping_risk: bool | None = None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repository_root() / path


def inspect_audio(path: Path) -> AudioInfo:
    if not path.exists():
        return AudioInfo(path=path, exists=False)

    info = sf.info(path)
    data, _ = sf.read(path, always_2d=True)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    return AudioInfo(
        path=path,
        exists=True,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration=float(info.duration),
        subtype=str(info.subtype),
        dtype=str(data.dtype),
        peak=peak,
        clipping_risk=peak >= 0.999,
    )


def format_info(label: str, info: AudioInfo) -> list[str]:
    if not info.exists:
        return [f"{label}: MISSING {info.path}"]
    return [
        f"{label}: {info.path}",
        f"  sample_rate: {info.sample_rate}",
        f"  channels: {info.channels}",
        f"  frames: {info.frames}",
        f"  duration_seconds: {info.duration:.6f}",
        f"  subtype: {info.subtype}",
        f"  dtype: {info.dtype}",
        f"  peak_amplitude: {info.peak:.6f}",
        f"  clipping_risk: {info.clipping_risk}",
    ]


def verify_pair(input_path: str | Path, output_path: str | Path, duration_tolerance: float = 0.01) -> tuple[bool, list[str]]:
    x_info = inspect_audio(resolve_path(input_path))
    y_info = inspect_audio(resolve_path(output_path))
    lines: list[str] = []
    failures: list[str] = []

    lines.extend(format_info("input", x_info))
    lines.extend(format_info("output", y_info))

    if not x_info.exists:
        failures.append("input WAV does not exist")
    if not y_info.exists:
        failures.append("output WAV does not exist")

    if x_info.exists and y_info.exists:
        if x_info.sample_rate != y_info.sample_rate:
            failures.append(f"sample rates differ: {x_info.sample_rate} vs {y_info.sample_rate}")
        if x_info.channels != y_info.channels:
            failures.append(f"channel counts differ: {x_info.channels} vs {y_info.channels}")
        duration_delta = abs((x_info.duration or 0.0) - (y_info.duration or 0.0))
        lines.append(f"duration_delta_seconds: {duration_delta:.6f}")
        if duration_delta > duration_tolerance:
            failures.append(
                f"durations differ by {duration_delta:.6f}s, tolerance is {duration_tolerance:.6f}s"
            )
        if x_info.clipping_risk:
            lines.append("WARNING: input peak is near full scale; clipping risk exists")
        if y_info.clipping_risk:
            lines.append("WARNING: output peak is near full scale; clipping risk exists")

    passed = not failures
    lines.append(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    for failure in failures:
        lines.append(f"FAIL: {failure}")
    return passed, lines


def write_log(lines: list[str], log_path: Path | None = None) -> Path:
    if log_path is None:
        log_path = repository_root() / "logs" / "common" / "verify_audio_pair.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"=== audio pair verification {datetime.now().astimezone().isoformat(timespec='seconds')} ===\n")
        for line in lines:
            handle.write(f"{line}\n")
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a paired input/output WAV dataset.")
    parser.add_argument("input_wav")
    parser.add_argument("output_wav")
    parser.add_argument("--duration-tolerance", type=float, default=0.01)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    passed, lines = verify_pair(args.input_wav, args.output_wav, args.duration_tolerance)
    for line in lines:
        print(line)
    write_log(lines, args.log)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
