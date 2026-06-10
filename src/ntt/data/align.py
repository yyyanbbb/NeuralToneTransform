from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def metadata_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_audio(path: str | Path) -> tuple[np.ndarray, int, sf.SoundFile]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    data, sample_rate = sf.read(resolved, always_2d=True, dtype="float64")
    return data, int(sample_rate), sf.info(resolved)


def peak_amplitude(data: np.ndarray) -> float:
    return float(np.max(np.abs(data))) if data.size else 0.0


def validate_audio_pair(
    dry: np.ndarray,
    wet: np.ndarray,
    dry_rate: int,
    wet_rate: int,
    *,
    duration_tolerance_seconds: float = 0.05,
) -> list[str]:
    warnings: list[str] = []
    if dry_rate != wet_rate:
        raise ValueError(f"Sample rates differ: dry={dry_rate}, wet={wet_rate}")
    if dry.ndim != 2 or wet.ndim != 2:
        raise ValueError("Audio must be loaded as 2D arrays shaped (samples, channels)")
    if dry.shape[1] != wet.shape[1]:
        raise ValueError(f"Channel counts differ: dry={dry.shape[1]}, wet={wet.shape[1]}")
    if not np.isfinite(dry).all():
        raise ValueError("Dry audio contains NaN or Inf")
    if not np.isfinite(wet).all():
        raise ValueError("Wet audio contains NaN or Inf")

    duration_delta = abs(len(dry) - len(wet)) / dry_rate
    if duration_delta > duration_tolerance_seconds:
        raise ValueError(
            f"Dry/wet durations differ by {duration_delta:.6f}s, "
            f"tolerance is {duration_tolerance_seconds:.6f}s"
        )

    dry_peak = peak_amplitude(dry)
    wet_peak = peak_amplitude(wet)
    if dry_peak >= 1.0:
        warnings.append("WARNING: dry audio peak amplitude is >= 1.0. Potential clipping detected.")
    if wet_peak >= 1.0:
        warnings.append("WARNING: wet audio peak amplitude is >= 1.0. Potential clipping detected.")
    return warnings


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float64, copy=False)
    return np.mean(data, axis=1, dtype=np.float64)


def estimate_delay_samples(
    dry: np.ndarray,
    wet: np.ndarray,
    *,
    max_lag_samples: int | None = None,
    analysis_samples: int = 262_144,
) -> int:
    """Estimate dry/wet delay in samples using FFT-based cross-correlation.

    Return value convention:
    - positive delay means the wet signal lags behind dry by that many samples;
      alignment should trim the first `delay` samples from wet and the tail of dry.
    - negative delay means dry lags behind wet; alignment should trim the first
      `abs(delay)` samples from dry and the tail of wet.
    """

    dry_mono = to_mono(dry)
    wet_mono = to_mono(wet)
    sample_count = min(len(dry_mono), len(wet_mono), analysis_samples)
    if sample_count <= 1:
        raise ValueError("Need at least two samples to estimate delay")

    dry_segment = dry_mono[:sample_count] - float(np.mean(dry_mono[:sample_count]))
    wet_segment = wet_mono[:sample_count] - float(np.mean(wet_mono[:sample_count]))
    if np.max(np.abs(dry_segment)) == 0.0 or np.max(np.abs(wet_segment)) == 0.0:
        raise ValueError("Cannot estimate delay from silent analysis segment")

    correlation = correlate(wet_segment, dry_segment, mode="full", method="fft")
    lags = correlation_lags(len(wet_segment), len(dry_segment), mode="full")

    if max_lag_samples is not None:
        if max_lag_samples < 0:
            raise ValueError("max_lag_samples must be non-negative")
        mask = np.abs(lags) <= max_lag_samples
        if not np.any(mask):
            raise ValueError("max_lag_samples removed all candidate lags")
        correlation = correlation[mask]
        lags = lags[mask]

    return int(lags[int(np.argmax(np.abs(correlation)))])


def trim_to_alignment(dry: np.ndarray, wet: np.ndarray, delay_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if delay_samples > 0:
        dry_trimmed = dry[:-delay_samples] if delay_samples < len(dry) else dry[:0]
        wet_trimmed = wet[delay_samples:]
    elif delay_samples < 0:
        offset = abs(delay_samples)
        dry_trimmed = dry[offset:]
        wet_trimmed = wet[:-offset] if offset < len(wet) else wet[:0]
    else:
        dry_trimmed = dry
        wet_trimmed = wet

    aligned_len = min(len(dry_trimmed), len(wet_trimmed))
    if aligned_len <= 0:
        raise ValueError(f"Delay {delay_samples} leaves no overlapping samples")
    return dry_trimmed[:aligned_len], wet_trimmed[:aligned_len]


def align_audio_pair(
    dry_path: str | Path,
    wet_path: str | Path,
    out_dir: str | Path,
    *,
    max_lag_samples: int | None = None,
    strict_clipping: bool = False,
) -> dict:
    dry_resolved = resolve_path(dry_path)
    wet_resolved = resolve_path(wet_path)
    out_resolved = resolve_path(out_dir)
    out_resolved.mkdir(parents=True, exist_ok=True)

    dry, dry_rate, _ = read_audio(dry_resolved)
    wet, wet_rate, _ = read_audio(wet_resolved)
    warnings = validate_audio_pair(dry, wet, dry_rate, wet_rate)
    dry_peak = peak_amplitude(dry)
    wet_peak = peak_amplitude(wet)
    dry_clipping_risk = dry_peak >= 1.0
    wet_clipping_risk = wet_peak >= 1.0
    if strict_clipping and (dry_clipping_risk or wet_clipping_risk):
        raise ValueError("Potential clipping detected and strict_clipping is enabled")
    for warning in warnings:
        print(warning)

    delay = estimate_delay_samples(dry, wet, max_lag_samples=max_lag_samples)
    aligned_dry, aligned_wet = trim_to_alignment(dry, wet, delay)

    output_dry = out_resolved / "aligned_dry.wav"
    output_wet = out_resolved / "aligned_wet.wav"
    sf.write(output_dry, aligned_dry, dry_rate)
    sf.write(output_wet, aligned_wet, dry_rate)

    metadata = {
        "dry_path": metadata_path(dry_resolved),
        "wet_path": metadata_path(wet_resolved),
        "output_dry_path": metadata_path(output_dry),
        "output_wet_path": metadata_path(output_wet),
        "sample_rate": dry_rate,
        "original_num_samples_dry": int(len(dry)),
        "original_num_samples_wet": int(len(wet)),
        "aligned_num_samples": int(len(aligned_dry)),
        "estimated_delay_samples": int(delay),
        "dry_peak": dry_peak,
        "wet_peak": wet_peak,
        "dry_clipping_risk": dry_clipping_risk,
        "wet_clipping_risk": wet_clipping_risk,
        "strict_clipping": bool(strict_clipping),
        "warnings": warnings,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out_resolved / "alignment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Align paired dry/wet WAV files.")
    parser.add_argument("--dry", required=True)
    parser.add_argument("--wet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-lag-samples", type=int, default=None)
    parser.add_argument("--strict-clipping", action="store_true")
    args = parser.parse_args()

    try:
        metadata = align_audio_pair(
            args.dry,
            args.wet,
            args.out_dir,
            max_lag_samples=args.max_lag_samples,
            strict_clipping=args.strict_clipping,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: alignment failed: {exc}") from exc

    print("Alignment complete")
    print(f"estimated_delay_samples: {metadata['estimated_delay_samples']}")
    print(f"aligned_num_samples: {metadata['aligned_num_samples']}")
    print(f"output_dry_path: {metadata['output_dry_path']}")
    print(f"output_wet_path: {metadata['output_wet_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
