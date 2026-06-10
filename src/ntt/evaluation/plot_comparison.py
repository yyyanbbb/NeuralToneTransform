from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def relative(path: str | Path) -> str:
    resolved = resolve_path(path)
    assert resolved is not None
    try:
        return resolved.resolve().relative_to(repository_root().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_mono_segment(path: Path, *, start_seconds: float, duration_seconds: float) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if not np.isfinite(data).all():
        raise ValueError(f"Audio contains NaN or Inf: {path}")
    start = max(0, int(start_seconds * sample_rate))
    end = min(data.shape[0], start + max(1, int(duration_seconds * sample_rate)))
    if start >= data.shape[0]:
        raise ValueError(f"Requested start {start_seconds} s exceeds audio length: {path}")
    return data[start:end, 0], int(sample_rate)


def plot_waveform_overlay(series: dict[str, np.ndarray], sample_rate: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    for name, samples in series.items():
        times = np.arange(samples.shape[0]) / sample_rate
        plt.plot(times, samples, linewidth=0.8, alpha=0.85, label=name)
    plt.title("Waveform Overlay")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_error_waveform(target: np.ndarray, predictions: dict[str, np.ndarray], sample_rate: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    if predictions:
        for name, samples in predictions.items():
            length = min(len(target), len(samples))
            times = np.arange(length) / sample_rate
            plt.plot(times, samples[:length] - target[:length], linewidth=0.8, alpha=0.85, label=name)
    else:
        plt.text(0.5, 0.5, "No prediction audio available", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title("Prediction Error Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Prediction - Target")
    if predictions:
        plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_spectrogram(samples: np.ndarray, sample_rate: int, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.specgram(samples, NFFT=1024, Fs=sample_rate, noverlap=768, cmap="magma")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def slug_for_name(name: str) -> str:
    mapping = {
        "A1": "a1",
        "A2 Lite": "a2_lite",
        "A2 Full": "a2_full",
        "TCN Small": "tcn_small",
        "TCN Medium": "tcn_medium",
        "TCN Large": "tcn_large",
    }
    return mapping.get(name, name.lower().replace("-", "_").replace(" ", "_"))


def write_figure_report(
    *,
    report_path: Path,
    generated: list[str],
    skipped: dict[str, str],
    start_seconds: float,
    duration_seconds: float,
) -> None:
    lines = [
        "# Figure Analysis",
        "",
        "## Generated Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated)
    if not generated:
        lines.append("- None")
    lines.extend(["", "## Skipped Predictions", ""])
    if skipped:
        lines.extend(f"- {name}: {reason}" for name, reason in skipped.items())
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Reading Waveform Overlay",
            "",
            f"The overlay shows the first selected window from {start_seconds:g} s for {duration_seconds:g} s. Closer waveform alignment to the target wet signal suggests better time-domain matching, but it does not fully capture perceived tone quality.",
            "",
            "## Reading Spectrograms",
            "",
            "Spectrograms show how energy is distributed over time and frequency. Differences between a prediction and the target can highlight harmonic, transient, or high-frequency mismatch.",
            "",
            "## Scope",
            "",
            "These figures visualize the current smoke/formal experiment artifacts. They are not a final subjective listening conclusion.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def create_comparison_figures(
    *,
    target_path: str | Path,
    predictions: dict[str, str | Path | None],
    out_dir: str | Path,
    report_path: str | Path | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float = 3.0,
) -> dict[str, Any]:
    resolved_target = resolve_path(target_path)
    if resolved_target is None or not resolved_target.is_file():
        raise FileNotFoundError(f"Target audio not found: {resolved_target}")
    resolved_out = resolve_path(out_dir)
    assert resolved_out is not None
    resolved_report = resolve_path(report_path) if report_path is not None else repository_root() / "reports" / "FIGURE_ANALYSIS.md"
    assert resolved_report is not None

    target, sample_rate = load_mono_segment(resolved_target, start_seconds=start_seconds, duration_seconds=duration_seconds)
    available: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}

    for name, path in predictions.items():
        resolved = resolve_path(path)
        if resolved is None or not resolved.is_file():
            skipped[name] = "prediction file missing"
            continue
        pred, pred_rate = load_mono_segment(resolved, start_seconds=start_seconds, duration_seconds=duration_seconds)
        if pred_rate != sample_rate:
            skipped[name] = f"sample rate {pred_rate} differs from target {sample_rate}"
            continue
        available[name] = pred

    generated: list[str] = []
    waveform_path = resolved_out / "waveform_overlay.png"
    plot_waveform_overlay({"Target Wet": target, **available}, sample_rate, waveform_path)
    generated.append(relative(waveform_path))

    error_path = resolved_out / "error_waveform.png"
    plot_error_waveform(target, available, sample_rate, error_path)
    generated.append(relative(error_path))

    target_spec_path = resolved_out / "spectrogram_target.png"
    plot_spectrogram(target, sample_rate, target_spec_path, "Spectrogram: Target Wet")
    generated.append(relative(target_spec_path))

    for name, samples in available.items():
        spec_path = resolved_out / f"spectrogram_{slug_for_name(name)}.png"
        plot_spectrogram(samples, sample_rate, spec_path, f"Spectrogram: {name}")
        generated.append(relative(spec_path))

    if "TCN Medium" in available:
        length = min(len(target), len(available["TCN Medium"]))
        error_spec_path = resolved_out / "spectrogram_error_tcn_medium.png"
        plot_spectrogram(available["TCN Medium"][:length] - target[:length], sample_rate, error_spec_path, "Spectrogram: TCN Medium Error")
        generated.append(relative(error_spec_path))

    write_figure_report(
        report_path=resolved_report,
        generated=generated,
        skipped=skipped,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
    print(f"figure report: {relative(resolved_report)}")
    for name, reason in skipped.items():
        print(f"WARNING: skipped {name}: {reason}")
    return {"generated": generated, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate waveform and spectrogram comparison figures.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--a1-pred")
    parser.add_argument("--a2-lite-pred")
    parser.add_argument("--a2-full-pred")
    parser.add_argument("--tcn-small-pred")
    parser.add_argument("--tcn-medium-pred")
    parser.add_argument("--tcn-large-pred")
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    args = parser.parse_args()

    try:
        create_comparison_figures(
            target_path=args.target,
            predictions={
                "A1": args.a1_pred,
                "A2 Lite": args.a2_lite_pred,
                "A2 Full": args.a2_full_pred,
                "TCN Small": args.tcn_small_pred,
                "TCN Medium": args.tcn_medium_pred,
                "TCN Large": args.tcn_large_pred,
            },
            out_dir=args.out_dir,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: plot comparison failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
