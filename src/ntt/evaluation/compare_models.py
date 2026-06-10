from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from src.ntt.evaluation.metrics import compute_metrics, load_audio
except ModuleNotFoundError:  # Support direct script execution.
    from metrics import compute_metrics, load_audio

try:
    import torch
except ModuleNotFoundError:  # Custom TCN metadata is optional for artifact-only usage.
    torch = None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_size(path: Path) -> str:
    return f"{path.stat().st_size} bytes" if path.is_file() else "missing"


def format_metric(value: float | str) -> str:
    return value if isinstance(value, str) else f"{value:.6g}"


def prediction_metrics(pred_path: Path | None, target_path: Path | None) -> tuple[dict[str, float | str], str]:
    empty = {"MSE": "TBD", "MAE": "TBD", "ESR": "TBD", "MRSTFT": "TBD", "SNR": "TBD"}
    if target_path is None or not target_path.is_file():
        return empty, "prediction audio not evaluated; target audio not available"
    if pred_path is None or not pred_path.is_file():
        return empty, "prediction audio not available"

    pred, pred_rate = load_audio(pred_path)
    target, target_rate = load_audio(target_path)
    if pred_rate != target_rate:
        return empty, f"prediction sample rate {pred_rate} differs from target {target_rate}"
    values = compute_metrics(pred, target, match_length=True)
    return {
        "MSE": values["MSE"],
        "MAE": values["MAE"],
        "ESR": values["ESR"],
        "MRSTFT": values["MRSTFT"],
        "SNR": values["SNR dB"],
    }, "audio metrics computed from prediction file"


def checkpoint_metadata(checkpoint_path: Path | None) -> tuple[dict[str, Any], str]:
    if checkpoint_path is None or not checkpoint_path.is_file():
        return {}, "checkpoint metadata not available"
    if torch is None:
        return {}, "checkpoint metadata not available; torch is not installed"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    return {
        "parameter_count": checkpoint.get("parameter_count"),
        "receptive_field": checkpoint.get("receptive_field"),
        "config": config,
    }, "checkpoint metadata loaded"


def custom_tcn_row(args: argparse.Namespace, target_path: Path | None) -> dict[str, str] | None:
    pred_path = resolve_path(args.custom_tcn_pred)
    checkpoint_path = resolve_path(args.custom_tcn_checkpoint)
    if pred_path is None and checkpoint_path is None:
        return None

    metrics, metric_note = prediction_metrics(pred_path, target_path)
    metadata, checkpoint_note = checkpoint_metadata(checkpoint_path)
    config = metadata.get("config", {})
    channels = config.get("channels", "TBD")
    skip_channels = config.get("skip_channels", "TBD")
    parameter_count = metadata.get("parameter_count", "TBD")
    receptive_field = metadata.get("receptive_field", "TBD")
    model_name = args.custom_tcn_name or config.get("model_name", "GatedTCN-Medium")

    return {
        "Model": model_name,
        "Architecture": "Custom causal dilated Gated TCN",
        "Channels": f"residual={channels}, skip={skip_channels}",
        "Training Version": "PyTorch custom training loop",
        "Output Path": relative(checkpoint_path) if checkpoint_path is not None else "TBD",
        "Prediction Path": relative(pred_path) if pred_path is not None else "TBD",
        "File Size": file_size(checkpoint_path) if checkpoint_path is not None else "missing",
        "Parameters": str(parameter_count),
        "Receptive Field": str(receptive_field),
        "MSE": format_metric(metrics["MSE"]),
        "MAE": format_metric(metrics["MAE"]),
        "ESR": format_metric(metrics["ESR"]),
        "MRSTFT": format_metric(metrics["MRSTFT"]),
        "SNR": format_metric(metrics["SNR"]),
        "Inference Notes": f"{checkpoint_note}; {metric_note}",
    }


def model_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    target_path = resolve_path(args.target)
    a1_model = repository_root() / "outputs" / "a1_baseline" / "model.nam"
    a2_model = repository_root() / "outputs" / "a2_baseline" / "model.nam"
    a2_report = repository_root() / "reports" / "a2_model_inspection.md"

    a1_prediction = resolve_path(args.a1_pred)
    a2_lite_prediction = resolve_path(args.a2_lite_pred)
    a2_full_prediction = resolve_path(args.a2_full_pred)
    a1_metrics, a1_note = prediction_metrics(a1_prediction, target_path)
    a2_lite_metrics, a2_lite_note = prediction_metrics(a2_lite_prediction, target_path)
    a2_full_metrics, a2_full_note = prediction_metrics(a2_full_prediction, target_path)
    a2_status = "PASS" if a2_model.is_file() and a2_report.is_file() and "OVERALL: PASS" in a2_report.read_text(encoding="utf-8") else "MISSING"

    rows = [
        {
            "Model": "A1 baseline",
            "Architecture": "legacy NAM WaveNet",
            "Channels": "single model",
            "Training Version": "neural-amp-modeler==0.12.2",
            "Output Path": "outputs/a1_baseline/model.nam",
            "Prediction Path": relative(a1_prediction) if a1_prediction is not None else "TBD",
            "File Size": file_size(a1_model),
            "Parameters": "TBD",
            "Receptive Field": "TBD",
            "MSE": format_metric(a1_metrics["MSE"]),
            "MAE": format_metric(a1_metrics["MAE"]),
            "ESR": format_metric(a1_metrics["ESR"]),
            "MRSTFT": format_metric(a1_metrics["MRSTFT"]),
            "SNR": format_metric(a1_metrics["SNR"]),
            "Inference Notes": "legacy NAM baseline artifact; " + a1_note,
        },
        {
            "Model": "A2-Lite baseline",
            "Architecture": "PackedWaveNet exported as SlimmableContainer",
            "Channels": "3",
            "Training Version": "neural-amp-modeler==0.13.0",
            "Output Path": "outputs/a2_baseline/model.nam",
            "Prediction Path": relative(a2_lite_prediction) if a2_lite_prediction is not None else "TBD",
            "File Size": file_size(a2_model),
            "Parameters": "TBD",
            "Receptive Field": "TBD",
            "MSE": format_metric(a2_lite_metrics["MSE"]),
            "MAE": format_metric(a2_lite_metrics["MAE"]),
            "ESR": format_metric(a2_lite_metrics["ESR"]),
            "MRSTFT": format_metric(a2_lite_metrics["MRSTFT"]),
            "SNR": format_metric(a2_lite_metrics["SNR"]),
            "Inference Notes": f"A2 smoke status {a2_status}; SlimmableContainer inspected; " + a2_lite_note,
        },
        {
            "Model": "A2-Full baseline",
            "Architecture": "PackedWaveNet exported as SlimmableContainer",
            "Channels": "8",
            "Training Version": "neural-amp-modeler==0.13.0",
            "Output Path": "outputs/a2_baseline/model.nam",
            "Prediction Path": relative(a2_full_prediction) if a2_full_prediction is not None else "TBD",
            "File Size": file_size(a2_model),
            "Parameters": "TBD",
            "Receptive Field": "TBD",
            "MSE": format_metric(a2_full_metrics["MSE"]),
            "MAE": format_metric(a2_full_metrics["MAE"]),
            "ESR": format_metric(a2_full_metrics["ESR"]),
            "MRSTFT": format_metric(a2_full_metrics["MRSTFT"]),
            "SNR": format_metric(a2_full_metrics["SNR"]),
            "Inference Notes": f"A2 smoke status {a2_status}; SlimmableContainer inspected; " + a2_full_note,
        },
    ]
    custom_row = custom_tcn_row(args, target_path)
    if custom_row is not None:
        rows.append(custom_row)
    return rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    columns = [
        "Model",
        "Architecture",
        "Channels",
        "Training Version",
        "Output Path",
        "Prediction Path",
        "File Size",
        "Parameters",
        "Receptive Field",
        "MSE",
        "MAE",
        "ESR",
        "MRSTFT",
        "SNR",
        "Inference Notes",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Experiment Comparison\n\n"
        "This table compares the currently available A1/A2 baseline artifacts and optional custom TCN predictions. "
        "Audio metrics are reported only when prediction audio is available; otherwise they are marked as TBD.\n\n"
        + markdown_table(rows)
        + "\n"
    )
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate A1/A2 experiment comparison table.")
    parser.add_argument("--artifact-only", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--a1-pred")
    parser.add_argument("--a2-lite-pred")
    parser.add_argument("--a2-full-pred")
    parser.add_argument("--custom-tcn-pred")
    parser.add_argument("--custom-tcn-name", default="GatedTCN-Medium")
    parser.add_argument("--custom-tcn-checkpoint")
    parser.add_argument("--output", default="reports/experiment_comparison.md")
    args = parser.parse_args()

    rows = model_rows(args)
    output_path = resolve_path(args.output)
    assert output_path is not None
    write_report(rows, output_path)
    print(f"comparison report: {relative(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
