from __future__ import annotations

import argparse
import json
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


def format_optional(value: Any) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


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


def benchmark_rtf(variant: str, pred_path: Path | None, checkpoint_path: Path | None) -> tuple[str, str]:
    candidates: list[Path] = []
    if checkpoint_path is not None:
        candidates.append(checkpoint_path.parent.parent / "benchmark.json")
    if pred_path is not None:
        candidates.append(pred_path.parent / "benchmark.json")
    candidates.append(repository_root() / "outputs" / "tcn_gated" / variant / "benchmark.json")
    for candidate in candidates:
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return format_optional(data.get("real_time_factor")), "benchmark metadata loaded"
    return "TBD", "benchmark metadata not available"


def json_rtf(path: str | Path) -> tuple[str, str]:
    resolved = resolve_path(path)
    if resolved is None or not resolved.is_file():
        return "TBD", "benchmark metadata not available"
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return format_optional(data.get("real_time_factor")), "benchmark metadata loaded"


def load_test_metrics(path: str | Path) -> tuple[dict[str, str], str]:
    empty = {
        "Test MSE": "TBD",
        "Test MAE": "TBD",
        "Test ESR": "TBD",
        "Test MRSTFT": "TBD",
        "Test SNR": "TBD",
    }
    resolved = resolve_path(path)
    if resolved is None or not resolved.is_file():
        return empty, "test metrics not available"
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        "Test MSE": format_optional(data.get("mse")),
        "Test MAE": format_optional(data.get("mae")),
        "Test ESR": format_optional(data.get("esr")),
        "Test MRSTFT": format_optional(data.get("mrstft")),
        "Test SNR": format_optional(data.get("snr")),
    }, "held-out test metrics loaded"


def tcn_training_note(variant: str) -> str:
    metrics_path = repository_root() / "outputs" / "tcn_gated" / variant / "training_metrics.json"
    if not metrics_path.is_file():
        return "training metrics not available"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    completed_epochs = data.get("completed_epochs", len(data.get("epochs", [])))
    num_epochs = data.get("num_epochs", data.get("training_config", {}).get("num_epochs"))
    training_config_path = str(data.get("training_config_path", ""))
    if completed_epochs == 20 and num_epochs == 20 and "formal" in training_config_path:
        return "Formal 20-epoch checkpoint"
    return "Smoke checkpoint, not formal training"


def tcn_variant_row(
    *,
    variant: str,
    default_name: str,
    pred_arg: str | Path | None,
    checkpoint_arg: str | Path | None,
    target_path: Path | None,
) -> dict[str, str]:
    pred_path = resolve_path(pred_arg)
    checkpoint_path = resolve_path(checkpoint_arg)
    metrics, metric_note = prediction_metrics(pred_path, target_path)
    metadata, checkpoint_note = checkpoint_metadata(checkpoint_path)
    config = metadata.get("config", {})
    if not config:
        config_path = repository_root() / "configs" / "tcn_gated" / f"{variant}.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
    channels = config.get("channels", "TBD") if isinstance(config, dict) else "TBD"
    skip_channels = config.get("skip_channels", "TBD") if isinstance(config, dict) else "TBD"
    parameter_count = metadata.get("parameter_count", "TBD")
    receptive_field = metadata.get("receptive_field", "TBD")
    model_name = config.get("model_name", default_name) if isinstance(config, dict) else default_name
    rtf, benchmark_note = benchmark_rtf(variant, pred_path, checkpoint_path)
    test_values, test_note = load_test_metrics(f"reports/test_metrics_tcn_{variant}.json")
    training_note = tcn_training_note(variant)

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
        **test_values,
        "RTF": rtf,
        "Inference Notes": f"{training_note}; {checkpoint_note}; {benchmark_note}; {test_note}; {metric_note}",
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
    a1_test, a1_test_note = load_test_metrics("reports/test_metrics_a1.json")
    a2_lite_test, a2_lite_test_note = load_test_metrics("reports/test_metrics_a2_lite.json")
    a2_full_test, a2_full_test_note = load_test_metrics("reports/test_metrics_a2_full.json")
    a1_rtf, a1_benchmark_note = json_rtf("outputs/a1_baseline/benchmark.json")
    a2_lite_rtf, a2_lite_benchmark_note = json_rtf("outputs/a2_baseline/a2_lite_benchmark.json")
    a2_full_rtf, a2_full_benchmark_note = json_rtf("outputs/a2_baseline/a2_full_benchmark.json")
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
            **a1_test,
            "RTF": a1_rtf,
            "Inference Notes": "legacy NAM baseline artifact; " + f"{a1_benchmark_note}; {a1_test_note}; {a1_note}",
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
            **a2_lite_test,
            "RTF": a2_lite_rtf,
            "Inference Notes": f"A2 smoke status {a2_status}; SlimmableContainer inspected; {a2_lite_benchmark_note}; {a2_lite_test_note}; " + a2_lite_note,
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
            **a2_full_test,
            "RTF": a2_full_rtf,
            "Inference Notes": f"A2 smoke status {a2_status}; SlimmableContainer inspected; {a2_full_benchmark_note}; {a2_full_test_note}; " + a2_full_note,
        },
    ]
    medium_pred = getattr(args, "tcn_medium_pred", None) or getattr(args, "custom_tcn_pred", None)
    medium_checkpoint = getattr(args, "tcn_medium_checkpoint", None) or getattr(args, "custom_tcn_checkpoint", None)
    rows.extend(
        [
            tcn_variant_row(
                variant="small",
                default_name="GatedTCN-Small",
                pred_arg=getattr(args, "tcn_small_pred", None),
                checkpoint_arg=getattr(args, "tcn_small_checkpoint", None),
                target_path=target_path,
            ),
            tcn_variant_row(
                variant="medium",
                default_name=getattr(args, "custom_tcn_name", None) or "GatedTCN-Medium",
                pred_arg=medium_pred,
                checkpoint_arg=medium_checkpoint,
                target_path=target_path,
            ),
            tcn_variant_row(
                variant="large",
                default_name="GatedTCN-Large",
                pred_arg=getattr(args, "tcn_large_pred", None),
                checkpoint_arg=getattr(args, "tcn_large_checkpoint", None),
                target_path=target_path,
            ),
        ]
    )
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
        "Test MSE",
        "Test MAE",
        "Test ESR",
        "Test MRSTFT",
        "Test SNR",
        "RTF",
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
        "This table compares A1/A2 baselines and custom TCN variants. Full-file metrics use aligned prediction audio when available; held-out test metrics come from `reports/test_metrics_*.json`. Missing metrics remain `TBD`.\n\n"
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
    parser.add_argument("--tcn-small-pred")
    parser.add_argument("--tcn-small-checkpoint")
    parser.add_argument("--tcn-medium-pred")
    parser.add_argument("--tcn-medium-checkpoint")
    parser.add_argument("--tcn-large-pred")
    parser.add_argument("--tcn-large-checkpoint")
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
