from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import soundfile as sf
import torch

from src.ntt.data.dataset import PairedAudioChunkDataset
from src.ntt.evaluation.metrics import compute_metrics
from src.ntt.tcn.checkpoint import load_checkpoint
from src.ntt.tcn.model import GatedTCN
from src.ntt.tcn.utils import created_at, resolve_path, sanitize_paths_for_json, select_device, write_json


METRIC_KEYS = {
    "MSE": "mse",
    "MAE": "mae",
    "ESR": "esr",
    "MRSTFT": "mrstft",
    "SNR dB": "snr",
}


def load_metadata(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Metadata file not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def test_items(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    items = [item for item in metadata.get("chunks", []) if item.get("split") == "test"]
    if not items:
        raise ValueError("Metadata contains no test split chunks")
    return items


def load_audio_channel_first(path: str | Path) -> tuple[np.ndarray, int]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    data, sample_rate = sf.read(resolved, always_2d=True, dtype="float32")
    if not np.isfinite(data).all():
        raise ValueError(f"Audio contains NaN or Inf: {resolved}")
    return data.T, int(sample_rate)


def average_metric_dicts(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("No metric values to average")
    return {key: float(np.mean([item[key] for item in values])) for key in values[0]}


def normalize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {output_key: float(metrics[input_key]) for input_key, output_key in METRIC_KEYS.items()}


def evaluate_prediction_file(
    *,
    metadata_path: str | Path,
    target_path: str | Path,
    prediction_path: str | Path,
    model_name: str,
) -> dict[str, Any]:
    metadata = load_metadata(metadata_path)
    items = test_items(metadata)
    target, target_rate = load_audio_channel_first(target_path)
    prediction, pred_rate = load_audio_channel_first(prediction_path)
    if target_rate != pred_rate:
        raise ValueError(f"Sample rates differ: target={target_rate}, prediction={pred_rate}")

    per_chunk: list[dict[str, float]] = []
    for item in items:
        start = int(item["start_sample"])
        end = int(item["end_sample"])
        if end > target.shape[-1] or end > prediction.shape[-1]:
            raise ValueError(f"Test chunk {start}:{end} exceeds target or prediction length")
        metrics = compute_metrics(prediction[..., start:end], target[..., start:end], match_length=True)
        per_chunk.append(normalize_metrics(metrics))

    summary = average_metric_dicts(per_chunk)
    return {
        "model_name": model_name,
        "split": "test",
        "mode": "prediction",
        "num_test_chunks": len(items),
        "metadata_path": metadata_path,
        "target_path": target_path,
        "prediction_path": prediction_path,
        "sample_rate": target_rate,
        "device": "cpu",
        "created_at": created_at(),
        **summary,
    }


@torch.inference_mode()
def evaluate_tcn_checkpoint(
    *,
    metadata_path: str | Path,
    checkpoint_path: str | Path,
    model_name: str,
    device_name: str,
) -> dict[str, Any]:
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint does not contain model config under key 'config'")
    model = GatedTCN.from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = PairedAudioChunkDataset(metadata_path, split="test")
    if len(dataset) == 0:
        raise ValueError("Test split has no chunks")

    per_chunk: list[dict[str, float]] = []
    sample_rate: int | None = None
    for item in dataset:
        dry = item["dry"].unsqueeze(0).to(device)
        wet = item["wet"]
        pred = model(dry).squeeze(0).detach().cpu()
        metrics = compute_metrics(pred, wet, match_length=True)
        per_chunk.append(normalize_metrics(metrics))
        sample_rate = int(item["sample_rate"])

    summary = average_metric_dicts(per_chunk)
    return {
        "model_name": model_name,
        "split": "test",
        "mode": "tcn_checkpoint",
        "num_test_chunks": len(dataset),
        "metadata_path": metadata_path,
        "checkpoint_path": checkpoint_path,
        "sample_rate": sample_rate,
        "device": device.type,
        "parameter_count": int(checkpoint.get("parameter_count", model.count_parameters())),
        "receptive_field": int(checkpoint.get("receptive_field", model.receptive_field())),
        "created_at": created_at(),
        **summary,
    }


def write_result(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_paths_for_json(payload)
    write_json(path, sanitized)
    return sanitized


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate held-out test split metrics.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--target")
    parser.add_argument("--prediction")
    parser.add_argument("--tcn-checkpoint")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    try:
        if args.tcn_checkpoint:
            result = evaluate_tcn_checkpoint(
                metadata_path=args.metadata,
                checkpoint_path=args.tcn_checkpoint,
                model_name=args.model_name,
                device_name=args.device,
            )
        else:
            if not args.target or not args.prediction:
                raise ValueError("prediction-file mode requires --target and --prediction")
            result = evaluate_prediction_file(
                metadata_path=args.metadata,
                target_path=args.target,
                prediction_path=args.prediction,
                model_name=args.model_name,
            )
        payload = write_result(args.out, result)
        print(f"test metrics: {args.out}")
        print(f"MSE: {payload['mse']:.6g}")
        print(f"ESR: {payload['esr']:.6g}")
        print(f"MRSTFT: {payload['mrstft']:.6g}")
        print(f"SNR: {payload['snr']:.6g}")
    except Exception as exc:
        raise SystemExit(f"ERROR: test split evaluation failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
