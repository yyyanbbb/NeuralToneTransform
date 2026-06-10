from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def _to_tensor(value: np.ndarray | torch.Tensor | Sequence[float]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(dtype=torch.float64, device="cpu")
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("Input contains NaN or Inf")
    return tensor


def prepare_pair(
    pred: np.ndarray | torch.Tensor | Sequence[float],
    target: np.ndarray | torch.Tensor | Sequence[float],
    *,
    match_length: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_tensor = _to_tensor(pred)
    target_tensor = _to_tensor(target)
    if pred_tensor.ndim == 0 or target_tensor.ndim == 0:
        raise ValueError("Inputs must contain at least one sample")

    if pred_tensor.shape != target_tensor.shape:
        if not match_length:
            raise ValueError(f"pred and target shapes differ: {tuple(pred_tensor.shape)} vs {tuple(target_tensor.shape)}")
        if pred_tensor.ndim != target_tensor.ndim:
            raise ValueError("match_length requires pred and target to have the same number of dimensions")
        if pred_tensor.shape[:-1] != target_tensor.shape[:-1]:
            raise ValueError("match_length can only trim the final sample dimension")
        min_len = min(pred_tensor.shape[-1], target_tensor.shape[-1])
        pred_tensor = pred_tensor[..., :min_len]
        target_tensor = target_tensor[..., :min_len]
    return pred_tensor, target_tensor


def mean_squared_error(pred, target, *, match_length: bool = False) -> float:
    pred_tensor, target_tensor = prepare_pair(pred, target, match_length=match_length)
    return float(torch.mean(torch.square(pred_tensor - target_tensor)).item())


def mean_absolute_error(pred, target, *, match_length: bool = False) -> float:
    pred_tensor, target_tensor = prepare_pair(pred, target, match_length=match_length)
    return float(torch.mean(torch.abs(pred_tensor - target_tensor)).item())


def error_to_signal_ratio(pred, target, eps: float = 1e-12, *, match_length: bool = False) -> float:
    pred_tensor, target_tensor = prepare_pair(pred, target, match_length=match_length)
    error_energy = torch.sum(torch.square(pred_tensor - target_tensor))
    signal_energy = torch.sum(torch.square(target_tensor))
    return float((error_energy / (signal_energy + eps)).item())


def normalized_mae(pred, target, eps: float = 1e-12, *, match_length: bool = False) -> float:
    pred_tensor, target_tensor = prepare_pair(pred, target, match_length=match_length)
    return float((torch.sum(torch.abs(pred_tensor - target_tensor)) / (torch.sum(torch.abs(target_tensor)) + eps)).item())


def snr_db(pred, target, eps: float = 1e-12, *, match_length: bool = False) -> float:
    pred_tensor, target_tensor = prepare_pair(pred, target, match_length=match_length)
    signal_energy = torch.sum(torch.square(target_tensor))
    noise_energy = torch.sum(torch.square(target_tensor - pred_tensor))
    return float((10.0 * torch.log10((signal_energy + eps) / (noise_energy + eps))).item())


def _channel_first(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Expected input shape (samples,) or (channels, samples), got {tuple(tensor.shape)}")


def multi_resolution_stft_loss(
    pred,
    target,
    fft_sizes=(512, 1024, 2048),
    hop_sizes=(128, 256, 512),
    win_lengths=(512, 1024, 2048),
    eps: float = 1e-7,
    *,
    match_length: bool = False,
) -> float:
    pred_tensor, target_tensor = prepare_pair(pred, target, match_length=match_length)
    pred_tensor = _channel_first(pred_tensor.to(dtype=torch.float32))
    target_tensor = _channel_first(target_tensor.to(dtype=torch.float32))
    if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
        raise ValueError("fft_sizes, hop_sizes, and win_lengths must have the same length")

    losses: list[torch.Tensor] = []
    for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
        if pred_tensor.shape[-1] < win_length:
            continue
        window = torch.hann_window(win_length, dtype=pred_tensor.dtype)
        pred_stft = torch.stft(
            pred_tensor,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_tensor,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        spectral_convergence = torch.linalg.vector_norm(target_mag - pred_mag, dim=(-2, -1)) / (
            torch.linalg.vector_norm(target_mag, dim=(-2, -1)) + eps
        )
        log_mag = torch.mean(torch.abs(torch.log(target_mag + eps) - torch.log(pred_mag + eps)), dim=(-2, -1))
        losses.append(torch.mean(spectral_convergence + log_mag))

    if not losses:
        raise ValueError("Input is shorter than all configured STFT window lengths")
    return float(torch.mean(torch.stack(losses)).item())


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    data, sample_rate = sf.read(resolved, always_2d=True, dtype="float32")
    tensor = torch.as_tensor(data.T, dtype=torch.float32)
    return tensor, int(sample_rate)


def compute_metrics(pred, target, *, match_length: bool = False) -> dict[str, float]:
    return {
        "MSE": mean_squared_error(pred, target, match_length=match_length),
        "MAE": mean_absolute_error(pred, target, match_length=match_length),
        "ESR": error_to_signal_ratio(pred, target, match_length=match_length),
        "Normalized MAE": normalized_mae(pred, target, match_length=match_length),
        "SNR dB": snr_db(pred, target, match_length=match_length),
        "MRSTFT": multi_resolution_stft_loss(pred, target, match_length=match_length),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute objective audio metrics.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--match-length", action="store_true")
    args = parser.parse_args()

    try:
        pred, pred_rate = load_audio(args.pred)
        target, target_rate = load_audio(args.target)
        if pred_rate != target_rate:
            raise ValueError(f"Sample rates differ: pred={pred_rate}, target={target_rate}")
        metrics = compute_metrics(pred, target, match_length=args.match_length)
    except Exception as exc:
        raise SystemExit(f"ERROR: metrics failed: {exc}") from exc

    for name, value in metrics.items():
        print(f"{name}: {value:.10g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
