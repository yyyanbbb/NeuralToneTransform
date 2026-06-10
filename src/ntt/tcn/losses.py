from __future__ import annotations

import torch
from torch import nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error for channel-first audio tensors."""

    _validate_pair(pred, target)
    return torch.mean(torch.square(pred - target))


def esr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Error-to-signal ratio loss for channel-first audio tensors."""

    _validate_pair(pred, target)
    error_energy = torch.sum(torch.square(pred - target))
    signal_energy = torch.sum(torch.square(target))
    return error_energy / (signal_energy + eps)


def mrstft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: tuple[int, ...] = (128, 256, 512),
    win_lengths: tuple[int, ...] = (512, 1024, 2048),
    eps: float = 1e-7,
) -> torch.Tensor:
    """Multi-resolution STFT loss with device-local Hann windows."""

    _validate_pair(pred, target)
    if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
        raise ValueError("fft_sizes, hop_sizes, and win_lengths must have the same length")
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    losses: list[torch.Tensor] = []

    for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
        if pred_flat.shape[-1] < win_length:
            continue
        window = torch.hann_window(win_length, dtype=pred.dtype, device=pred.device)
        pred_stft = torch.stft(
            pred_flat,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_flat,
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
    return torch.mean(torch.stack(losses))


class CompositeToneLoss(nn.Module):
    """Weighted MSE + ESR + multi-resolution STFT training objective."""

    def __init__(self, *, mse_weight: float = 1.0, esr_weight: float = 0.1, mrstft_weight: float = 0.001) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.esr_weight = float(esr_weight)
        self.mrstft_weight = float(mrstft_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mse = mse_loss(pred, target)
        esr = esr_loss(pred, target)
        mrstft = mrstft_loss(pred, target)
        total = self.mse_weight * mse + self.esr_weight * esr + self.mrstft_weight * mrstft
        return total, {"mse": mse, "esr": esr, "mrstft": mrstft}


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"Expected shape (batch, channels, samples), got {tuple(pred.shape)}")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("pred and target must not contain NaN or Inf")
