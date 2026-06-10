from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from src.ntt.tcn.blocks import GatedTCNBlock


class GatedTCN(nn.Module):
    """Custom causal dilated gated TCN for dry-to-wet audio prediction."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 32,
        skip_channels: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] | None = None,
        dropout: float = 0.0,
        final_activation: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]
        if not dilations:
            raise ValueError("dilations must contain at least one value")
        if any(dilation < 1 for dilation in dilations):
            raise ValueError("all dilations must be positive")
        if final_activation not in (None, "tanh"):
            raise ValueError("final_activation must be None or 'tanh'")

        self.model_name = model_name or "GatedTCN"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilations = [int(dilation) for dilation in dilations]
        self.dropout = dropout
        self.final_activation = final_activation

        self.input_projection = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                GatedTCNBlock(
                    channels=channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in self.dilations
            ]
        )
        self.output_head = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(skip_channels, out_channels, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, channels, samples), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")

        y = self.input_projection(x)
        skip_sum: torch.Tensor | None = None
        for block in self.blocks:
            y, skip = block(y)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        if skip_sum is None:
            raise RuntimeError("GatedTCN has no blocks")
        output = self.output_head(skip_sum)
        if self.final_activation == "tanh":
            output = torch.tanh(output)
        if output.shape[-1] != x.shape[-1]:
            raise RuntimeError(f"GatedTCN changed length from {x.shape[-1]} to {output.shape[-1]}")
        return output

    def receptive_field(self) -> int:
        return 1 + sum((self.kernel_size - 1) * dilation for dilation in self.dilations)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GatedTCN":
        allowed_keys = {
            "model_name",
            "in_channels",
            "out_channels",
            "channels",
            "skip_channels",
            "kernel_size",
            "dilations",
            "dropout",
            "final_activation",
        }
        model_kwargs = {key: value for key, value in config.items() if key in allowed_keys}
        return cls(**model_kwargs)
