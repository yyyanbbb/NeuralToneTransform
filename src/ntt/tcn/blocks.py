from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """One-dimensional causal convolution with explicit left-only padding.

    Inputs and outputs use channel-first audio tensors shaped
    ``(batch, channels, samples)``. For ``kernel_size > 1`` or ``dilation > 1``,
    padding is applied only to the left side of the time axis so the output at
    time ``t`` cannot depend on future input samples.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if dilation < 1:
            raise ValueError("dilation must be positive")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, channels, samples), got {tuple(x.shape)}")
        padded = F.pad(x, (self.left_padding, 0))
        y = self.conv(padded)
        if y.shape[-1] != x.shape[-1]:
            raise RuntimeError(f"CausalConv1d changed length from {x.shape[-1]} to {y.shape[-1]}")
        return y


class GatedTCNBlock(nn.Module):
    """Causal dilated gated TCN block with residual and skip projections.

    The block computes a causal dilated convolution from ``channels`` to
    ``2 * channels``, splits the result into filter and gate tensors, applies
    ``tanh(filter) * sigmoid(gate)``, then projects the gated activation into
    a residual update and a skip connection.

    Args:
        channels: Number of residual channels in and out of the block.
        skip_channels: Number of channels emitted by the skip projection.
        kernel_size: Temporal kernel size for the causal dilated convolution.
        dilation: Dilation factor for the causal convolution.
        dropout: Optional dropout probability applied after gating.

    Returns:
        A tuple ``(output, skip)`` where ``output`` has the same shape as the
        input and ``skip`` has shape ``(batch, skip_channels, samples)``.
    """

    def __init__(
        self,
        *,
        channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if skip_channels < 1:
            raise ValueError("skip_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.causal_conv = CausalConv1d(
            channels,
            2 * channels,
            kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.residual_projection = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_projection = nn.Conv1d(channels, skip_channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.residual_projection, self.skip_projection):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.causal_conv(x)
        filter_part, gate_part = torch.chunk(y, 2, dim=1)
        gated = torch.tanh(filter_part) * torch.sigmoid(gate_part)
        gated = self.dropout(gated)
        residual = self.residual_projection(gated)
        skip = self.skip_projection(gated)
        return x + residual, skip
