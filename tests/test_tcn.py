from __future__ import annotations

import unittest

import torch

from src.ntt.tcn.blocks import CausalConv1d, GatedTCNBlock
from src.ntt.tcn.losses import CompositeToneLoss, esr_loss, mrstft_loss
from src.ntt.tcn.model import GatedTCN


class TCNBlockTests(unittest.TestCase):
    def test_causal_conv_preserves_length_and_does_not_read_future(self) -> None:
        conv = CausalConv1d(1, 1, kernel_size=3, dilation=2, bias=False)
        with torch.no_grad():
            conv.conv.weight.fill_(1.0)

        x = torch.zeros(1, 1, 12)
        x[..., 8] = 1.0
        y = conv(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.equal(y[..., :8], torch.zeros_like(y[..., :8])))
        self.assertGreater(float(y[..., 8].item()), 0.0)

    def test_gated_tcn_block_returns_residual_and_skip_with_matching_time_axis(self) -> None:
        block = GatedTCNBlock(channels=4, skip_channels=6, kernel_size=3, dilation=4, dropout=0.1)
        x = torch.randn(2, 4, 32)

        residual, skip = block(x)

        self.assertEqual(tuple(residual.shape), tuple(x.shape))
        self.assertEqual(tuple(skip.shape), (2, 6, 32))


class GatedTCNModelTests(unittest.TestCase):
    def test_model_preserves_input_shape_and_reports_receptive_field(self) -> None:
        model = GatedTCN(
            in_channels=1,
            out_channels=1,
            channels=8,
            skip_channels=8,
            kernel_size=3,
            dilations=[1, 2, 4],
            dropout=0.0,
        )
        x = torch.randn(2, 1, 64)

        y = model(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(model.receptive_field(), 1 + 2 * (1 + 2 + 4))
        self.assertGreater(model.count_parameters(), 0)

    def test_from_config_builds_model(self) -> None:
        model = GatedTCN.from_config(
            {
                "in_channels": 1,
                "out_channels": 1,
                "channels": 4,
                "skip_channels": 4,
                "kernel_size": 5,
                "dilations": [1, 3],
                "dropout": 0.0,
                "final_activation": "tanh",
            }
        )

        self.assertEqual(model.receptive_field(), 1 + 4 * (1 + 3))
        self.assertEqual(tuple(model(torch.randn(1, 1, 40)).shape), (1, 1, 40))


class TCNLossTests(unittest.TestCase):
    def test_esr_loss_is_differentiable(self) -> None:
        pred = torch.tensor([[[0.0, 0.5, 1.0]]], requires_grad=True)
        target = torch.tensor([[[0.0, 1.0, 1.0]]])

        loss = esr_loss(pred, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(pred.grad)

    def test_mrstft_loss_handles_shorter_fft_configuration(self) -> None:
        target = torch.sin(torch.linspace(0, 8 * torch.pi, 1024)).view(1, 1, -1)
        pred = (target * 0.9).clone().requires_grad_(True)

        loss = mrstft_loss(
            pred,
            target,
            fft_sizes=(128, 256),
            hop_sizes=(32, 64),
            win_lengths=(128, 256),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(pred.grad)

    def test_composite_loss_returns_total_and_component_tensors(self) -> None:
        pred = torch.randn(2, 1, 512, requires_grad=True)
        target = torch.randn(2, 1, 512)
        loss_fn = CompositeToneLoss(mse_weight=1.0, esr_weight=0.1, mrstft_weight=0.01)

        total, components = loss_fn(pred, target)

        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(components), {"mse", "esr", "mrstft"})
        total.backward()
        self.assertIsNotNone(pred.grad)


if __name__ == "__main__":
    unittest.main()
