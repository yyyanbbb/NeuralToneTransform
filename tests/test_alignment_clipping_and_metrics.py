from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

from src.ntt.data.align import align_audio_pair
from src.ntt.evaluation.metrics import (
    error_to_signal_ratio,
    mean_absolute_error,
    mean_squared_error,
    multi_resolution_stft_loss,
    normalized_mae,
    snr_db,
)


class AlignmentClippingAndMetricsTests(unittest.TestCase):
    def test_alignment_records_clipping_warning_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dry_path = root / "dry.wav"
            wet_path = root / "wet.wav"
            dry = np.zeros(2048, dtype=np.float32)
            wet = np.zeros(2048, dtype=np.float32)
            dry[128:256] = 1.0
            wet[128:256] = 0.5
            sf.write(dry_path, dry, 48_000, subtype="FLOAT")
            sf.write(wet_path, wet, 48_000, subtype="FLOAT")

            metadata = align_audio_pair(dry_path, wet_path, root / "aligned")

            self.assertTrue(metadata["dry_clipping_risk"])
            self.assertFalse(metadata["wet_clipping_risk"])
            self.assertFalse(metadata["strict_clipping"])
            self.assertTrue(any("dry audio peak amplitude" in warning for warning in metadata["warnings"]))

    def test_alignment_strict_clipping_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dry_path = root / "dry.wav"
            wet_path = root / "wet.wav"
            dry = np.zeros(2048, dtype=np.float32)
            wet = np.zeros(2048, dtype=np.float32)
            dry[128:256] = 1.0
            wet[128:256] = 0.5
            sf.write(dry_path, dry, 48_000, subtype="FLOAT")
            sf.write(wet_path, wet, 48_000, subtype="FLOAT")

            with self.assertRaisesRegex(ValueError, "Potential clipping detected"):
                align_audio_pair(dry_path, wet_path, root / "aligned", strict_clipping=True)

    def test_metrics_match_definitions(self) -> None:
        pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        target = np.array([1.0, 1.0, 5.0], dtype=np.float32)

        self.assertAlmostEqual(mean_squared_error(pred, target), 5.0 / 3.0)
        self.assertAlmostEqual(mean_absolute_error(pred, target), 1.0)
        self.assertAlmostEqual(error_to_signal_ratio(pred, target), 5.0 / 27.0)
        self.assertAlmostEqual(normalized_mae(pred, target), 3.0 / 7.0)
        self.assertAlmostEqual(snr_db(pred, target), 10.0 * math.log10(27.0 / 5.0))

    def test_mrstft_accepts_torch_channel_first_input(self) -> None:
        target = torch.sin(torch.linspace(0, 8 * math.pi, 4096)).unsqueeze(0)
        pred = target * 0.95
        loss = multi_resolution_stft_loss(
            pred,
            target,
            fft_sizes=(256, 512),
            hop_sizes=(64, 128),
            win_lengths=(256, 512),
        )
        self.assertGreaterEqual(loss, 0.0)
        self.assertTrue(math.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
