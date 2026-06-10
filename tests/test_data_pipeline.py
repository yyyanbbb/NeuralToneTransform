from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

from src.ntt.data.align import align_audio_pair, estimate_delay_samples
from src.ntt.data.chunk_audio import chunk_audio_pair
from src.ntt.data.dataset import PairedAudioChunkDataset


class DataPipelineTests(unittest.TestCase):
    def test_estimates_positive_wet_delay_and_writes_aligned_pair(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dry_path = root / "dry.wav"
            wet_path = root / "wet.wav"
            out_dir = root / "aligned"
            sample_rate = 48_000
            dry = np.zeros(2048, dtype=np.float32)
            dry[200:260] = 0.5
            wet = np.concatenate([np.zeros(17, dtype=np.float32), dry[:-17]])
            sf.write(dry_path, dry, sample_rate)
            sf.write(wet_path, wet, sample_rate)

            delay = estimate_delay_samples(dry, wet, max_lag_samples=128)
            self.assertEqual(delay, 17)

            metadata = align_audio_pair(dry_path, wet_path, out_dir)

            self.assertEqual(metadata["estimated_delay_samples"], 17)
            self.assertEqual(metadata["aligned_num_samples"], 2031)
            self.assertTrue((out_dir / "aligned_dry.wav").is_file())
            self.assertTrue((out_dir / "aligned_wet.wav").is_file())

    def test_chunks_in_time_order_and_dataset_loads_first_train_item(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_dir = root / "aligned"
            chunks_dir = root / "chunks"
            aligned_dir.mkdir()
            sample_rate = 48_000
            dry = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
            wet = dry * 0.25
            dry_path = aligned_dir / "aligned_dry.wav"
            wet_path = aligned_dir / "aligned_wet.wav"
            sf.write(dry_path, dry, sample_rate)
            sf.write(wet_path, wet, sample_rate)

            metadata = chunk_audio_pair(
                dry_path,
                wet_path,
                chunks_dir,
                chunk_size=16,
                hop_size=16,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
            )

            self.assertEqual(metadata["total_chunks"], 4)
            self.assertEqual(metadata["train_chunks"], 2)
            self.assertEqual(metadata["val_chunks"], 1)
            self.assertEqual(metadata["test_chunks"], 1)
            starts = [item["start_sample"] for item in metadata["chunks"]]
            self.assertEqual(starts, [0, 16, 32, 48])

            dataset = PairedAudioChunkDataset(chunks_dir / "metadata.json", split="train")
            item = dataset[0]
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(item["dry"].shape), (1, 16))
            self.assertEqual(tuple(item["wet"].shape), (1, 16))
            self.assertEqual(item["sample_rate"], sample_rate)
            self.assertEqual(item["dry"].dtype, torch.float32)

            loaded = json.loads((chunks_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["chunks"][0]["split"], "train")


if __name__ == "__main__":
    unittest.main()
