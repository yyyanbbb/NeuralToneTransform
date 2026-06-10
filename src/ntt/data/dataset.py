from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "val", "test"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def load_audio_tensor(path: Path, *, dtype: torch.dtype, normalize: bool, target_peak: float) -> tuple[torch.Tensor, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Audio chunk not found: {path}")
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if not np.isfinite(data).all():
        raise ValueError(f"Audio chunk contains NaN or Inf: {path}")
    tensor = torch.as_tensor(data.T, dtype=dtype)
    if normalize:
        peak = torch.max(torch.abs(tensor))
        if peak > 0:
            tensor = tensor * (target_peak / peak)
    return tensor, int(sample_rate)


class PairedAudioChunkDataset(Dataset):
    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        *,
        normalize: bool = False,
        target_peak: float = 0.95,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
        if target_peak <= 0:
            raise ValueError("target_peak must be positive")

        self.metadata_path = resolve_path(metadata_path)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Chunk metadata not found: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.split = split
        self.normalize = normalize
        self.target_peak = target_peak
        self.dtype = dtype
        self.items = [item for item in self.metadata.get("chunks", []) if item.get("split") == split]

        if not isinstance(self.items, list):
            raise ValueError("metadata field 'chunks' must be a list")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        dry_path = resolve_path(item["dry_path"])
        wet_path = resolve_path(item["wet_path"])
        dry, dry_rate = load_audio_tensor(
            dry_path,
            dtype=self.dtype,
            normalize=self.normalize,
            target_peak=self.target_peak,
        )
        wet, wet_rate = load_audio_tensor(
            wet_path,
            dtype=self.dtype,
            normalize=self.normalize,
            target_peak=self.target_peak,
        )

        if dry_rate != wet_rate:
            raise ValueError(f"Sample rates differ for item {index}: dry={dry_rate}, wet={wet_rate}")
        if dry.shape != wet.shape:
            raise ValueError(f"Dry/wet tensor shapes differ for item {index}: dry={dry.shape}, wet={wet.shape}")

        return {
            "dry": dry,
            "wet": wet,
            "dry_path": item["dry_path"],
            "wet_path": item["wet_path"],
            "sample_rate": dry_rate,
        }


def parse_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test paired audio chunk dataset loading.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="train")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--target-peak", type=float, default=0.95)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    try:
        dataset = PairedAudioChunkDataset(
            args.metadata,
            args.split,
            normalize=args.normalize,
            target_peak=args.target_peak,
            dtype=parse_dtype(args.dtype),
        )
        print(f"dataset length: {len(dataset)}")
        if len(dataset) == 0:
            return 0
        item = dataset[0]
        print(f"first item dry shape: {tuple(item['dry'].shape)}")
        print(f"first item wet shape: {tuple(item['wet'].shape)}")
        print(f"sample rate: {item['sample_rate']}")
    except Exception as exc:
        raise SystemExit(f"ERROR: dataset smoke test failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
