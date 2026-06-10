from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


SPLITS = ("train", "val", "test")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def metadata_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    data, sample_rate = sf.read(resolved, always_2d=True, dtype="float64")
    return data, int(sample_rate)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios):.12f}")


def split_for_index(index: int, train_count: int, val_count: int) -> str:
    if index < train_count:
        return "train"
    if index < train_count + val_count:
        return "val"
    return "test"


def chunk_audio_pair(
    dry_path: str | Path,
    wet_path: str | Path,
    out_dir: str | Path,
    *,
    chunk_size: int,
    hop_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool = False,
    seed: int = 0,
) -> dict:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    validate_ratios(train_ratio, val_ratio, test_ratio)

    dry_resolved = resolve_path(dry_path)
    wet_resolved = resolve_path(wet_path)
    out_resolved = resolve_path(out_dir)
    dry, dry_rate = read_audio(dry_resolved)
    wet, wet_rate = read_audio(wet_resolved)

    if dry_rate != wet_rate:
        raise ValueError(f"Sample rates differ: dry={dry_rate}, wet={wet_rate}")
    if dry.shape != wet.shape:
        raise ValueError(f"Dry/wet shapes differ: dry={dry.shape}, wet={wet.shape}")
    if len(dry) < chunk_size:
        raise ValueError(f"Audio length {len(dry)} is shorter than chunk_size {chunk_size}")

    for split in SPLITS:
        (out_resolved / split).mkdir(parents=True, exist_ok=True)

    starts = list(range(0, len(dry) - chunk_size + 1, hop_size))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(starts)

    total_chunks = len(starts)
    train_count = int(total_chunks * train_ratio)
    val_count = int(total_chunks * val_ratio)
    test_count = total_chunks - train_count - val_count

    chunks: list[dict] = []
    split_counters = {"train": 0, "val": 0, "test": 0}

    for global_index, start in enumerate(starts):
        split = split_for_index(global_index, train_count, val_count)
        split_index = split_counters[split]
        split_counters[split] += 1
        end = start + chunk_size
        dry_chunk_path = out_resolved / split / f"dry_{split_index:06d}.wav"
        wet_chunk_path = out_resolved / split / f"wet_{split_index:06d}.wav"
        sf.write(dry_chunk_path, dry[start:end], dry_rate)
        sf.write(wet_chunk_path, wet[start:end], dry_rate)
        chunks.append(
            {
                "split": split,
                "dry_path": metadata_path(dry_chunk_path),
                "wet_path": metadata_path(wet_chunk_path),
                "start_sample": int(start),
                "end_sample": int(end),
            }
        )

    metadata = {
        "source_dry": metadata_path(dry_resolved),
        "source_wet": metadata_path(wet_resolved),
        "sample_rate": dry_rate,
        "chunk_size": int(chunk_size),
        "hop_size": int(hop_size),
        "total_chunks": int(total_chunks),
        "train_chunks": int(split_counters["train"]),
        "val_chunks": int(split_counters["val"]),
        "test_chunks": int(split_counters["test"]),
        "split_ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "shuffle": bool(shuffle),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chunks": chunks,
    }
    (out_resolved / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk aligned dry/wet WAV files.")
    parser.add_argument("--dry", required=True)
    parser.add_argument("--wet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--hop-size", type=int, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        metadata = chunk_audio_pair(
            args.dry,
            args.wet,
            args.out_dir,
            chunk_size=args.chunk_size,
            hop_size=args.hop_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: chunking failed: {exc}") from exc

    print("Chunking complete")
    print(f"total_chunks: {metadata['total_chunks']}")
    print(f"train_chunks: {metadata['train_chunks']}")
    print(f"val_chunks: {metadata['val_chunks']}")
    print(f"test_chunks: {metadata['test_chunks']}")
    print(f"metadata: {metadata_path(resolve_path(args.out_dir) / 'metadata.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
