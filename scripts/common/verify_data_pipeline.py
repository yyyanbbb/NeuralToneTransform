from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_ALIGNMENT_FIELDS = {
    "dry_path",
    "wet_path",
    "output_dry_path",
    "output_wet_path",
    "sample_rate",
    "original_num_samples_dry",
    "original_num_samples_wet",
    "aligned_num_samples",
    "estimated_delay_samples",
    "dry_peak",
    "wet_peak",
    "dry_clipping_risk",
    "wet_clipping_risk",
    "strict_clipping",
    "warnings",
    "created_at",
}

REQUIRED_CHUNK_FIELDS = {
    "source_dry",
    "source_wet",
    "sample_rate",
    "chunk_size",
    "hop_size",
    "total_chunks",
    "train_chunks",
    "val_chunks",
    "test_chunks",
    "split_ratios",
    "chunks",
}


@dataclass
class Result:
    passed: bool
    message: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_required_fields(payload: dict[str, Any], required: set[str], label: str) -> list[Result]:
    return [
        Result(field in payload, f"{label} contains field: {field}")
        for field in sorted(required)
    ]


def run() -> tuple[int, list[str]]:
    root = repository_root()
    sys.path.insert(0, str(root))
    from src.ntt.data.dataset import PairedAudioChunkDataset

    checks: list[Result] = []
    aligned_dry = root / "data" / "aligned" / "aligned_dry.wav"
    aligned_wet = root / "data" / "aligned" / "aligned_wet.wav"
    alignment_metadata_path = root / "data" / "aligned" / "alignment_metadata.json"
    chunk_metadata_path = root / "data" / "chunks" / "metadata.json"

    checks.extend(
        [
            Result(aligned_dry.is_file(), "data/aligned/aligned_dry.wav exists"),
            Result(aligned_wet.is_file(), "data/aligned/aligned_wet.wav exists"),
            Result(alignment_metadata_path.is_file(), "data/aligned/alignment_metadata.json exists"),
            Result(chunk_metadata_path.is_file(), "data/chunks/metadata.json exists"),
        ]
    )

    alignment_metadata: dict[str, Any] = {}
    chunk_metadata: dict[str, Any] = {}
    if alignment_metadata_path.is_file():
        try:
            alignment_metadata = load_json(alignment_metadata_path)
            checks.append(Result(True, "alignment metadata is valid JSON"))
            checks.extend(check_required_fields(alignment_metadata, REQUIRED_ALIGNMENT_FIELDS, "alignment metadata"))
        except Exception as exc:
            checks.append(Result(False, f"alignment metadata JSON parse failed: {exc}"))

    if chunk_metadata_path.is_file():
        try:
            chunk_metadata = load_json(chunk_metadata_path)
            checks.append(Result(True, "chunk metadata is valid JSON"))
            checks.extend(check_required_fields(chunk_metadata, REQUIRED_CHUNK_FIELDS, "chunk metadata"))
            for split in ("train", "val", "test"):
                count = int(chunk_metadata.get(f"{split}_chunks", 0))
                checks.append(Result(count > 0, f"{split} split has at least one chunk"))
        except Exception as exc:
            checks.append(Result(False, f"chunk metadata JSON parse failed: {exc}"))

    try:
        dataset = PairedAudioChunkDataset(chunk_metadata_path, split="train")
        checks.append(Result(len(dataset) > 0, "train dataset is non-empty"))
        if len(dataset) > 0:
            item = dataset[0]
            checks.append(Result(tuple(item["dry"].shape) == tuple(item["wet"].shape), "first dry/wet tensor shapes match"))
            checks.append(Result(item["dry"].ndim == 2, "first dry tensor shape is (channels, samples)"))
            checks.append(Result(item["sample_rate"] == chunk_metadata.get("sample_rate"), "dataset sample rate matches metadata"))
    except Exception as exc:
        checks.append(Result(False, f"dataset smoke load failed: {exc}"))

    pass_count = sum(1 for check in checks if check.passed)
    fail_count = len(checks) - pass_count
    lines = ["=== Data Pipeline Verification ==="]
    for check in checks:
        lines.append(f"[{'PASS' if check.passed else 'FAIL'}] {check.message}")
    lines.extend(["", "=== Summary ===", f"pass_count: {pass_count}", f"fail_count: {fail_count}"])
    lines.append(f"OVERALL: {'PASS' if fail_count == 0 else 'FAIL'}")
    return fail_count, lines


def main() -> int:
    failures, lines = run()
    for line in lines:
        print(line)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
