from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+"),
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
    re.compile(r"/mnt/c/Users/"),
]


@dataclass
class CheckResult:
    passed: bool
    message: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def file_exists(root: Path, relative: str) -> bool:
    return (root / relative).is_file()


def dir_exists(root: Path, relative: str) -> bool:
    return (root / relative).is_dir()


def config_has_private_absolute_path(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS)


def json_loads(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def target_checks(root: Path, target: str) -> list[CheckResult]:
    if target == "a1":
        config_paths = [
            "configs/a1_baseline/data.json",
            "configs/a1_baseline/model.json",
            "configs/a1_baseline/learning.json",
        ]
        checks = [
            CheckResult(dir_exists(root, ".venv"), ".venv directory exists"),
            CheckResult(file_exists(root, "requirements-a1.txt"), "requirements-a1.txt exists"),
            CheckResult(file_exists(root, "data/raw/baseline_input.wav"), "baseline input WAV exists"),
            CheckResult(file_exists(root, "data/raw/baseline_output.wav"), "baseline output WAV exists"),
            CheckResult(file_exists(root, "outputs/a1_baseline/model.nam"), "A1 canonical model exists"),
            CheckResult(dir_exists(root, "logs/a1"), "logs/a1 directory exists"),
        ]
    elif target == "a2":
        config_paths = [
            "configs/a2_baseline/data.json",
            "configs/a2_baseline/model_packed.json",
            "configs/a2_baseline/learning.json",
        ]
        checks = [
            CheckResult(dir_exists(root, ".venv-a2"), ".venv-a2 directory exists"),
            CheckResult(file_exists(root, "requirements-a2.txt"), "requirements-a2.txt exists"),
            CheckResult(file_exists(root, "data/raw/baseline_input.wav"), "baseline input WAV exists"),
            CheckResult(file_exists(root, "data/raw/baseline_output.wav"), "baseline output WAV exists"),
            CheckResult(file_exists(root, "outputs/a2_baseline/model.nam"), "A2 canonical model exists"),
            CheckResult(dir_exists(root, "logs/a2"), "logs/a2 directory exists"),
            CheckResult(file_exists(root, "reports/a2_model_inspection.md"), "A2 inspection report exists"),
        ]
    else:
        raise ValueError(f"Unsupported target: {target}")

    for config_path in config_paths:
        full_path = root / config_path
        checks.append(CheckResult(full_path.is_file(), f"{config_path} exists"))
        if full_path.is_file():
            checks.append(CheckResult(json_loads(full_path), f"{config_path} is valid JSON"))
            checks.append(
                CheckResult(
                    not config_has_private_absolute_path(full_path),
                    f"{config_path} does not contain private absolute paths",
                )
            )

    return checks


def run(target: str) -> tuple[int, list[str]]:
    root = repository_root()
    targets = ["a1", "a2"] if target == "all" else [target]
    lines = [
        "=== NeuralToneTransform Reproducibility Verification ===",
        f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"project root: {root}",
        f"target: {target}",
    ]
    failures = 0
    passes = 0

    for item in targets:
        lines.append("")
        lines.append(f"[{item.upper()}]")
        for result in target_checks(root, item):
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"[{status}] {result.message}")
            if result.passed:
                passes += 1
            else:
                failures += 1

    lines.extend(["", "=== Summary ===", f"pass_count: {passes}", f"fail_count: {failures}"])
    lines.append(f"OVERALL: {'PASS' if failures == 0 else 'FAIL'}")
    return failures, lines


def write_log(lines: list[str]) -> Path:
    log_path = repository_root() / "logs" / "common" / "verify_reproducibility.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify A1/A2 reproducibility artifacts.")
    parser.add_argument("--target", choices=["a1", "a2", "all"], default="all")
    args = parser.parse_args()

    failures, lines = run(args.target)
    for line in lines:
        print(line)
    write_log(lines)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
