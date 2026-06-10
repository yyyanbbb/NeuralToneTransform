from __future__ import annotations

import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path


DEFAULT_PACKAGES = [
    "torch",
    "torchaudio",
    "librosa",
    "matplotlib",
    "tensorboard",
    "soundfile",
    "scipy",
    "numpy",
    "pandas",
    "neural-amp-modeler",
]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_log(lines: list[str]) -> None:
    log_dir = repository_root() / "logs" / "common"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "print_dependency_versions.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"=== dependency version check {datetime.now().astimezone().isoformat(timespec='seconds')} ===\n")
        for line in lines:
            handle.write(f"{line}\n")


def main() -> int:
    packages = sys.argv[1:] or DEFAULT_PACKAGES
    lines: list[str] = []
    missing: list[str] = []

    for package_name in packages:
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            line = f"{package_name}: NOT INSTALLED"
            missing.append(package_name)
        else:
            line = f"{package_name}=={version}"
        lines.append(line)
        print(line)

    write_log(lines)
    if missing:
        print(f"Missing packages: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
