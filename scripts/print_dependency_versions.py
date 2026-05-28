from __future__ import annotations

import sys
from importlib import metadata


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


def main() -> int:
    packages = sys.argv[1:] or DEFAULT_PACKAGES
    for package_name in packages:
        print(f"{package_name}=={metadata.version(package_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
