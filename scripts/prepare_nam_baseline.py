from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


INPUT_URL = "https://drive.google.com/uc?export=download&id=1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G"
OUTPUT_URL = "https://drive.google.com/uc?export=download&id=1NrpQLBbCDHyu0RPsne4YcjIpi5-rEP6w"


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"reuse existing file: {dst}")
        return
    print(f"downloading: {url}")
    with urlopen(url) as resp:
        data = resp.read()
    dst.write_bytes(data)
    print(f"saved: {dst} ({len(data)} bytes)")


def relpath_for_config(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def package_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    cfg_dir = root / "configs" / "nam_baseline"
    out_dir = root / "outputs" / "nam_baseline"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = raw_dir / "baseline_input.wav"
    y_path = raw_dir / "baseline_output.wav"
    print("=== Step2 Prepare NAM Baseline ===")
    print(f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"python executable: {sys.executable}")
    print(f"project root: {root}")
    print(f"nam version: {package_version('nam')}")
    download(INPUT_URL, x_path)
    download(OUTPUT_URL, y_path)

    data_cfg = {
        "_notes": ["adapted from nam_full_configs/data/single_pair.json"],
        "train": {"start_seconds": None, "stop_seconds": -9.0, "ny": 8192},
        "validation": {"start_seconds": -9.0, "stop_seconds": None, "ny": None},
        "common": {
            "x_path": relpath_for_config(x_path, root),
            "y_path": relpath_for_config(y_path, root),
            "delay": 0,
        },
    }

    model_cfg = {
        "_notes": ["adapted from nam_full_configs/models/wavenet.json"],
        "net": {
            "name": "WaveNet",
            "config": {
                "layers_configs": [
                    {
                        "condition_size": 1,
                        "input_size": 1,
                        "channels": 16,
                        "head_size": 8,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False,
                    },
                    {
                        "condition_size": 1,
                        "input_size": 16,
                        "channels": 8,
                        "head_size": 1,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True,
                    },
                ],
                "head_scale": 0.02,
            },
        },
        "optimizer": {"lr": 0.004},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
    }

    learning_cfg = {
        "_notes": ["adapted from nam_full_configs/learning/demo.json"],
        "train_dataloader": {
            "batch_size": 16,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {},
        "trainer": {
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": 10,
            "enable_progress_bar": False,
        },
        "trainer_fit_kwargs": {},
    }

    (cfg_dir / "data.json").write_text(json.dumps(data_cfg, indent=2), encoding="utf-8")
    (cfg_dir / "model.json").write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")
    (cfg_dir / "learning.json").write_text(json.dumps(learning_cfg, indent=2), encoding="utf-8")
    print(f"baseline input: {x_path}")
    print(f"baseline output: {y_path}")
    print(f"data config: {cfg_dir / 'data.json'}")
    print(f"model config: {cfg_dir / 'model.json'}")
    print(f"learning config: {cfg_dir / 'learning.json'}")
    print(f"output directory: {out_dir}")


if __name__ == "__main__":
    main()
