from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


INPUT_URL = "https://drive.google.com/uc?export=download&id=1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G"
OUTPUT_URL = "https://drive.google.com/uc?export=download&id=1NrpQLBbCDHyu0RPsne4YcjIpi5-rEP6w"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    root = repository_root()
    raw_dir = root / "data" / "raw"
    cfg_dir = root / "configs" / "a1_baseline"
    out_dir = root / "outputs" / "a1_baseline"
    log_dir = root / "logs" / "a1"
    for directory in (raw_dir, cfg_dir, out_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "prepare_a1_baseline.log"
    sys.path.insert(0, str(root / "scripts" / "common"))
    from verify_audio_pair import verify_pair, write_log as write_audio_log

    x_path = raw_dir / "baseline_input.wav"
    y_path = raw_dir / "baseline_output.wav"
    lines = [
        "=== A1 Prepare NAM Baseline ===",
        f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"python executable: {sys.executable}",
        f"project root: {root}",
        f"nam version: {package_version('nam')}",
    ]

    try:
        for line in lines:
            print(line)
        download(INPUT_URL, x_path)
        download(OUTPUT_URL, y_path)

        passed, audio_lines = verify_pair(x_path, y_path)
        write_audio_log(audio_lines, log_dir / "verify_audio_pair_a1.log")
        for line in audio_lines:
            print(line)
        if not passed:
            raise RuntimeError("A1 baseline audio pair verification failed")

        data_cfg = {
            "_notes": ["A1 legacy NAM baseline; adapted from nam_full_configs/data/single_pair.json"],
            "train": {"start_seconds": None, "stop_seconds": -9.0, "ny": 8192},
            "validation": {"start_seconds": -9.0, "stop_seconds": None, "ny": None},
            "common": {
                "x_path": relpath_for_config(x_path, root),
                "y_path": relpath_for_config(y_path, root),
                "delay": 0,
            },
        }

        model_cfg = {
            "_notes": ["A1 legacy WaveNet baseline; adapted from nam_full_configs/models/wavenet.json"],
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
            "_notes": ["A1 smoke/demo learning config; adapted from nam_full_configs/learning/demo.json"],
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

        write_json(cfg_dir / "data.json", data_cfg)
        write_json(cfg_dir / "model.json", model_cfg)
        write_json(cfg_dir / "learning.json", learning_cfg)
        result_lines = [
            f"baseline input: {x_path}",
            f"baseline output: {y_path}",
            f"data config: {cfg_dir / 'data.json'}",
            f"model config: {cfg_dir / 'model.json'}",
            f"learning config: {cfg_dir / 'learning.json'}",
            f"output directory: {out_dir}",
        ]
        for line in result_lines:
            print(line)
        log_path.write_text("\n".join(lines + audio_lines + result_lines) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        message = f"ERROR: {exc}"
        print(message, file=sys.stderr)
        log_path.write_text("\n".join(lines + [message]) + "\n", encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
