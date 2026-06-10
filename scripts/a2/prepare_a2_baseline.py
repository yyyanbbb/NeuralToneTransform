from __future__ import annotations

import importlib
import json
import shutil
import sys
from datetime import datetime
from importlib import resources
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


def find_official_packed_config() -> Path | None:
    try:
        package_root = Path(str(resources.files("nam")))
    except Exception:
        return None

    preferred_names = {
        "config_model_packed.json",
        "model_packed.json",
        "packed_wavenet.json",
        "packed-wavenet.json",
    }
    candidates = [path for path in package_root.rglob("*.json") if path.name.lower() in preferred_names]
    if candidates:
        return candidates[0]
    packed_candidates = [path for path in package_root.rglob("*.json") if "packed" in path.name.lower()]
    return packed_candidates[0] if packed_candidates else None


def placeholder_packed_config() -> dict:
    return {
        "_notes": [
            "PLACEHOLDER A2 config generated because no official packed config was found in the installed nam package.",
            "Manually compare this file with the official neural-amp-modeler 0.13.0 PackedWaveNet config before full training.",
        ],
        "_manual_review_required": True,
        "net": {
            "name": "PackedWaveNet",
            "config": {
                "container": "SlimmableContainer",
                "submodels": [
                    {"name": "a2_lite", "channels": 3, "description": "low CPU / hardware target"},
                    {"name": "a2_full", "channels": 8, "description": "desktop plugin / high quality target"},
                ],
                "packed_training": True,
                "masking": "separate Lite and Full weight blocks during packed training",
                "sample_rate": 48000,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "activation": "Tanh",
            },
        },
        "optimizer": {"lr": 0.004},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    root = repository_root()
    raw_dir = root / "data" / "raw"
    cfg_dir = root / "configs" / "a2_baseline"
    out_dir = root / "outputs" / "a2_baseline"
    log_dir = root / "logs" / "a2"
    for directory in (raw_dir, cfg_dir, out_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "prepare_a2_baseline.log"
    sys.path.insert(0, str(root / "scripts" / "common"))
    from verify_audio_pair import verify_pair, write_log as write_audio_log

    lines = [
        "=== A2 Prepare PackedWaveNet Baseline ===",
        f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"python executable: {sys.executable}",
        f"project root: {root}",
    ]

    try:
        try:
            lines.append(f"nam version: {package_version('nam')}")
        except Exception as exc:
            lines.append(f"nam version: unavailable ({exc})")

        x_path = raw_dir / "baseline_input.wav"
        y_path = raw_dir / "baseline_output.wav"
        for line in lines:
            print(line)
        download(INPUT_URL, x_path)
        download(OUTPUT_URL, y_path)

        passed, audio_lines = verify_pair(x_path, y_path)
        write_audio_log(audio_lines, log_dir / "verify_audio_pair_a2.log")
        for line in audio_lines:
            print(line)
        if not passed:
            raise RuntimeError("A2 baseline audio pair verification failed")

        data_cfg = {
            "_notes": ["A2 PackedWaveNet baseline data config; reuses A1 baseline audio pair."],
            "train": {"start_seconds": None, "stop_seconds": -9.0, "ny": 8192},
            "validation": {"start_seconds": -9.0, "stop_seconds": None, "ny": None},
            "common": {
                "x_path": relpath_for_config(x_path, root),
                "y_path": relpath_for_config(y_path, root),
                "delay": 0,
            },
        }
        write_json(cfg_dir / "data.json", data_cfg)

        official_config = find_official_packed_config()
        model_path = cfg_dir / "model_packed.json"
        if official_config is not None:
            shutil.copyfile(official_config, model_path)
            model_message = f"copied official packed config from: {official_config}"
        else:
            write_json(model_path, placeholder_packed_config())
            model_message = "official packed config not found; wrote placeholder model_packed.json"

        learning_cfg = {
            "_notes": [
                "A2 smoke-test learning config. Increase max_epochs and tune batch size for full reproduction."
            ],
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
                "max_epochs": 1,
                "enable_progress_bar": False,
            },
            "trainer_fit_kwargs": {},
        }
        write_json(cfg_dir / "learning.json", learning_cfg)

        result_lines = [
            model_message,
            f"baseline input: {x_path}",
            f"baseline output: {y_path}",
            f"data config: {cfg_dir / 'data.json'}",
            f"model config: {model_path}",
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
