from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


KEYWORDS = ["SlimmableContainer", "WaveNet", "channels", "3", "8"]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def readable_strings(payload: bytes, min_length: int = 4) -> list[str]:
    text = payload.decode("utf-8", errors="ignore")
    return re.findall(r"[\x20-\x7E]{" + str(min_length) + r",}", text)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(repository_root()).as_posix()
    except ValueError:
        return str(path)


def extract_submodel_channels(parsed_json: object) -> list[int]:
    if not isinstance(parsed_json, dict):
        return []
    config = parsed_json.get("config")
    if not isinstance(config, dict):
        return []
    submodels = config.get("submodels")
    if not isinstance(submodels, list):
        return []

    channels: list[int] = []
    for submodel in submodels:
        if not isinstance(submodel, dict):
            continue
        model = submodel.get("model")
        if not isinstance(model, dict):
            continue
        model_config = model.get("config")
        if not isinstance(model_config, dict):
            continue
        layers = model_config.get("layers")
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if isinstance(layer, dict) and isinstance(layer.get("channels"), int):
                channels.append(layer["channels"])
                break
    return channels


def inspect_model(model_path: Path) -> tuple[int, str]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# A2 Model Inspection",
        "",
        f"- timestamp: {now}",
        f"- model_path: {display_path(model_path)}",
    ]

    if not model_path.is_file():
        lines.extend(["", "OVERALL: FAIL", f"ERROR: model file not found: {model_path}"])
        return 1, "\n".join(lines) + "\n"

    payload = model_path.read_bytes()
    strings = readable_strings(payload[: min(len(payload), 2_000_000)])
    joined = "\n".join(strings)
    found = {keyword: (keyword in joined) for keyword in KEYWORDS}
    parsed_json = None

    try:
        parsed_json = json.loads(payload.decode("utf-8"))
    except Exception:
        parsed_json = None

    architecture = parsed_json.get("architecture") if isinstance(parsed_json, dict) else None
    channels = extract_submodel_channels(parsed_json)

    lines.extend(
        [
            f"- file_size_bytes: {model_path.stat().st_size}",
            f"- first_32_bytes_hex: {payload[:32].hex()}",
            f"- parsed_as_json: {parsed_json is not None}",
            f"- architecture: {architecture if architecture is not None else 'unknown'}",
            f"- submodel_channels: {channels if channels else 'unknown'}",
            "",
            "## Keyword Checks",
            "",
        ]
    )
    for keyword in KEYWORDS:
        lines.append(f"- {keyword}: {'FOUND' if found[keyword] else 'NOT FOUND'}")

    lines.extend(["", "## Readable String Snippets", ""])
    for snippet in strings[:40]:
        lines.append(f"- `{snippet[:160]}`")

    if parsed_json is not None:
        lines.extend(["", "## Parsed JSON Top Level", ""])
        if isinstance(parsed_json, dict):
            for key in parsed_json.keys():
                lines.append(f"- `{key}`")

    slim_found = found["SlimmableContainer"]
    lines.extend(
        [
            "",
            f"SlimmableContainer observed: {'YES' if slim_found else 'NO'}",
            f"OVERALL: {'PASS' if slim_found else 'FAIL'}",
        ]
    )
    return 0 if slim_found else 1, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an exported A2 .nam file.")
    parser.add_argument("--model", default="outputs/a2_baseline/model.nam")
    parser.add_argument("--report", default="reports/a2_model_inspection.md")
    args = parser.parse_args()

    root = repository_root()
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    exit_code, report = inspect_model(resolve_path(args.model))
    report_path.write_text(report, encoding="utf-8")

    step3_path = root / "reports" / "STEP3_A2_COMPLETION_REPORT.md"
    if step3_path.exists():
        with step3_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n## Latest A2 Model Inspection\n\n")
            handle.write(report)
    print(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
