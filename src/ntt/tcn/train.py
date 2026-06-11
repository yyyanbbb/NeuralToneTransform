from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.utils.data import DataLoader

from src.ntt.data.dataset import PairedAudioChunkDataset
from src.ntt.tcn.checkpoint import save_checkpoint
from src.ntt.tcn.losses import CompositeToneLoss
from src.ntt.tcn.model import GatedTCN
from src.ntt.tcn.utils import (
    created_at,
    cuda_device_name,
    load_json,
    resolve_path,
    sanitize_paths_for_json,
    select_device,
    set_seed,
    to_repo_relative,
    write_json,
)


def build_loss(config: dict[str, Any]) -> CompositeToneLoss:
    weights = config.get("loss_weights", {})
    return CompositeToneLoss(
        mse_weight=weights.get("mse", 1.0),
        esr_weight=weights.get("esr", 0.1),
        mrstft_weight=weights.get("mrstft", 0.001),
    )


def average_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    if count == 0:
        raise RuntimeError("No batches were processed")
    return {key: value / count for key, value in total.items()}


def run_epoch(
    *,
    model: GatedTCN,
    loader: DataLoader,
    loss_fn: CompositeToneLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "mse": 0.0, "esr": 0.0, "mrstft": 0.0}
    batch_count = 0

    for batch_index, batch in enumerate(loader, start=1):
        dry = batch["dry"].to(device, non_blocking=True)
        wet = batch["wet"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred = model(dry)
            loss, components = loss_fn(pred, wet)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm is not None and gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        totals["total"] += float(loss.detach().cpu().item())
        for name in ("mse", "esr", "mrstft"):
            totals[name] += float(components[name].detach().cpu().item())
        batch_count += 1
        if max_batches is not None and batch_index >= max_batches:
            break

    return average_metrics(totals, batch_count)


def train(model_config_path: str | Path, training_config_path: str | Path) -> dict[str, Any]:
    training_start = time.perf_counter()
    training_start_time = created_at()
    model_config = load_json(model_config_path)
    training_config = load_json(training_config_path)
    set_seed(training_config.get("seed"))
    device = select_device(training_config.get("device", "auto"))
    print(f"using device: {device.type}")
    device_name = cuda_device_name(device)
    if device_name is not None:
        print(f"cuda device: {device_name}")

    train_dataset = PairedAudioChunkDataset(training_config["metadata_path"], split="train")
    val_dataset = PairedAudioChunkDataset(training_config["metadata_path"], split="val")
    if len(train_dataset) == 0:
        raise RuntimeError("Training split has no chunks")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation split has no chunks")

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": int(training_config.get("batch_size", 4)),
        "num_workers": int(training_config.get("num_workers", 0)),
        "pin_memory": use_cuda,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = GatedTCN.from_config(model_config).to(device)
    parameter_count = model.count_parameters()
    receptive_field = model.receptive_field()
    print(f"model: {model_config.get('model_name', 'GatedTCN')}")
    print(f"parameter count: {parameter_count}")
    print(f"receptive field: {receptive_field} samples")

    loss_fn = build_loss(training_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-6)),
    )

    output_dir = resolve_path(training_config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    json_model_config = sanitize_paths_for_json(model_config)
    json_training_config = sanitize_paths_for_json(training_config)
    metrics: dict[str, Any] = {
        "model_name": model_config.get("model_name", "GatedTCN"),
        "model_config_path": to_repo_relative(model_config_path),
        "training_config_path": to_repo_relative(training_config_path),
        "metadata_path": to_repo_relative(training_config["metadata_path"]),
        "output_dir": to_repo_relative(output_dir),
        "model_config": json_model_config,
        "training_config": json_training_config,
        "device": device.type,
        "cuda_device_name": device_name,
        "num_epochs": int(training_config.get("num_epochs", 1)),
        "completed_epochs": 0,
        "best_epoch": None,
        "parameter_count": parameter_count,
        "receptive_field": receptive_field,
        "created_at": created_at(),
        "training_start_time": training_start_time,
        "epochs": [],
    }

    num_epochs = int(training_config.get("num_epochs", 1))
    gradient_clip_norm = training_config.get("gradient_clip_norm")
    max_train_batches = training_config.get("max_train_batches")
    max_val_batches = training_config.get("max_val_batches")
    best_epoch: int | None = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
            max_batches=max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
            gradient_clip_norm=None,
            max_batches=max_val_batches,
        )
        is_best = val_metrics["total"] <= best_val_loss
        if is_best:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "is_best": is_best,
        }
        metrics["epochs"].append(epoch_metrics)
        metrics["completed_epochs"] = epoch
        metrics["best_epoch"] = best_epoch

        print(
            "epoch {epoch}/{num_epochs} "
            "train total={train_total:.6g} mse={train_mse:.6g} esr={train_esr:.6g} mrstft={train_mrstft:.6g} "
            "val total={val_total:.6g} mse={val_mse:.6g} esr={val_esr:.6g} mrstft={val_mrstft:.6g}".format(
                epoch=epoch,
                num_epochs=num_epochs,
                train_total=train_metrics["total"],
                train_mse=train_metrics["mse"],
                train_esr=train_metrics["esr"],
                train_mrstft=train_metrics["mrstft"],
                val_total=val_metrics["total"],
                val_mse=val_metrics["mse"],
                val_esr=val_metrics["esr"],
                val_mrstft=val_metrics["mrstft"],
            )
        )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=json_model_config,
            best_val_loss=best_val_loss,
            parameter_count=parameter_count,
            receptive_field=receptive_field,
            training_config=json_training_config,
        )
        if training_config.get("save_every_epoch", True):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=json_model_config,
                best_val_loss=best_val_loss,
                parameter_count=parameter_count,
                receptive_field=receptive_field,
                training_config=json_training_config,
            )
        if epoch_metrics["is_best"]:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=json_model_config,
                best_val_loss=best_val_loss,
                parameter_count=parameter_count,
                receptive_field=receptive_field,
                training_config=json_training_config,
            )

    metrics["best_val_loss"] = best_val_loss
    metrics["train_loss_history"] = [epoch["train"]["total"] for epoch in metrics["epochs"]]
    metrics["val_loss_history"] = [epoch["val"]["total"] for epoch in metrics["epochs"]]
    metrics["training_end_time"] = created_at()
    metrics["total_training_time_seconds"] = time.perf_counter() - training_start
    metrics["training_time_seconds"] = metrics["total_training_time_seconds"]
    write_json(output_dir / "training_metrics.json", metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the custom Gated TCN model.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    args = parser.parse_args()

    try:
        train(args.model_config, args.training_config)
    except Exception as exc:
        raise SystemExit(f"ERROR: TCN training failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
