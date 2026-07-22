"""Metrics and result serialization shared by experiments."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    error = prediction - target
    mae = error.abs().mean().item()
    mse = error.square().mean().item()
    rmse = math.sqrt(mse)
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {"mae": mae, "mse": mse, "rmse": rmse, "psnr_db": psnr}


def complex_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Residual statistics for complex k-space tensors."""
    error = (prediction - target).abs()
    target_energy = target.abs().square().sum().clamp_min(1e-12)
    return {
        "kspace_mae": error.mean().item(),
        "kspace_mse": error.square().mean().item(),
        "kspace_relative_l2": (error.square().sum() / target_energy).sqrt().item(),
    }


def serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def save_json(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable(payload), handle, indent=2, sort_keys=True)
    print(f"Saved metrics to {path}")
