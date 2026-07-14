"""Metrics and result serialization shared by evaluation experiments."""

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


def tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    pred_flat = prediction.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    diff = pred_flat - target_flat
    target_norm = target_flat.norm(dim=1).clamp_min(1e-12)
    pred_norm = pred_flat.norm(dim=1).clamp_min(1e-12)
    relative_l2 = diff.norm(dim=1) / target_norm
    cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1).clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(cosine))
    radius_relative_error = (pred_norm - target_norm).abs() / target_norm
    return {
        **image_metrics(prediction, target),
        "relative_l2": relative_l2.mean().item(),
        "cosine_similarity": cosine.mean().item(),
        "angle_degrees": angle_deg.mean().item(),
        "radius_relative_error": radius_relative_error.mean().item(),
    }


def print_noise_stats(name: str, noise: torch.Tensor) -> dict[str, float]:
    flat = noise.float().flatten(start_dim=1)
    expected_radius = math.sqrt(flat.shape[1])
    radii = flat.norm(dim=1)
    stats = {
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "radius_mean": radii.mean().item(),
        "radius_std": radii.std(unbiased=False).item(),
        "sqrt_dimension": expected_radius,
    }
    print(
        f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"radius={stats['radius_mean']:.2f} +/- {stats['radius_std']:.2f} "
        f"(sqrt(dim)={expected_radius:.2f})"
    )
    return stats


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
