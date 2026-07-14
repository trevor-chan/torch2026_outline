"""Atomic training checkpoint save and resume helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flow_interpolation.utils.training import EMA, normalize_state_dict, unwrap_compiled_model


CHECKPOINT_VERSION = 1
STEP_PATTERN = re.compile(r"(?:^|_)step_?(\d+)(?:\.[^.]+)?$")


@dataclass(frozen=True)
class CheckpointLoadResult:
    path: Path
    step: int
    full_state: bool


def checkpoint_step(path: str | os.PathLike[str]) -> int:
    match = STEP_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else 0


def find_latest_checkpoint(path: str | os.PathLike[str]) -> Path | None:
    """Resolve a checkpoint file, checkpoint directory, or run directory."""
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    checkpoint_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    if not checkpoint_dir.is_dir():
        return None

    candidates = [
        candidate
        for candidate in checkpoint_dir.iterdir()
        if candidate.is_file() and candidate.suffix in {".pt", ".pth"}
    ]
    if not candidates:
        return None
    model_candidates = [candidate for candidate in candidates if "ema" not in candidate.stem.lower()]
    if model_candidates:
        candidates = model_candidates
    return max(candidates, key=lambda candidate: (checkpoint_step(candidate), candidate.stat().st_mtime)).resolve()


def workdir_from_checkpoint(path: str | os.PathLike[str]) -> Path:
    checkpoint_path = Path(path).resolve()
    return checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent


def save_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": int(step),
        "model_state_dict": unwrap_compiled_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "extra": extra or {},
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    print(f"Checkpoint saved to {path}")
    return path


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    ema: EMA | None,
    device: torch.device | str,
    strict: bool = True,
    restore_rng: bool = True,
) -> CheckpointLoadResult:
    """Load a full training checkpoint or a legacy model-only state dict."""
    checkpoint_path = Path(path).resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    full_state = isinstance(payload, dict) and "model_state_dict" in payload

    if full_state:
        model_state = payload["model_state_dict"]
    elif isinstance(payload, dict):
        model_state = payload.get("state_dict", payload.get("model", payload))
    else:
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")

    target_model = unwrap_compiled_model(model)
    target_model.load_state_dict(normalize_state_dict(model_state), strict=strict)

    if full_state and optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if ema is not None:
        if full_state and payload.get("ema_state_dict") is not None:
            ema.load_state_dict(payload["ema_state_dict"])
        else:
            ema.copy_from_model()

    if full_state and restore_rng:
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])

    step = int(payload.get("step", checkpoint_step(checkpoint_path))) if full_state else checkpoint_step(checkpoint_path)
    kind = "training state" if full_state else "legacy model weights"
    print(f"Loaded {kind} from {checkpoint_path} at step {step}")
    return CheckpointLoadResult(path=checkpoint_path, step=step, full_state=full_state)
