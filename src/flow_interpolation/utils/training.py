"""Shared training performance, memory, model, and EMA utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psutil
import torch


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module from one or more torch.compile wrappers."""
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip common compile/distributed prefixes from checkpoint keys."""
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.removeprefix("module.").removeprefix("_orig_mod.")
        normalized[key] = value
    return normalized


def save_model(model: torch.nn.Module, path: str | os.PathLike[str] = "model.pth") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_compiled_model(model).state_dict(), path)
    print(f"Model saved to {path}")
    return path


def load_model(
    model: torch.nn.Module,
    path: str | os.PathLike[str] = "model.pth",
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    state_dict = torch.load(path, map_location=device, weights_only=True)
    unwrap_compiled_model(model).load_state_dict(normalize_state_dict(state_dict))
    return model


def estimate_training_flops(
    model: torch.nn.Module,
    data: torch.Tensor,
    conditioning: torch.Tensor | None = None,
    *,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
    criterion: torch.nn.Module | None = None,
) -> float:
    """Estimate model forward and backward FLOPs for one training batch."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        profile_model = unwrap_compiled_model(model)
        if device is not None:
            data = data.to(device)
            if conditioning is not None:
                conditioning = conditioning.to(device)
        was_training = profile_model.training
        profile_model.train()
        profile_model.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad(), torch.autocast(
                data.device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ), FlopCounterMode(display=False) as flop_counter:
                if criterion is None:
                    time = torch.rand(data.shape[0], device=data.device, dtype=data.dtype)
                    loss = profile_model(data, conditioning, time).sum()
                else:
                    loss = criterion(profile_model, data, conditioning)
                loss.backward()
            flops = float(flop_counter.get_total_flops())
        finally:
            profile_model.zero_grad(set_to_none=True)
            profile_model.train(was_training)

        print(f"Estimated training FLOPs per batch (forward + backward): {flops / 1e9:.2f} GFLOPs")
        return flops
    except Exception as error:
        print(f"Could not estimate training FLOPs: {error}")
        return 0.0


# Compatibility alias for callers outside the trainer.
estimate_flops = estimate_training_flops


class EMA:
    """Exponential moving average over trainable model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device=None):
        self.model = model
        self.decay = decay
        trainable = self._trainable_params()
        if not trainable:
            raise ValueError("Model has no trainable parameters for EMA to track")
        self.device = device if device is not None else trainable[0].device
        self.shadow_params = [parameter.detach().clone().to(self.device) for parameter in trainable]
        self.collected_params: list[torch.Tensor] = []

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def update(self) -> None:
        for shadow, parameter in zip(self.shadow_params, self._trainable_params(), strict=True):
            shadow.lerp_(parameter.detach().to(shadow.device), 1.0 - self.decay)

    @torch.no_grad()
    def copy_from_model(self) -> None:
        trainable = self._trainable_params()
        if len(trainable) != len(self.shadow_params):
            raise ValueError("EMA parameter count changed after initialization")
        for shadow, parameter in zip(self.shadow_params, trainable, strict=True):
            shadow.copy_(parameter.detach().to(device=shadow.device, dtype=shadow.dtype))

    @torch.no_grad()
    def store(self) -> None:
        trainable = self._trainable_params()
        self.collected_params = [parameter.detach().clone() for parameter in trainable]
        for shadow, parameter in zip(self.shadow_params, trainable, strict=True):
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        if len(self.collected_params) != len(self.shadow_params):
            raise RuntimeError("EMA.restore() called without a matching EMA.store()")
        for collected, parameter in zip(self.collected_params, self._trainable_params(), strict=True):
            parameter.copy_(collected.to(device=parameter.device, dtype=parameter.dtype))
        self.collected_params = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow_params": [parameter.detach().clone() for parameter in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        shadows = state_dict["shadow_params"]
        if len(shadows) != len(self.shadow_params):
            raise ValueError(
                f"EMA checkpoint has {len(shadows)} parameters; expected {len(self.shadow_params)}"
            )
        self.decay = float(state_dict.get("decay", self.decay))
        for destination, source in zip(self.shadow_params, shadows, strict=True):
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def get_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """Return process and CUDA memory statistics in GiB."""
    stats = {"cpu_gb": psutil.Process(os.getpid()).memory_info().rss / (1024**3)}
    if torch.cuda.is_available():
        device_index = 0 if device is None or device.index is None else device.index
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated(device_index) / (1024**3)
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved(device_index) / (1024**3)
    return stats


def format_memory_stats(stats: dict[str, float]) -> str:
    values = [f"CPU: {stats['cpu_gb']:.2f} GiB"] if "cpu_gb" in stats else []
    if "gpu_allocated_gb" in stats:
        values.append(
            f"GPU: {stats['gpu_allocated_gb']:.2f} GiB allocated / "
            f"{stats['gpu_reserved_gb']:.2f} GiB reserved"
        )
    return " | ".join(values) if values else "Memory stats unavailable"


def get_memory_usage(device: torch.device | None = None) -> str:
    return format_memory_stats(get_memory_stats(device))


def calculate_mfu(flops_per_batch: float, batch_time: float, peak_flops: float) -> float:
    """Calculate training MFU as a percentage from forward-plus-backward FLOPs."""
    if flops_per_batch <= 0.0 or batch_time <= 0.0 or peak_flops <= 0.0:
        return 0.0
    return 100.0 * flops_per_batch / batch_time / peak_flops
