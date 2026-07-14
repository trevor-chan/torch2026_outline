from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from tqdm import tqdm

from flow_interpolation.models import TransformerDiffusionModel


@dataclass(frozen=True)
class FlowSettings:
    data_time: float
    noise_time: float
    ode_steps: int
    solver: str
    encode_batch_size: int
    decode_batch_size: int


@dataclass(frozen=True)
class ModelSettings:
    checkpoint: str
    image_size: int
    model_dim: int
    num_layers: int
    num_heads: int
    time_embed_dim: Optional[int]
    rope_theta: float
    strict_load: bool
    compile_model: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.removeprefix("module.").removeprefix("_orig_mod.")
        state_dict[key] = value

    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")


def build_model(settings: ModelSettings, device: torch.device) -> torch.nn.Module:
    model = TransformerDiffusionModel(
        in_channels=3,
        dim=settings.model_dim,
        depth=settings.num_layers,
        num_heads=settings.num_heads,
        time_embed_dim=settings.time_embed_dim,
        rope_theta=settings.rope_theta,
    ).to(device)
    load_checkpoint(
        model,
        checkpoint_path=settings.checkpoint,
        device=device,
        strict=settings.strict_load,
    )
    model.eval()
    if settings.compile_model:
        model = torch.compile(model)
    return model


def validate_flow_settings(settings: FlowSettings) -> None:
    if settings.data_time < 0.0:
        raise ValueError("data_time must be non-negative")
    if not 0.0 < settings.noise_time <= 1.0:
        raise ValueError("noise_time must satisfy 0 < noise_time <= 1")
    if settings.data_time >= settings.noise_time:
        raise ValueError("data_time must be smaller than noise_time")
    if settings.ode_steps <= 0:
        raise ValueError("ode_steps must be positive")
    if settings.solver not in {"euler", "heun"}:
        raise ValueError(f"Unknown solver: {settings.solver}")


@torch.no_grad()
def integrate_flow(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_start: float,
    t_end: float,
    num_steps: int,
    solver: str = "heun",
    desc: Optional[str] = None,
) -> torch.Tensor:
    """Integrate dx/dt = v_theta(x, t) in either time direction."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError(f"Unknown solver: {solver}")

    x = x.clone()
    times = torch.linspace(t_start, t_end, num_steps + 1, device=x.device, dtype=x.dtype)
    iterator: Iterable[tuple[torch.Tensor, torch.Tensor]] = zip(times[:-1], times[1:])
    if desc is not None:
        iterator = tqdm(iterator, total=num_steps, desc=desc)

    for t_curr, t_next in iterator:
        dt = t_next - t_curr
        v_curr = model(x, None, t_curr.expand(x.shape[0]))
        if solver == "euler":
            x = x + dt * v_curr
        else:
            x_pred = x + dt * v_curr
            v_next = model(x_pred, None, t_next.expand(x.shape[0]))
            x = x + 0.5 * dt * (v_curr + v_next)
    return x


def perturb_to_p_eps(
    images: torch.Tensor,
    data_time: float,
    eps_noise: torch.Tensor,
) -> torch.Tensor:
    """Move clean images to the training marginal x_t=(1-t)x_0+t eps."""
    return (1.0 - data_time) * images + data_time * eps_noise.to(images.device, images.dtype)


def make_boundary_noise(
    samples: torch.Tensor,
    mode: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create epsilon-boundary noise, shared across frames or independent per frame."""
    if mode == "shared":
        shape = (1, *samples.shape[1:])
    elif mode == "independent":
        shape = samples.shape
    else:
        raise ValueError(f"Unknown boundary-noise mode: {mode}")
    return torch.randn(shape, device=samples.device, dtype=samples.dtype, generator=generator)


@torch.no_grad()
def encode_in_chunks(
    model: torch.nn.Module,
    samples: torch.Tensor,
    settings: FlowSettings,
    device: torch.device,
    *,
    eps_noise: Optional[torch.Tensor] = None,
    perturb: bool = True,
    desc: str = "Encoding data to noise",
) -> torch.Tensor:
    """Encode frames from data_time to noise_time.

    When ``perturb`` is true, ``samples`` are interpreted as clean x_0 and are first
    moved to p(data_time) with ``eps_noise``. When false, samples are assumed to
    already live at data_time; this is required for true cycle-consistency tests.
    """
    chunks: list[torch.Tensor] = []
    split_samples = samples.split(settings.encode_batch_size)
    offset = 0
    for chunk in tqdm(split_samples, desc=desc):
        chunk = chunk.to(device)
        if perturb:
            if eps_noise is None:
                raise ValueError("eps_noise is required when perturb=True")
            if eps_noise.shape[0] == samples.shape[0]:
                chunk_eps = eps_noise[offset : offset + chunk.shape[0]].to(device)
            else:
                chunk_eps = eps_noise.to(device)
            chunk = perturb_to_p_eps(chunk, settings.data_time, chunk_eps)
        encoded = integrate_flow(
            model,
            chunk,
            t_start=settings.data_time,
            t_end=settings.noise_time,
            num_steps=settings.ode_steps,
            solver=settings.solver,
        )
        chunks.append(encoded.cpu())
        offset += chunk.shape[0]
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def decode_in_chunks(
    model: torch.nn.Module,
    latents: torch.Tensor,
    settings: FlowSettings,
    device: torch.device,
    *,
    desc: str = "Decoding noise to data",
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for chunk in tqdm(latents.split(settings.decode_batch_size), desc=desc):
        decoded = integrate_flow(
            model,
            chunk.to(device),
            t_start=settings.noise_time,
            t_end=settings.data_time,
            num_steps=settings.ode_steps,
            solver=settings.solver,
        )
        chunks.append(decoded.cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def predict_clean_and_noise(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover RF endpoint estimates from x_t and the predicted velocity.

    For x_t=(1-t)x_0+t z and v=z-x_0:
      x_0 = x_t - t v
      z   = x_t + (1-t) v
    """
    if isinstance(t, float):
        t_batch = x_t.new_full((x_t.shape[0],), t)
    else:
        t_batch = t.to(device=x_t.device, dtype=x_t.dtype)
        if t_batch.ndim == 0:
            t_batch = t_batch.expand(x_t.shape[0])
    v = model(x_t, None, t_batch)
    t_view = t_batch.view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
    x0_hat = x_t - t_view * v
    z_hat = x_t + (1.0 - t_view) * v
    return x0_hat, z_hat, v


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


def nearest_observed_timeline(
    observed_values: torch.Tensor,
    observed_indices: torch.Tensor,
    total_frames: int,
) -> torch.Tensor:
    frame_ids = torch.arange(total_frames)
    distances = (frame_ids[:, None] - observed_indices[None, :]).abs()
    nearest = distances.argmin(dim=1)
    return observed_values[nearest]


def missing_mask(total_frames: int, observed_indices: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(total_frames, dtype=torch.bool)
    mask[observed_indices.cpu()] = False
    return mask


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
