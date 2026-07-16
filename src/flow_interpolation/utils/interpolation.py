"""Linear and hyperspherical interpolation primitives."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _flatten_vectors(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    if x.ndim < 2:
        raise ValueError("Expected a leading vector axis and at least one feature axis.")
    return x.reshape(x.shape[0], -1), shape


def slerp_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor | Iterable[float],
    *,
    mode: str = "iscs",
    eps: float = 1e-8,
) -> torch.Tensor:
    """SLERP one pair of arbitrary tensors at multiple interpolation weights.

    ``mode='iscs'`` applies sine weights directly to the original vectors, matching
    ISCS. ``mode='radius-lerp'`` interpolates unit directions and endpoint radii.
    """
    if a.shape != b.shape:
        raise ValueError(f"Endpoint shapes differ: {a.shape} versus {b.shape}")
    if mode not in {"iscs", "radius-lerp"}:
        raise ValueError(f"Unknown SLERP mode: {mode}")
    weights = torch.as_tensor(weights, device=a.device, dtype=a.dtype).flatten()
    a_flat = a.reshape(1, -1)
    b_flat = b.reshape(1, -1)
    a_norm = a_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    b_norm = b_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    cosine = ((a_flat * b_flat).sum(dim=1, keepdim=True) / (a_norm * b_norm)).clamp(
        -1.0 + 1e-7,
        1.0 - 1e-7,
    )
    omega = torch.acos(cosine)
    sin_omega = torch.sin(omega)
    w = weights[:, None]

    if float(sin_omega.abs().max()) < 1e-6:
        out = (1.0 - w) * a_flat + w * b_flat
        return out.reshape(weights.numel(), *a.shape)

    left = torch.sin((1.0 - w) * omega) / (sin_omega + eps)
    right = torch.sin(w * omega) / (sin_omega + eps)
    if mode == "iscs":
        out = left * a_flat + right * b_flat
    else:
        a_unit = a_flat / a_norm
        b_unit = b_flat / b_norm
        direction = left * a_unit + right * b_unit
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(eps)
        radius = (1.0 - w) * a_norm + w * b_norm
        out = direction * radius
    return out.reshape(weights.numel(), *a.shape)


def slerp_path(
    keyframes: torch.Tensor,
    samples_per_segment: int,
    *,
    mode: str = "iscs",
) -> torch.Tensor:
    if keyframes.shape[0] < 2:
        raise ValueError("At least two keyframes are required.")
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be positive")
    weights = torch.linspace(
        0.0,
        1.0,
        samples_per_segment + 1,
        device=keyframes.device,
        dtype=keyframes.dtype,
    )
    pieces: list[torch.Tensor] = []
    for index in range(keyframes.shape[0] - 1):
        segment = slerp_pair(keyframes[index], keyframes[index + 1], weights, mode=mode)
        if index > 0:
            segment = segment[1:]
        pieces.append(segment)
    return torch.cat(pieces, dim=0)


def _sphere_log(base: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cosine = torch.dot(base, target).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cosine)
    tangent = target - cosine * base
    tangent_norm = tangent.norm()
    if float(theta.abs()) < 1e-6 or float(tangent_norm) < eps:
        return torch.zeros_like(base)
    return tangent * (theta / tangent_norm)


def _sphere_exp(base: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    length = tangent.norm()
    if float(length) < eps:
        return base
    result = torch.cos(length) * base + torch.sin(length) * tangent / length
    return result / result.norm().clamp_min(eps)


def _unit_slerp(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cosine = torch.dot(a, b).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cosine)
    sin_theta = torch.sin(theta)
    if float(sin_theta.abs()) < 1e-6:
        out = (1.0 - weight) * a + weight * b
    else:
        out = (
            torch.sin((1.0 - weight) * theta) / (sin_theta + eps) * a
            + torch.sin(weight * theta) / (sin_theta + eps) * b
        )
    return out / out.norm().clamp_min(eps)


def squad_path(keyframes: torch.Tensor, samples_per_segment: int, eps: float = 1e-8) -> torch.Tensor:
    """Generalized SQUAD on the hypersphere with linearly interpolated radius.

    The direction uses Shoemake's SQUAD construction, replacing quaternion log/exp
    with Riemannian log/exp on S^(D-1). This gives a smooth multi-keyframe path and
    avoids independent pairwise tangent changes. Endpoint controls are clamped to
    the endpoint directions.
    """
    if keyframes.shape[0] < 2:
        raise ValueError("At least two keyframes are required.")
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be positive")

    flat, original_shape = _flatten_vectors(keyframes)
    radii = flat.norm(dim=1).clamp_min(eps)
    directions = flat / radii[:, None]
    controls = directions.clone()
    for index in range(1, directions.shape[0] - 1):
        tangent = -0.25 * (
            _sphere_log(directions[index], directions[index - 1], eps=eps)
            + _sphere_log(directions[index], directions[index + 1], eps=eps)
        )
        controls[index] = _sphere_exp(directions[index], tangent, eps=eps)

    weights = torch.linspace(
        0.0,
        1.0,
        samples_per_segment + 1,
        device=keyframes.device,
        dtype=keyframes.dtype,
    )
    output: list[torch.Tensor] = []
    for segment_index in range(directions.shape[0] - 1):
        for weight_index, weight in enumerate(weights):
            if segment_index > 0 and weight_index == 0:
                continue
            direct = _unit_slerp(
                directions[segment_index], directions[segment_index + 1], weight, eps=eps
            )
            control = _unit_slerp(
                controls[segment_index], controls[segment_index + 1], weight, eps=eps
            )
            blend = 2.0 * weight * (1.0 - weight)
            direction = _unit_slerp(direct, control, blend, eps=eps)
            radius = torch.lerp(radii[segment_index], radii[segment_index + 1], weight)
            output.append(direction * radius)
    stacked = torch.stack(output, dim=0)
    return stacked.reshape(stacked.shape[0], *original_shape[1:])


def squad_segment(
    keyframes: torch.Tensor,
    segment_index: int,
    weights: torch.Tensor | Iterable[float],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Evaluate one generalized-SQUAD segment at arbitrary weights."""
    if keyframes.shape[0] < 2:
        raise ValueError("At least two keyframes are required.")
    if not 0 <= segment_index < keyframes.shape[0] - 1:
        raise ValueError("segment_index is outside the keyframe path")

    flat, original_shape = _flatten_vectors(keyframes)
    radii = flat.norm(dim=1).clamp_min(eps)
    directions = flat / radii[:, None]
    controls = directions.clone()
    for index in range(1, directions.shape[0] - 1):
        tangent = -0.25 * (
            _sphere_log(directions[index], directions[index - 1], eps=eps)
            + _sphere_log(directions[index], directions[index + 1], eps=eps)
        )
        controls[index] = _sphere_exp(directions[index], tangent, eps=eps)

    weights = torch.as_tensor(
        weights,
        device=keyframes.device,
        dtype=keyframes.dtype,
    ).flatten()
    output = []
    for weight in weights:
        direct = _unit_slerp(
            directions[segment_index],
            directions[segment_index + 1],
            weight,
            eps=eps,
        )
        control = _unit_slerp(
            controls[segment_index],
            controls[segment_index + 1],
            weight,
            eps=eps,
        )
        blend = 2.0 * weight * (1.0 - weight)
        direction = _unit_slerp(direct, control, blend, eps=eps)
        radius = torch.lerp(radii[segment_index], radii[segment_index + 1], weight)
        output.append(direction * radius)
    stacked = torch.stack(output, dim=0)
    return stacked.reshape(stacked.shape[0], *original_shape[1:])


def interpolation_segment(
    keyframes: torch.Tensor,
    segment_index: int,
    weights: torch.Tensor | Iterable[float],
    method: str,
    *,
    slerp_mode: str = "iscs",
) -> torch.Tensor:
    """Evaluate one interpolation segment at arbitrary local coordinates."""
    if not 0 <= segment_index < keyframes.shape[0] - 1:
        raise ValueError("segment_index is outside the keyframe path")
    weights = torch.as_tensor(
        weights,
        device=keyframes.device,
        dtype=keyframes.dtype,
    ).flatten()
    if method == "lerp":
        view = weights.view(-1, *([1] * (keyframes.ndim - 1)))
        return torch.lerp(
            keyframes[segment_index],
            keyframes[segment_index + 1],
            view,
        )
    if method == "slerp":
        return slerp_pair(
            keyframes[segment_index],
            keyframes[segment_index + 1],
            weights,
            mode=slerp_mode,
        )
    if method == "squad":
        return squad_segment(keyframes, segment_index, weights)
    raise ValueError(f"Unknown interpolation method: {method}")


def interpolate_keyframes(
    keyframes: torch.Tensor,
    samples_per_segment: int,
    method: str,
    *,
    slerp_mode: str = "iscs",
) -> torch.Tensor:
    if method == "slerp":
        return slerp_path(keyframes, samples_per_segment, mode=slerp_mode)
    if method == "squad":
        return squad_path(keyframes, samples_per_segment)
    if method == "lerp":
        weights = torch.linspace(
            0.0,
            1.0,
            samples_per_segment + 1,
            device=keyframes.device,
            dtype=keyframes.dtype,
        )
        pieces = []
        for index in range(keyframes.shape[0] - 1):
            view = weights.view(-1, *([1] * (keyframes.ndim - 1)))
            segment = torch.lerp(keyframes[index], keyframes[index + 1], view)
            if index > 0:
                segment = segment[1:]
            pieces.append(segment)
        return torch.cat(pieces, dim=0)
    raise ValueError(f"Unknown interpolation method: {method}")


def global_slerp_noise(
    num_frames: int,
    frame_shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    mode: str = "iscs",
) -> torch.Tensor:
    """Sample two Gaussian anchors and one SLERP path across the full timeline."""
    if num_frames < 2:
        return torch.randn((num_frames, *frame_shape), device=device, dtype=dtype, generator=generator)
    anchors = torch.randn((2, *frame_shape), device=device, dtype=dtype, generator=generator)
    return slerp_path(anchors, samples_per_segment=num_frames - 1, mode=mode)
