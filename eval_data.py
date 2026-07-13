from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dataset import BouncingBallVideoDataset


DEFAULT_TRAINING_COLOR_WALK_STD = 0.1


@dataclass(frozen=True)
class CadenceInfo:
    training_frame_dt: float
    high_frame_dt: float
    requested_ratio: float
    endpoint_stride: int
    actual_endpoint_dt: float
    endpoint_dt_error: float
    relative_error: float
    rounding_policy: str


@dataclass(frozen=True)
class SequenceData:
    frames: torch.Tensor
    observed_indices: torch.Tensor
    observed_frames: torch.Tensor
    cadence: CadenceInfo
    high_rate_color_walk_std: float
    start_index: int

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def num_intervals(self) -> int:
        return int(self.observed_indices.numel() - 1)


def resolve_cadence(
    training_frame_dt: float,
    high_frame_dt: float,
    rounding_policy: str = "nearest",
    tolerance: float = 1e-9,
) -> CadenceInfo:
    if training_frame_dt <= 0.0 or high_frame_dt <= 0.0:
        raise ValueError("Frame spacings must be positive.")
    ratio = training_frame_dt / high_frame_dt
    if rounding_policy == "nearest":
        # Explicit half-up rounding avoids Python's banker's-rounding behavior.
        stride = int(math.floor(ratio + 0.5))
    elif rounding_policy == "floor":
        stride = int(math.floor(ratio))
    elif rounding_policy == "ceil":
        stride = int(math.ceil(ratio))
    elif rounding_policy == "exact":
        stride = int(round(ratio))
        if not math.isclose(ratio, stride, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"training_frame_dt/high_frame_dt={ratio:.9f} is not integral. "
                "Choose floor, ceil, or nearest, or change the requested spacings."
            )
    else:
        raise ValueError(f"Unknown stride rounding policy: {rounding_policy}")
    stride = max(1, stride)
    actual = stride * high_frame_dt
    error = actual - training_frame_dt
    return CadenceInfo(
        training_frame_dt=training_frame_dt,
        high_frame_dt=high_frame_dt,
        requested_ratio=ratio,
        endpoint_stride=stride,
        actual_endpoint_dt=actual,
        endpoint_dt_error=error,
        relative_error=error / training_frame_dt,
        rounding_policy=rounding_policy,
    )


def print_cadence(cadence: CadenceInfo) -> None:
    print(
        "Cadence: "
        f"requested training_dt/high_dt={cadence.requested_ratio:.6f}; "
        f"policy={cadence.rounding_policy}; stride={cadence.endpoint_stride}; "
        f"requested endpoint dt={cadence.training_frame_dt:.6f}s; "
        f"actual endpoint dt={cadence.actual_endpoint_dt:.6f}s; "
        f"error={cadence.endpoint_dt_error:+.6f}s "
        f"({100.0 * cadence.relative_error:+.3f}%)."
    )


def scale_color_walk_std(
    training_color_walk_std: float,
    training_frame_dt: float,
    high_frame_dt: float,
) -> float:
    if training_color_walk_std < 0.0:
        raise ValueError("training_color_walk_std must be non-negative")
    return training_color_walk_std * math.sqrt(high_frame_dt / training_frame_dt)


def build_sequence(
    *,
    image_size: int,
    seed: int,
    start_index: int,
    num_intervals: int,
    training_frame_dt: float,
    high_frame_dt: float,
    training_color_walk_std: float = DEFAULT_TRAINING_COLOR_WALK_STD,
    color_walk_std: float | None = None,
    stride_rounding: str = "nearest",
) -> SequenceData:
    if num_intervals <= 0:
        raise ValueError("num_intervals must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    cadence = resolve_cadence(
        training_frame_dt=training_frame_dt,
        high_frame_dt=high_frame_dt,
        rounding_policy=stride_rounding,
    )
    actual_color_std = (
        color_walk_std
        if color_walk_std is not None
        else scale_color_walk_std(
            training_color_walk_std=training_color_walk_std,
            training_frame_dt=training_frame_dt,
            high_frame_dt=high_frame_dt,
        )
    )
    sequence_length = num_intervals * cadence.endpoint_stride + 1
    required_samples = start_index + sequence_length
    dataset = BouncingBallVideoDataset(
        num_samples=required_samples,
        image_size=image_size,
        seed=seed,
        frame_dt=high_frame_dt,
        color_walk_std=actual_color_std,
        write_video=False,
    )
    frames = dataset.samples[start_index : start_index + sequence_length].contiguous()
    observed_indices = torch.arange(
        0,
        sequence_length,
        cadence.endpoint_stride,
        dtype=torch.long,
    )
    observed_frames = frames[observed_indices]
    print_cadence(cadence)
    print(
        f"Color walk: training std={training_color_walk_std:.6f} per "
        f"{training_frame_dt:.6f}s; high-rate std={actual_color_std:.6f} per "
        f"{high_frame_dt:.6f}s."
    )
    return SequenceData(
        frames=frames,
        observed_indices=observed_indices,
        observed_frames=observed_frames,
        cadence=cadence,
        high_rate_color_walk_std=actual_color_std,
        start_index=start_index,
    )
