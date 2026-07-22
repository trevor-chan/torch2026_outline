"""Progressive temporal binning schedules.

A bin of width ``w`` centered at frame ``i`` asks the scene, rendered at a single
time, to explain every observation acquired within ``+/- w/2`` frames. Widening
the bin widens the effective sampling mask (the union over its members) at the
cost of asking the model to match a temporally averaged scene; narrowing it
trades that coverage back for temporal resolution.

Widths here are in frames and are always odd, so a bin is symmetric about its
center: ``width = 2 * half_width + 1``, and ``width = 1`` is the exact per-frame
objective with no binning at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

ScheduleKind = Literal["constant", "linear", "exponential", "step"]


@dataclass(frozen=True)
class BinSchedule:
    """Bin width as a function of optimization step.

    Args:
        start_width: bin width at step 0, in frames.
        end_width: bin width at ``anneal_steps`` and after.
        anneal_steps: steps over which the width travels from start to end. Set
            to 0 (with equal widths) for a fixed-width condition.
        kind: interpolation between the two widths. ``exponential`` is the
            default because coverage of the mask union saturates quickly with
            width, so equal *ratios* of width are closer to equal increments of
            difficulty than equal differences are.
        step_count: number of discrete plateaus when ``kind="step"``.
    """

    start_width: int = 25
    end_width: int = 1
    anneal_steps: int = 0
    kind: ScheduleKind = "exponential"
    step_count: int = 5

    def __post_init__(self) -> None:
        if self.start_width < 1 or self.end_width < 1:
            raise ValueError("bin widths must be at least 1")
        if self.anneal_steps < 0:
            raise ValueError("anneal_steps must be non-negative")
        if self.step_count < 1:
            raise ValueError("step_count must be at least 1")

    def width_at(self, step: int) -> int:
        """Odd bin width in frames at ``step``."""
        if self.anneal_steps <= 0 or self.kind == "constant":
            return _odd(self.start_width if step < self.anneal_steps else self.end_width)

        progress = min(max(step / self.anneal_steps, 0.0), 1.0)
        if self.kind == "step":
            # Hold each plateau for an equal number of steps; the final plateau
            # is end_width itself.
            plateau = min(int(progress * self.step_count), self.step_count - 1)
            progress = plateau / max(self.step_count - 1, 1)
        if self.kind == "linear":
            width = self.start_width + progress * (self.end_width - self.start_width)
        else:
            log_start = math.log(self.start_width)
            log_end = math.log(self.end_width)
            width = math.exp(log_start + progress * (log_end - log_start))
        return _odd(width)

    def half_width_at(self, step: int) -> int:
        return (self.width_at(step) - 1) // 2


def _odd(width: float) -> int:
    """Round to the nearest odd integer at least 1."""
    rounded = int(round(width))
    if rounded % 2 == 0:
        rounded += 1
    return max(rounded, 1)


def build_bin_schedule(
    condition: str,
    *,
    num_frames: int,
    max_steps: int,
    start_width: int = 25,
    end_width: int = 1,
    anneal_fraction: float = 0.5,
    kind: ScheduleKind = "exponential",
) -> BinSchedule:
    """Construct one of the three experimental conditions.

    ``wide`` and ``narrow`` are the fixed-width controls; ``curriculum`` anneals
    from ``start_width`` down to ``end_width`` over ``anneal_fraction`` of the
    run, then holds at ``end_width`` for the remainder so the two curricula and
    the narrow baseline finish on the same objective.
    """
    start_width = min(start_width, _odd(num_frames))
    if condition == "wide":
        return BinSchedule(start_width=start_width, end_width=start_width, kind="constant")
    if condition == "narrow":
        return BinSchedule(start_width=end_width, end_width=end_width, kind="constant")
    if condition == "curriculum":
        if not 0.0 < anneal_fraction <= 1.0:
            raise ValueError("anneal_fraction must lie in (0, 1]")
        return BinSchedule(
            start_width=start_width,
            end_width=end_width,
            anneal_steps=max(1, int(round(max_steps * anneal_fraction))),
            kind=kind,
        )
    raise ValueError(f"Unknown binning condition: {condition}")


def bin_window(
    center_indices: torch.Tensor,
    half_width: int,
    num_frames: int,
) -> torch.Tensor:
    """Frame indices covered by each bin; returns ``[B, 2 * half_width + 1]``.

    Bins are clamped at the sequence boundaries rather than wrapped, so the
    first and last frames see a one-sided window. Wrapping would imply the scene
    is periodic, which it is not.
    """
    if half_width < 0:
        raise ValueError("half_width must be non-negative")
    offsets = torch.arange(
        -half_width,
        half_width + 1,
        device=center_indices.device,
    )
    return (center_indices.view(-1, 1) + offsets.view(1, -1)).clamp(0, num_frames - 1)
