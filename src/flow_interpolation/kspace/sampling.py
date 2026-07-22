"""Per-frame k-space sampling masks.

Every generator returns a boolean mask of shape ``[height, width]`` laid out in
the centered convention of :mod:`flow_interpolation.kspace.transforms`, so index
``(height // 2, width // 2)`` is DC.

Masks are drawn independently per frame so that the union over a temporal window
covers substantially more of k-space than any single frame does. That is the
property progressive temporal binning exploits: widening the bin widens the
effective sampling pattern.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from flow_interpolation.kspace.transforms import kspace_radius

GOLDEN_ANGLE_DEGREES = 111.246117975


def _validate_rate(sampling_rate: float) -> None:
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling_rate must lie in (0, 1]")


def _center_block(
    height: int,
    width: int,
    center_fraction: float,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Fully sampled square block around DC, as a boolean mask."""
    mask = torch.zeros(height, width, dtype=torch.bool, device=device)
    if center_fraction <= 0.0:
        return mask
    # center_fraction is a fraction of the total grid area, so the block side is
    # its square root; this keeps its meaning stable across mask families.
    side = math.sqrt(center_fraction)
    half_y = int(round(height * side / 2.0))
    half_x = int(round(width * side / 2.0))
    if half_y == 0 or half_x == 0:
        return mask
    cy, cx = height // 2, width // 2
    mask[max(cy - half_y, 0) : cy + half_y, max(cx - half_x, 0) : cx + half_x] = True
    return mask


def _fill_to_budget(
    mask: torch.Tensor,
    weights: torch.Tensor,
    budget: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Add unsampled entries, drawn without replacement under ``weights``.

    ``budget`` counts the total number of ``True`` entries in the returned mask,
    so any center block already present is charged against it.
    """
    remaining = budget - int(mask.sum().item())
    if remaining <= 0:
        return mask
    candidate_weights = weights.flatten().clone()
    candidate_weights[mask.flatten()] = 0.0
    available = int((candidate_weights > 0).sum().item())
    if available == 0:
        return mask
    chosen = torch.multinomial(
        candidate_weights,
        num_samples=min(remaining, available),
        replacement=False,
        generator=generator,
    )
    flat = mask.flatten()
    flat[chosen] = True
    return flat.view_as(mask)


def uniform_random_mask(
    height: int,
    width: int,
    *,
    sampling_rate: float,
    center_fraction: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Uniformly random points, with an exact sample count."""
    _validate_rate(sampling_rate)
    mask = _center_block(height, width, center_fraction, device)
    weights = torch.ones(height, width, device=device)
    budget = max(1, int(round(sampling_rate * height * width)))
    return _fill_to_budget(mask, weights, budget, generator)


def variable_density_mask(
    height: int,
    width: int,
    *,
    sampling_rate: float,
    center_fraction: float = 0.0,
    decay: float = 3.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Random points with density falling off as ``(1 + r)^-decay`` from DC.

    Natural images concentrate energy at low frequencies, so this recovers far
    more signal per sample than a uniform draw. It is also the closest of these
    families to what variable-density MRI acquisitions actually do.
    """
    _validate_rate(sampling_rate)
    if decay < 0.0:
        raise ValueError("decay must be non-negative")
    mask = _center_block(height, width, center_fraction, device)
    radius = kspace_radius(height, width, device=device)
    weights = (1.0 + radius).pow(-decay)
    budget = max(1, int(round(sampling_rate * height * width)))
    return _fill_to_budget(mask, weights, budget, generator)


def cartesian_line_mask(
    height: int,
    width: int,
    *,
    sampling_rate: float,
    center_fraction: float = 0.0,
    axis: int = 0,
    decay: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Fully sampled parallel lines, the standard Cartesian undersampling.

    ``axis=0`` keeps whole rows (undersampling along the vertical phase-encode
    direction), ``axis=1`` keeps whole columns. Line selection is
    variable-density in the phase-encode coordinate.
    """
    _validate_rate(sampling_rate)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (rows) or 1 (columns)")
    length = height if axis == 0 else width
    num_lines = max(1, int(round(sampling_rate * length)))

    positions = torch.arange(length, device=device, dtype=torch.float32)
    offset = (positions - length // 2).abs() / max(length // 2, 1)
    weights = (1.0 + offset).pow(-decay)

    selected = torch.zeros(length, dtype=torch.bool, device=device)
    center_side = math.sqrt(center_fraction) if center_fraction > 0.0 else 0.0
    half = int(round(length * center_side / 2.0))
    if half > 0:
        selected[max(length // 2 - half, 0) : length // 2 + half] = True
    selected = _fill_to_budget(selected.unsqueeze(0), weights.unsqueeze(0), num_lines, generator)
    selected = selected.squeeze(0)

    mask = torch.zeros(height, width, dtype=torch.bool, device=device)
    if axis == 0:
        mask[selected, :] = True
    else:
        mask[:, selected] = True
    return mask


def radial_mask(
    height: int,
    width: int,
    *,
    sampling_rate: float,
    num_spokes: Optional[int] = None,
    angle_offset: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Spokes through DC, rasterized onto the Cartesian grid.

    A crude stand-in for a non-Cartesian trajectory: it reproduces the
    characteristic dense-center / sparse-periphery coverage and the streaking
    artifact structure without requiring gridding or a NUFFT. ``sampling_rate``
    is honored approximately, since rasterized spokes overlap near DC.
    """
    _validate_rate(sampling_rate)
    del generator  # spoke angles are deterministic given the offset
    if num_spokes is None:
        # A spoke covers ~max(height, width) grid points before overlap; solve
        # for the count that lands near the requested rate.
        diameter = max(height, width)
        num_spokes = max(1, int(round(sampling_rate * height * width / diameter)))

    mask = torch.zeros(height, width, dtype=torch.bool, device=device)
    cy, cx = height // 2, width // 2
    steps = 2 * max(height, width)
    radii = torch.linspace(-0.5, 0.5, steps, device=device)
    for spoke in range(num_spokes):
        angle = math.radians(angle_offset + spoke * 180.0 / num_spokes)
        ys = (radii * height * math.sin(angle)).round().long() + cy
        xs = (radii * width * math.cos(angle)).round().long() + cx
        inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        mask[ys[inside], xs[inside]] = True
    return mask


MASK_FAMILIES: dict[str, Callable[..., torch.Tensor]] = {
    "uniform": uniform_random_mask,
    "variable-density": variable_density_mask,
    "cartesian": cartesian_line_mask,
    "radial": radial_mask,
}


def build_mask_sequence(
    num_frames: int,
    height: int,
    width: int,
    *,
    sampling_rate: float,
    family: str = "variable-density",
    center_fraction: float = 0.0,
    seed: int = 0,
    device: torch.device | str | None = None,
    **family_kwargs,
) -> torch.Tensor:
    """Draw one independent mask per frame; returns ``[num_frames, H, W]`` bool.

    Radial masks rotate by the golden angle between frames instead of being
    redrawn, which is the standard way to make consecutive spokes complementary.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if family not in MASK_FAMILIES:
        raise ValueError(f"Unknown mask family: {family}. Choose from {sorted(MASK_FAMILIES)}")
    generate = MASK_FAMILIES[family]
    # Masks are always drawn on the CPU so a single seeded generator gives the
    # same pattern regardless of where the fit will run; the stack moves after.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    masks = []
    for frame in range(num_frames):
        kwargs = dict(family_kwargs)
        if family == "radial":
            kwargs.setdefault("angle_offset", frame * GOLDEN_ANGLE_DEGREES)
        else:
            kwargs.setdefault("center_fraction", center_fraction)
        masks.append(
            generate(
                height,
                width,
                sampling_rate=sampling_rate,
                generator=generator,
                **kwargs,
            )
        )
    return torch.stack(masks, dim=0).to(device)
