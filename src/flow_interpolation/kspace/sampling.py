"""k-space sampling masks.

Every generator returns boolean masks laid out in the centered convention of
:mod:`flow_interpolation.kspace.transforms`, so index ``(height // 2,
width // 2)`` is DC.

There are two kinds. *Per-frame* families (:data:`MASK_FAMILIES`) draw each
frame independently: the union over a temporal window still grows as the window
widens, but only in expectation, and repeated draws waste samples on
frequencies the window already covers. *Sequence* families
(:data:`SEQUENCE_MASK_FAMILIES`) generate the whole run at once and can
coordinate across frames to avoid that waste.

Either way, the union over a window covering more than any single frame is the
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


def poisson_density_mask(
    height: int,
    width: int,
    *,
    sampling_rate: float,
    center_fraction: float = 0.0,
    lam: float = 1.0,
    radial_scale: float = 6.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Random points whose radial density follows a Poisson pmf.

    Distance from DC is mapped to a continuous Poisson count
    ``k = radius * radial_scale``, and the sampling weight is
    ``p(k; lam) = lam^k e^-lam / k!`` evaluated via ``lgamma``. The factorial
    makes the tail fall off super-exponentially, so this concentrates on the
    low frequencies far more aggressively than the power-law
    :func:`variable_density_mask` at comparable settings — useful for asking how
    much the periphery is worth at a fixed budget.

    ``lam`` controls where the density peaks: the Poisson pmf is maximized at
    ``k = floor(lam)``, so ``lam <= 1`` (the default) decays monotonically from
    DC as intended. Larger values deliberately peak on an annulus of mid
    frequencies instead, which is a valid pattern but no longer center-biased.

    Weights are never exactly zero, so the periphery is still reachable once the
    high-weight entries are exhausted; the budget is always met exactly.
    """
    _validate_rate(sampling_rate)
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    if radial_scale <= 0.0:
        raise ValueError("radial_scale must be positive")

    mask = _center_block(height, width, center_fraction, device)
    counts = kspace_radius(height, width, device=device) * radial_scale
    # Computed in log space: k! overflows quickly once radial_scale is large.
    log_weights = (
        counts * math.log(lam) - lam - torch.lgamma(counts + 1.0)
    )
    weights = (log_weights - log_weights.max()).exp()
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


def without_replacement_sequence(
    num_frames: int,
    height: int,
    width: int,
    *,
    sampling_rate: float,
    center_fraction: float = 0.0,
    decay: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Deal k-space points across frames without replacement.

    Points are drawn without replacement from a shuffled pool and dealt out to
    consecutive frames; when the pool empties it is reshuffled and dealing
    continues. Every frame gets exactly the same number of samples, but no
    frequency repeats until every other frequency has been used once, so
    coverage over time is far more even than independent per-frame draws.

    That matters here because independent draws waste samples: by the
    coupon-collector argument a window of ``1 / sampling_rate`` independent
    frames covers only about ``1 - 1/e`` (63%) of k-space, while the same window
    of dealt frames covers all of it. Under progressive binning, the bin can
    therefore be narrowed further before coverage collapses.

    ``decay`` biases the *order* of the deal toward low frequencies via
    ``(1 + r)^-decay``, exactly as :func:`variable_density_mask` biases its
    draw. Because every point is still dealt exactly once per cycle, this
    changes which frequencies arrive early in a partial window without
    weakening the full-cycle coverage guarantee at all. ``decay=0`` is a uniform
    deal.

    A ``center_fraction`` block is sampled in every frame and excluded from the
    pool, since re-dealing points that are already always present would waste
    the budget.
    """
    _validate_rate(sampling_rate)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if decay < 0.0:
        raise ValueError("decay must be non-negative")

    total = height * width
    budget = max(1, int(round(sampling_rate * total)))
    center = _center_block(height, width, center_fraction, device)
    center_flat = center.flatten()

    weights = (1.0 + kspace_radius(height, width, device=device)).pow(-decay).flatten()
    weights = weights.masked_fill(center_flat, 0.0)
    pool_size = int((weights > 0).sum().item())
    # The center block is free every frame, so only the remainder is dealt.
    per_frame = max(min(budget - int(center_flat.sum().item()), pool_size), 0)

    def fresh_pool() -> torch.Tensor:
        return torch.multinomial(weights, pool_size, replacement=False, generator=generator)

    masks = torch.zeros(num_frames, total, dtype=torch.bool, device=device)
    masks[:, center_flat] = True
    pool = fresh_pool()
    for frame in range(num_frames):
        if per_frame == 0:
            continue
        if pool.numel() >= per_frame:
            chosen, pool = pool[:per_frame], pool[per_frame:]
        else:
            # The cycle ended mid-frame. Take what is left, reshuffle, and move
            # the just-used points to the back of the new cycle so they are not
            # drawn twice in a row while every other point still gets its turn.
            leftover = pool
            refreshed = fresh_pool()
            used = torch.isin(refreshed, leftover)
            refreshed = torch.cat((refreshed[~used], refreshed[used]))
            needed = per_frame - leftover.numel()
            chosen = torch.cat((leftover, refreshed[:needed]))
            pool = refreshed[needed:]
        masks[frame, chosen] = True

    return masks.view(num_frames, height, width)


MASK_FAMILIES: dict[str, Callable[..., torch.Tensor]] = {
    "uniform": uniform_random_mask,
    "variable-density": variable_density_mask,
    "poisson": poisson_density_mask,
    "cartesian": cartesian_line_mask,
    "radial": radial_mask,
}

SEQUENCE_MASK_FAMILIES: dict[str, Callable[..., torch.Tensor]] = {
    "without-replacement": without_replacement_sequence,
}

MASK_FAMILY_NAMES: tuple[str, ...] = tuple(
    sorted({*MASK_FAMILIES, *SEQUENCE_MASK_FAMILIES})
)


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
    """Build the per-frame masks for a run; returns ``[num_frames, H, W]`` bool.

    Per-frame families are drawn independently for each frame. Radial masks
    rotate by the golden angle between frames instead of being redrawn, which is
    the standard way to make consecutive spokes complementary. Sequence families
    generate the whole run in one call so they can coordinate across frames.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if family not in MASK_FAMILIES and family not in SEQUENCE_MASK_FAMILIES:
        raise ValueError(f"Unknown mask family: {family}. Choose from {list(MASK_FAMILY_NAMES)}")

    if family in SEQUENCE_MASK_FAMILIES:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return SEQUENCE_MASK_FAMILIES[family](
            num_frames,
            height,
            width,
            sampling_rate=sampling_rate,
            center_fraction=center_fraction,
            generator=generator,
            **family_kwargs,
        ).to(device)

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
