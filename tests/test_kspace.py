"""Correctness of the Fourier forward model, masks, and observation simulation."""

from __future__ import annotations

import pytest
import torch

from flow_interpolation.kspace import (
    MASK_FAMILY_NAMES,
    build_dynamic_kspace,
    build_mask_sequence,
    cartesian_line_mask,
    fft2c,
    ifft2c,
    poisson_density_mask,
    radial_mask,
    uniform_random_mask,
    variable_density_mask,
    zero_filled_reconstruction,
)
from flow_interpolation.kspace.transforms import kspace_radius


def _frames(num_frames: int = 6, size: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(num_frames, 3, size, size, generator=generator)


def test_fft_roundtrip_is_exact():
    frames = _frames()
    recovered = ifft2c(fft2c(frames))
    assert torch.allclose(recovered.real, frames, atol=1e-5)
    assert recovered.imag.abs().max() < 1e-5


def test_transform_is_unitary():
    frames = _frames()
    assert torch.allclose(
        fft2c(frames).abs().square().sum(),
        frames.square().sum(),
        rtol=1e-5,
    )


def test_dc_sits_at_the_center():
    frames = torch.ones(1, 1, 16, 16)
    spectrum = fft2c(frames)
    peak = spectrum.abs().flatten().argmax().item()
    assert divmod(peak, 16) == (8, 8)
    assert kspace_radius(16, 16)[8, 8].item() == 0.0


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.25])
def test_uniform_mask_hits_the_requested_rate_exactly(rate):
    generator = torch.Generator().manual_seed(0)
    mask = uniform_random_mask(32, 32, sampling_rate=rate, generator=generator)
    assert mask.sum().item() == round(rate * 32 * 32)


def test_center_block_is_always_sampled_and_charged_to_the_budget():
    generator = torch.Generator().manual_seed(0)
    mask = uniform_random_mask(
        32, 32, sampling_rate=0.1, center_fraction=0.02, generator=generator
    )
    assert mask.sum().item() == round(0.1 * 32 * 32)
    assert mask[16, 16]


def test_variable_density_concentrates_samples_near_dc():
    generator = torch.Generator().manual_seed(0)
    mask = variable_density_mask(32, 32, sampling_rate=0.1, decay=3.0, generator=generator)
    radius = kspace_radius(32, 32)
    assert radius[mask].mean() < radius.mean()


def test_cartesian_mask_keeps_whole_lines():
    generator = torch.Generator().manual_seed(0)
    mask = cartesian_line_mask(32, 24, sampling_rate=0.25, generator=generator)
    row_sums = mask.sum(dim=1)
    assert set(row_sums.unique().tolist()) <= {0, 24}
    assert (row_sums > 0).sum().item() == 8


def test_radial_mask_passes_through_dc():
    mask = radial_mask(32, 32, sampling_rate=0.1)
    assert mask[16, 16]
    assert 0.02 < mask.float().mean().item() < 0.4


def test_masks_differ_across_frames_so_the_union_covers_more():
    masks = build_mask_sequence(20, 32, 32, sampling_rate=0.1, family="uniform", seed=0)
    per_frame = masks[0].float().mean().item()
    union = masks.any(dim=0).float().mean().item()
    assert union > 3 * per_frame


def test_mask_sequence_is_seed_reproducible():
    first = build_mask_sequence(5, 16, 16, sampling_rate=0.2, seed=3)
    second = build_mask_sequence(5, 16, 16, sampling_rate=0.2, seed=3)
    third = build_mask_sequence(5, 16, 16, sampling_rate=0.2, seed=4)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_observations_are_zero_outside_the_mask():
    data = build_dynamic_kspace(_frames(), sampling_rate=0.1, family="uniform", seed=0)
    unobserved = data.kspace[~data.masks.unsqueeze(1).expand_as(data.kspace)]
    assert unobserved.abs().max().item() == 0.0


def test_full_sampling_reconstructs_the_frames_exactly():
    frames = _frames()
    data = build_dynamic_kspace(frames, sampling_rate=1.0, family="uniform", seed=0)
    assert torch.allclose(zero_filled_reconstruction(data), frames, atol=1e-5)


def test_window_coverage_grows_with_bin_width():
    data = build_dynamic_kspace(_frames(num_frames=32), sampling_rate=0.1, seed=0)
    coverages = [data.coverage_for_window(half) for half in (0, 2, 8)]
    assert coverages == sorted(coverages)
    assert coverages[0] < coverages[-1]


def test_times_are_normalized_and_ordered():
    data = build_dynamic_kspace(_frames(num_frames=10), sampling_rate=0.1, seed=0)
    assert data.times[0].item() == 0.0
    assert data.times[-1].item() == 1.0
    assert torch.all(data.times.diff() > 0)


def test_poisson_mask_is_center_biased_and_hits_the_budget():
    generator = torch.Generator().manual_seed(0)
    mask = poisson_density_mask(32, 32, sampling_rate=0.1, generator=generator)
    radius = kspace_radius(32, 32)
    assert mask.sum().item() == round(0.1 * 32 * 32)
    assert radius[mask].mean() < radius.mean()


def test_poisson_concentrates_harder_than_the_power_law():
    generator = torch.Generator().manual_seed(0)
    poisson = poisson_density_mask(32, 32, sampling_rate=0.1, generator=generator)
    power_law = variable_density_mask(32, 32, sampling_rate=0.1, decay=3.0, generator=generator)
    radius = kspace_radius(32, 32)
    # The factorial tail falls off faster than (1 + r)^-3.
    assert radius[poisson].mean() < radius[power_law].mean()


def test_large_lam_moves_the_poisson_peak_off_dc():
    generator = torch.Generator().manual_seed(0)
    radius = kspace_radius(32, 32)
    centered = poisson_density_mask(32, 32, sampling_rate=0.1, lam=1.0, generator=generator)
    annulus = poisson_density_mask(32, 32, sampling_rate=0.1, lam=4.0, generator=generator)
    assert radius[annulus].mean() > radius[centered].mean()


def test_poisson_rejects_degenerate_parameters():
    with pytest.raises(ValueError):
        poisson_density_mask(16, 16, sampling_rate=0.1, lam=0.0)
    with pytest.raises(ValueError):
        poisson_density_mask(16, 16, sampling_rate=0.1, radial_scale=0.0)


def test_without_replacement_gives_every_frame_the_same_count():
    masks = build_mask_sequence(
        30, 32, 32, sampling_rate=0.1, family="without-replacement", seed=0
    )
    counts = masks.flatten(1).sum(dim=1)
    assert counts.unique().numel() == 1
    assert counts[0].item() == round(0.1 * 32 * 32)


# 1/8 of a 32x32 grid is 128 points, so a cycle is exactly 8 frames with no
# remainder. Rates that do not divide the grid leave a partial cycle, which
# blurs the once-per-cycle guarantee across the boundary.
CYCLE_RATE = 0.125
CYCLE_FRAMES = 8


def test_without_replacement_covers_k_space_exactly_once_per_cycle():
    masks = build_mask_sequence(
        CYCLE_FRAMES, 32, 32, sampling_rate=CYCLE_RATE, family="without-replacement", seed=0
    )
    assert masks.sum(dim=0).max().item() == 1
    assert masks.any(dim=0).all()


def test_without_replacement_beats_independent_draws_on_window_coverage():
    kwargs = dict(sampling_rate=CYCLE_RATE, seed=0)
    dealt = build_mask_sequence(CYCLE_FRAMES, 32, 32, family="without-replacement", **kwargs)
    independent = build_mask_sequence(CYCLE_FRAMES, 32, 32, family="uniform", **kwargs)
    # Coupon collector: independent draws leave roughly 1/e of k-space unseen.
    assert dealt.any(dim=0).float().mean().item() == 1.0
    assert independent.any(dim=0).float().mean().item() < 0.7


def test_without_replacement_repeats_only_after_a_full_cycle():
    masks = build_mask_sequence(
        CYCLE_FRAMES + 1, 32, 32, sampling_rate=CYCLE_RATE, family="without-replacement", seed=0
    )
    # The extra frame starts a new cycle, so at most one repeat per frequency.
    assert masks.sum(dim=0).max().item() == 2
    assert masks[:CYCLE_FRAMES].sum(dim=0).max().item() == 1


def test_without_replacement_decay_biases_order_but_not_cycle_coverage():
    biased = build_mask_sequence(
        CYCLE_FRAMES,
        32,
        32,
        sampling_rate=CYCLE_RATE,
        family="without-replacement",
        seed=0,
        decay=3.0,
    )
    radius = kspace_radius(32, 32)
    # The first frame of the cycle takes the low frequencies...
    assert radius[biased[0]].mean() < radius.mean()
    # ...but the full cycle still covers everything exactly once.
    assert biased.sum(dim=0).max().item() == 1
    assert biased.any(dim=0).all()


def test_without_replacement_center_block_repeats_every_frame():
    masks = build_mask_sequence(
        10,
        32,
        32,
        sampling_rate=0.1,
        family="without-replacement",
        center_fraction=0.02,
        seed=0,
    )
    assert masks[:, 16, 16].all()
    counts = masks.flatten(1).sum(dim=1)
    assert counts.unique().numel() == 1


def test_sequence_families_are_seed_reproducible():
    first = build_mask_sequence(8, 16, 16, sampling_rate=0.2, family="without-replacement", seed=3)
    second = build_mask_sequence(8, 16, 16, sampling_rate=0.2, family="without-replacement", seed=3)
    third = build_mask_sequence(8, 16, 16, sampling_rate=0.2, family="without-replacement", seed=4)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_every_advertised_family_builds_a_sequence():
    for family in MASK_FAMILY_NAMES:
        masks = build_mask_sequence(6, 16, 16, sampling_rate=0.15, family=family, seed=0)
        assert masks.shape == (6, 16, 16)
        assert masks.dtype == torch.bool
        assert masks.any()


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="Unknown mask family"):
        build_mask_sequence(4, 16, 16, sampling_rate=0.1, family="nonsense")
