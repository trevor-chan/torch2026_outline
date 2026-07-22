"""Correctness of the Fourier forward model, masks, and observation simulation."""

from __future__ import annotations

import pytest
import torch

from flow_interpolation.kspace import (
    build_dynamic_kspace,
    build_mask_sequence,
    cartesian_line_mask,
    fft2c,
    ifft2c,
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
