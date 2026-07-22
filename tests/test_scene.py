"""Scene models, binning schedules, and the k-space fitting objective."""

from __future__ import annotations

import pytest
import torch

from flow_interpolation.kspace import build_dynamic_kspace, fft2c, ifft2c
from flow_interpolation.scene import (
    BinSchedule,
    ReconstructionVisualizer,
    bin_window,
    build_bin_schedule,
    build_scene_model,
    kspace_consistency_loss,
    log_magnitude,
    reconstruction_panels,
    spatial_tv,
    temporal_tv,
)
from flow_interpolation.scene.fit import SceneFitter


def _frames(num_frames: int = 8, size: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(num_frames, 3, size, size, generator=generator)


@pytest.mark.parametrize("name", ["fourier-mlp", "kplanes"])
def test_scene_models_render_the_expected_shape_and_range(name):
    model = build_scene_model(name, height=16, width=12, channels=3)
    rendered = model.render(torch.tensor([0.0, 0.5, 1.0]))
    assert rendered.shape == (3, 3, 16, 12)
    assert torch.isfinite(rendered).all()


@pytest.mark.parametrize("name", ["fourier-mlp", "kplanes"])
def test_scene_models_are_differentiable_through_the_kspace_loss(name):
    model = build_scene_model(name, height=16, width=16, channels=3)
    data = build_dynamic_kspace(_frames(), sampling_rate=0.2, seed=0)
    window = bin_window(torch.tensor([1, 4]), half_width=1, num_frames=data.num_frames)
    loss = kspace_consistency_loss(
        model.render(data.times[torch.tensor([1, 4])]),
        data.kspace[window],
        data.masks[window],
    )
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("name", ["fourier-mlp", "kplanes"])
def test_scene_models_vary_with_time(name):
    model = build_scene_model(name, height=16, width=16, channels=3)
    # The output head is zero-initialized, so an untrained model renders zero at
    # every time by construction; randomize it to expose the time dependence
    # carried by the features underneath.
    head = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
    torch.nn.init.normal_(head.weight, std=0.1)

    first = model.render(torch.tensor([0.0]))
    last = model.render(torch.tensor([1.0]))
    assert not torch.allclose(first, last)


def test_consistency_loss_is_zero_for_a_perfect_reconstruction():
    frames = _frames()
    data = build_dynamic_kspace(frames, sampling_rate=0.2, seed=0)
    window = bin_window(torch.arange(data.num_frames), half_width=0, num_frames=data.num_frames)
    loss = kspace_consistency_loss(frames, data.kspace[window], data.masks[window])
    assert loss.item() < 1e-9


def test_consistency_loss_ignores_unobserved_frequencies():
    frames = _frames()
    data = build_dynamic_kspace(frames, sampling_rate=0.2, family="uniform", seed=0)
    window = bin_window(torch.arange(data.num_frames), half_width=0, num_frames=data.num_frames)

    # Perturb only the unobserved part of the spectrum; the loss must not move.
    # The perturbed image is kept complex: projecting it back to the reals would
    # symmetrize the perturbation and leak it into the observed frequencies.
    unobserved = (~data.masks).unsqueeze(1)
    perturbation = torch.randn_like(frames) * unobserved
    altered = ifft2c(fft2c(frames) + perturbation)
    loss = kspace_consistency_loss(altered, data.kspace[window], data.masks[window])
    assert loss.item() < 1e-8


def test_bin_window_clamps_at_the_sequence_boundaries():
    window = bin_window(torch.tensor([0, 5, 9]), half_width=2, num_frames=10)
    assert window.shape == (3, 5)
    assert window[0].tolist() == [0, 0, 0, 1, 2]
    assert window[1].tolist() == [3, 4, 5, 6, 7]
    assert window[2].tolist() == [7, 8, 9, 9, 9]


def test_bin_widths_are_odd_and_monotone_under_the_curriculum():
    schedule = BinSchedule(start_width=25, end_width=1, anneal_steps=1_000)
    widths = [schedule.width_at(step) for step in range(0, 1_200, 50)]
    assert all(width % 2 == 1 for width in widths)
    assert widths == sorted(widths, reverse=True)
    assert widths[0] == 25
    assert widths[-1] == 1


def test_fixed_conditions_never_change_width():
    wide = build_bin_schedule("wide", num_frames=100, max_steps=1_000, start_width=25)
    narrow = build_bin_schedule("narrow", num_frames=100, max_steps=1_000, end_width=1)
    assert {wide.width_at(step) for step in (0, 500, 1_000)} == {25}
    assert {narrow.width_at(step) for step in (0, 500, 1_000)} == {1}


def test_curriculum_ends_on_the_narrow_objective():
    schedule = build_bin_schedule(
        "curriculum", num_frames=100, max_steps=1_000, start_width=25, end_width=1
    )
    assert schedule.width_at(0) == 25
    assert schedule.width_at(1_000) == 1


def test_start_width_is_clamped_to_the_sequence_length():
    schedule = build_bin_schedule("wide", num_frames=9, max_steps=100, start_width=101)
    assert schedule.width_at(0) <= 9


def test_regularizers_penalize_variation():
    constant = torch.ones(4, 3, 8, 8)
    noisy = torch.rand(4, 3, 8, 8)
    assert spatial_tv(constant).item() == 0.0
    assert temporal_tv(constant).item() == 0.0
    assert spatial_tv(noisy) > 0
    assert temporal_tv(noisy) > 0


def test_fitting_reduces_the_data_consistency_loss():
    data = build_dynamic_kspace(_frames(num_frames=8), sampling_rate=0.3, seed=0)
    model = build_scene_model("kplanes", height=16, width=16, channels=3, resolutions=(16,))
    fitter = SceneFitter(
        model=model,
        data=data,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        schedule=BinSchedule(start_width=1, end_width=1, kind="constant"),
        device=torch.device("cpu"),
        max_steps=0,
        batch_size=4,
        eval_interval=0,
    )
    before = fitter.step(0)["consistency"]
    for step in range(1, 60):
        fitter.step(step)
    after = fitter.step(60)["consistency"]
    assert after < before


def test_log_magnitude_maps_the_reference_peak_to_white():
    spectrum = fft2c(_frames())
    reference = spectrum.abs().max().item()
    displayed = log_magnitude(spectrum, reference=reference)
    assert displayed.shape == (8, 3, 16, 16)
    assert displayed.min() >= 0.0 and displayed.max() <= 1.0
    # The DC bin carries the peak, so it nearly saturates -- only nearly, since
    # the display averages the three channels while the reference is the
    # per-channel peak. Zeros map to black.
    assert displayed.max().item() > 0.99
    assert log_magnitude(torch.zeros_like(spectrum), reference=reference).max().item() == 0.0


def test_unobserved_frequencies_are_black_in_the_measured_panel():
    data = build_dynamic_kspace(_frames(), sampling_rate=0.2, family="uniform", seed=0)
    displayed = log_magnitude(data.kspace, reference=fft2c(data.frames).abs().max().item())
    unobserved = displayed[:, 0][~data.masks]
    assert unobserved.max().item() == 0.0


def test_reconstruction_panel_has_two_rows_and_four_columns():
    data = build_dynamic_kspace(_frames(), sampling_rate=0.2, seed=0)
    panels = reconstruction_panels(
        data.frames, data.kspace, data.frames, display_scale=1, gap=2
    )
    assert panels.shape == (8, 3, 2 * 16 + 2, 4 * 16 + 3 * 2)


def test_panel_residuals_vanish_for_a_perfect_reconstruction():
    data = build_dynamic_kspace(_frames(), sampling_rate=0.2, seed=0)
    panels = reconstruction_panels(data.frames, data.kspace, data.frames, gap=0)
    width = 16
    image_residual = panels[:, :, :width, 3 * width :]
    kspace_residual = panels[:, :, width:, 3 * width :]
    assert image_residual.abs().max().item() < 1e-5
    assert kspace_residual.abs().max().item() < 1e-4


def _fitter_for(data):
    model = build_scene_model("kplanes", height=16, width=16, channels=3, resolutions=(16,))
    return SceneFitter(
        model=model,
        data=data,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        schedule=BinSchedule(start_width=1, end_width=1, kind="constant"),
        device=torch.device("cpu"),
        eval_interval=0,
    )


def test_visualizer_writes_snippet_videos_only_on_its_interval(tmp_path):
    data = build_dynamic_kspace(_frames(num_frames=12), sampling_rate=0.2, seed=0)
    visualizer = ReconstructionVisualizer(
        data=data, call_every=5, output_dir=tmp_path, snippet_frames=4
    )
    for step in range(1, 11):
        visualizer(step, _fitter_for(data))
    assert sorted(path.name for path in tmp_path.glob("*.mp4")) == [
        "snippet_000000005.mp4",
        "snippet_000000010.mp4",
    ]


def test_snippet_is_contiguous_and_centered_by_default():
    data = build_dynamic_kspace(_frames(num_frames=12), sampling_rate=0.2, seed=0)
    visualizer = ReconstructionVisualizer(data=data, snippet_frames=4)
    assert visualizer._source_indices.tolist() == [4, 5, 6, 7]


def test_snippet_start_is_clamped_to_keep_the_window_in_range():
    data = build_dynamic_kspace(_frames(num_frames=12), sampling_rate=0.2, seed=0)
    visualizer = ReconstructionVisualizer(data=data, snippet_frames=4, snippet_start=99)
    assert visualizer._source_indices.tolist() == [8, 9, 10, 11]


def test_upsampling_adds_query_times_between_observations():
    data = build_dynamic_kspace(_frames(num_frames=12), sampling_rate=0.2, seed=0)
    visualizer = ReconstructionVisualizer(data=data, snippet_frames=4, snippet_upsample=3)
    # Three query times per observation interval, spanning the same window.
    assert visualizer._query_times.shape[0] == (4 - 1) * 3 + 1
    assert visualizer._query_times[0] == data.times[4]
    assert visualizer._query_times[-1] == data.times[7]
    # Measured columns hold the nearest observation between query times.
    assert visualizer._source_indices.tolist() == [4, 4, 5, 5, 5, 6, 6, 6, 7, 7]
    # Playback speeds up so the snippet still runs at the acquisition's rate.
    assert visualizer.playback_fps == visualizer.fps * 3


def test_snippet_panels_cover_every_query_time():
    data = build_dynamic_kspace(_frames(num_frames=12), sampling_rate=0.2, seed=0)
    visualizer = ReconstructionVisualizer(data=data, snippet_frames=4, snippet_upsample=2)
    panels = visualizer.snippet_panels(_fitter_for(data))
    assert panels.shape[0] == visualizer._query_times.shape[0]
