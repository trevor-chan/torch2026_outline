from __future__ import annotations

import torch

from flow_interpolation.data import BouncingBallVideoDataset, build_sequence


def _dataset(background_noise_std: float, seed: int = 7) -> BouncingBallVideoDataset:
    return BouncingBallVideoDataset(
        num_samples=5,
        image_size=16,
        seed=seed,
        background_noise_std=background_noise_std,
        write_video=False,
    )


def test_background_noise_is_deterministic_and_does_not_change_default() -> None:
    default = _dataset(0.0)
    explicit_zero = _dataset(0.0)
    first = _dataset(0.01)
    second = _dataset(0.01)

    torch.testing.assert_close(default.samples, explicit_zero.samples)
    torch.testing.assert_close(first.samples, second.samples)
    assert not torch.equal(first.samples, default.samples)
    assert float(first.samples.min()) >= 0.0
    assert float(first.samples.max()) <= 1.0


def test_background_noise_uses_independent_rng_from_trajectory() -> None:
    clean = _dataset(0.0)
    noisy = _dataset(0.01)

    # Fully opaque foreground pixels are independent of the background field.
    opaque_foreground = clean.samples == 1.0
    assert opaque_foreground.any()
    torch.testing.assert_close(noisy.samples[opaque_foreground], clean.samples[opaque_foreground])


def test_sequence_threads_background_noise_configuration() -> None:
    sequence = build_sequence(
        image_size=16,
        seed=3,
        start_index=0,
        num_intervals=2,
        training_frame_dt=0.25,
        high_frame_dt=0.25,
        background_noise_std=0.01,
        stride_rounding="exact",
    )

    assert sequence.background_noise_std == 0.01
    assert sequence.frames.shape[0] == 3
