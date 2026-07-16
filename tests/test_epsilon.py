from __future__ import annotations

import torch

from flow_interpolation.data import CadenceInfo, SequenceData
from flow_interpolation.evaluation.experiments.epsilon import (
    _comparison_metrics,
    _trajectory_snr_metrics,
    _variance_maps,
    run_epsilon_ablation,
)
from flow_interpolation.utils.flow import FlowSettings


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x, conditioning, t):
        del conditioning, t
        return torch.zeros_like(x)


def _sequence() -> SequenceData:
    frames = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 2.0], [2.0, 2.0]]],
        ]
    )
    observed_indices = torch.tensor([0, 1, 2])
    return SequenceData(
        frames=frames,
        observed_indices=observed_indices,
        observed_frames=frames,
        cadence=CadenceInfo(
            training_frame_dt=1.0,
            high_frame_dt=1.0,
            requested_ratio=1.0,
            endpoint_stride=1,
            actual_endpoint_dt=1.0,
            endpoint_dt_error=0.0,
            relative_error=0.0,
            rounding_policy="exact",
        ),
        high_rate_color_walk_std=0.0,
        start_index=0,
    )


def test_variance_maps_separate_image_and_boundary_axes() -> None:
    latents = torch.tensor(
        [
            [[[[0.0]]], [[[2.0]]]],
            [[[[2.0]]], [[[4.0]]]],
        ]
    )
    maps = _variance_maps(latents)
    torch.testing.assert_close(maps["population"], torch.tensor([[[1.0]]]))
    torch.testing.assert_close(maps["boundary"], torch.tensor([[[1.0]]]))
    torch.testing.assert_close(maps["total"], torch.tensor([[[2.0]]]))
    torch.testing.assert_close(maps["coordinate_mean_square"], torch.tensor([[[4.0]]]))
    torch.testing.assert_close(maps["second_moment"], torch.tensor([[[6.0]]]))


def test_centered_comparison_removes_common_latent_shift() -> None:
    reference = torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 1, 2)
    shifted = reference + 5.0
    metrics = _comparison_metrics(shifted, reference)
    assert metrics["latent_rmse"] == 5.0
    assert metrics["trajectory_centered_rmse"] == 0.0
    assert metrics["trajectory_centered_rmse_over_reference_rms"] == 0.0
    assert metrics["trajectory_step_rmse"] == 0.0
    assert metrics["trajectory_step_rmse_over_reference_rms"] == 0.0


def test_trajectory_snr_matches_vector_energy_definition() -> None:
    latents = torch.tensor(
        [
            [[[[0.0]]], [[[2.0]]], [[[4.0]]]],
            [[[[2.0]]], [[[4.0]]], [[[6.0]]]],
        ]
    )
    metrics = _trajectory_snr_metrics(latents, frame_spacing=0.5)

    # Per frame, the two draws are one unit from their mean: V_enc=1.
    assert metrics["encoding_variability_l2_squared"] == 1.0
    # The draw-mean trajectory advances by two units per frame: V_time=4.
    assert metrics["temporal_signal_l2_squared"] == 4.0
    assert metrics["trajectory_signal_to_encoding_noise_ratio"] == 4.0
    assert metrics["trajectory_encoding_noise_to_signal_ratio"] == 0.25
    assert metrics["temporal_signal_l2_squared_per_time_squared"] == 16.0


def test_epsilon_ablation_writes_metrics_csv_plots_and_tensors(tmp_path) -> None:
    payload = run_epsilon_ablation(
        model=_ZeroVelocity().eval(),
        device=torch.device("cpu"),
        sequence=_sequence(),
        flow=FlowSettings(0.1, 0.9, 1, "euler", 64, 64),
        epsilons=[0.0, 0.2],
        num_boundary_samples=3,
        boundary_noise_mode="shared",
        frame_source="observed",
        seed=7,
        output_dir=str(tmp_path),
        save_tensors=True,
    )

    assert payload["effective_epsilons"] == [0.0, 0.1, 0.2]
    assert payload["reference_epsilon"] == 0.1
    assert payload["image_background_noise_std"] == 0.0
    assert payload["trajectory_snr_definition"]["signal_to_noise_ratio"] == (
        "temporal_signal / encoding_variability"
    )
    assert payload["rows"][0]["boundary_variance_mean"] == 0.0
    assert payload["prior_reference"]["coordinate_variance"] == 1.0
    for row in payload["rows"]:
        assert abs(
            row["coordinate_second_moment_mean"]
            - row["total_variance_mean"]
            - row["coordinate_mean_square_mean"]
        ) < 1e-6
    assert (tmp_path / "epsilon_ablation_metrics.json").is_file()
    assert (tmp_path / "epsilon_ablation.csv").is_file()
    assert (tmp_path / "epsilon_ablation_summary.png").is_file()
    assert (tmp_path / "epsilon_variance_maps.png").is_file()
    assert (tmp_path / "epsilon_trajectory_snr.png").is_file()
    assert (tmp_path / "epsilon_ablation_tensors.pt").is_file()
