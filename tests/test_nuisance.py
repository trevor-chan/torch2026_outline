from __future__ import annotations

import torch

from flow_interpolation.data import CadenceInfo, SequenceData
from flow_interpolation.evaluation.experiments.nuisance import (
    run_nuisance_dimension_analysis,
)
from flow_interpolation.utils.flow import FlowSettings
from flow_interpolation.utils.temporal_nuisance import analyze_temporal_nuisance


def test_centered_difference_svd_recovers_low_rank_motion_and_preserves_endpoints() -> None:
    time = torch.arange(12, dtype=torch.float32)
    differences = torch.stack(
        [
            0.2 + torch.sin(2.0 * torch.pi * time / time.numel()),
            -0.1 + torch.cos(4.0 * torch.pi * time / time.numel()),
            torch.full_like(time, 0.3),
        ],
        dim=1,
    )
    trajectory = torch.cat(
        [torch.zeros(1, 3), torch.cumsum(differences, dim=0)],
        dim=0,
    )
    analysis = analyze_temporal_nuisance(
        trajectory,
        sample_spacing=0.1,
        svd_ranks=[0, 1, 2],
        fourier_harmonics=[0, 1, 2],
    )

    assert analysis["summary"]["svd_rank_for_99_percent_centered_energy"] == 2
    torch.testing.assert_close(
        analysis["svd_reconstructions"][2],
        trajectory,
        atol=1e-5,
        rtol=1e-5,
    )
    for reconstruction in analysis["svd_reconstructions"].values():
        torch.testing.assert_close(reconstruction[0], trajectory[0])
        torch.testing.assert_close(reconstruction[-1], trajectory[-1], atol=1e-5, rtol=1e-5)


def test_fourier_low_pass_recovers_known_harmonics() -> None:
    time = torch.arange(16, dtype=torch.float32)
    differences = (
        0.25
        + torch.sin(2.0 * torch.pi * time / time.numel())
        + 0.5 * torch.sin(6.0 * torch.pi * time / time.numel())
    )[:, None]
    trajectory = torch.cat(
        [torch.zeros(1, 1), torch.cumsum(differences, dim=0)],
        dim=0,
    )
    analysis = analyze_temporal_nuisance(
        trajectory,
        sample_spacing=0.25,
        svd_ranks=[0],
        fourier_harmonics=[0, 1, 3],
    )

    assert analysis["fourier_metrics"][1][
        "centered_difference_energy_retained_fraction"
    ] > 0.79
    torch.testing.assert_close(
        analysis["fourier_reconstructions"][3],
        trajectory,
        atol=1e-5,
        rtol=1e-5,
    )


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x, conditioning, t):
        del conditioning, t
        return torch.zeros_like(x)


def test_nuisance_experiment_writes_compact_metrics_and_sweep(tmp_path) -> None:
    frames = torch.rand(5, 3, 4, 4)
    observed_indices = torch.tensor([0, 2, 4])
    sequence = SequenceData(
        frames=frames,
        observed_indices=observed_indices,
        observed_frames=frames[observed_indices],
        cadence=CadenceInfo(
            training_frame_dt=0.2,
            high_frame_dt=0.1,
            requested_ratio=2.0,
            endpoint_stride=2,
            actual_endpoint_dt=0.2,
            endpoint_dt_error=0.0,
            relative_error=0.0,
            rounding_policy="exact",
        ),
        high_rate_color_walk_std=0.1,
        start_index=0,
    )
    payload = run_nuisance_dimension_analysis(
        model=_ZeroVelocity(),
        device=torch.device("cpu"),
        sequence=sequence,
        flow=FlowSettings(0.0, 1.0, 1, "euler", 5, 5),
        boundary_noise_mode="shared",
        seed=7,
        svd_ranks=[0, 1],
        fourier_harmonics=[0, 1],
        output_dir=str(tmp_path),
        plot_results=False,
        write_videos=False,
    )

    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "sweep.csv").is_file()
    assert payload["summary"]["frame_count"] == 5
    assert payload["sweeps"]["svd"][0]["parameter"] == 0
    assert payload["sweeps"]["fourier"][0]["parameter"] == 0
    assert "foreground_weighted_rmse" in payload["sweeps"]["svd"][0][
        "decoded_vs_dense_reference"
    ]
    header = (tmp_path / "sweep.csv").read_text().splitlines()[0]
    assert "decoded_vs_dense_reference_activity_centroid_error_pixels" in header
