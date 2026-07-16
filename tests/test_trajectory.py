from __future__ import annotations

import torch

from flow_interpolation.data import CadenceInfo, SequenceData
from flow_interpolation.evaluation.experiments.trajectory import run_trajectory_analysis
from flow_interpolation.utils.flow import FlowSettings
from flow_interpolation.utils.trajectory import (
    analyze_trajectory_at_stride,
    subspace_residuals,
    trajectory_geometry,
)


def test_endpoint_plane_residual_detects_out_of_plane_motion() -> None:
    in_plane = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    indices = torch.tensor([0, 4])
    plane_metrics = subspace_residuals(in_plane, indices, local_keyframe_count=2)
    assert float(plane_metrics["relative_l2"].max()) < 1e-6

    out_of_plane = in_plane.clone()
    out_of_plane[2, 2] = 1.0
    residual_metrics = subspace_residuals(out_of_plane, indices, local_keyframe_count=2)
    assert float(residual_metrics["relative_l2"][2]) > 0.7
    assert float(residual_metrics["energy_fraction"][2]) > 0.5


def test_straight_trajectory_has_zero_interior_acceleration_and_curvature() -> None:
    trajectory = torch.stack([torch.tensor([float(index), 1.0]) for index in range(5)])
    geometry = trajectory_geometry(trajectory, sample_spacing=0.25)

    torch.testing.assert_close(geometry["acceleration_l2"][1:-1], torch.zeros(3))
    torch.testing.assert_close(geometry["curvature"][1:-1], torch.zeros(3))
    torch.testing.assert_close(
        geometry["speed_l2"],
        torch.full((5,), 4.0),
    )


def test_lerp_recovers_piecewise_linear_reference() -> None:
    trajectory = torch.stack([torch.tensor([float(index), 2.0 * index]) for index in range(7)])
    analysis = analyze_trajectory_at_stride(
        trajectory,
        keyframe_stride=3,
        sample_spacing=1.0,
        methods=["lerp"],
        slerp_mode="iscs",
    )

    assert analysis["ignored_tail_frames"] == 0
    torch.testing.assert_close(analysis["predictions"]["lerp"], trajectory)
    torch.testing.assert_close(
        analysis["method_metrics"]["lerp"]["relative_l2"],
        torch.zeros(7),
    )
    torch.testing.assert_close(
        analysis["method_metrics"]["lerp"]["tangent_cosine_similarity"],
        torch.ones(7),
    )


def test_stride_analysis_reports_ignored_incomplete_tail() -> None:
    trajectory = torch.randn(10, 4)
    analysis = analyze_trajectory_at_stride(
        trajectory,
        keyframe_stride=4,
        sample_spacing=1.0,
        methods=["slerp", "squad"],
        slerp_mode="radius-lerp",
    )

    assert analysis["reference"].shape[0] == 9
    assert analysis["ignored_tail_frames"] == 1
    assert analysis["keyframe_indices"].tolist() == [0, 4, 8]


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x, conditioning, t):
        del conditioning, t
        return torch.zeros_like(x)


def test_trajectory_experiment_writes_density_summary_and_per_frame_csv(tmp_path) -> None:
    frames = torch.randn(5, 1, 2, 2)
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
    output_json = tmp_path / "metrics.json"
    output_csv = tmp_path / "per_frame.csv"

    payload = run_trajectory_analysis(
        model=_ZeroVelocity(),
        device=torch.device("cpu"),
        sequence=sequence,
        flow=FlowSettings(0.0, 1.0, 1, "euler", 5, 5),
        methods=["lerp", "squad"],
        slerp_mode="radius-lerp",
        boundary_noise_mode="shared",
        seed=3,
        output_json=str(output_json),
        output_csv=str(output_csv),
        keyframe_strides=[1, 2],
    )

    assert output_json.is_file()
    assert output_csv.is_file()
    assert (tmp_path / "plots" / "reference_geometry.png").is_file()
    assert (tmp_path / "plots" / "density_summary.png").is_file()
    assert (tmp_path / "plots" / "paths_and_residuals_stride_0001.png").is_file()
    assert (tmp_path / "plots" / "paths_and_residuals_stride_0002.png").is_file()
    assert payload["reference_definition"].startswith("Dense dataset frames")
    assert payload["density_sweep"]["2"]["methods"].keys() == {"lerp", "squad"}
    assert payload["artifacts"]["plots"] == [
        "plots/reference_geometry.png",
        "plots/density_summary.png",
        "plots/paths_and_residuals_stride_0001.png",
        "plots/paths_and_residuals_stride_0002.png",
    ]
    assert payload["density_sweep"]["1"]["methods"]["lerp"]["intermediate_frames"][
        "relative_l2"
    ]["mean"] is None
    header = output_csv.read_text().splitlines()[0]
    assert "endpoint_plane_relative_l2" in header
    assert "tangent_cosine_similarity" in header
