from __future__ import annotations

import torch

from flow_interpolation.data import CadenceInfo, SequenceData
from flow_interpolation.evaluation.experiments.trajectory import run_trajectory_analysis
from flow_interpolation.utils.flow import FlowSettings
from flow_interpolation.utils.interpolation import interpolation_segment
from flow_interpolation.utils.trajectory import (
    analyze_trajectory_at_stride,
    closest_point_error_decomposition,
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
    torch.testing.assert_close(
        analysis["method_metrics"]["lerp"]["timing_error_absolute"],
        torch.zeros(7),
    )
    torch.testing.assert_close(
        analysis["method_metrics"]["lerp"]["closest_point_orthogonal_rmse"],
        torch.zeros(7),
    )


def test_closest_point_decomposition_separates_lerp_time_warp_from_geometry() -> None:
    keyframes = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    nominal = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    true_alpha = nominal.square()
    reference = torch.stack([10.0 * true_alpha, torch.zeros_like(true_alpha)], dim=1)
    result = closest_point_error_decomposition(
        reference,
        keyframes,
        segment_indices=torch.zeros(5, dtype=torch.long),
        interpolation_coordinate=nominal,
        method="lerp",
        slerp_mode="iscs",
        keyframe_stride=4,
        sample_spacing=0.1,
    )

    torch.testing.assert_close(result["closest_point_alpha"], true_alpha)
    torch.testing.assert_close(
        result["timing_error_absolute"],
        (true_alpha - nominal).abs(),
    )
    torch.testing.assert_close(
        result["closest_point_orthogonal_l2"],
        torch.zeros(5),
    )


def test_closest_point_decomposition_detects_orthogonal_lerp_error() -> None:
    keyframes = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    reference = torch.tensor([[0.0, 0.0], [0.5, 2.0], [1.0, 0.0]])
    result = closest_point_error_decomposition(
        reference,
        keyframes,
        segment_indices=torch.zeros(3, dtype=torch.long),
        interpolation_coordinate=torch.tensor([0.0, 0.5, 1.0]),
        method="lerp",
        slerp_mode="iscs",
        keyframe_stride=2,
        sample_spacing=1.0,
    )

    assert result["timing_error_absolute"][1] == 0.0
    assert result["closest_point_orthogonal_l2"][1] == 2.0


def test_slerp_closest_point_recovers_nonuniform_parameterization() -> None:
    keyframes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    nominal = torch.linspace(0.0, 1.0, 5)
    true_alpha = nominal.square()
    reference = interpolation_segment(
        keyframes,
        0,
        true_alpha,
        "slerp",
        slerp_mode="radius-lerp",
    )
    result = closest_point_error_decomposition(
        reference,
        keyframes,
        segment_indices=torch.zeros(5, dtype=torch.long),
        interpolation_coordinate=nominal,
        method="slerp",
        slerp_mode="radius-lerp",
        keyframe_stride=4,
        sample_spacing=0.1,
        coarse_samples=65,
        refinement_steps=20,
    )

    torch.testing.assert_close(
        result["closest_point_alpha"],
        true_alpha,
        atol=2e-4,
        rtol=0.0,
    )
    assert float(result["closest_point_orthogonal_l2"].max()) < 5e-4


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
    assert (tmp_path / "plots" / "timing_geometry_stride_0001.png").is_file()
    assert (tmp_path / "plots" / "timing_geometry_stride_0002.png").is_file()
    assert payload["reference_definition"].startswith("Dense dataset frames")
    assert payload["density_sweep"]["2"]["methods"].keys() == {"lerp", "squad"}
    assert payload["artifacts"]["plots"] == [
        "plots/reference_geometry.png",
        "plots/density_summary.png",
        "plots/paths_and_residuals_stride_0001.png",
        "plots/timing_geometry_stride_0001.png",
        "plots/paths_and_residuals_stride_0002.png",
        "plots/timing_geometry_stride_0002.png",
    ]
    assert payload["density_sweep"]["1"]["methods"]["lerp"]["intermediate_frames"][
        "relative_l2"
    ]["mean"] is None
    header = output_csv.read_text().splitlines()[0]
    assert "endpoint_plane_relative_l2" in header
    assert "tangent_cosine_similarity" in header
    assert "timing_error_absolute" in header
    assert "closest_point_orthogonal_rmse" in header
