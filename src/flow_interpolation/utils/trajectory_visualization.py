"""Static plots for dense latent-trajectory diagnostics."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


def _pyplot():
    # Matplotlib tries to create a user cache at import time, which is often not
    # writable on compute nodes. Keep its cache in the process-local temp area.
    cache = Path(tempfile.gettempdir()) / "flow_interpolation_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _as_numpy(values: torch.Tensor):
    return values.detach().float().cpu().numpy()


def _project_paths(analysis: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Project reference and predicted paths onto a shared two-dimensional PCA basis."""
    named_paths = {"dense reference": analysis["reference"], **analysis["predictions"]}
    flattened = {
        name: values.detach().float().cpu().flatten(start_dim=1)
        for name, values in named_paths.items()
    }
    combined = torch.cat(list(flattened.values()), dim=0)
    center = combined.mean(dim=0, keepdim=True)
    centered = combined - center
    _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    component_count = min(2, right_vectors.shape[0])
    components = right_vectors[:component_count].T
    projected = {
        name: (values - center) @ components
        for name, values in flattened.items()
    }
    if component_count < 2:
        projected = {
            name: torch.nn.functional.pad(values, (0, 2 - component_count))
            for name, values in projected.items()
        }
    explained = singular_values.square()
    explained = explained[:2] / explained.sum().clamp_min(1e-12)
    explained = torch.nn.functional.pad(explained, (0, 2 - explained.numel()))
    return projected, explained


def save_stride_diagnostic_plot(
    analysis: dict,
    *,
    keyframe_stride: int,
    sample_spacing: float,
    output_path: str | Path,
) -> Path:
    """Plot projected paths and full-dimensional residual metrics for one stride."""
    plt = _pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = analysis["frame_indices"].float() * sample_spacing
    time_values = _as_numpy(time)
    projected, explained = _project_paths(analysis)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    path_axis, error_axis, residual_axis, alignment_axis = axes.flatten()

    for name, values in projected.items():
        style = {"color": "black", "linewidth": 2.2} if name == "dense reference" else {}
        path_axis.plot(_as_numpy(values[:, 0]), _as_numpy(values[:, 1]), label=name, **style)
    reference_projection = projected["dense reference"]
    keyframe_indices = analysis["keyframe_indices"]
    path_axis.scatter(
        _as_numpy(reference_projection[keyframe_indices, 0]),
        _as_numpy(reference_projection[keyframe_indices, 1]),
        marker="o",
        s=34,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="keyframes",
    )
    path_axis.set_title("Latent paths (shared PCA projection)")
    path_axis.set_xlabel(f"PC1 ({100.0 * float(explained[0]):.1f}% variance)")
    path_axis.set_ylabel(f"PC2 ({100.0 * float(explained[1]):.1f}% variance)")
    path_axis.legend(fontsize=8)

    for method, metrics in analysis["method_metrics"].items():
        error_axis.plot(time_values, _as_numpy(metrics["relative_l2"]), label=method)
    error_axis.set_title("Path error in full latent space")
    error_axis.set_xlabel("Sequence time")
    error_axis.set_ylabel("Relative L2")
    error_axis.legend(fontsize=8)

    residual_axis.plot(
        time_values,
        _as_numpy(analysis["endpoint_plane_residual"]["relative_l2"]),
        label="outside endpoint plane",
    )
    residual_axis.plot(
        time_values,
        _as_numpy(analysis["local_four_keyframe_subspace_residual"]["relative_l2"]),
        label="outside local 4-keyframe span",
    )
    residual_axis.set_title("Reference subspace residuals")
    residual_axis.set_xlabel("Sequence time")
    residual_axis.set_ylabel("Relative L2")
    residual_axis.legend(fontsize=8)

    for method, metrics in analysis["method_metrics"].items():
        alignment_axis.plot(
            time_values,
            _as_numpy(metrics["tangent_angle_degrees"]),
            label=f"{method}: tangent angle",
        )
    alignment_axis.set_title("Local trajectory alignment")
    alignment_axis.set_xlabel("Sequence time")
    alignment_axis.set_ylabel("Angle (degrees)")
    alignment_axis.legend(fontsize=8)

    for axis in axes.flatten():
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"Trajectory diagnostic: stride {keyframe_stride} "
        f"({keyframe_stride * sample_spacing:g} time units)",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_reference_geometry_plot(
    geometry: dict[str, torch.Tensor],
    *,
    sample_spacing: float,
    output_path: str | Path,
) -> Path:
    """Plot radial, angular, and differential geometry of the dense reference path."""
    plt = _pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = torch.arange(geometry["radius"].shape[0]).float() * sample_spacing
    time_values = _as_numpy(time)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    radius_axis, fraction_axis, derivative_axis, turning_axis = axes.flatten()
    radius_axis.plot(time_values, _as_numpy(geometry["radius"]), color="black")
    radius_axis.set_title("Latent radius")
    radius_axis.set_ylabel("L2 norm")

    fraction_axis.plot(
        time_values, _as_numpy(geometry["radial_step_fraction"]), label="radial"
    )
    fraction_axis.plot(
        time_values, _as_numpy(geometry["angular_step_fraction"]), label="angular"
    )
    fraction_axis.set_title("Step decomposition")
    fraction_axis.set_ylabel("Fraction")
    fraction_axis.set_ylim(-0.02, 1.02)
    fraction_axis.legend(fontsize=8)

    derivative_axis.plot(
        time_values, _as_numpy(geometry["speed_rms"]), label="speed RMS"
    )
    acceleration_axis = derivative_axis.twinx()
    acceleration_axis.plot(
        time_values,
        _as_numpy(geometry["acceleration_rms"]),
        color="tab:red",
        alpha=0.75,
        label="acceleration RMS",
    )
    derivative_axis.set_title("Temporal derivatives")
    derivative_axis.set_ylabel("Speed RMS")
    acceleration_axis.set_ylabel("Acceleration RMS", color="tab:red")
    lines = derivative_axis.lines + acceleration_axis.lines
    derivative_axis.legend(lines, [line.get_label() for line in lines], fontsize=8)

    turning_axis.plot(
        time_values,
        _as_numpy(geometry["turning_angle_degrees"]),
        label="turning angle (degrees)",
    )
    curvature_axis = turning_axis.twinx()
    curvature_axis.plot(
        time_values,
        _as_numpy(geometry["curvature"]),
        color="tab:red",
        alpha=0.75,
        label="curvature",
    )
    turning_axis.set_title("Direction change")
    turning_axis.set_ylabel("Turning angle (degrees)")
    curvature_axis.set_ylabel("Curvature", color="tab:red")
    lines = turning_axis.lines + curvature_axis.lines
    turning_axis.legend(lines, [line.get_label() for line in lines], fontsize=8)

    for axis in axes.flatten():
        axis.set_xlabel("Sequence time")
        axis.grid(alpha=0.25)
    figure.suptitle("Dense encoded-reference geometry", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _summary_means(
    density_sweep: dict[str, dict],
    methods: list[str],
    metric: str,
) -> tuple[list[float], dict[str, list[float]]]:
    ordered = sorted(density_sweep.values(), key=lambda item: item["keyframe_spacing"])
    spacing = [float(item["keyframe_spacing"]) for item in ordered]
    values = {
        method: [
            _float_or_nan(item["methods"][method]["intermediate_frames"][metric]["mean"])
            for item in ordered
        ]
        for method in methods
    }
    return spacing, values


def _float_or_nan(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)


def save_density_summary_plot(
    density_sweep: dict[str, dict],
    *,
    methods: list[str],
    output_path: str | Path,
) -> Path:
    """Plot key metrics as a function of sparse-keyframe spacing."""
    plt = _pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(density_sweep.values(), key=lambda item: item["keyframe_spacing"])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specifications = [
        ("relative_l2", "Interpolation error", "Mean relative L2"),
        ("tangent_angle_degrees", "Tangent mismatch", "Mean angle (degrees)"),
        ("speed_relative_error", "Speed mismatch", "Mean relative error"),
    ]
    for axis, (metric, title, ylabel) in zip(axes.flatten()[:3], specifications):
        spacing, method_values = _summary_means(density_sweep, methods, metric)
        for method, values in method_values.items():
            axis.plot(spacing, values, marker="o", label=method)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)

    residual_axis = axes.flatten()[3]
    spacing = [float(item["keyframe_spacing"]) for item in ordered]
    residual_axis.plot(
        spacing,
        [
            _float_or_nan(item["endpoint_plane_residual"]["relative_l2"]["mean"])
            for item in ordered
        ],
        marker="o",
        label="endpoint plane",
    )
    residual_axis.plot(
        spacing,
        [
            _float_or_nan(
                item["local_four_keyframe_subspace_residual"]["relative_l2"]["mean"]
            )
            for item in ordered
        ],
        marker="o",
        label="local 4-keyframe span",
    )
    residual_axis.set_title("Reference subspace residual")
    residual_axis.set_ylabel("Mean relative L2")
    residual_axis.legend(fontsize=8)

    for axis in axes.flatten():
        axis.set_xlabel("Keyframe spacing")
        axis.grid(alpha=0.25)
    figure.suptitle("Trajectory metrics across keyframe densities", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_trajectory_plots(
    analyses: dict[int, dict],
    density_sweep: dict[str, dict],
    reference_geometry: dict[str, torch.Tensor],
    *,
    methods: list[str],
    sample_spacing: float,
    output_dir: str | Path,
) -> list[Path]:
    """Write all standard trajectory-analysis figures and return their paths."""
    output_dir = Path(output_dir)
    paths = [
        save_reference_geometry_plot(
            reference_geometry,
            sample_spacing=sample_spacing,
            output_path=output_dir / "reference_geometry.png",
        ),
        save_density_summary_plot(
            density_sweep,
            methods=methods,
            output_path=output_dir / "density_summary.png",
        ),
    ]
    paths.extend(
        save_stride_diagnostic_plot(
            analysis,
            keyframe_stride=stride,
            sample_spacing=sample_spacing,
            output_path=output_dir / f"paths_and_residuals_stride_{stride:04d}.png",
        )
        for stride, analysis in analyses.items()
    )
    for path in paths:
        print(f"Saved trajectory plot to {path}")
    return paths
