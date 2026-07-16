"""Geometry diagnostics for densely sampled latent trajectories."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flow_interpolation.utils.interpolation import (
    interpolate_keyframes,
    interpolation_segment,
)


def flatten_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.ndim < 2:
        raise ValueError("A trajectory needs a frame axis and at least one feature axis")
    if trajectory.shape[0] < 2:
        raise ValueError("A trajectory needs at least two frames")
    return trajectory.float().flatten(start_dim=1)


def trajectory_geometry(
    trajectory: torch.Tensor,
    sample_spacing: float,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compute time-parameterized radial, angular, and differential geometry."""
    if sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive")
    values = flatten_trajectory(trajectory)
    frame_count, dimension = values.shape
    radii = values.norm(dim=1)
    directions = values / radii[:, None].clamp_min(eps)

    tangent = torch.zeros_like(values)
    tangent[0] = (values[1] - values[0]) / sample_spacing
    tangent[-1] = (values[-1] - values[-2]) / sample_spacing
    if frame_count > 2:
        tangent[1:-1] = (values[2:] - values[:-2]) / (2.0 * sample_spacing)

    acceleration = torch.zeros_like(values)
    if frame_count > 2:
        acceleration[1:-1] = (
            values[2:] - 2.0 * values[1:-1] + values[:-2]
        ) / (sample_spacing**2)

    speed_l2 = tangent.norm(dim=1)
    speed_rms = tangent.square().mean(dim=1).sqrt()
    acceleration_l2 = acceleration.norm(dim=1)
    acceleration_rms = acceleration.square().mean(dim=1).sqrt()

    radial_speed = torch.zeros(frame_count, dtype=values.dtype, device=values.device)
    angular_speed_degrees = torch.zeros_like(radial_speed)
    step_distance_rms = torch.zeros_like(radial_speed)
    radial_step_fraction = torch.zeros_like(radial_speed)
    angular_step_fraction = torch.zeros_like(radial_speed)
    step_radii = radii[1:] - radii[:-1]
    step_angles = torch.acos((directions[1:] * directions[:-1]).sum(dim=1).clamp(-1.0, 1.0))
    angular_arc = 0.5 * (radii[1:] + radii[:-1]) * step_angles
    radial_angular_norm = torch.sqrt(step_radii.square() + angular_arc.square()).clamp_min(eps)
    radial_speed[1:] = step_radii / sample_spacing
    angular_speed_degrees[1:] = torch.rad2deg(step_angles) / sample_spacing
    step_distance_rms[1:] = (values[1:] - values[:-1]).square().mean(dim=1).sqrt()
    radial_step_fraction[1:] = step_radii.abs() / radial_angular_norm
    angular_step_fraction[1:] = angular_arc.abs() / radial_angular_norm

    curvature = torch.zeros_like(radial_speed)
    turning_angle_degrees = torch.zeros_like(radial_speed)
    if frame_count > 2:
        interior_velocity = tangent[1:-1]
        interior_acceleration = acceleration[1:-1]
        speed_squared = interior_velocity.square().sum(dim=1).clamp_min(eps)
        parallel_scale = (
            (interior_acceleration * interior_velocity).sum(dim=1) / speed_squared
        )
        normal_acceleration = interior_acceleration - parallel_scale[:, None] * interior_velocity
        curvature[1:-1] = normal_acceleration.norm(dim=1) / speed_squared

        incoming = values[1:-1] - values[:-2]
        outgoing = values[2:] - values[1:-1]
        turning_cosine = F.cosine_similarity(incoming, outgoing, dim=1).clamp(-1.0, 1.0)
        turning_angle_degrees[1:-1] = torch.rad2deg(torch.acos(turning_cosine))

    return {
        "radius": radii,
        "radial_speed": radial_speed,
        "angular_speed_degrees": angular_speed_degrees,
        "step_distance_rms": step_distance_rms,
        "radial_step_fraction": radial_step_fraction,
        "angular_step_fraction": angular_step_fraction,
        "speed_l2": speed_l2,
        "speed_rms": speed_rms,
        "acceleration_l2": acceleration_l2,
        "acceleration_rms": acceleration_rms,
        "curvature": curvature,
        "turning_angle_degrees": turning_angle_degrees,
        "tangent": tangent,
    }


def latent_error_metrics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    prediction_flat = flatten_trajectory(prediction)
    reference_flat = flatten_trajectory(reference)
    if prediction_flat.shape != reference_flat.shape:
        raise ValueError(
            f"Trajectory shapes differ: {prediction_flat.shape} versus {reference_flat.shape}"
        )
    difference = prediction_flat - reference_flat
    reference_norm = reference_flat.norm(dim=1).clamp_min(eps)
    prediction_norm = prediction_flat.norm(dim=1).clamp_min(eps)
    cosine = F.cosine_similarity(prediction_flat, reference_flat, dim=1).clamp(-1.0, 1.0)
    return {
        "relative_l2": difference.norm(dim=1) / reference_norm,
        "rmse": difference.square().mean(dim=1).sqrt(),
        "cosine_similarity": cosine,
        "angle_degrees": torch.rad2deg(torch.acos(cosine)),
        "radius_relative_error": (prediction_norm - reference_norm).abs() / reference_norm,
    }


def _project_onto_span(
    targets: torch.Tensor,
    basis_vectors: torch.Tensor,
    *,
    eps: float,
) -> dict[str, torch.Tensor]:
    targets_flat = targets.float().flatten(start_dim=1)
    basis = basis_vectors.float().flatten(start_dim=1).T
    coefficients = torch.linalg.lstsq(basis, targets_flat.T).solution
    projection = (basis @ coefficients).T
    residual = targets_flat - projection
    target_energy = targets_flat.square().sum(dim=1).clamp_min(eps)
    residual_energy = residual.square().sum(dim=1)
    return {
        "rmse": residual.square().mean(dim=1).sqrt(),
        "relative_l2": residual.norm(dim=1) / target_energy.sqrt(),
        "energy_fraction": residual_energy / target_energy,
    }


def subspace_residuals(
    reference: torch.Tensor,
    keyframe_indices: torch.Tensor,
    *,
    local_keyframe_count: int,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Measure reference-path energy outside local keyframe linear spans.

    ``local_keyframe_count=2`` gives the endpoint plane. A value of four gives the
    local neighborhood available to a four-keyframe spline such as SQUAD.
    """
    if local_keyframe_count < 2:
        raise ValueError("local_keyframe_count must be at least two")
    indices = keyframe_indices.to(dtype=torch.long, device="cpu")
    if indices.ndim != 1 or indices.numel() < 2:
        raise ValueError("At least two keyframe indices are required")
    if int(indices[0]) != 0 or int(indices[-1]) >= reference.shape[0]:
        raise ValueError("Keyframe indices must start at zero and lie inside the trajectory")

    output = {
        name: torch.empty(int(indices[-1]) + 1, dtype=torch.float32)
        for name in ("rmse", "relative_l2", "energy_fraction")
    }
    keyframes = reference[indices]
    for segment_index in range(indices.numel() - 1):
        start = int(indices[segment_index])
        end = int(indices[segment_index + 1])
        half_width = local_keyframe_count // 2
        first_key = max(0, segment_index - (half_width - 1))
        last_key = min(indices.numel(), first_key + local_keyframe_count)
        first_key = max(0, last_key - local_keyframe_count)
        basis = keyframes[first_key:last_key]
        first_frame = start if segment_index == 0 else start + 1
        projected = _project_onto_span(reference[first_frame : end + 1], basis, eps=eps)
        for name, values in projected.items():
            output[name][first_frame : end + 1] = values.cpu()
    return output


def _closest_lerp_points(
    targets: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets_flat = targets.float().flatten(start_dim=1)
    left_flat = left.float().flatten()
    direction = right.float().flatten() - left_flat
    denominator = direction.square().sum()
    if float(denominator) <= eps:
        alpha = torch.zeros(targets.shape[0], dtype=targets_flat.dtype)
    else:
        alpha = ((targets_flat - left_flat) * direction).sum(dim=1) / denominator
        alpha = alpha.clamp(0.0, 1.0)
    closest = left_flat + alpha[:, None] * direction
    return alpha, closest.reshape(targets.shape)


def _closest_curved_points(
    targets: torch.Tensor,
    keyframes: torch.Tensor,
    segment_index: int,
    method: str,
    *,
    slerp_mode: str,
    coarse_samples: int,
    refinement_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coarse_samples < 3:
        raise ValueError("coarse_samples must be at least three")
    if refinement_steps < 0:
        raise ValueError("refinement_steps must be non-negative")
    targets_flat = targets.float().flatten(start_dim=1)
    grid = torch.linspace(0.0, 1.0, coarse_samples, dtype=keyframes.dtype)
    coarse_curve = interpolation_segment(
        keyframes,
        segment_index,
        grid,
        method,
        slerp_mode=slerp_mode,
    ).float().flatten(start_dim=1)
    distances = torch.cdist(targets_flat, coarse_curve).square()
    best_indices = distances.argmin(dim=1)
    best_alpha = grid[best_indices].float()
    best_points = coarse_curve[best_indices]
    best_distance = distances.gather(1, best_indices[:, None]).squeeze(1)

    step = 1.0 / (coarse_samples - 1)
    lower = (best_alpha - step).clamp_min(0.0)
    upper = (best_alpha + step).clamp_max(1.0)
    lower = torch.where(best_indices == 0, best_alpha, lower)
    upper = torch.where(best_indices == coarse_samples - 1, best_alpha, upper)
    inverse_phi = (5.0**0.5 - 1.0) / 2.0

    for _ in range(refinement_steps):
        left_alpha = upper - inverse_phi * (upper - lower)
        right_alpha = lower + inverse_phi * (upper - lower)
        left_points = interpolation_segment(
            keyframes,
            segment_index,
            left_alpha,
            method,
            slerp_mode=slerp_mode,
        ).float().flatten(start_dim=1)
        right_points = interpolation_segment(
            keyframes,
            segment_index,
            right_alpha,
            method,
            slerp_mode=slerp_mode,
        ).float().flatten(start_dim=1)
        left_distance = (left_points - targets_flat).square().sum(dim=1)
        right_distance = (right_points - targets_flat).square().sum(dim=1)
        choose_left = left_distance <= right_distance
        upper = torch.where(choose_left, right_alpha, upper)
        lower = torch.where(choose_left, lower, left_alpha)

    refined_alpha = 0.5 * (lower + upper)
    refined_points = interpolation_segment(
        keyframes,
        segment_index,
        refined_alpha,
        method,
        slerp_mode=slerp_mode,
    ).float().flatten(start_dim=1)
    refined_distance = (refined_points - targets_flat).square().sum(dim=1)
    use_refined = refined_distance < best_distance
    alpha = torch.where(use_refined, refined_alpha, best_alpha)
    points = torch.where(use_refined[:, None], refined_points, best_points)
    return alpha, points.reshape(targets.shape)


def closest_point_error_decomposition(
    reference: torch.Tensor,
    keyframes: torch.Tensor,
    segment_indices: torch.Tensor,
    interpolation_coordinate: torch.Tensor,
    *,
    method: str,
    slerp_mode: str,
    keyframe_stride: int,
    sample_spacing: float,
    coarse_samples: int = 129,
    refinement_steps: int = 24,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Separate same-time path error into timing and closest-curve residuals."""
    frame_count = reference.shape[0]
    closest_alpha = torch.empty(frame_count, dtype=torch.float32)
    closest_points = torch.empty_like(reference, dtype=torch.float32)

    for segment_index in range(keyframes.shape[0] - 1):
        frame_mask = segment_indices == segment_index
        targets = reference[frame_mask]
        if method == "lerp":
            alpha, points = _closest_lerp_points(
                targets,
                keyframes[segment_index],
                keyframes[segment_index + 1],
                eps=eps,
            )
        else:
            alpha, points = _closest_curved_points(
                targets,
                keyframes,
                segment_index,
                method,
                slerp_mode=slerp_mode,
                coarse_samples=coarse_samples,
                refinement_steps=refinement_steps,
            )
        closest_alpha[frame_mask] = alpha.cpu()
        closest_points[frame_mask] = points.cpu()

    reference_flat = reference.float().flatten(start_dim=1)
    closest_flat = closest_points.flatten(start_dim=1)
    residual = reference_flat - closest_flat
    residual_l2 = residual.norm(dim=1)
    reference_l2 = reference_flat.norm(dim=1).clamp_min(eps)
    signed_timing_error = closest_alpha - interpolation_coordinate.float()
    return {
        "closest_point_alpha": closest_alpha,
        "signed_timing_error": signed_timing_error,
        "timing_error_absolute": signed_timing_error.abs(),
        "timing_error_frames": signed_timing_error.abs() * keyframe_stride,
        "timing_error_time": (
            signed_timing_error.abs() * keyframe_stride * sample_spacing
        ),
        "closest_point_orthogonal_l2": residual_l2,
        "closest_point_orthogonal_rmse": residual.square().mean(dim=1).sqrt(),
        "closest_point_orthogonal_relative_l2": residual_l2 / reference_l2,
    }


def analyze_trajectory_at_stride(
    reference: torch.Tensor,
    *,
    keyframe_stride: int,
    sample_spacing: float,
    methods: list[str],
    slerp_mode: str,
    closest_point_samples: int = 129,
    closest_point_refinement_steps: int = 24,
) -> dict:
    """Compare sparse-keyframe paths with a dense encoded reference trajectory."""
    if keyframe_stride <= 0:
        raise ValueError("keyframe_stride must be positive")
    original_frame_count = reference.shape[0]
    last_frame = ((reference.shape[0] - 1) // keyframe_stride) * keyframe_stride
    if last_frame < keyframe_stride:
        raise ValueError(
            f"keyframe_stride={keyframe_stride} is too large for {reference.shape[0]} frames"
        )
    reference = reference[: last_frame + 1].cpu()
    keyframe_indices = torch.arange(0, last_frame + 1, keyframe_stride, dtype=torch.long)
    keyframes = reference[keyframe_indices]
    reference_geometry = trajectory_geometry(reference, sample_spacing)
    endpoint_plane = subspace_residuals(
        reference,
        keyframe_indices,
        local_keyframe_count=2,
    )
    local_four_keyframe = subspace_residuals(
        reference,
        keyframe_indices,
        local_keyframe_count=4,
    )

    frame_indices = torch.arange(last_frame + 1, dtype=torch.long)
    segment_indices = torch.div(
        (frame_indices - 1).clamp_min(0),
        keyframe_stride,
        rounding_mode="floor",
    ).clamp_max(keyframe_indices.numel() - 2)
    segment_starts = segment_indices * keyframe_stride
    interpolation_coordinate = (frame_indices - segment_starts).float() / keyframe_stride
    interpolation_coordinate[-1] = 1.0
    observed_mask = torch.zeros(last_frame + 1, dtype=torch.bool)
    observed_mask[keyframe_indices] = True

    predictions = {}
    method_metrics = {}
    for method in methods:
        prediction = interpolate_keyframes(
            keyframes,
            keyframe_stride,
            method,
            slerp_mode=slerp_mode,
        ).cpu()
        geometry = trajectory_geometry(prediction, sample_spacing)
        errors = latent_error_metrics(prediction, reference)
        closest_point = closest_point_error_decomposition(
            reference,
            keyframes,
            segment_indices,
            interpolation_coordinate,
            method=method,
            slerp_mode=slerp_mode,
            keyframe_stride=keyframe_stride,
            sample_spacing=sample_spacing,
            coarse_samples=closest_point_samples,
            refinement_steps=closest_point_refinement_steps,
        )
        tangent_cosine = F.cosine_similarity(
            geometry["tangent"],
            reference_geometry["tangent"],
            dim=1,
        ).clamp(-1.0, 1.0)
        subspace_lower_bound = (
            local_four_keyframe if method == "squad" else endpoint_plane
        )["rmse"]
        method_metrics[method] = {
            **errors,
            **closest_point,
            "same_time_to_closest_rmse_ratio": errors["rmse"]
            / closest_point["closest_point_orthogonal_rmse"].clamp_min(1e-12),
            "closest_rmse_fraction_of_same_time_error": closest_point[
                "closest_point_orthogonal_rmse"
            ]
            / errors["rmse"].clamp_min(1e-12),
            "tangent_cosine_similarity": tangent_cosine,
            "tangent_angle_degrees": torch.rad2deg(torch.acos(tangent_cosine)),
            "speed_relative_error": (
                geometry["speed_l2"] - reference_geometry["speed_l2"]
            ).abs()
            / reference_geometry["speed_l2"].clamp_min(1e-12),
            "subspace_lower_bound_rmse": subspace_lower_bound,
            "rmse_above_subspace_lower_bound": (
                errors["rmse"] - subspace_lower_bound
            ).clamp_min(0.0),
            "rmse_to_subspace_lower_bound_ratio": errors["rmse"]
            / subspace_lower_bound.clamp_min(1e-12),
        }
        predictions[method] = prediction

    return {
        "reference": reference,
        "keyframe_indices": keyframe_indices,
        "keyframes": keyframes,
        "frame_indices": frame_indices,
        "segment_indices": segment_indices,
        "interpolation_coordinate": interpolation_coordinate,
        "observed_mask": observed_mask,
        "reference_geometry": reference_geometry,
        "endpoint_plane_residual": endpoint_plane,
        "local_four_keyframe_subspace_residual": local_four_keyframe,
        "predictions": predictions,
        "method_metrics": method_metrics,
        "closest_point_settings": {
            "coarse_samples": closest_point_samples,
            "refinement_steps": closest_point_refinement_steps,
        },
        "ignored_tail_frames": int(original_frame_count - last_frame - 1),
    }
