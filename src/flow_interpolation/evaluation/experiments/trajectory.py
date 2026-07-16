"""Dense encoded-trajectory analysis against sparse latent interpolators."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import torch

from flow_interpolation.data import SequenceData
from flow_interpolation.utils.flow import (
    FlowSettings,
    decode_in_chunks,
    encode_in_chunks,
    make_boundary_noise,
)
from flow_interpolation.utils.metrics import image_metrics, save_json
from flow_interpolation.utils.trajectory import analyze_trajectory_at_stride, trajectory_geometry
from flow_interpolation.utils.trajectory_visualization import save_trajectory_plots
from flow_interpolation.utils.visualization import make_comparison_video_frames, write_video


def _statistics(values: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float | int | None]:
    values = values.detach().float().cpu()
    if mask is not None:
        values = values[mask.cpu()]
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.numel()),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "median": values.median().item(),
        "p95": torch.quantile(values, 0.95).item(),
        "max": values.max().item(),
    }


def _summarize_metrics(
    metrics: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, dict[str, float | int | None]]:
    return {name: _statistics(values, mask) for name, values in metrics.items()}


def _reference_geometry_summary(
    geometry: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | int | None]]:
    frame_count = geometry["radius"].shape[0]
    all_frames = torch.ones(frame_count, dtype=torch.bool)
    steps = all_frames.clone()
    steps[0] = False
    interior = all_frames.clone()
    interior[[0, -1]] = False
    masks = {
        "radius": all_frames,
        "radial_speed": steps,
        "angular_speed_degrees": steps,
        "step_distance_rms": steps,
        "radial_step_fraction": steps,
        "angular_step_fraction": steps,
        "speed_l2": all_frames,
        "speed_rms": all_frames,
        "acceleration_l2": interior,
        "acceleration_rms": interior,
        "curvature": interior,
        "turning_angle_degrees": interior,
    }
    return {name: _statistics(geometry[name], mask) for name, mask in masks.items()}


def _coordinate_summary(
    analysis: dict,
    method: str,
    keyframe_stride: int,
) -> list[dict]:
    frame_indices = analysis["frame_indices"]
    segment_indices = analysis["segment_indices"]
    offsets = frame_indices - segment_indices * keyframe_stride
    rows = []
    for offset in range(keyframe_stride + 1):
        mask = offsets == offset
        if not mask.any():
            continue
        rows.append(
            {
                "offset": offset,
                "coordinate": offset / keyframe_stride,
                "metrics": _summarize_metrics(analysis["method_metrics"][method], mask),
            }
        )
    return rows


def _default_keyframe_strides(sequence: SequenceData) -> list[int]:
    base = sequence.cadence.endpoint_stride
    maximum = sequence.num_frames - 1
    return sorted(
        {
            stride
            for stride in (max(1, base // 2), base, base * 2)
            if 1 <= stride <= maximum
        }
    )


def _write_per_frame_csv(
    path: str | Path,
    analyses: dict[int, dict],
    methods: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry_names = [
        "radius",
        "radial_speed",
        "angular_speed_degrees",
        "step_distance_rms",
        "radial_step_fraction",
        "angular_step_fraction",
        "speed_l2",
        "speed_rms",
        "acceleration_l2",
        "acceleration_rms",
        "curvature",
        "turning_angle_degrees",
    ]
    residual_names = ["rmse", "relative_l2", "energy_fraction"]
    metric_names = list(next(iter(analyses.values()))["method_metrics"][methods[0]].keys())
    fieldnames = [
        "keyframe_stride",
        "frame_index",
        "segment_index",
        "interpolation_coordinate",
        "is_keyframe",
        "method",
        *[f"reference_{name}" for name in geometry_names],
        *[f"endpoint_plane_{name}" for name in residual_names],
        *[f"local_four_keyframe_subspace_{name}" for name in residual_names],
        *metric_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stride, analysis in analyses.items():
            for method in methods:
                for index in range(analysis["reference"].shape[0]):
                    writer.writerow(
                        {
                            "keyframe_stride": stride,
                            "frame_index": int(analysis["frame_indices"][index]),
                            "segment_index": int(analysis["segment_indices"][index]),
                            "interpolation_coordinate": float(
                                analysis["interpolation_coordinate"][index]
                            ),
                            "is_keyframe": bool(analysis["observed_mask"][index]),
                            "method": method,
                            **{
                                f"reference_{name}": float(
                                    analysis["reference_geometry"][name][index]
                                )
                                for name in geometry_names
                            },
                            **{
                                f"endpoint_plane_{name}": float(
                                    analysis["endpoint_plane_residual"][name][index]
                                )
                                for name in residual_names
                            },
                            **{
                                f"local_four_keyframe_subspace_{name}": float(
                                    analysis["local_four_keyframe_subspace_residual"][name][index]
                                )
                                for name in residual_names
                            },
                            **{
                                name: float(values[index])
                                for name, values in analysis["method_metrics"][method].items()
                            },
                        }
                    )
    print(f"Saved per-frame trajectory diagnostics to {path}")


def _image_metric_splits(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
) -> dict[str, dict[str, float] | None]:
    missing_mask = ~observed_mask
    return {
        "all_frames": image_metrics(prediction, target),
        "intermediate_frames": (
            image_metrics(prediction[missing_mask], target[missing_mask])
            if missing_mask.any()
            else None
        ),
        "keyframes": image_metrics(prediction[observed_mask], target[observed_mask]),
    }


def _decode_and_render_paths(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    reference_latents: torch.Tensor,
    analyses: dict[int, dict],
    density_sweep: dict[str, dict],
    methods: list[str],
    output_dir: Path,
    video_fps: float,
    display_scale: int,
    gap: int,
    residual_scale: float,
) -> list[Path]:
    decoded_reference = decode_in_chunks(
        model,
        reference_latents,
        flow,
        device,
        desc="Decoding dense reference trajectory",
    ).clamp(0.0, 1.0)
    video_paths = []
    for stride, analysis in analyses.items():
        frame_count = analysis["reference"].shape[0]
        target = sequence.frames[:frame_count]
        observed_mask = analysis["observed_mask"]
        decoded_predictions: OrderedDict[str, torch.Tensor] = OrderedDict()
        decoded_metrics = {
            "dense_encoded_roundtrip": _image_metric_splits(
                decoded_reference[:frame_count], target, observed_mask
            )
        }
        for method in methods:
            prediction = decode_in_chunks(
                model,
                analysis["predictions"][method],
                flow,
                device,
                desc=f"Decoding {method} trajectory at stride {stride}",
            ).clamp(0.0, 1.0)
            decoded_predictions[method] = prediction
            decoded_metrics[method] = _image_metric_splits(prediction, target, observed_mask)

        video_frames = make_comparison_video_frames(
            target,
            decoded_reference[:frame_count],
            decoded_predictions,
            residual_scale=residual_scale,
            display_scale=display_scale,
            gap=gap,
        )
        video_path = output_dir / f"decoded_paths_stride_{stride:04d}.mp4"
        write_video(video_frames, str(video_path), fps=video_fps)
        video_paths.append(video_path)
        density_sweep[str(stride)]["decoded_image_metrics"] = decoded_metrics
    return video_paths


@torch.no_grad()
def run_trajectory_analysis(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    methods: list[str],
    slerp_mode: str,
    boundary_noise_mode: str,
    seed: int,
    output_json: str,
    output_csv: str,
    output_tensors: str | None = None,
    keyframe_strides: list[int] | None = None,
    closest_point_samples: int = 129,
    closest_point_refinement_steps: int = 24,
    plot_paths: bool = True,
    decode_paths: bool = False,
    video_fps: float = 10.0,
    display_scale: int = 8,
    gap: int = 2,
    residual_scale: float = 4.0,
) -> dict:
    """Use the dense encoded sequence as an empirical oracle latent trajectory."""
    strides = sorted(set(keyframe_strides or _default_keyframe_strides(sequence)))
    invalid = [stride for stride in strides if stride <= 0 or stride >= sequence.num_frames]
    if invalid:
        raise ValueError(
            f"Keyframe strides must be in [1, {sequence.num_frames - 1}]; got {invalid}"
        )

    generator = torch.Generator(device=device).manual_seed(seed)
    frames_device = sequence.frames.to(device)
    eps_noise = make_boundary_noise(frames_device, boundary_noise_mode, generator=generator)
    reference_latents = encode_in_chunks(
        model,
        sequence.frames,
        flow,
        device,
        eps_noise=eps_noise.cpu(),
        perturb=True,
        desc="Encoding dense reference trajectory",
    )
    analyses = {
        stride: analyze_trajectory_at_stride(
            reference_latents,
            keyframe_stride=stride,
            sample_spacing=sequence.cadence.high_frame_dt,
            methods=methods,
            slerp_mode=slerp_mode,
            closest_point_samples=closest_point_samples,
            closest_point_refinement_steps=closest_point_refinement_steps,
        )
        for stride in strides
    }

    density_sweep = {}
    for stride, analysis in analyses.items():
        missing = ~analysis["observed_mask"]
        method_summary = {}
        for method in methods:
            metrics = analysis["method_metrics"][method]
            method_summary[method] = {
                "all_frames": _summarize_metrics(metrics, torch.ones_like(missing)),
                "intermediate_frames": _summarize_metrics(metrics, missing),
                "by_interpolation_coordinate": _coordinate_summary(analysis, method, stride),
            }
        density_sweep[str(stride)] = {
            "keyframe_stride": stride,
            "keyframe_spacing": stride * sequence.cadence.high_frame_dt,
            "analyzed_frame_count": int(analysis["reference"].shape[0]),
            "ignored_tail_frames": analysis["ignored_tail_frames"],
            "keyframe_indices": analysis["keyframe_indices"],
            "reference_geometry": _reference_geometry_summary(analysis["reference_geometry"]),
            "endpoint_plane_residual": _summarize_metrics(
                analysis["endpoint_plane_residual"], missing
            ),
            "local_four_keyframe_subspace_residual": _summarize_metrics(
                analysis["local_four_keyframe_subspace_residual"], missing
            ),
            "methods": method_summary,
        }
        print(
            f"Trajectory stride {stride}: "
            f"endpoint-plane residual={density_sweep[str(stride)]['endpoint_plane_residual']['relative_l2']['mean']}, "
            f"method relative L2="
            f"{ {method: method_summary[method]['intermediate_frames']['relative_l2']['mean'] for method in methods} }"
        )

    full_geometry = trajectory_geometry(reference_latents, sequence.cadence.high_frame_dt)
    output_root = Path(output_json).parent
    artifact_paths: dict[str, list[str]] = {"plots": [], "decoded_videos": []}
    if plot_paths:
        paths = save_trajectory_plots(
            analyses,
            density_sweep,
            full_geometry,
            methods=methods,
            sample_spacing=sequence.cadence.high_frame_dt,
            output_dir=output_root / "plots",
        )
        artifact_paths["plots"] = [str(path.relative_to(output_root)) for path in paths]
    if decode_paths:
        paths = _decode_and_render_paths(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            reference_latents=reference_latents,
            analyses=analyses,
            density_sweep=density_sweep,
            methods=methods,
            output_dir=output_root / "videos",
            video_fps=video_fps,
            display_scale=display_scale,
            gap=gap,
            residual_scale=residual_scale,
        )
        artifact_paths["decoded_videos"] = [
            str(path.relative_to(output_root)) for path in paths
        ]
    payload = {
        "reference_definition": (
            "Dense dataset frames encoded through the deterministic flow. This is an empirical "
            "oracle for the observed process, not a globally optimal path in noise space."
        ),
        "methods": methods,
        "slerp_mode": slerp_mode,
        "boundary_noise_mode": boundary_noise_mode,
        "boundary_noise_note": (
            "Shared boundary noise preserves temporal comparability; independent noise adds "
            "framewise perturbations to the measured trajectory."
        ),
        "closest_point_definition": (
            "For each dense reference frame, minimize distance to the candidate "
            "interpolation curve within the surrounding keyframe segment. LERP uses "
            "exact orthogonal projection; SLERP and SQUAD use coarse bracketing and "
            "bounded one-dimensional refinement."
        ),
        "closest_point_settings": {
            "coarse_samples": closest_point_samples,
            "refinement_steps": closest_point_refinement_steps,
        },
        "dense_frame_count": sequence.num_frames,
        "dense_frame_spacing": sequence.cadence.high_frame_dt,
        "cadence": sequence.cadence,
        "full_reference_geometry": _reference_geometry_summary(full_geometry),
        "density_sweep": density_sweep,
        "artifacts": artifact_paths,
        "decoded_video_panel_order": (
            [
                "ground_truth",
                "decoded_dense_reference",
                *[
                    item
                    for method in methods
                    for item in (method, f"{method}_absolute_residual")
                ],
            ]
            if decode_paths
            else None
        ),
    }
    save_json(payload, output_json)
    _write_per_frame_csv(output_csv, analyses, methods)

    if output_tensors is not None:
        tensor_path = Path(output_tensors)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "reference_latents": reference_latents,
                "boundary_noise": eps_noise.cpu(),
                "analyses": analyses,
            },
            tensor_path,
        )
        print(f"Saved trajectory analysis tensors to {tensor_path}")
    return payload


run_latent_geodesic_evaluation = run_trajectory_analysis
