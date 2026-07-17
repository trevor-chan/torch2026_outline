"""Test whether temporal disorder in terminal latents is visually dispensable."""

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
from flow_interpolation.utils.nuisance_visualization import (
    save_retention_plot,
    save_temporal_spectrum_plot,
)
from flow_interpolation.utils.temporal_nuisance import analyze_temporal_nuisance
from flow_interpolation.utils.visualization import make_comparison_video_frames, write_video


def _activity_map(images: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    images = images.detach().float().cpu()
    flattened = images.flatten(start_dim=2)
    background = flattened.median(dim=2).values[:, :, None, None]
    activity = (images - background).abs().mean(dim=1)
    if normalize:
        activity = activity / activity.flatten(start_dim=1).amax(dim=1)[
            :, None, None
        ].clamp_min(1e-12)
    return activity


def _activity_centroid(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = _activity_map(images, normalize=False)
    height, width = weights.shape[-2:]
    y = torch.arange(height, dtype=weights.dtype)
    x = torch.arange(width, dtype=weights.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    mass = weights.sum(dim=(-2, -1)).clamp_min(1e-12)
    centroid_x = (weights * grid_x).sum(dim=(-2, -1)) / mass
    centroid_y = (weights * grid_y).sum(dim=(-2, -1)) / mass
    return torch.stack([centroid_x, centroid_y], dim=1), mass


def _decoded_information_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    region_reference: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    weights = _activity_map(region_reference, normalize=True)
    error_energy = (prediction - target).square().mean(dim=1)
    foreground_denominator = weights.sum().clamp_min(1e-12)
    background_weights = 1.0 - weights
    background_denominator = background_weights.sum().clamp_min(1e-12)
    prediction_centroid, prediction_mass = _activity_centroid(prediction)
    target_centroid, target_mass = _activity_centroid(target)
    centroid_error = (prediction_centroid - target_centroid).norm(dim=1)
    if centroid_error.numel() > 1:
        prediction_steps = prediction_centroid[1:] - prediction_centroid[:-1]
        target_steps = target_centroid[1:] - target_centroid[:-1]
        centroid_step_error = (prediction_steps - target_steps).norm(dim=1)
        centroid_step_rmse = float(centroid_step_error.square().mean().sqrt().item())
    else:
        centroid_step_rmse = 0.0
    return {
        **image_metrics(prediction, target),
        "foreground_weighted_rmse": float(
            ((error_energy * weights).sum() / foreground_denominator).sqrt().item()
        ),
        "background_weighted_rmse": float(
            (
                (error_energy * background_weights).sum() / background_denominator
            )
            .sqrt()
            .item()
        ),
        "activity_centroid_error_pixels": float(centroid_error.mean().item()),
        "activity_centroid_error_p95_pixels": float(
            torch.quantile(centroid_error, 0.95).item()
        ),
        "activity_centroid_step_rmse_pixels": centroid_step_rmse,
        "activity_mass_relative_error": float(
            ((prediction_mass - target_mass).abs() / target_mass.clamp_min(1e-12))
            .mean()
            .item()
        ),
    }


def _compact_video_predictions(
    decoded: OrderedDict[str, torch.Tensor],
    maximum_panels: int = 4,
) -> OrderedDict[str, torch.Tensor]:
    if len(decoded) <= maximum_panels:
        return decoded
    indices = torch.linspace(0, len(decoded) - 1, maximum_panels).round().long().tolist()
    items = list(decoded.items())
    return OrderedDict(items[index] for index in sorted(set(indices)))


def _write_sweep_csv(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for transform, sweep in payload["sweeps"].items():
        for item in sweep:
            rows.append(
                {
                    "transform": transform,
                    "parameter": item["parameter"],
                    "label": item["label"],
                    **{f"latent_{key}": value for key, value in item["latent"].items()},
                    **{
                        f"decoded_vs_ground_truth_{key}": value
                        for key, value in item["decoded_vs_ground_truth"].items()
                    },
                    **{
                        f"decoded_vs_dense_reference_{key}": value
                        for key, value in item["decoded_vs_dense_reference"].items()
                    },
                }
            )
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    if not fieldnames:
        fieldnames = ["transform", "parameter", "label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved nuisance-dimension sweep to {path}")


@torch.no_grad()
def run_nuisance_dimension_analysis(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    boundary_noise_mode: str,
    seed: int,
    svd_ranks: list[int],
    fourier_harmonics: list[int],
    output_dir: str,
    plot_results: bool = True,
    write_videos: bool = True,
    video_fps: float = 10.0,
    display_scale: int = 8,
    gap: int = 2,
    residual_scale: float = 4.0,
    save_tensors: bool = False,
) -> dict:
    """Encode, temporally compress, decode, and quantify lost image information."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    frames_device = sequence.frames.to(device)
    boundary_noise = make_boundary_noise(
        frames_device,
        boundary_noise_mode,
        generator=generator,
    )
    reference_latents = encode_in_chunks(
        model,
        sequence.frames,
        flow,
        device,
        eps_noise=boundary_noise.cpu(),
        perturb=True,
        desc="Encoding dense trajectory for nuisance analysis",
    )
    analysis = analyze_temporal_nuisance(
        reference_latents,
        sample_spacing=sequence.cadence.high_frame_dt,
        svd_ranks=svd_ranks,
        fourier_harmonics=fourier_harmonics,
    )
    decoded_reference = decode_in_chunks(
        model,
        reference_latents,
        flow,
        device,
        desc="Decoding dense latent reference",
    ).clamp(0.0, 1.0)

    maximum_rank = analysis["summary"]["maximum_centered_rank"]
    maximum_harmonic = analysis["summary"]["maximum_fourier_harmonic"]
    reconstruction_specs: list[tuple[str, int, str, torch.Tensor]] = []
    for rank, reconstruction in analysis["svd_reconstructions"].items():
        if rank != maximum_rank:
            reconstruction_specs.append(("svd", rank, f"rank_{rank:04d}", reconstruction))
    for harmonic, reconstruction in analysis["fourier_reconstructions"].items():
        if harmonic != maximum_harmonic:
            reconstruction_specs.append(
                ("fourier", harmonic, f"harmonic_{harmonic:04d}", reconstruction)
            )

    decoded_by_key: dict[tuple[str, int], torch.Tensor] = {}
    if reconstruction_specs:
        stacked = torch.cat([item[3] for item in reconstruction_specs], dim=0)
        decoded_stacked = decode_in_chunks(
            model,
            stacked,
            flow,
            device,
            desc="Decoding filtered latent trajectories",
        ).clamp(0.0, 1.0)
        frame_count = sequence.num_frames
        for index, (transform, parameter, _, _) in enumerate(reconstruction_specs):
            decoded_by_key[(transform, parameter)] = decoded_stacked[
                index * frame_count : (index + 1) * frame_count
            ]

    target = sequence.frames
    region_reference = target
    baseline_metrics = _decoded_information_metrics(
        decoded_reference,
        target,
        region_reference=region_reference,
    )
    sweeps: dict[str, list[dict]] = {"svd": [], "fourier": []}
    decoded_groups: dict[str, OrderedDict[str, torch.Tensor]] = {
        "svd": OrderedDict(),
        "fourier": OrderedDict(),
    }
    for transform, metrics, maximum in (
        ("svd", analysis["svd_metrics"], maximum_rank),
        ("fourier", analysis["fourier_metrics"], maximum_harmonic),
    ):
        for parameter, latent_metrics in metrics.items():
            label = (
                f"rank_{parameter:04d}"
                if transform == "svd"
                else f"harmonic_{parameter:04d}"
            )
            decoded = (
                decoded_reference
                if parameter == maximum
                else decoded_by_key[(transform, parameter)]
            )
            ground_truth_metrics = _decoded_information_metrics(
                decoded,
                target,
                region_reference=region_reference,
            )
            dense_reference_metrics = _decoded_information_metrics(
                decoded,
                decoded_reference,
                region_reference=region_reference,
            )
            ground_truth_metrics["rmse_increase_over_dense_decode"] = (
                ground_truth_metrics["rmse"] - baseline_metrics["rmse"]
            )
            ground_truth_metrics["psnr_drop_from_dense_decode_db"] = (
                baseline_metrics["psnr_db"] - ground_truth_metrics["psnr_db"]
            )
            sweeps[transform].append(
                {
                    "parameter": parameter,
                    "label": label,
                    "latent": latent_metrics,
                    "decoded_vs_ground_truth": ground_truth_metrics,
                    "decoded_vs_dense_reference": dense_reference_metrics,
                }
            )
            if parameter != maximum:
                decoded_groups[transform][label] = decoded

    payload = {
        "definition": (
            "Adjacent terminal-latent differences are decomposed either by a "
            "mean-preserving centered SVD or by temporal low-pass Fourier filtering. "
            "Reconstructed differences are integrated from the original first latent "
            "and endpoint-corrected before deterministic decoding."
        ),
        "interpretation": {
            "nuisance_evidence": (
                "Strong latent compression with small decoded-vs-dense-reference error, "
                "especially small foreground and centroid error, indicates that discarded "
                "dimensions mostly encode visually minor nuisance variation."
            ),
            "signal_evidence": (
                "Rapid growth in foreground or activity-centroid error as rank/bandwidth "
                "is reduced indicates that the removed components carry physical motion."
            ),
        },
        "boundary_noise_mode": boundary_noise_mode,
        "dense_frame_spacing": sequence.cadence.high_frame_dt,
        "cadence": sequence.cadence,
        "summary": analysis["summary"],
        "svd_components": analysis["svd_components"],
        "fourier_bins": analysis["fourier_bins"],
        "dense_encoded_roundtrip": baseline_metrics,
        "sweeps": sweeps,
        "artifacts": {"plots": [], "videos": []},
        "video_panel_order": {},
    }

    if plot_results:
        plot_paths = [
            save_temporal_spectrum_plot(
                payload,
                output_root / "plots" / "temporal_spectrum.png",
            ),
            save_retention_plot(
                payload,
                output_root / "plots" / "retention_vs_image_loss.png",
            ),
        ]
        payload["artifacts"]["plots"] = [
            str(path.relative_to(output_root)) for path in plot_paths
        ]
        for path in plot_paths:
            print(f"Saved nuisance-dimension plot to {path}")

    if write_videos:
        for transform, predictions in decoded_groups.items():
            selected = _compact_video_predictions(predictions)
            if not selected:
                continue
            frames = make_comparison_video_frames(
                target,
                decoded_reference,
                selected,
                residual_scale=residual_scale,
                display_scale=display_scale,
                gap=gap,
            )
            video_path = output_root / "videos" / f"{transform}_sweep.mp4"
            write_video(frames, str(video_path), fps=video_fps)
            payload["artifacts"]["videos"].append(str(video_path.relative_to(output_root)))
            payload["video_panel_order"][transform] = [
                "ground_truth",
                "decoded_dense_reference",
                *[
                    item
                    for label in selected
                    for item in (label, f"{label}_absolute_residual")
                ],
            ]

    save_json(payload, output_root / "metrics.json")
    _write_sweep_csv(output_root / "sweep.csv", payload)
    if save_tensors:
        torch.save(
            {
                "reference_latents": reference_latents,
                "boundary_noise": boundary_noise.cpu(),
                "difference_matrix": analysis["difference_matrix"],
                "mean_difference": analysis["mean_difference"],
                "singular_values": analysis["singular_values"],
                "svd_reconstructions": analysis["svd_reconstructions"],
                "fourier_reconstructions": analysis["fourier_reconstructions"],
                "decoded_reference": decoded_reference,
                "decoded_reconstructions": decoded_by_key,
            },
            output_root / "tensors.pt",
        )
        print(f"Saved nuisance-dimension tensors to {output_root / 'tensors.pt'}")
    return payload
