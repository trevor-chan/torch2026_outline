"""Ablate the image-boundary epsilon used for data-to-noise ODE inversion."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from flow_interpolation.data import SequenceData
from flow_interpolation.utils.flow import FlowSettings, encode_in_chunks
from flow_interpolation.utils.metrics import save_json


def _pyplot():
    cache = Path(tempfile.gettempdir()) / "flow_interpolation_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _epsilon_label(value: float) -> str:
    return f"{value:.6g}"


def _set_epsilon_scale(axis, epsilons: list[float]) -> None:
    if all(value > 0.0 for value in epsilons):
        axis.set_xscale("log")
        return
    positive = [value for value in epsilons if value > 0.0]
    linear_threshold = min(positive) / 10.0 if positive else 1e-6
    axis.set_xscale("symlog", linthresh=linear_threshold)


def _rms(values: torch.Tensor) -> float:
    return values.detach().float().square().mean().sqrt().item()


def _mean_variance_by_channel(variance_map: torch.Tensor) -> list[float]:
    return variance_map.detach().float().mean(dim=(-2, -1)).tolist()


def _latent_summary(
    latents: torch.Tensor,
    population_variance_map: torch.Tensor,
    boundary_variance_map: torch.Tensor,
    total_variance_map: torch.Tensor,
    coordinate_mean_square_map: torch.Tensor,
    second_moment_map: torch.Tensor,
) -> dict:
    flat = latents.detach().float().flatten(start_dim=2)
    radii = flat.norm(dim=2)
    latent_variance = flat.var(unbiased=False).item()
    latent_second_moment = flat.square().mean().item()
    normalized_radius = radii / math.sqrt(flat.shape[-1])
    return {
        "latent_mean": flat.mean().item(),
        "latent_std": math.sqrt(latent_variance),
        "latent_variance": latent_variance,
        "latent_second_moment": latent_second_moment,
        "radius_mean": radii.mean().item(),
        "radius_std": radii.std(unbiased=False).item(),
        "radius_over_sqrt_dimension_mean": normalized_radius.mean().item(),
        "radius_over_sqrt_dimension_std": normalized_radius.std(unbiased=False).item(),
        "population_variance_mean": population_variance_map.mean().item(),
        "boundary_variance_mean": boundary_variance_map.mean().item(),
        "total_variance_mean": total_variance_map.mean().item(),
        "coordinate_mean_square_mean": coordinate_mean_square_map.mean().item(),
        "coordinate_second_moment_mean": second_moment_map.mean().item(),
        "population_variance_by_channel": _mean_variance_by_channel(
            population_variance_map
        ),
        "boundary_variance_by_channel": _mean_variance_by_channel(boundary_variance_map),
        "total_variance_by_channel": _mean_variance_by_channel(total_variance_map),
        "coordinate_mean_square_by_channel": _mean_variance_by_channel(
            coordinate_mean_square_map
        ),
        "coordinate_second_moment_by_channel": _mean_variance_by_channel(
            second_moment_map
        ),
    }


def _comparison_metrics(latents: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = latents - reference
    reference_rms = _rms(reference)
    centered = latents - latents.mean(dim=1, keepdim=True)
    reference_centered = reference - reference.mean(dim=1, keepdim=True)
    centered_rmse = _rms(centered - reference_centered)
    reference_centered_rms = _rms(reference_centered)
    if latents.shape[1] > 1:
        steps = latents[:, 1:] - latents[:, :-1]
        reference_steps = reference[:, 1:] - reference[:, :-1]
        step_rmse = _rms(steps - reference_steps)
        reference_step_rms = _rms(reference_steps)
    else:
        step_rmse = 0.0
        reference_step_rms = 0.0
    rmse = _rms(difference)
    return {
        "latent_rmse": rmse,
        "latent_rmse_over_reference_rms": rmse / max(reference_rms, 1e-12),
        "trajectory_centered_rmse": centered_rmse,
        "trajectory_centered_rmse_over_reference_rms": centered_rmse
        / max(reference_centered_rms, 1e-12),
        "trajectory_step_rmse": step_rmse,
        "trajectory_step_rmse_over_reference_rms": step_rmse
        / max(reference_step_rms, 1e-12),
    }


def _trajectory_snr_metrics(
    latents: torch.Tensor,
    *,
    frame_spacing: float,
) -> dict[str, float | None]:
    """Compare boundary-seed variability with mean temporal motion.

    ``latents`` has shape [boundary draws, frames, channels, height, width].
    Both energies use squared L2 norms in the full latent dimension. Their
    per-coordinate versions divide by the latent dimension and have the same ratio.
    """
    values = latents.detach().float()
    latent_dimension = math.prod(values.shape[2:])
    mean_trajectory = values.mean(dim=0)
    encoding_residual = values - mean_trajectory.unsqueeze(0)
    encoding_energy_by_frame = (
        encoding_residual.flatten(start_dim=2).square().sum(dim=2).mean(dim=0)
    )
    encoding_variability = encoding_energy_by_frame.mean().item()

    if values.shape[1] > 1:
        temporal_steps = mean_trajectory[1:] - mean_trajectory[:-1]
        temporal_energy_by_step = temporal_steps.flatten(start_dim=1).square().sum(dim=1)
        temporal_signal = temporal_energy_by_step.mean().item()
    else:
        temporal_signal = 0.0

    encoding_per_coordinate = encoding_variability / latent_dimension
    temporal_per_coordinate = temporal_signal / latent_dimension
    if encoding_variability > 0.0:
        signal_to_noise = temporal_signal / encoding_variability
        noise_to_signal = encoding_variability / max(temporal_signal, 1e-30)
        snr_db = 10.0 * math.log10(max(signal_to_noise, 1e-30))
    else:
        signal_to_noise = None
        noise_to_signal = 0.0 if temporal_signal > 0.0 else None
        snr_db = None

    return {
        "encoding_variability_l2_squared": encoding_variability,
        "encoding_variability_per_coordinate": encoding_per_coordinate,
        "encoding_variability_rms_per_coordinate": math.sqrt(encoding_per_coordinate),
        "temporal_signal_l2_squared": temporal_signal,
        "temporal_signal_per_coordinate": temporal_per_coordinate,
        "temporal_signal_rms_per_coordinate": math.sqrt(temporal_per_coordinate),
        "trajectory_signal_to_encoding_noise_ratio": signal_to_noise,
        "trajectory_encoding_noise_to_signal_ratio": noise_to_signal,
        "trajectory_snr_db": snr_db,
        "frame_spacing": frame_spacing,
        "temporal_signal_l2_squared_per_time_squared": (
            temporal_signal / (frame_spacing * frame_spacing)
        ),
    }


def _variance_maps(latents: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return CxHxW maps separating variance and second-moment contributions.

    ``latents`` has shape [boundary draws, images, channels, height, width].
    """
    flattened = latents.flatten(0, 1)
    coordinate_mean = flattened.mean(dim=0)
    return {
        "population": latents.var(dim=1, unbiased=False).mean(dim=0),
        "boundary": latents.var(dim=0, unbiased=False).mean(dim=0),
        "total": flattened.var(dim=0, unbiased=False),
        "coordinate_mean_square": coordinate_mean.square(),
        "second_moment": flattened.square().mean(dim=0),
    }


def _pairwise_comparisons(
    latents_by_epsilon: dict[float, torch.Tensor],
    epsilons: list[float],
) -> dict[str, list[list[float]]]:
    metric_names = (
        "latent_rmse",
        "latent_rmse_over_reference_rms",
        "trajectory_centered_rmse",
        "trajectory_centered_rmse_over_reference_rms",
        "trajectory_step_rmse",
        "trajectory_step_rmse_over_reference_rms",
    )
    matrices = {name: [] for name in metric_names}
    for row_epsilon in epsilons:
        row_values = {name: [] for name in metric_names}
        for column_epsilon in epsilons:
            metrics = _comparison_metrics(
                latents_by_epsilon[column_epsilon],
                latents_by_epsilon[row_epsilon],
            )
            for name in metric_names:
                row_values[name].append(metrics[name])
        for name in metric_names:
            matrices[name].append(row_values[name])
    return matrices


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epsilon",
        "is_reference",
        "latent_mean",
        "latent_std",
        "latent_variance",
        "latent_second_moment",
        "radius_mean",
        "radius_std",
        "radius_over_sqrt_dimension_mean",
        "radius_over_sqrt_dimension_std",
        "population_variance_mean",
        "boundary_variance_mean",
        "total_variance_mean",
        "coordinate_mean_square_mean",
        "coordinate_second_moment_mean",
        "latent_rmse",
        "latent_rmse_over_reference_rms",
        "trajectory_centered_rmse",
        "trajectory_centered_rmse_over_reference_rms",
        "trajectory_step_rmse",
        "trajectory_step_rmse_over_reference_rms",
        "encoding_variability_l2_squared",
        "encoding_variability_per_coordinate",
        "encoding_variability_rms_per_coordinate",
        "temporal_signal_l2_squared",
        "temporal_signal_per_coordinate",
        "temporal_signal_rms_per_coordinate",
        "trajectory_signal_to_encoding_noise_ratio",
        "trajectory_encoding_noise_to_signal_ratio",
        "trajectory_snr_db",
        "frame_spacing",
        "temporal_signal_l2_squared_per_time_squared",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fieldnames} for row in rows)
    print(f"Saved epsilon-ablation summary to {path}")


def _plot_trajectory_snr(rows: list[dict], output_path: Path) -> None:
    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epsilons = [row["epsilon"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    energy_axis, ratio_axis = axes

    energy_axis.plot(
        epsilons,
        [row["encoding_variability_per_coordinate"] for row in rows],
        marker="o",
        label=r"$V_{\mathrm{enc}}/d$",
    )
    energy_axis.plot(
        epsilons,
        [row["temporal_signal_per_coordinate"] for row in rows],
        marker="o",
        label=r"$V_{\mathrm{time}}/d$",
    )
    energy_axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="unit prior variance",
    )
    energy_axis.set_yscale("log")
    energy_axis.set_title("Encoding variability vs temporal signal")
    energy_axis.set_ylabel("Mean squared energy per latent coordinate")
    energy_axis.legend(fontsize=8)

    ratios = [
        (
            float("nan")
            if row["trajectory_signal_to_encoding_noise_ratio"] is None
            else row["trajectory_signal_to_encoding_noise_ratio"]
        )
        for row in rows
    ]
    ratio_axis.plot(epsilons, ratios, marker="o", color="tab:purple")
    ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ratio_axis.set_yscale("log")
    ratio_axis.set_title(r"Trajectory SNR: $V_{\mathrm{time}}/V_{\mathrm{enc}}$")
    ratio_axis.set_ylabel("Signal-to-encoding-noise ratio")

    for axis in axes:
        _set_epsilon_scale(axis, epsilons)
        axis.set_xlabel("Image-boundary epsilon")
        axis.grid(alpha=0.25)
    figure.suptitle("Temporal latent signal relative to boundary-seed variability")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved trajectory-SNR plot to {output_path}")


def _plot_summary(
    *,
    rows: list[dict],
    epsilons: list[float],
    pairwise_centered_rmse: list[list[float]],
    output_path: Path,
) -> None:
    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = epsilons
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    variance_axis, stats_axis, displacement_axis, pairwise_axis = axes.flatten()

    for key, label in (
        ("population_variance_mean", "across images"),
        ("boundary_variance_mean", "across boundary draws"),
        ("total_variance_mean", "per-coordinate combined"),
        ("latent_variance", "global over all entries"),
    ):
        variance_axis.plot(x, [row[key] for row in rows], marker="o", label=label)
    variance_axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="N(0,I) total/global variance",
    )
    variance_axis.set_title("Mean coordinate variance")
    variance_axis.set_ylabel("Variance")
    variance_axis.set_yscale("log")
    variance_axis.legend(fontsize=8)

    stats_axis.plot(x, [row["latent_std"] for row in rows], marker="o", label="latent std")
    stats_axis.plot(
        x,
        [row["radius_over_sqrt_dimension_mean"] for row in rows],
        marker="o",
        label="radius / sqrt(d)",
    )
    stats_axis.plot(
        x,
        [math.sqrt(row["latent_second_moment"]) for row in rows],
        marker="o",
        label="sqrt(second moment)",
    )
    stats_axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="N(0,I) expectation",
    )
    stats_axis.set_title("Terminal latent scale")
    stats_axis.set_ylabel("Scale")
    stats_axis.legend(fontsize=8)

    for key, label in (
        ("latent_rmse", "absolute latent"),
        ("trajectory_centered_rmse", "centered trajectory"),
        ("trajectory_step_rmse", "trajectory steps"),
    ):
        displacement_axis.plot(x, [row[key] for row in rows], marker="o", label=label)
    displacement_axis.set_title("Difference from reference epsilon")
    displacement_axis.set_ylabel("RMSE (N(0,I) standard deviations)")
    displacement_axis.set_yscale("symlog", linthresh=1e-8)
    displacement_axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="one prior standard deviation",
    )
    displacement_axis.axhline(
        0.1,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label="0.1 prior standard deviations",
    )
    displacement_axis.legend(fontsize=8)

    image = pairwise_axis.imshow(pairwise_centered_rmse, cmap="magma")
    pairwise_axis.set_title("Pairwise centered RMSE (prior sigma units)")
    pairwise_axis.set_xticks(
        range(len(epsilons)), [_epsilon_label(value) for value in epsilons]
    )
    pairwise_axis.set_yticks(
        range(len(epsilons)), [_epsilon_label(value) for value in epsilons]
    )
    pairwise_axis.set_xlabel("Compared epsilon")
    pairwise_axis.set_ylabel("Reference epsilon")
    figure.colorbar(image, ax=pairwise_axis, shrink=0.85)

    for axis in (variance_axis, stats_axis, displacement_axis):
        _set_epsilon_scale(axis, epsilons)
        axis.set_xlabel("Image-boundary epsilon")
        axis.grid(alpha=0.25)
    figure.suptitle("Image-to-noise epsilon ablation", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved epsilon-ablation plot to {output_path}")


def _plot_variance_maps(
    variance_maps: dict[float, dict[str, torch.Tensor]],
    epsilons: list[float],
    output_path: Path,
) -> None:
    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_names = ("population", "boundary", "total", "second_moment")
    channel_averages = {
        epsilon: {
            name: variance_maps[epsilon][name].detach().float().mean(dim=0)
            for name in map_names
        }
        for epsilon in epsilons
    }
    log_floor = 1e-12
    limits = {}
    for name in map_names:
        values = torch.stack(
            [torch.log10(channel_averages[epsilon][name] + log_floor) for epsilon in epsilons]
        )
        limits[name] = (values.min().item(), values.max().item())

    figure, axes = plt.subplots(
        len(epsilons),
        len(map_names),
        figsize=(13, max(2.4 * len(epsilons), 4.0)),
        squeeze=False,
        constrained_layout=True,
    )
    last_images = {}
    for row_index, epsilon in enumerate(epsilons):
        for column_index, name in enumerate(map_names):
            axis = axes[row_index][column_index]
            values = torch.log10(channel_averages[epsilon][name] + log_floor).cpu().numpy()
            vmin, vmax = limits[name]
            if math.isclose(vmin, vmax):
                vmax = vmin + 1.0
            last_images[name] = axis.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                title = {
                    "population": "Across-image variance",
                    "boundary": "Boundary-draw variance",
                    "total": "Combined coordinate variance",
                    "second_moment": "Coordinate second moment",
                }[name]
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(f"eps={_epsilon_label(epsilon)}")
    for column_index, name in enumerate(map_names):
        figure.colorbar(
            last_images[name],
            ax=axes[:, column_index].tolist(),
            shrink=0.8,
            label="log10(value); unit prior scale is 0",
        )
    figure.suptitle(
        "Channel-averaged terminal latent maps; unit prior variance/moment is log10(1)=0",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved epsilon variance maps to {output_path}")


@torch.no_grad()
def run_epsilon_ablation(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    epsilons: list[float],
    num_boundary_samples: int,
    boundary_noise_mode: str,
    frame_source: str,
    seed: int,
    output_dir: str,
    save_tensors: bool,
) -> dict:
    """Measure terminal-latent variance and trajectory sensitivity across epsilons."""
    if num_boundary_samples <= 0:
        raise ValueError("num_boundary_samples must be positive")
    if boundary_noise_mode not in {"shared", "independent"}:
        raise ValueError(f"Unknown boundary-noise mode: {boundary_noise_mode}")
    if frame_source == "observed":
        frames = sequence.observed_frames
        frame_indices = sequence.observed_indices
    elif frame_source == "dense":
        frames = sequence.frames
        frame_indices = torch.arange(sequence.num_frames)
    else:
        raise ValueError(f"Unknown frame source: {frame_source}")

    requested_epsilons = list(epsilons)
    effective_epsilons = sorted(set([*requested_epsilons, flow.data_time]))
    invalid = [
        epsilon
        for epsilon in effective_epsilons
        if epsilon < 0.0 or epsilon >= flow.noise_time
    ]
    if invalid:
        raise ValueError(
            f"Every epsilon must be in [0, {flow.noise_time}); got {invalid}"
        )

    generator = torch.Generator(device=device).manual_seed(seed)
    sample_shape = frames.shape[1:]
    if boundary_noise_mode == "shared":
        boundary_noise = torch.randn(
            (num_boundary_samples, 1, *sample_shape),
            device=device,
            dtype=frames.dtype,
            generator=generator,
        ).expand(num_boundary_samples, frames.shape[0], *sample_shape)
    else:
        boundary_noise = torch.randn(
            (num_boundary_samples, frames.shape[0], *sample_shape),
            device=device,
            dtype=frames.dtype,
            generator=generator,
        )
    boundary_noise = boundary_noise.contiguous().cpu()
    repeated_frames = frames.repeat((num_boundary_samples, 1, 1, 1))
    flattened_noise = boundary_noise.flatten(0, 1)

    latents_by_epsilon: dict[float, torch.Tensor] = {}
    variance_maps: dict[float, dict[str, torch.Tensor]] = {}
    latent_dimension = int(math.prod(sample_shape))
    for epsilon in effective_epsilons:
        epsilon_flow = replace(flow, data_time=epsilon)
        encoded = encode_in_chunks(
            model,
            repeated_frames,
            epsilon_flow,
            device,
            eps_noise=flattened_noise,
            perturb=True,
            desc=f"Encoding epsilon={_epsilon_label(epsilon)}",
        )
        if not torch.isfinite(encoded).all():
            raise RuntimeError(
                f"Non-finite terminal latents were produced for epsilon={epsilon:g}"
            )
        latents = encoded.reshape(num_boundary_samples, frames.shape[0], *sample_shape)
        latents_by_epsilon[epsilon] = latents
        variance_maps[epsilon] = _variance_maps(latents)

    reference = latents_by_epsilon[flow.data_time]
    frame_spacing = (
        sequence.cadence.high_frame_dt
        if frame_source == "dense"
        else sequence.cadence.actual_endpoint_dt
    )
    rows = []
    for epsilon in effective_epsilons:
        maps = variance_maps[epsilon]
        row = {
            "epsilon": epsilon,
            "is_reference": epsilon == flow.data_time,
            "latent_dimension": latent_dimension,
            **_latent_summary(
                latents_by_epsilon[epsilon],
                maps["population"],
                maps["boundary"],
                maps["total"],
                maps["coordinate_mean_square"],
                maps["second_moment"],
            ),
            **_comparison_metrics(latents_by_epsilon[epsilon], reference),
            **_trajectory_snr_metrics(
                latents_by_epsilon[epsilon],
                frame_spacing=frame_spacing,
            ),
        }
        rows.append(row)
        print(
            f"Epsilon {epsilon:g}: latent std={row['latent_std']:.5f}, "
            f"population var={row['population_variance_mean']:.6g}, "
            f"boundary var={row['boundary_variance_mean']:.6g}, "
            f"centered trajectory RMSE={row['trajectory_centered_rmse']:.6g}, "
            f"trajectory SNR={row['trajectory_signal_to_encoding_noise_ratio']}"
        )

    pairwise = _pairwise_comparisons(latents_by_epsilon, effective_epsilons)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "epsilon_ablation.csv"
    summary_plot = output_root / "epsilon_ablation_summary.png"
    maps_plot = output_root / "epsilon_variance_maps.png"
    snr_plot = output_root / "epsilon_trajectory_snr.png"
    _write_summary_csv(summary_csv, rows)
    _plot_summary(
        rows=rows,
        epsilons=effective_epsilons,
        pairwise_centered_rmse=pairwise["trajectory_centered_rmse"],
        output_path=summary_plot,
    )
    _plot_variance_maps(variance_maps, effective_epsilons, maps_plot)
    _plot_trajectory_snr(rows, snr_plot)

    payload = {
        "purpose": (
            "Ablate the clean-image boundary epsilon while reusing identical boundary-noise "
            "draws across epsilon values. Across-image variance measures the encoded sample "
            "distribution; boundary-draw variance measures stochastic sensitivity for fixed "
            "images; centered and step RMSE isolate changes in trajectory geometry from a "
            "common latent shift. Raw latent RMSE is measured in prior standard-deviation "
            "units because the target terminal distribution is N(0,I). Trajectory SNR "
            "compares mean adjacent-frame latent motion with within-frame variability "
            "over boundary-noise draws."
        ),
        "trajectory_snr_definition": {
            "encoding_variability": (
                "E_k E_r ||z_k^(r) - mean_r[z_k^(r)]||_2^2"
            ),
            "temporal_signal": (
                "E_k ||mean_r[z_(k+1)^(r)] - mean_r[z_k^(r)]||_2^2"
            ),
            "signal_to_noise_ratio": "temporal_signal / encoding_variability",
            "note": (
                "The ratio is unchanged by dividing both energies by latent dimension. "
                "It depends on frame spacing, which is recorded for each row."
            ),
        },
        "prior_reference": {
            "distribution": "standard normal N(0,I)",
            "coordinate_mean": 0.0,
            "coordinate_variance": 1.0,
            "coordinate_standard_deviation": 1.0,
            "coordinate_second_moment": 1.0,
            "expected_radius_approximation": math.sqrt(latent_dimension),
            "expected_radius_over_sqrt_dimension_approximation": 1.0,
            "interpretation_note": (
                "Global latent variance pools every coordinate and sample. Combined "
                "coordinate variance first estimates variance separately at each CxHxW "
                "location and then averages; it can differ because coordinate means are "
                "not identical. Coordinate second moment equals coordinate variance plus "
                "squared coordinate mean and is directly comparable with E[z_j^2]=1. "
                "Across-image and boundary-draw variances are two conditional summaries, "
                "not additive components of the combined variance."
            ),
        },
        "requested_epsilons": requested_epsilons,
        "effective_epsilons": effective_epsilons,
        "reference_epsilon": flow.data_time,
        "num_boundary_samples": num_boundary_samples,
        "boundary_noise_mode": boundary_noise_mode,
        "frame_source": frame_source,
        "frame_count": int(frames.shape[0]),
        "frame_indices": frame_indices,
        "image_background_noise_std": sequence.background_noise_std,
        "noise_time": flow.noise_time,
        "ode_steps": flow.ode_steps,
        "solver": flow.solver,
        "rows": rows,
        "pairwise": {
            "epsilon_order": effective_epsilons,
            **pairwise,
        },
        "artifacts": {
            "summary_csv": summary_csv.name,
            "summary_plot": summary_plot.name,
            "variance_maps_plot": maps_plot.name,
            "trajectory_snr_plot": snr_plot.name,
            "tensors": "epsilon_ablation_tensors.pt" if save_tensors else None,
        },
    }
    save_json(payload, output_root / "epsilon_ablation_metrics.json")
    if save_tensors:
        tensor_path = output_root / "epsilon_ablation_tensors.pt"
        torch.save(
            {
                "frames": frames,
                "frame_indices": frame_indices,
                "boundary_noise": boundary_noise,
                "latents_by_epsilon": latents_by_epsilon,
                "variance_maps": variance_maps,
            },
            tensor_path,
        )
        print(f"Saved epsilon-ablation tensors to {tensor_path}")
    return payload
