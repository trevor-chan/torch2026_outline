"""Temporal decompositions for separating smooth signal from latent nuisance variation."""

from __future__ import annotations

import torch


def _flatten_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.ndim < 2 or trajectory.shape[0] < 2:
        raise ValueError("trajectory must contain at least two frames and one feature axis")
    return trajectory.detach().float().cpu().flatten(start_dim=1)


def _restore_trajectory(
    initial: torch.Tensor,
    differences: torch.Tensor,
    sample_shape: torch.Size,
) -> torch.Tensor:
    flat = torch.cat(
        [
            initial[None],
            initial[None] + torch.cumsum(differences, dim=0),
        ],
        dim=0,
    )
    return flat.reshape(flat.shape[0], *sample_shape)


def _endpoint_correct(
    reconstruction: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Distribute numerical endpoint drift uniformly without changing temporal rank."""
    correction = (reference.sum(dim=0) - reconstruction.sum(dim=0)) / reference.shape[0]
    return reconstruction + correction


def _trajectory_metrics(
    reconstruction: torch.Tensor,
    reference_trajectory: torch.Tensor,
    reconstructed_differences: torch.Tensor,
    reference_differences: torch.Tensor,
    centered_reference_differences: torch.Tensor,
) -> dict[str, float]:
    residual = reconstruction - reference_trajectory
    difference_residual = reconstructed_differences - reference_differences
    reference_energy = reference_differences.square().sum().clamp_min(1e-12)
    centered_energy = centered_reference_differences.square().sum().clamp_min(1e-12)
    reconstructed_centered = (
        reconstructed_differences - reconstructed_differences.mean(dim=0, keepdim=True)
    )
    retained_centered = (
        1.0
        - (reconstructed_centered - centered_reference_differences)
        .square()
        .sum()
        / centered_energy
    ).clamp(0.0, 1.0)
    return {
        "difference_energy_retained_fraction": float(
            (1.0 - difference_residual.square().sum() / reference_energy)
            .clamp(0.0, 1.0)
            .item()
        ),
        "centered_difference_energy_retained_fraction": float(retained_centered.item()),
        "difference_residual_rmse": float(difference_residual.square().mean().sqrt().item()),
        "trajectory_rmse": float(residual.square().mean().sqrt().item()),
        "trajectory_relative_l2": float(
            (
                residual.norm(dim=1)
                / reference_trajectory.norm(dim=1).clamp_min(1e-12)
            )
            .mean()
            .item()
        ),
        "endpoint_rmse": float(residual[-1].square().mean().sqrt().item()),
    }


def _rfft_power(values: torch.Tensor) -> torch.Tensor:
    """Return one-sided Fourier power with Parseval-correct interior weighting."""
    coefficients = torch.fft.rfft(values, dim=0)
    power = coefficients.abs().square().sum(dim=1)
    if values.shape[0] > 1 and power.numel() > 1:
        weights = torch.full_like(power, 2.0)
        weights[0] = 1.0
        if values.shape[0] % 2 == 0:
            weights[-1] = 1.0
        power = power * weights
    return power


def _rank_for_fraction(cumulative: torch.Tensor, fraction: float) -> int:
    if cumulative.numel() == 0:
        return 0
    index = torch.searchsorted(cumulative, torch.tensor(fraction, dtype=cumulative.dtype))
    return min(int(index) + 1, cumulative.numel())


def _harmonic_for_fraction(cumulative: torch.Tensor, fraction: float) -> int:
    if cumulative.numel() == 0:
        return 0
    index = torch.searchsorted(cumulative, torch.tensor(fraction, dtype=cumulative.dtype))
    return min(int(index), cumulative.numel() - 1)


def analyze_temporal_nuisance(
    trajectory: torch.Tensor,
    *,
    sample_spacing: float,
    svd_ranks: list[int],
    fourier_harmonics: list[int],
) -> dict:
    """Reconstruct a trajectory from low-rank or low-frequency temporal differences.

    The SVD is applied to mean-centered adjacent differences. The mean difference is
    always retained, making rank zero a constant-velocity, endpoint-preserving path.
    Fourier reconstructions likewise always retain the DC component.
    """
    if sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive")
    if any(rank < 0 for rank in svd_ranks):
        raise ValueError("SVD ranks must be non-negative")
    if any(harmonic < 0 for harmonic in fourier_harmonics):
        raise ValueError("Fourier harmonic counts must be non-negative")

    sample_shape = trajectory.shape[1:]
    values = _flatten_trajectory(trajectory)
    differences = values[1:] - values[:-1]
    mean_difference = differences.mean(dim=0, keepdim=True)
    centered = differences - mean_difference
    u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    squared_singular_values = singular_values.square()
    centered_energy = squared_singular_values.sum().clamp_min(1e-12)
    component_energy_fraction = squared_singular_values / centered_energy
    cumulative_component_energy = torch.cumsum(component_energy_fraction, dim=0).clamp_max(1.0)
    maximum_rank = int(singular_values.numel())
    resolved_ranks = sorted({min(rank, maximum_rank) for rank in svd_ranks} | {maximum_rank})

    temporal_frequencies = torch.fft.rfftfreq(differences.shape[0], d=sample_spacing)
    component_rows = []
    for index in range(maximum_rank):
        temporal_coefficient = u[:, index] * singular_values[index]
        power = _rfft_power(temporal_coefficient[:, None])
        non_dc = power.clone()
        non_dc[0] = 0.0
        denominator = non_dc.sum()
        if denominator > 0.0:
            dominant_index = int(non_dc.argmax())
            centroid_frequency = float(
                (temporal_frequencies * non_dc / denominator).sum().item()
            )
            dominant_frequency = float(temporal_frequencies[dominant_index].item())
        else:
            centroid_frequency = 0.0
            dominant_frequency = 0.0
        component_rows.append(
            {
                "component": index + 1,
                "singular_value": float(singular_values[index].item()),
                "energy_fraction": float(component_energy_fraction[index].item()),
                "cumulative_energy_fraction": float(cumulative_component_energy[index].item()),
                "dominant_frequency": dominant_frequency,
                "spectral_centroid_frequency": centroid_frequency,
            }
        )

    svd_reconstructions: dict[int, torch.Tensor] = {}
    svd_metrics: dict[int, dict[str, float]] = {}
    for rank in resolved_ranks:
        if rank == 0:
            reconstructed_centered = torch.zeros_like(centered)
        else:
            reconstructed_centered = (
                u[:, :rank] * singular_values[:rank]
            ) @ vh[:rank]
        reconstructed_differences = _endpoint_correct(
            reconstructed_centered + mean_difference,
            differences,
        )
        reconstructed_trajectory = _restore_trajectory(
            values[0],
            reconstructed_differences,
            sample_shape,
        )
        svd_reconstructions[rank] = reconstructed_trajectory
        svd_metrics[rank] = {
            "rank": rank,
            "rank_fraction": rank / max(maximum_rank, 1),
            **_trajectory_metrics(
                reconstructed_trajectory.flatten(start_dim=1),
                values,
                reconstructed_differences,
                differences,
                centered,
            ),
        }

    coefficients = torch.fft.rfft(differences, dim=0)
    fourier_power = _rfft_power(differences)
    total_fourier_power = fourier_power.sum().clamp_min(1e-12)
    fourier_energy_fraction = fourier_power / total_fourier_power
    cumulative_fourier_energy = torch.cumsum(fourier_energy_fraction, dim=0).clamp_max(1.0)
    maximum_harmonic = int(coefficients.shape[0] - 1)
    resolved_harmonics = sorted(
        {min(harmonic, maximum_harmonic) for harmonic in fourier_harmonics}
        | {maximum_harmonic}
    )
    fourier_rows = [
        {
            "harmonic": index,
            "frequency": float(temporal_frequencies[index].item()),
            "period": (
                None
                if index == 0
                else float(1.0 / temporal_frequencies[index].item())
            ),
            "energy_fraction": float(fourier_energy_fraction[index].item()),
            "cumulative_energy_fraction": float(cumulative_fourier_energy[index].item()),
        }
        for index in range(coefficients.shape[0])
    ]

    fourier_reconstructions: dict[int, torch.Tensor] = {}
    fourier_metrics: dict[int, dict[str, float]] = {}
    for harmonic in resolved_harmonics:
        filtered_coefficients = torch.zeros_like(coefficients)
        filtered_coefficients[: harmonic + 1] = coefficients[: harmonic + 1]
        reconstructed_differences = torch.fft.irfft(
            filtered_coefficients,
            n=differences.shape[0],
            dim=0,
        )
        reconstructed_differences = _endpoint_correct(reconstructed_differences, differences)
        reconstructed_trajectory = _restore_trajectory(
            values[0],
            reconstructed_differences,
            sample_shape,
        )
        fourier_reconstructions[harmonic] = reconstructed_trajectory
        cutoff_frequency = float(temporal_frequencies[harmonic].item())
        fourier_metrics[harmonic] = {
            "harmonic": harmonic,
            "cutoff_frequency": cutoff_frequency,
            "cutoff_period": None if harmonic == 0 else 1.0 / cutoff_frequency,
            **_trajectory_metrics(
                reconstructed_trajectory.flatten(start_dim=1),
                values,
                reconstructed_differences,
                differences,
                centered,
            ),
        }

    participation_ratio = float(
        squared_singular_values.sum().square().div(
            squared_singular_values.square().sum().clamp_min(1e-12)
        )
    )
    non_dc_fourier = fourier_power[1:]
    non_dc_total = non_dc_fourier.sum().clamp_min(1e-12)
    non_dc_cumulative = torch.cumsum(non_dc_fourier / non_dc_total, dim=0)
    return {
        "difference_matrix": differences,
        "mean_difference": mean_difference,
        "centered_difference_matrix": centered,
        "singular_values": singular_values,
        "svd_components": component_rows,
        "fourier_bins": fourier_rows,
        "svd_reconstructions": svd_reconstructions,
        "fourier_reconstructions": fourier_reconstructions,
        "svd_metrics": svd_metrics,
        "fourier_metrics": fourier_metrics,
        "summary": {
            "frame_count": int(values.shape[0]),
            "latent_dimension": int(values.shape[1]),
            "difference_count": int(differences.shape[0]),
            "duration": float((values.shape[0] - 1) * sample_spacing),
            "nyquist_frequency": float(0.5 / sample_spacing),
            "maximum_centered_rank": maximum_rank,
            "maximum_fourier_harmonic": maximum_harmonic,
            "centered_difference_effective_rank": participation_ratio,
            "svd_rank_for_50_percent_centered_energy": _rank_for_fraction(
                cumulative_component_energy, 0.50
            ),
            "svd_rank_for_90_percent_centered_energy": _rank_for_fraction(
                cumulative_component_energy, 0.90
            ),
            "svd_rank_for_95_percent_centered_energy": _rank_for_fraction(
                cumulative_component_energy, 0.95
            ),
            "svd_rank_for_99_percent_centered_energy": _rank_for_fraction(
                cumulative_component_energy, 0.99
            ),
            "fourier_harmonic_for_50_percent_non_dc_energy": (
                _harmonic_for_fraction(non_dc_cumulative, 0.50) + 1
                if non_dc_fourier.numel()
                else 0
            ),
            "fourier_harmonic_for_90_percent_non_dc_energy": (
                _harmonic_for_fraction(non_dc_cumulative, 0.90) + 1
                if non_dc_fourier.numel()
                else 0
            ),
            "fourier_harmonic_for_95_percent_non_dc_energy": (
                _harmonic_for_fraction(non_dc_cumulative, 0.95) + 1
                if non_dc_fourier.numel()
                else 0
            ),
            "fourier_harmonic_for_99_percent_non_dc_energy": (
                _harmonic_for_fraction(non_dc_cumulative, 0.99) + 1
                if non_dc_fourier.numel()
                else 0
            ),
            "mean_difference_rms": float(mean_difference.square().mean().sqrt().item()),
            "centered_difference_rms": float(centered.square().mean().sqrt().item()),
            "difference_rms": float(differences.square().mean().sqrt().item()),
        },
    }
