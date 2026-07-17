"""Plots for temporal latent nuisance-dimension diagnostics."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _pyplot():
    cache = Path(tempfile.gettempdir()) / "flow_interpolation_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_temporal_spectrum_plot(payload: dict, output_path: str | Path) -> Path:
    plt = _pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    components = payload["svd_components"]
    bins = payload["fourier_bins"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    singular_axis, cumulative_axis, power_axis, frequency_cumulative_axis = axes.flatten()

    component_index = [item["component"] for item in components]
    singular_axis.semilogy(
        component_index,
        [item["singular_value"] for item in components],
        marker=".",
    )
    singular_axis.set_title("Centered difference singular spectrum")
    singular_axis.set_xlabel("Component")
    singular_axis.set_ylabel("Singular value")

    cumulative_axis.plot(
        component_index,
        [item["cumulative_energy_fraction"] for item in components],
    )
    for threshold in (0.5, 0.9, 0.95, 0.99):
        cumulative_axis.axhline(threshold, color="gray", linewidth=0.8, alpha=0.5)
    cumulative_axis.set_title("SVD energy retained")
    cumulative_axis.set_xlabel("Rank")
    cumulative_axis.set_ylabel("Centered difference energy")
    cumulative_axis.set_ylim(0.0, 1.02)

    non_dc_bins = bins[1:]
    frequencies = [item["frequency"] for item in non_dc_bins]
    power_axis.semilogy(
        frequencies,
        [max(item["energy_fraction"], 1e-12) for item in non_dc_bins],
        marker=".",
    )
    power_axis.set_title("Temporal Fourier spectrum of differences")
    power_axis.set_xlabel("Frequency")
    power_axis.set_ylabel("Total difference-energy fraction")

    frequency_cumulative_axis.plot(
        [item["frequency"] for item in bins],
        [item["cumulative_energy_fraction"] for item in bins],
    )
    for threshold in (0.5, 0.9, 0.95, 0.99):
        frequency_cumulative_axis.axhline(
            threshold, color="gray", linewidth=0.8, alpha=0.5
        )
    frequency_cumulative_axis.set_title("Low-pass energy retained")
    frequency_cumulative_axis.set_xlabel("Cutoff frequency")
    frequency_cumulative_axis.set_ylabel("Difference energy")
    frequency_cumulative_axis.set_ylim(0.0, 1.02)

    for axis in axes.flatten():
        axis.grid(alpha=0.25)
    figure.suptitle("Temporal structure of the terminal-latent trajectory", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_retention_plot(payload: dict, output_path: str | Path) -> Path:
    plt = _pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    latent_axis, image_axis, region_axis, motion_axis = axes.flatten()

    for transform, color in (("svd", "tab:blue"), ("fourier", "tab:orange")):
        rows = payload["sweeps"][transform]
        retained = [
            row["latent"]["centered_difference_energy_retained_fraction"]
            for row in rows
        ]
        latent_axis.plot(
            retained,
            [row["latent"]["trajectory_rmse"] for row in rows],
            marker="o",
            color=color,
            label=transform,
        )
        image_axis.plot(
            retained,
            [row["decoded_vs_dense_reference"]["rmse"] for row in rows],
            marker="o",
            color=color,
            label=transform,
        )
        region_axis.plot(
            retained,
            [
                row["decoded_vs_dense_reference"]["foreground_weighted_rmse"]
                for row in rows
            ],
            marker="o",
            color=color,
            linestyle="-",
            label=f"{transform}: foreground",
        )
        region_axis.plot(
            retained,
            [
                row["decoded_vs_dense_reference"]["background_weighted_rmse"]
                for row in rows
            ],
            marker="o",
            color=color,
            linestyle="--",
            label=f"{transform}: background",
        )
        motion_axis.plot(
            retained,
            [
                row["decoded_vs_dense_reference"]["activity_centroid_error_pixels"]
                for row in rows
            ],
            marker="o",
            color=color,
            label=transform,
        )

    latent_axis.set_title("Latent trajectory loss")
    latent_axis.set_ylabel("RMSE from dense latent trajectory")
    image_axis.set_title("Filter-induced decoded image loss")
    image_axis.set_ylabel("RMSE from decoded dense trajectory")
    region_axis.set_title("Where decoded information is lost")
    region_axis.set_ylabel("Weighted image RMSE")
    motion_axis.set_title("Physical-motion proxy")
    motion_axis.set_ylabel("Soft activity-centroid error (pixels)")
    for axis in axes.flatten():
        axis.set_xlabel("Centered difference energy retained")
        axis.set_xlim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Latent compression versus decoded information retention", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
