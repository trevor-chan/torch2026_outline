"""Compatibility imports for the renamed trajectory analysis experiment."""

from flow_interpolation.evaluation.experiments.trajectory import (
    run_latent_geodesic_evaluation,
    run_trajectory_analysis,
)

__all__ = ["run_latent_geodesic_evaluation", "run_trajectory_analysis"]
