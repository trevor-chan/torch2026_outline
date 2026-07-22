"""Data-consistency and regularization terms for scene fitting."""

from __future__ import annotations

import torch

from flow_interpolation.kspace.transforms import fft2c


def kspace_consistency_loss(
    rendered: torch.Tensor,
    observed_kspace: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Masked complex residual between the rendered scene and observations.

    Args:
        rendered: ``[B, C, H, W]`` real images from the scene model.
        observed_kspace: ``[B, W_bin, C, H, W]`` complex measurements, already
            masked, for each frame in each bin.
        masks: ``[B, W_bin, H, W]`` boolean sampling patterns for those frames.

    The mean is taken over sampled entries only. Normalizing by the number of
    observations rather than the grid size keeps the loss scale independent of
    the sampling rate and of the bin width, so a curriculum that changes the bin
    width mid-run does not silently rescale the gradient.
    """
    if rendered.ndim != 4:
        raise ValueError(f"Expected rendered [B, C, H, W], got {tuple(rendered.shape)}")
    if observed_kspace.ndim != 5:
        raise ValueError(
            f"Expected observed_kspace [B, W, C, H, W], got {tuple(observed_kspace.shape)}"
        )

    predicted = fft2c(rendered).unsqueeze(1)
    mask = masks.unsqueeze(2)
    residual = (predicted - observed_kspace) * mask
    observation_count = mask.expand_as(residual).sum().clamp_min(1)
    return residual.abs().square().sum() / observation_count


def spatial_tv(rendered: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation over the two spatial axes."""
    dy = rendered[..., 1:, :] - rendered[..., :-1, :]
    dx = rendered[..., :, 1:] - rendered[..., :, :-1]
    return dy.abs().mean() + dx.abs().mean()


def temporal_tv(rendered: torch.Tensor) -> torch.Tensor:
    """Total variation along the batch axis.

    Only meaningful when the rendered batch is a set of *adjacent* query times,
    which the fitting loop arranges by rendering a short consecutive run of
    times whenever this term is active.
    """
    if rendered.shape[0] < 2:
        return rendered.new_zeros(())
    return (rendered[1:] - rendered[:-1]).abs().mean()
