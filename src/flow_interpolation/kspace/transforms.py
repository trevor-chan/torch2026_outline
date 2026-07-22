"""Centered orthonormal 2D Fourier transforms.

The DC term sits at the center of the array (``fftshift`` convention) so that
mask families expressed in terms of distance from the center behave the way
their names suggest. The ``"ortho"`` normalization makes the transform unitary,
so a masked k-space residual is directly comparable in magnitude to an image
residual and Parseval's identity holds without extra bookkeeping.

The transforms operate on the trailing two dimensions, so any leading batch,
frame, or channel axes pass through untouched.
"""

from __future__ import annotations

import torch

_SPATIAL_DIMS = (-2, -1)


def fft2c(images: torch.Tensor) -> torch.Tensor:
    """Map images to centered, orthonormal k-space.

    ``images`` may be real or complex; the result is always complex.
    """
    shifted = torch.fft.ifftshift(images, dim=_SPATIAL_DIMS)
    spectrum = torch.fft.fft2(shifted, dim=_SPATIAL_DIMS, norm="ortho")
    return torch.fft.fftshift(spectrum, dim=_SPATIAL_DIMS)


def ifft2c(spectrum: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`fft2c`. Returns a complex tensor."""
    shifted = torch.fft.ifftshift(spectrum, dim=_SPATIAL_DIMS)
    images = torch.fft.ifft2(shifted, dim=_SPATIAL_DIMS, norm="ortho")
    return torch.fft.fftshift(images, dim=_SPATIAL_DIMS)


def kspace_grid(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized frequency coordinates matching the :func:`fft2c` layout.

    Returns ``(ky, kx)`` grids of shape ``[height, width]``, each spanning
    roughly ``[-1, 1]`` with zero at the DC location. Even-sized axes place DC
    at index ``n // 2``, one sample past center, exactly as ``fftshift`` does.
    """
    ky = (torch.arange(height, device=device, dtype=dtype) - height // 2) / max(height // 2, 1)
    kx = (torch.arange(width, device=device, dtype=dtype) - width // 2) / max(width // 2, 1)
    return torch.meshgrid(ky, kx, indexing="ij")


def kspace_radius(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Euclidean distance from DC on the normalized frequency grid."""
    ky, kx = kspace_grid(height, width, device=device, dtype=dtype)
    return torch.sqrt(ky.square() + kx.square())
