"""Simulated sparse dynamic k-space observations.

The measurement pipeline is: render a dense frame sequence, transform each frame
to k-space, then keep only the frequencies in that frame's mask. The dense frames
are carried alongside as ground truth for evaluation and are never shown to the
optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from flow_interpolation.kspace.sampling import build_mask_sequence
from flow_interpolation.kspace.transforms import fft2c, ifft2c


@dataclass
class DynamicKSpaceData:
    """Sparse k-space observations of a dynamic scene.

    Attributes:
        kspace: ``[N, C, H, W]`` complex, already multiplied by the mask so
            unobserved entries are exactly zero.
        masks: ``[N, H, W]`` boolean sampling patterns, shared across channels.
        times: ``[N]`` acquisition time of each observation, normalized to
            ``[0, 1]`` across the sequence.
        frames: ``[N, C, H, W]`` ground-truth images. Evaluation only.
        frame_dt: spacing between consecutive observations in the same units as
            ``times``, i.e. ``1 / (N - 1)``.
    """

    kspace: torch.Tensor
    masks: torch.Tensor
    times: torch.Tensor
    frames: torch.Tensor
    frame_dt: float

    @property
    def num_frames(self) -> int:
        return int(self.kspace.shape[0])

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(self.kspace.shape[1:])  # type: ignore[return-value]

    @property
    def sampling_rate(self) -> float:
        """Per-frame fraction of k-space actually observed."""
        return self.masks.float().mean().item()

    @property
    def union_coverage(self) -> float:
        """Fraction of k-space observed at least once across the sequence."""
        return self.masks.any(dim=0).float().mean().item()

    def coverage_for_window(self, half_width: int) -> float:
        """Mean coverage of the mask union over windows of ``+/- half_width``.

        This is the quantity progressive binning trades against temporal
        resolution: it rises toward 1 as the bin widens.
        """
        if half_width < 0:
            raise ValueError("half_width must be non-negative")
        coverages = []
        for center in range(self.num_frames):
            low = max(center - half_width, 0)
            high = min(center + half_width + 1, self.num_frames)
            coverages.append(self.masks[low:high].any(dim=0).float().mean())
        return torch.stack(coverages).mean().item()

    def to(self, device: torch.device | str) -> "DynamicKSpaceData":
        return DynamicKSpaceData(
            kspace=self.kspace.to(device),
            masks=self.masks.to(device),
            times=self.times.to(device),
            frames=self.frames.to(device),
            frame_dt=self.frame_dt,
        )


def build_dynamic_kspace(
    frames: torch.Tensor,
    *,
    sampling_rate: float = 0.1,
    family: str = "variable-density",
    center_fraction: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 0,
    **family_kwargs,
) -> DynamicKSpaceData:
    """Simulate sparse k-space observations from a dense frame sequence.

    Args:
        frames: ``[N, C, H, W]`` images in ``[0, 1]``. Assumed real-valued, i.e.
            zero phase everywhere, so the resulting k-space is Hermitian.
        sampling_rate: fraction of the k-space grid retained per frame.
        family: mask family, see :data:`~.sampling.MASK_FAMILIES`.
        center_fraction: fraction of the grid area around DC to always sample.
        noise_std: standard deviation per real and imaginary component of the
            complex Gaussian measurement noise added before masking.
        seed: seeds both the masks and the measurement noise.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected frames of shape [N, C, H, W], got {tuple(frames.shape)}")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")

    num_frames, _, height, width = frames.shape
    masks = build_mask_sequence(
        num_frames,
        height,
        width,
        sampling_rate=sampling_rate,
        family=family,
        center_fraction=center_fraction,
        seed=seed,
        **family_kwargs,
    )

    kspace = fft2c(frames)
    if noise_std > 0.0:
        # Independent real/imaginary noise, seeded apart from the mask stream so
        # changing the SNR leaves the sampling pattern untouched.
        generator = torch.Generator(device="cpu").manual_seed(seed + 7_919)
        real = torch.randn(kspace.shape, generator=generator, dtype=frames.dtype)
        imaginary = torch.randn(kspace.shape, generator=generator, dtype=frames.dtype)
        kspace = kspace + noise_std * torch.complex(real, imaginary).to(kspace.device)

    kspace = kspace * masks.unsqueeze(1)
    times = (
        torch.arange(num_frames, dtype=torch.float32) / max(num_frames - 1, 1)
        if num_frames > 1
        else torch.zeros(1)
    )
    return DynamicKSpaceData(
        kspace=kspace,
        masks=masks,
        times=times,
        frames=frames,
        frame_dt=1.0 / max(num_frames - 1, 1),
    )


def zero_filled_reconstruction(data: DynamicKSpaceData) -> torch.Tensor:
    """Per-frame inverse FFT of the masked measurements. The naive baseline."""
    return ifft2c(data.kspace).abs()


def temporal_average_reconstruction(
    data: DynamicKSpaceData,
    half_width: Optional[int] = None,
) -> torch.Tensor:
    """Reconstruct each frame from the pooled measurements in its window.

    Each observed frequency is averaged over the frames in the window that
    sampled it, which is exactly the static solution progressive binning starts
    from. ``half_width=None`` pools the entire sequence into one static image.
    """
    counts = data.masks.unsqueeze(1).to(data.kspace.real.dtype)
    if half_width is None:
        pooled = data.kspace.sum(dim=0, keepdim=True) / counts.sum(dim=0, keepdim=True).clamp_min(1)
        pooled = pooled.expand_as(data.kspace)
    else:
        pooled = torch.empty_like(data.kspace)
        for center in range(data.num_frames):
            low = max(center - half_width, 0)
            high = min(center + half_width + 1, data.num_frames)
            window_counts = counts[low:high].sum(dim=0).clamp_min(1)
            pooled[center] = data.kspace[low:high].sum(dim=0) / window_counts
    return ifft2c(pooled).abs()
