"""Progress videos comparing ground truth, measurements, and the fitted scene.

Every frame of the video is a 2x4 panel:

```
          ground truth      measured        fitted          residual
image  |  dense frame   |  zero-filled  |  s_theta(t)   |  |fit - truth| |
k-space|  full spectrum |  sparse M.k   |  F[s_theta]   |  |dK|          |
```

Putting image and k-space side by side is the point: the two rows fail in
visibly different ways. A reconstruction that looks plausible but ignores the
measurements shows up as structure in the k-space residual at *observed*
frequencies, while over-smoothing shows up as missing energy at high radius.
Neither is obvious from the image row alone.

All k-space panels share one log-magnitude mapping, fixed to the ground-truth
peak, so brightness is comparable across columns and stable across steps —
a per-panel autoscale would make the fit appear to converge simply because its
own dynamic range changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from flow_interpolation.kspace.dataset import DynamicKSpaceData
from flow_interpolation.kspace.transforms import fft2c, ifft2c
from flow_interpolation.utils.visualization import concat_with_gap, write_video


def log_magnitude(
    spectrum: torch.Tensor,
    *,
    reference: float,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Map complex k-space to a displayable ``[N, 3, H, W]`` grayscale image.

    Magnitudes are averaged over color channels — per-channel k-space carries no
    structure the eye can use — and compressed as ``log1p(|K| / (eps * ref))``,
    normalized so ``|K| = reference`` maps to white. The default ``epsilon``
    gives roughly 60 dB of visible dynamic range, enough to see both the DC peak
    and the sampling pattern out at the corners.
    """
    if reference <= 0.0:
        raise ValueError("reference must be positive")
    magnitude = spectrum.abs()
    if magnitude.ndim == 4:
        magnitude = magnitude.mean(dim=1, keepdim=True)
    scaled = torch.log1p(magnitude / (epsilon * reference)) / math.log1p(1.0 / epsilon)
    return scaled.clamp(0.0, 1.0).expand(-1, 3, -1, -1)


def reconstruction_panels(
    ground_truth: torch.Tensor,
    observed_kspace: torch.Tensor,
    rendered: torch.Tensor,
    *,
    reference: Optional[float] = None,
    residual_scale: float = 4.0,
    display_scale: int = 1,
    gap: int = 2,
) -> torch.Tensor:
    """Build the 2x4 comparison panel for every frame.

    Args:
        ground_truth: ``[N, C, H, W]`` dense reference frames.
        observed_kspace: ``[N, C, H, W]`` complex, masked measurements.
        rendered: ``[N, C, H, W]`` scene output at the same times.
        reference: peak magnitude for the k-space color map; defaults to the
            ground-truth peak.
        residual_scale: gain applied to the image residual so small errors stay
            visible. The k-space residual uses the shared log map instead.

    Returns ``[N, 3, 2 * H', 4 * W']`` uint8-ready float frames in ``[0, 1]``.
    """
    if not (ground_truth.shape == rendered.shape == observed_kspace.shape):
        raise ValueError("ground_truth, observed_kspace, and rendered must share a shape")

    truth_kspace = fft2c(ground_truth)
    rendered_kspace = fft2c(rendered)
    if reference is None:
        reference = truth_kspace.abs().max().item()

    zero_filled = ifft2c(observed_kspace).abs()
    image_row = [
        ground_truth.clamp(0.0, 1.0),
        zero_filled.clamp(0.0, 1.0),
        rendered.clamp(0.0, 1.0),
        (rendered.clamp(0.0, 1.0) - ground_truth.clamp(0.0, 1.0))
        .abs()
        .mul(residual_scale)
        .clamp(0.0, 1.0),
    ]
    kspace_row = [
        log_magnitude(truth_kspace, reference=reference),
        log_magnitude(observed_kspace, reference=reference),
        log_magnitude(rendered_kspace, reference=reference),
        log_magnitude(rendered_kspace - truth_kspace, reference=reference),
    ]

    top = concat_with_gap(image_row, dim=-1, gap=gap)
    bottom = concat_with_gap(kspace_row, dim=-1, gap=gap)
    panel = concat_with_gap([top, bottom], dim=-2, gap=gap)
    if display_scale > 1:
        panel = panel.repeat_interleave(display_scale, dim=-2).repeat_interleave(
            display_scale, dim=-1
        )
    return panel


def panels_to_video_frames(panels: torch.Tensor) -> torch.Tensor:
    """Convert ``[N, 3, H, W]`` float panels to ``[N, H, W, 3]`` uint8 frames."""
    return panels.permute(0, 2, 3, 1).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)


@dataclass
class ReconstructionVisualizer:
    """Periodically write a comparison video over a short snippet of time.

    Motion is the thing being reconstructed, so progress is far easier to judge
    from a moving snippet than from stills: temporal artifacts — jitter, a
    lagging or smeared ball, the trail detaching — are obvious in motion and
    nearly invisible frame by frame.

    The snippet is a *fixed* contiguous window, the same one at every interval,
    so successive videos are directly comparable across the run. It defaults to
    the middle of the sequence, away from the one-sided bins at the boundaries.

    Follows the repo's callback convention: it is called every step and returns
    immediately unless ``call_every`` divides the step.
    """

    data: DynamicKSpaceData
    call_every: int = 2_000
    output_dir: Optional[Path] = None
    writer: Optional[SummaryWriter] = None
    snippet_frames: int = 24
    snippet_start: Optional[int] = None
    snippet_upsample: int = 1
    fps: float = 10.0
    residual_scale: float = 4.0
    display_scale: int = 4
    tensorboard_tag: str = "reconstruction/panels"
    _query_times: torch.Tensor = field(init=False)
    _source_indices: torch.Tensor = field(init=False)
    _reference: float = field(init=False)

    def __post_init__(self) -> None:
        if self.call_every < 0:
            raise ValueError("call_every must be non-negative")
        if self.snippet_frames < 1:
            raise ValueError("snippet_frames must be at least 1")
        if self.snippet_upsample < 1:
            raise ValueError("snippet_upsample must be at least 1")

        count = min(self.snippet_frames, self.data.num_frames)
        if self.snippet_start is None:
            start = max((self.data.num_frames - count) // 2, 0)
        else:
            start = min(max(self.snippet_start, 0), self.data.num_frames - count)
        indices = torch.arange(start, start + count)

        if self.snippet_upsample == 1:
            self._query_times = self.data.times.cpu()[indices]
            self._source_indices = indices
        else:
            # Render between observations to show what a continuous-time model
            # actually claims. The measured columns can only hold their nearest
            # observation, so between observation times the residual column
            # mixes reconstruction error with real motion; read it strictly on
            # the frames that were sampled.
            times = self.data.times.cpu()
            steps = (count - 1) * self.snippet_upsample + 1
            self._query_times = torch.linspace(times[indices[0]], times[indices[-1]], steps)
            nearest = (self._query_times.view(-1, 1) - times.view(1, -1)).abs().argmin(dim=1)
            self._source_indices = nearest

        self._reference = fft2c(self.data.frames.cpu()).abs().max().item()

    @property
    def playback_fps(self) -> float:
        """Frame rate that plays the snippet at the acquisition's real speed."""
        return self.fps * self.snippet_upsample

    def panels_for(self, rendered: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return reconstruction_panels(
            self.data.frames.cpu()[indices],
            self.data.kspace.cpu()[indices],
            rendered.cpu(),
            reference=self._reference,
            residual_scale=self.residual_scale,
            display_scale=self.display_scale,
        )

    def snippet_panels(self, fitter) -> torch.Tensor:
        rendered = fitter.render_sequence(self._query_times)
        return self.panels_for(rendered, self._source_indices)

    def __call__(self, step: int, fitter) -> None:
        if self.call_every <= 0 or step % self.call_every != 0:
            return
        panels = self.snippet_panels(fitter)

        if self.output_dir is not None:
            path = Path(self.output_dir) / f"snippet_{step:09d}.mp4"
            write_video(panels_to_video_frames(panels), str(path), fps=self.playback_fps)
        if self.writer is not None:
            # TensorBoard's add_video needs moviepy, which this project does not
            # depend on; the middle frame of the snippet keeps a scrubable
            # record in TensorBoard while the video itself lands on disk.
            self.writer.add_images(
                self.tensorboard_tag, panels[panels.shape[0] // 2].unsqueeze(0), step
            )
            self.writer.flush()

    def write_sequence_video(self, fitter, path: str | Path, chunk: int = 32) -> None:
        """Render every observation time as a panel and write the full video."""
        indices = torch.arange(self.data.num_frames)
        frames = []
        for start in range(0, indices.shape[0], chunk):
            batch = indices[start : start + chunk]
            rendered = fitter.render_sequence(self.data.times.cpu()[batch])
            frames.append(panels_to_video_frames(self.panels_for(rendered, batch)))
        write_video(torch.cat(frames, dim=0), str(path), fps=self.fps)
