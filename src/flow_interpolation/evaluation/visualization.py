from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import torch


def residual_image(prediction: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    return (prediction.clamp(0.0, 1.0) - target.clamp(0.0, 1.0)).abs().mul(scale).clamp(0.0, 1.0)


def concat_with_gap(
    images: list[torch.Tensor],
    dim: int,
    gap: int,
    value: float = 0.06,
) -> torch.Tensor:
    if not images:
        raise ValueError("At least one image is required")
    if len(images) == 1 or gap <= 0:
        return torch.cat(images, dim=dim)
    output = images[0]
    for image in images[1:]:
        gap_shape = list(output.shape)
        gap_shape[dim] = gap
        gap_tensor = output.new_full(gap_shape, value)
        output = torch.cat([output, gap_tensor, image], dim=dim)
    return output


def make_comparison_video_frames(
    ground_truth: torch.Tensor,
    reference: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    *,
    residual_scale: float,
    display_scale: int,
    gap: int,
) -> torch.Tensor:
    """Create frames ordered as GT, reference, prediction, residual, ..."""
    ordered = OrderedDict(predictions)
    if any(value.shape != ground_truth.shape for value in ordered.values()):
        raise ValueError("Every prediction must have the same shape as ground_truth")
    if reference.shape != ground_truth.shape:
        raise ValueError("reference must have the same shape as ground_truth")

    frames: list[torch.Tensor] = []
    for index in range(ground_truth.shape[0]):
        panels = [ground_truth[index].clamp(0.0, 1.0), reference[index].clamp(0.0, 1.0)]
        for prediction in ordered.values():
            panels.extend(
                [
                    prediction[index].clamp(0.0, 1.0),
                    residual_image(prediction[index], ground_truth[index], residual_scale),
                ]
            )
        panel = concat_with_gap(panels, dim=-1, gap=gap)
        if display_scale > 1:
            panel = panel.repeat_interleave(display_scale, dim=-2).repeat_interleave(
                display_scale, dim=-1
            )
        frame = panel.permute(1, 2, 0).mul(255.0).round().to(torch.uint8)
        frames.append(frame)
    return torch.stack(frames, dim=0)


def write_video(frames: torch.Tensor, path: str, fps: float) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("Install imageio-ffmpeg to write MP4 files.") from error

    frames = frames.cpu().contiguous()
    height, width = frames.shape[1:3]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        writer = imageio_ffmpeg.write_frames(
            path,
            size=(width, height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        for frame in frames:
            writer.send(frame.numpy().tobytes())
    finally:
        if writer is not None:
            writer.close()
    print(f"Saved video to {path}")
