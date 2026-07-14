from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from flow_interpolation.evaluation.common import (
    FlowSettings,
    decode_in_chunks,
    encode_in_chunks,
    image_metrics,
    make_boundary_noise,
    missing_mask,
    nearest_observed_timeline,
    print_noise_stats,
    save_json,
)
from flow_interpolation.evaluation.data import SequenceData
from flow_interpolation.evaluation.geometry import interpolate_keyframes
from flow_interpolation.evaluation.visualization import make_comparison_video_frames, write_video


def _temporal_metrics(frames: torch.Tensor) -> dict[str, float]:
    if frames.shape[0] < 2:
        return {"step_mae": 0.0, "acceleration_mae": 0.0}
    velocity = frames[1:] - frames[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    return {
        "step_mae": velocity.abs().mean().item(),
        "acceleration_mae": acceleration.abs().mean().item() if acceleration.numel() else 0.0,
    }


@torch.no_grad()
def run_latent_interpolation_evaluation(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    methods: list[str],
    slerp_mode: str,
    boundary_noise_mode: str,
    seed: int,
    output_dir: str,
    video_fps: float,
    display_scale: int,
    gap: int,
    residual_scale: float,
    save_tensors: bool,
) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    observed_device = sequence.observed_frames.to(device)
    eps_noise = make_boundary_noise(observed_device, boundary_noise_mode, generator=generator)
    keyframe_latents = encode_in_chunks(
        model,
        sequence.observed_frames,
        flow,
        device,
        eps_noise=eps_noise.cpu(),
        perturb=True,
        desc="Encoding observed keyframes",
    )
    print_noise_stats("Encoded keyframe latents", keyframe_latents)

    missing = missing_mask(sequence.num_frames, sequence.observed_indices)
    nearest = nearest_observed_timeline(
        sequence.observed_frames,
        sequence.observed_indices,
        sequence.num_frames,
    )
    predictions: OrderedDict[str, torch.Tensor] = OrderedDict()
    metrics: dict[str, dict] = {}
    latent_paths: dict[str, torch.Tensor] = {}

    for method in methods:
        latent_path = interpolate_keyframes(
            keyframe_latents,
            sequence.cadence.endpoint_stride,
            method,
            slerp_mode=slerp_mode,
        ).cpu()
        decoded_at_eps = decode_in_chunks(
            model,
            latent_path,
            flow,
            device,
            desc=f"Decoding {method} path",
        )
        prediction = decoded_at_eps.clamp(0.0, 1.0)
        predictions[method] = prediction
        latent_paths[method] = latent_path
        metrics[method] = {
            "all_frames": image_metrics(prediction, sequence.frames),
            "missing_frames": image_metrics(prediction[missing], sequence.frames[missing]),
            "observed_frames": image_metrics(
                prediction[sequence.observed_indices],
                sequence.observed_frames,
            ),
            "temporal": _temporal_metrics(prediction),
        }
        print(f"Latent interpolation [{method}]: {metrics[method]}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    frames = make_comparison_video_frames(
        sequence.frames,
        nearest,
        predictions,
        residual_scale=residual_scale,
        display_scale=display_scale,
        gap=gap,
    )
    video_path = output_root / "latent_interpolation.mp4"
    write_video(frames, str(video_path), fps=video_fps)

    payload = {
        "methods": methods,
        "slerp_mode": slerp_mode,
        "boundary_noise_mode": boundary_noise_mode,
        "cadence": sequence.cadence,
        "video_panel_order": [
            "ground_truth",
            "nearest_observed",
            *[item for method in methods for item in (method, f"{method}_absolute_residual")],
        ],
        "metrics": metrics,
    }
    save_json(payload, output_root / "latent_interpolation_metrics.json")
    if save_tensors:
        torch.save(
            {
                "frames": sequence.frames,
                "observed_indices": sequence.observed_indices,
                "keyframe_latents": keyframe_latents,
                "latent_paths": latent_paths,
                "predictions": dict(predictions),
            },
            output_root / "latent_interpolation_tensors.pt",
        )
        print(f"Saved interpolation tensors to {output_root / 'latent_interpolation_tensors.pt'}")
    return payload
