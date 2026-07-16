"""Deterministic latent interpolation experiment."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import torch

from flow_interpolation.data import SequenceData, missing_mask, nearest_observed_timeline
from flow_interpolation.utils.flow import (
    FlowSettings,
    decode_in_chunks,
    encode_in_chunks,
    make_boundary_noise,
    perturb_to_p_eps,
)
from flow_interpolation.utils.interpolation import interpolate_keyframes
from flow_interpolation.utils.metrics import image_metrics, print_noise_stats, save_json
from flow_interpolation.utils.visualization import make_comparison_video_frames, write_video


def _temporal_metrics(frames: torch.Tensor) -> dict[str, float]:
    if frames.shape[0] < 2:
        return {"step_mae": 0.0, "acceleration_mae": 0.0}
    velocity = frames[1:] - frames[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    return {
        "step_mae": velocity.abs().mean().item(),
        "acceleration_mae": acceleration.abs().mean().item() if acceleration.numel() else 0.0,
    }


def interpolation_time_from_tau(flow: FlowSettings, tau: float) -> float:
    """Map a [0,1] path fraction to the configured rectified-flow time interval."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1]")
    return flow.data_time + tau * (flow.noise_time - flow.data_time)


def _flow_at_tau(flow: FlowSettings, tau: float) -> tuple[FlowSettings | None, float, int]:
    interpolation_time = interpolation_time_from_tau(flow, tau)
    if tau == 0.0:
        return None, interpolation_time, 0
    if tau == 1.0:
        return flow, flow.noise_time, flow.ode_steps
    steps = max(1, int(math.ceil(flow.ode_steps * tau)))
    return (
        replace(flow, noise_time=interpolation_time, ode_steps=steps),
        interpolation_time,
        steps,
    )


@torch.no_grad()
def encode_keyframes_to_tau(
    model: torch.nn.Module,
    samples: torch.Tensor,
    flow: FlowSettings,
    device: torch.device,
    *,
    tau: float,
    eps_noise: torch.Tensor,
    desc: str,
) -> tuple[torch.Tensor, float, int]:
    """Move clean keyframes to the interpolation state selected by ``tau``."""
    partial_flow, interpolation_time, steps = _flow_at_tau(flow, tau)
    if partial_flow is None:
        states = perturb_to_p_eps(samples, flow.data_time, eps_noise)
        return states.cpu(), interpolation_time, steps
    states = encode_in_chunks(
        model,
        samples,
        partial_flow,
        device,
        eps_noise=eps_noise,
        perturb=True,
        desc=desc,
    )
    return states, interpolation_time, steps


@torch.no_grad()
def decode_from_tau(
    model: torch.nn.Module,
    states: torch.Tensor,
    flow: FlowSettings,
    device: torch.device,
    *,
    tau: float,
    desc: str,
) -> tuple[torch.Tensor, int]:
    """Decode interpolation states from the selected tau back to data_time."""
    partial_flow, _, steps = _flow_at_tau(flow, tau)
    if partial_flow is None:
        return states.clone().cpu(), steps
    return (
        decode_in_chunks(
            model,
            states,
            partial_flow,
            device,
            desc=desc,
        ),
        steps,
    )


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
    tau: float = 1.0,
) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    observed_device = sequence.observed_frames.to(device)
    eps_noise = make_boundary_noise(observed_device, boundary_noise_mode, generator=generator)
    keyframe_states, interpolation_time, encode_steps = encode_keyframes_to_tau(
        model,
        sequence.observed_frames,
        flow,
        device,
        tau=tau,
        eps_noise=eps_noise.cpu(),
        desc=f"Encoding observed keyframes to tau={tau:g}",
    )
    keyframe_state_stats = print_noise_stats(
        f"Encoded keyframe states at tau={tau:g}",
        keyframe_states,
    )

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
            keyframe_states,
            sequence.cadence.endpoint_stride,
            method,
            slerp_mode=slerp_mode,
        ).cpu()
        decoded_at_eps, decode_steps = decode_from_tau(
            model,
            latent_path,
            flow,
            device,
            tau=tau,
            desc=f"Decoding {method} path from tau={tau:g}",
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
            "transport": {
                "tau": tau,
                "interpolation_time": interpolation_time,
                "encode_steps": encode_steps,
                "decode_steps": decode_steps,
            },
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
        "tau": tau,
        "interpolation_time": interpolation_time,
        "definition": (
            "tau is the fraction of the configured data_time-to-noise_time ODE "
            "interval traversed before interpolation. tau=0 interpolates at the "
            "epsilon-perturbed data boundary; tau=1 interpolates at noise_time."
        ),
        "transport": {
            "data_time": flow.data_time,
            "noise_time": flow.noise_time,
            "interpolation_time": interpolation_time,
            "full_ode_steps": flow.ode_steps,
            "encode_steps": encode_steps,
            "decode_steps": encode_steps,
        },
        "keyframe_state_stats": keyframe_state_stats,
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
                "keyframe_states": keyframe_states,
                "keyframe_latents": keyframe_states,
                "latent_paths": latent_paths,
                "predictions": dict(predictions),
                "tau": tau,
                "interpolation_time": interpolation_time,
            },
            output_root / "latent_interpolation_tensors.pt",
        )
        print(f"Saved interpolation tensors to {output_root / 'latent_interpolation_tensors.pt'}")
    return payload
