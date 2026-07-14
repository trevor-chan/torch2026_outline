from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from eval_common import (
    FlowSettings,
    encode_in_chunks,
    image_metrics,
    integrate_flow,
    make_boundary_noise,
    missing_mask,
    nearest_observed_timeline,
    print_noise_stats,
    save_json,
    tensor_metrics,
)
from eval_data import SequenceData
from eval_geometry import interpolate_keyframes
from eval_visualization import make_comparison_video_frames, write_video


IMAGE_INTERPOLATION_METHODS = {"linear", "smoothstep", "catmull-rom"}


def _temporal_metrics(frames: torch.Tensor) -> dict[str, float]:
    if frames.shape[0] < 2:
        return {"step_mae": 0.0, "acceleration_mae": 0.0}
    velocity = frames[1:] - frames[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    return {
        "step_mae": velocity.abs().mean().item(),
        "acceleration_mae": acceleration.abs().mean().item() if acceleration.numel() else 0.0,
    }


def _prediction_metrics(
    prediction: torch.Tensor,
    sequence: SequenceData,
    missing: torch.Tensor,
) -> dict[str, dict[str, float]]:
    return {
        "all_frames": image_metrics(prediction, sequence.frames),
        "missing_frames": image_metrics(prediction[missing], sequence.frames[missing]),
        "observed_frames": image_metrics(
            prediction[sequence.observed_indices],
            sequence.observed_frames,
        ),
        "temporal": _temporal_metrics(prediction),
    }


def _segment_weights(
    samples_per_segment: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be positive")
    return torch.linspace(
        0.0,
        1.0,
        samples_per_segment + 1,
        device=device,
        dtype=dtype,
    )


def interpolate_images(
    keyframes: torch.Tensor,
    samples_per_segment: int,
    method: str,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """Interpolate image keyframes on the dense evaluation timeline.

    ``linear`` is ordinary pixelwise interpolation. ``smoothstep`` uses the same
    endpoint values but replaces the segment coordinate ``u`` by
    ``u^2 (3 - 2u)``, giving zero endpoint velocity within each segment.
    ``catmull-rom`` is a four-point cubic with duplicated end controls.
    """
    if method not in IMAGE_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown image interpolation method {method!r}; "
            f"expected one of {sorted(IMAGE_INTERPOLATION_METHODS)}"
        )
    if keyframes.shape[0] < 2:
        raise ValueError("At least two image keyframes are required")

    weights = _segment_weights(
        samples_per_segment,
        device=keyframes.device,
        dtype=keyframes.dtype,
    )
    view = weights.view(-1, *([1] * (keyframes.ndim - 1)))
    pieces: list[torch.Tensor] = []

    for index in range(keyframes.shape[0] - 1):
        left = keyframes[index]
        right = keyframes[index + 1]

        if method == "linear":
            segment = torch.lerp(left, right, view)
        elif method == "smoothstep":
            smooth = view.square() * (3.0 - 2.0 * view)
            segment = torch.lerp(left, right, smooth)
        else:
            p0 = keyframes[max(index - 1, 0)]
            p1 = left
            p2 = right
            p3 = keyframes[min(index + 2, keyframes.shape[0] - 1)]
            u = view
            u2 = u.square()
            u3 = u2 * u
            segment = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * u
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
            )

        if index > 0:
            segment = segment[1:]
        pieces.append(segment)

    output = torch.cat(pieces, dim=0)
    return output.clamp(0.0, 1.0) if clamp else output


def compose_hybrid_state(
    image_path: torch.Tensor,
    noise_path: torch.Tensor,
    mix_time: float,
) -> torch.Tensor:
    """Compose x_t=(1-t)x_image+t z_squad at a chosen RF time."""
    if image_path.shape != noise_path.shape:
        raise ValueError(
            f"Image and noise paths must have matching shapes; got "
            f"{image_path.shape} and {noise_path.shape}"
        )
    if not 0.0 <= mix_time <= 1.0:
        raise ValueError("mix_time must lie in [0, 1]")
    return (1.0 - mix_time) * image_path + mix_time * noise_path


def _scaled_step_count(flow: FlowSettings, t_start: float) -> int:
    """Preserve the nominal full-path step size for a partial decode."""
    if t_start < flow.data_time:
        raise ValueError("t_start must be at or above flow.data_time")
    full_distance = flow.noise_time - flow.data_time
    remaining_distance = t_start - flow.data_time
    if remaining_distance <= 0.0:
        return 0
    return max(1, int(math.ceil(flow.ode_steps * remaining_distance / full_distance)))


@torch.no_grad()
def decode_from_time_in_chunks(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    t_start: float,
    flow: FlowSettings,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, int]:
    """Decode arbitrary states from ``t_start`` to the configured data boundary."""
    num_steps = _scaled_step_count(flow, t_start)
    if num_steps == 0:
        return states.clone().cpu(), 0

    chunks: list[torch.Tensor] = []
    for chunk in tqdm(states.split(flow.decode_batch_size), desc=desc):
        decoded = integrate_flow(
            model,
            chunk.to(device),
            t_start=t_start,
            t_end=flow.data_time,
            num_steps=num_steps,
            solver=flow.solver,
        )
        chunks.append(decoded.cpu())
    return torch.cat(chunks, dim=0), num_steps


@torch.no_grad()
def _squad_state_at_time(
    model: torch.nn.Module,
    squad_latents: torch.Tensor,
    *,
    t_target: float,
    flow: FlowSettings,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Follow the ordinary SQUAD ODE trajectory to an intermediate time."""
    if math.isclose(t_target, flow.noise_time, rel_tol=0.0, abs_tol=1e-12):
        return squad_latents.clone().cpu(), 0
    if not flow.data_time <= t_target <= flow.noise_time:
        raise ValueError("t_target must be inside the configured flow interval")

    full_distance = flow.noise_time - flow.data_time
    distance = flow.noise_time - t_target
    num_steps = max(1, int(math.ceil(flow.ode_steps * distance / full_distance)))
    chunks: list[torch.Tensor] = []
    for chunk in tqdm(
        squad_latents.split(flow.decode_batch_size),
        desc=f"Tracing SQUAD path to t={t_target:.4f}",
    ):
        state = integrate_flow(
            model,
            chunk.to(device),
            t_start=flow.noise_time,
            t_end=t_target,
            num_steps=num_steps,
            solver=flow.solver,
        )
        chunks.append(state.cpu())
    return torch.cat(chunks, dim=0), num_steps


def _format_mix_time(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


@torch.no_grad()
def run_hybrid_latent_interpolation_evaluation(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    mix_times: Iterable[float],
    image_methods: list[str],
    slerp_mode: str,
    boundary_noise_mode: str,
    seed: int,
    hard_keyframes: bool,
    clamp_image_interpolation: bool,
    compare_start_states: bool,
    output_dir: str,
    video_fps: float,
    display_scale: int,
    gap: int,
    residual_scale: float,
    save_tensors: bool,
) -> dict:
    """Evaluate image/noise hybrid interpolation.

    Observed frames are encoded and connected with a SQUAD path in terminal-noise
    space. Separately, the observed images are interpolated on the dense timeline.
    For each requested ``mix_time`` the sampler constructs

        x_t = (1 - t) * x_image_interpolated + t * z_squad

    and integrates the learned ODE from ``t`` to ``flow.data_time``.
    """
    mix_times = [float(value) for value in mix_times]
    if not mix_times:
        raise ValueError("At least one mix time is required")
    for value in mix_times:
        if not flow.data_time <= value <= flow.noise_time:
            raise ValueError(
                f"mix_time={value} is outside the configured interval "
                f"[{flow.data_time}, {flow.noise_time}]"
            )
    unknown_methods = set(image_methods) - IMAGE_INTERPOLATION_METHODS
    if unknown_methods:
        raise ValueError(f"Unknown image interpolation methods: {sorted(unknown_methods)}")

    generator = torch.Generator(device=device).manual_seed(seed)
    observed_device = sequence.observed_frames.to(device)
    eps_noise = make_boundary_noise(
        observed_device,
        boundary_noise_mode,
        generator=generator,
    )
    keyframe_latents = encode_in_chunks(
        model,
        sequence.observed_frames,
        flow,
        device,
        eps_noise=eps_noise.cpu(),
        perturb=True,
        desc="Encoding hybrid keyframes",
    )
    keyframe_noise_stats = print_noise_stats("Encoded hybrid keyframes", keyframe_latents)

    squad_latents = interpolate_keyframes(
        keyframe_latents,
        sequence.cadence.endpoint_stride,
        "squad",
        slerp_mode=slerp_mode,
    ).cpu()
    if squad_latents.shape[0] != sequence.num_frames:
        raise RuntimeError(
            f"SQUAD path has {squad_latents.shape[0]} frames; expected {sequence.num_frames}"
        )

    deterministic_squad, squad_decode_steps = decode_from_time_in_chunks(
        model,
        squad_latents,
        t_start=flow.noise_time,
        flow=flow,
        device=device,
        desc="Decoding deterministic SQUAD baseline",
    )
    deterministic_squad = deterministic_squad.clamp(0.0, 1.0)

    missing = missing_mask(sequence.num_frames, sequence.observed_indices)
    nearest = nearest_observed_timeline(
        sequence.observed_frames,
        sequence.observed_indices,
        sequence.num_frames,
    )

    predictions: OrderedDict[str, torch.Tensor] = OrderedDict()
    predictions["deterministic_squad"] = deterministic_squad
    metrics: dict[str, dict] = {
        "deterministic_squad": _prediction_metrics(
            deterministic_squad,
            sequence,
            missing,
        )
    }
    image_paths: dict[str, torch.Tensor] = {}
    hybrid_states: dict[str, torch.Tensor] = {}
    start_state_metrics: dict[str, dict] = {}
    squad_states_at_mix: dict[str, torch.Tensor] = {}
    squad_trace_steps: dict[str, int] = {}

    for image_method in image_methods:
        image_path = interpolate_images(
            sequence.observed_frames,
            sequence.cadence.endpoint_stride,
            image_method,
            clamp=clamp_image_interpolation,
        ).cpu()
        if image_path.shape[0] != sequence.num_frames:
            raise RuntimeError(
                f"Image path has {image_path.shape[0]} frames; expected {sequence.num_frames}"
            )
        image_paths[image_method] = image_path
        image_name = f"image_{image_method.replace('-', '_')}"
        predictions[image_name] = image_path.clamp(0.0, 1.0)
        metrics[image_name] = _prediction_metrics(predictions[image_name], sequence, missing)
        print(f"Image interpolation [{image_method}]: {metrics[image_name]}")

        for mix_time in mix_times:
            time_key = _format_mix_time(mix_time)
            name = f"hybrid_{image_method.replace('-', '_')}_t{time_key}"
            hybrid_state = compose_hybrid_state(image_path, squad_latents, mix_time).cpu()
            hybrid_states[name] = hybrid_state

            decoded, remaining_steps = decode_from_time_in_chunks(
                model,
                hybrid_state,
                t_start=mix_time,
                flow=flow,
                device=device,
                desc=f"Decoding {name}",
            )
            prediction = decoded.clamp(0.0, 1.0)
            if hard_keyframes:
                prediction[sequence.observed_indices] = sequence.observed_frames
            predictions[name] = prediction

            method_metrics = _prediction_metrics(prediction, sequence, missing)
            method_metrics["mixing"] = {
                "mix_time": mix_time,
                "image_weight": 1.0 - mix_time,
                "noise_weight": mix_time,
                "remaining_ode_steps": remaining_steps,
            }
            method_metrics["difference_from_deterministic_squad"] = {
                "all_frames": image_metrics(prediction, deterministic_squad),
                "missing_frames": image_metrics(
                    prediction[missing],
                    deterministic_squad[missing],
                ),
            }

            if compare_start_states:
                if time_key not in squad_states_at_mix:
                    squad_state, trace_steps = _squad_state_at_time(
                        model,
                        squad_latents,
                        t_target=mix_time,
                        flow=flow,
                        device=device,
                    )
                    squad_states_at_mix[time_key] = squad_state
                    squad_trace_steps[time_key] = trace_steps
                reference_state = squad_states_at_mix[time_key]
                state_metrics = tensor_metrics(hybrid_state, reference_state)
                start_state_metrics[name] = {
                    "hybrid_vs_ordinary_squad_state": state_metrics,
                    "squad_trace_steps": squad_trace_steps[time_key],
                }
                method_metrics["start_state"] = start_state_metrics[name]

            metrics[name] = method_metrics
            print(f"Hybrid interpolation [{name}]: {method_metrics}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    video_frames = make_comparison_video_frames(
        sequence.frames,
        nearest,
        predictions,
        residual_scale=residual_scale,
        display_scale=display_scale,
        gap=gap,
    )
    video_path = output_root / "hybrid_latent_interpolation.mp4"
    write_video(video_frames, str(video_path), fps=video_fps)

    payload = {
        "definition": (
            "For each mix time t, construct x_t=(1-t)*x_image_interpolated+"
            "t*z_squad and integrate the rectified-flow ODE from t to data_time."
        ),
        "mix_times": mix_times,
        "image_methods": image_methods,
        "slerp_mode": slerp_mode,
        "boundary_noise_mode": boundary_noise_mode,
        "hard_keyframes": hard_keyframes,
        "clamp_image_interpolation": clamp_image_interpolation,
        "compare_start_states": compare_start_states,
        "flow": {
            "data_time": flow.data_time,
            "noise_time": flow.noise_time,
            "ode_steps": flow.ode_steps,
            "solver": flow.solver,
            "squad_baseline_decode_steps": squad_decode_steps,
        },
        "cadence": sequence.cadence,
        "keyframe_noise_stats": keyframe_noise_stats,
        "video_panel_order": [
            "ground_truth",
            "nearest_observed",
            *[
                item
                for prediction_name in predictions
                for item in (prediction_name, f"{prediction_name}_absolute_residual")
            ],
        ],
        "metrics": metrics,
    }
    save_json(payload, output_root / "hybrid_latent_interpolation_metrics.json")

    if save_tensors:
        tensor_payload = {
            "frames": sequence.frames,
            "observed_indices": sequence.observed_indices,
            "observed_frames": sequence.observed_frames,
            "keyframe_latents": keyframe_latents,
            "squad_latents": squad_latents,
            "image_paths": image_paths,
            "hybrid_states": hybrid_states,
            "predictions": dict(predictions),
        }
        if compare_start_states:
            tensor_payload["squad_states_at_mix"] = squad_states_at_mix
        tensor_path = output_root / "hybrid_latent_interpolation_tensors.pt"
        torch.save(tensor_payload, tensor_path)
        print(f"Saved hybrid interpolation tensors to {tensor_path}")

    return payload
