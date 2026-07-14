from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from pathlib import Path
import torch
from tqdm import tqdm

from flow_interpolation.evaluation.common import (
    FlowSettings,
    decode_in_chunks,
    encode_in_chunks,
    image_metrics,
    make_boundary_noise,
    missing_mask,
    nearest_observed_timeline,
    predict_clean_and_noise,
    print_noise_stats,
    save_json,
)
from flow_interpolation.evaluation.data import SequenceData
from flow_interpolation.evaluation.geometry import global_slerp_noise, interpolate_keyframes, slerp_pair
from flow_interpolation.evaluation.latent import _temporal_metrics
from flow_interpolation.evaluation.visualization import make_comparison_video_frames, write_video


def bridge_envelope(
    num_frames: int,
    observed_indices: torch.Tensor,
    *,
    kind: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a scalar stochasticity envelope that vanishes at every keyframe.

    The envelope is built independently inside each observed-frame interval. Every
    supported profile is zero at s=0 and s=1 and has a maximum of approximately 1
    near the interval midpoint.
    """
    if kind not in {"sine", "brownian", "quadratic"}:
        raise ValueError(f"Unknown bridge envelope: {kind}")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    indices = observed_indices.to(device=device, dtype=torch.long)
    if indices.ndim != 1 or indices.numel() < 2:
        raise ValueError("At least two one-dimensional observed indices are required")
    if int(indices[0]) != 0 or int(indices[-1]) != num_frames - 1:
        raise ValueError("Observed indices must include the first and final frame")
    if not bool(torch.all(indices[1:] > indices[:-1])):
        raise ValueError("Observed indices must be strictly increasing")

    envelope = torch.zeros(num_frames, device=device, dtype=dtype)
    for left_tensor, right_tensor in zip(indices[:-1], indices[1:]):
        left = int(left_tensor)
        right = int(right_tensor)
        length = right - left
        s = torch.linspace(0.0, 1.0, length + 1, device=device, dtype=dtype)
        if kind == "sine":
            values = torch.sin(math.pi * s)
        elif kind == "brownian":
            values = 2.0 * torch.sqrt((s * (1.0 - s)).clamp_min(0.0))
        else:
            values = 4.0 * s * (1.0 - s)
        envelope[left : right + 1] = values

    envelope[indices] = 0.0
    return envelope.clamp(0.0, 1.0)


def sample_bridge_innovation(
    *,
    mode: str,
    num_frames: int,
    frame_shape: tuple[int, ...],
    observed_indices: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    slerp_mode: str,
) -> torch.Tensor:
    """Sample a temporally structured unit-variance innovation field.

    The returned field is later multiplied by ``bridge_envelope``. Therefore its
    values at keyframes do not affect the result. ``piecewise-slerp`` draws a new
    pair of Gaussian anchors inside every observed interval, which avoids coupling
    unrelated intervals while retaining smooth changes within each interval.
    """
    shape = (num_frames, *frame_shape)
    if mode == "independent":
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    if mode == "global-slerp":
        return global_slerp_noise(
            num_frames,
            frame_shape,
            device=device,
            dtype=dtype,
            generator=generator,
            mode=slerp_mode,
        )
    if mode not in {"piecewise-slerp", "segment-shared"}:
        raise ValueError(f"Unknown bridge innovation mode: {mode}")

    indices = observed_indices.to(device=device, dtype=torch.long)
    output = torch.empty(shape, device=device, dtype=dtype)
    for segment_index, (left_tensor, right_tensor) in enumerate(zip(indices[:-1], indices[1:])):
        left = int(left_tensor)
        right = int(right_tensor)
        length = right - left
        if mode == "segment-shared":
            value = torch.randn(frame_shape, device=device, dtype=dtype, generator=generator)
            segment = value.unsqueeze(0).expand(length + 1, *frame_shape)
        else:
            anchors = torch.randn(
                (2, *frame_shape),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            weights = torch.linspace(0.0, 1.0, length + 1, device=device, dtype=dtype)
            segment = slerp_pair(anchors[0], anchors[1], weights, mode=slerp_mode)
        if segment_index == 0:
            output[left : right + 1] = segment
        else:
            # The shared boundary is immaterial because its envelope is zero.
            output[left + 1 : right + 1] = segment[1:]
    return output


def _batch_radius_slerp(
    current: torch.Tensor,
    target: torch.Tensor,
    strength: float | torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical interpolation for one corresponding pair per batch element."""
    if current.shape != target.shape:
        raise ValueError(f"Shape mismatch: {current.shape} versus {target.shape}")
    batch = current.shape[0]
    current_flat = current.reshape(batch, -1)
    target_flat = target.reshape(batch, -1)
    current_radius = current_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    target_radius = target_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    current_unit = current_flat / current_radius
    target_unit = target_flat / target_radius
    cosine = (current_unit * target_unit).sum(dim=1, keepdim=True).clamp(
        -1.0 + 1e-7,
        1.0 - 1e-7,
    )
    angle = torch.acos(cosine)
    sin_angle = torch.sin(angle)

    weight = torch.as_tensor(strength, device=current.device, dtype=current.dtype)
    if weight.ndim == 0:
        weight = weight.expand(batch)
    if weight.shape != (batch,):
        raise ValueError(f"Expected scalar or [{batch}] strengths, got {tuple(weight.shape)}")
    weight = weight.clamp(0.0, 1.0).view(batch, 1)

    linear_direction = (1.0 - weight) * current_unit + weight * target_unit
    spherical_direction = (
        torch.sin((1.0 - weight) * angle) / sin_angle.clamp_min(eps) * current_unit
        + torch.sin(weight * angle) / sin_angle.clamp_min(eps) * target_unit
    )
    near_collinear = sin_angle.abs() < 1e-6
    direction = torch.where(near_collinear, linear_direction, spherical_direction)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(eps)
    radius = torch.lerp(current_radius, target_radius, weight)
    return (direction * radius).reshape_as(current)


def _blend_bridge_target(
    current: torch.Tensor,
    target: torch.Tensor,
    strength: float,
    mode: str,
) -> torch.Tensor:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("bridge strength must be in [0, 1]")
    if mode == "lerp":
        return torch.lerp(current, target, strength)
    if mode == "slerp":
        return _batch_radius_slerp(current, target, strength)
    raise ValueError(f"Unknown bridge blend mode: {mode}")


def _variance_preserving_residual_mix(
    center: torch.Tensor,
    innovation: torch.Tensor,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    """Mix a bridge center and innovation with per-frame amplitudes in [0, 1]."""
    if center.shape != innovation.shape:
        raise ValueError(f"Shape mismatch: {center.shape} versus {innovation.shape}")
    if amplitude.shape != (center.shape[0],):
        raise ValueError(
            f"Expected one amplitude per frame ({center.shape[0]}), got {tuple(amplitude.shape)}"
        )
    view = amplitude.clamp(0.0, 1.0).view(center.shape[0], *([1] * (center.ndim - 1)))
    return torch.sqrt((1.0 - view.square()).clamp_min(0.0)) * center + view * innovation


def _project_observations(
    x0_hat: torch.Tensor,
    sequence: SequenceData,
    strength: float,
) -> torch.Tensor:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("data-consistency strength must be in [0, 1]")
    output = x0_hat.clone()
    indices = sequence.observed_indices.to(x0_hat.device)
    observations = sequence.observed_frames.to(x0_hat.device, x0_hat.dtype)
    output[indices] = torch.lerp(output[indices], observations, strength)
    return output


@torch.no_grad()
def _predict_endpoints_in_chunks(
    model: torch.nn.Module,
    states: torch.Tensor,
    t: float,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean: list[torch.Tensor] = []
    noise: list[torch.Tensor] = []
    for chunk in tqdm(states.split(batch_size), desc=desc):
        x0_hat, z_hat, _ = predict_clean_and_noise(model, chunk.to(device), t)
        clean.append(x0_hat.cpu())
        noise.append(z_hat.cpu())
    return torch.cat(clean, dim=0), torch.cat(noise, dim=0)


@torch.no_grad()
def build_squad_bridge(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    boundary_noise_mode: str,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Encode keyframes and construct the deterministic SQUAD reference bridge."""
    generator = torch.Generator(device=device).manual_seed(seed)
    observed_device = sequence.observed_frames.to(device)
    eps_noise = make_boundary_noise(observed_device, boundary_noise_mode, generator=generator)
    keyframe_states = encode_in_chunks(
        model,
        sequence.observed_frames,
        flow,
        device,
        eps_noise=eps_noise.cpu(),
        perturb=True,
        desc="Encoding bridge keyframes",
    )
    print_noise_stats("Encoded bridge keyframes", keyframe_states)
    bridge_states = interpolate_keyframes(
        keyframe_states,
        sequence.cadence.endpoint_stride,
        "squad",
    ).cpu()
    if bridge_states.shape[0] != sequence.num_frames:
        raise RuntimeError(
            f"SQUAD path has {bridge_states.shape[0]} frames; expected {sequence.num_frames}"
        )
    bridge_x0_hat, bridge_z_hat = _predict_endpoints_in_chunks(
        model,
        bridge_states,
        flow.noise_time,
        flow.decode_batch_size,
        device,
        "Estimating SQUAD bridge endpoints",
    )
    deterministic = decode_in_chunks(
        model,
        bridge_states,
        flow,
        device,
        desc="Decoding deterministic SQUAD baseline",
    ).clamp(0.0, 1.0)
    return {
        "keyframe_states": keyframe_states,
        "bridge_states": bridge_states,
        "bridge_x0_hat": bridge_x0_hat,
        "bridge_z_hat": bridge_z_hat,
        "deterministic": deterministic,
    }


@torch.no_grad()
def sample_endpoint_bridge(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    bridge_states: torch.Tensor,
    bridge_x0_hat: torch.Tensor,
    bridge_z_hat: torch.Tensor,
    sampler: str,
    stochasticity: float,
    innovation_mode: str,
    envelope_kind: str,
    bridge_strength: float,
    bridge_power: float,
    noise_power: float,
    bridge_blend: str,
    noise_refresh: str,
    dc_strength: float,
    slerp_mode: str,
    seed: int,
    clip_x0: bool,
) -> torch.Tensor:
    """Sample around an endpoint-conditioned SQUAD bridge.

    ``sampler='init'`` adds one variance-preserving bridge residual at the terminal
    state and then follows the ordinary deterministic ODE.

    ``sampler='iterative'`` begins from the same initialization, but at every
    reverse step it also pulls the model's inferred terminal endpoint ``z_hat``
    toward the fixed SQUAD terminal-noise bridge before re-composing ``x_t``. The
    guidance and stochastic residual both decay toward the data boundary.
    """
    if sampler not in {"init", "iterative"}:
        raise ValueError(f"Unknown endpoint-bridge sampler: {sampler}")
    if not 0.0 <= stochasticity <= 1.0:
        raise ValueError("stochasticity must be in [0, 1]")
    if bridge_power < 0.0 or noise_power < 0.0:
        raise ValueError("bridge_power and noise_power must be non-negative")
    if noise_refresh not in {"fixed", "fresh"}:
        raise ValueError("noise_refresh must be fixed or fresh")

    bridge_states = bridge_states.to(device)
    bridge_x0_hat = bridge_x0_hat.to(device)
    bridge_z_hat = bridge_z_hat.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    frame_shape = tuple(bridge_states.shape[1:])
    envelope = bridge_envelope(
        sequence.num_frames,
        sequence.observed_indices,
        kind=envelope_kind,
        device=device,
        dtype=bridge_states.dtype,
    )

    fixed_innovation = sample_bridge_innovation(
        mode=innovation_mode,
        num_frames=sequence.num_frames,
        frame_shape=frame_shape,
        observed_indices=sequence.observed_indices,
        device=device,
        dtype=bridge_states.dtype,
        generator=generator,
        slerp_mode=slerp_mode,
    )
    initial_amplitude = stochasticity * envelope
    z_initial = _variance_preserving_residual_mix(
        bridge_z_hat,
        fixed_innovation,
        initial_amplitude,
    )
    # This exactly reconstructs bridge_states at stochasticity=0 because x0_hat
    # and z_hat were inferred from the same state and time.
    x_t = (1.0 - flow.noise_time) * bridge_x0_hat + flow.noise_time * z_initial

    if sampler == "init":
        prediction = decode_in_chunks(
            model,
            x_t.cpu(),
            flow,
            device,
            desc=f"Decoding stochastic SQUAD initialization [eta={stochasticity:.3f}]",
        )
        prediction = _project_observations(prediction.to(device), sequence, dc_strength)
        return prediction.cpu().clamp(0.0, 1.0)

    times = torch.linspace(
        flow.noise_time,
        flow.data_time,
        flow.ode_steps + 1,
        device=device,
        dtype=x_t.dtype,
    )
    denominator = max(flow.noise_time - flow.data_time, 1e-12)
    for t_curr, t_next in tqdm(
        zip(times[:-1], times[1:]),
        total=flow.ode_steps,
        desc=f"Iterative endpoint bridge [eta={stochasticity:.3f}]",
    ):
        # First take an ordinary deterministic ODE step. Re-estimating x0/z at
        # t_next and re-composing them is exactly identity when both bridge
        # guidance and stochasticity are zero, so this branch respects the
        # selected Euler/Heun solver rather than silently falling back to Euler.
        _, _, v_curr = predict_clean_and_noise(model, x_t, t_curr)
        dt = t_next - t_curr
        if flow.solver == "euler":
            x_proposal = x_t + dt * v_curr
        elif flow.solver == "heun":
            x_euler = x_t + dt * v_curr
            _, _, v_next = predict_clean_and_noise(model, x_euler, t_next)
            x_proposal = x_t + 0.5 * dt * (v_curr + v_next)
        else:
            raise ValueError(f"Unknown ODE solver: {flow.solver}")

        x0_hat, z_hat, _ = predict_clean_and_noise(model, x_proposal, t_next)
        if clip_x0:
            x0_hat = x0_hat.clamp(0.0, 1.0)
        x0_dc = _project_observations(x0_hat, sequence, dc_strength)

        normalized_time = float(((t_next - flow.data_time) / denominator).clamp(0.0, 1.0))
        current_bridge_strength = bridge_strength * normalized_time**bridge_power
        z_center = _blend_bridge_target(
            z_hat,
            bridge_z_hat,
            current_bridge_strength,
            bridge_blend,
        )

        if noise_refresh == "fresh":
            innovation = sample_bridge_innovation(
                mode=innovation_mode,
                num_frames=sequence.num_frames,
                frame_shape=frame_shape,
                observed_indices=sequence.observed_indices,
                device=device,
                dtype=x_t.dtype,
                generator=generator,
                slerp_mode=slerp_mode,
            )
        else:
            innovation = fixed_innovation
        amplitude = stochasticity * normalized_time**noise_power * envelope
        z_next = _variance_preserving_residual_mix(z_center, innovation, amplitude)
        x_t = (1.0 - t_next) * x0_dc + t_next * z_next

    x0_final, _, _ = predict_clean_and_noise(model, x_t, flow.data_time)
    prediction = _project_observations(x0_final, sequence, dc_strength)
    return prediction.cpu().clamp(0.0, 1.0)


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


def _aggregate_numbers(items: list[dict]) -> dict:
    if not items:
        return {}
    output: dict = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], dict):
            output[key] = _aggregate_numbers(values)
        else:
            tensor = torch.tensor(values, dtype=torch.float64)
            output[key] = {
                "mean": tensor.mean().item(),
                "std": tensor.std(unbiased=False).item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
            }
    return output


def _diversity_metrics(
    samples: torch.Tensor,
    deterministic: torch.Tensor,
    missing: torch.Tensor,
) -> dict[str, float]:
    # samples: [S, T, C, H, W]
    missing_samples = samples[:, missing]
    pixel_std = missing_samples.float().std(dim=0, unbiased=False).mean().item()
    deviation = samples[:, missing] - deterministic[None, missing]
    deviation_rmse = deviation.float().square().mean(dim=(1, 2, 3, 4)).sqrt().mean().item()
    pairwise: list[float] = []
    for left, right in itertools.combinations(range(samples.shape[0]), 2):
        difference = missing_samples[left].float() - missing_samples[right].float()
        pairwise.append(difference.square().mean().sqrt().item())
    return {
        "missing_pixel_std": pixel_std,
        "missing_rmse_from_deterministic_squad": deviation_rmse,
        "missing_pairwise_rmse": sum(pairwise) / len(pairwise) if pairwise else 0.0,
    }


@torch.no_grad()
def run_endpoint_bridge_evaluation(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    samplers: list[str],
    stochasticities: list[float],
    num_samples: int,
    innovation_mode: str,
    envelope_kind: str,
    bridge_strength: float,
    bridge_power: float,
    noise_power: float,
    bridge_blend: str,
    noise_refresh: str,
    dc_strength: float,
    slerp_mode: str,
    boundary_noise_mode: str,
    seed: int,
    clip_x0: bool,
    output_dir: str,
    video_fps: float,
    display_scale: int,
    gap: int,
    residual_scale: float,
    save_tensors: bool,
) -> dict:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not stochasticities:
        raise ValueError("At least one stochasticity value is required")

    bridge = build_squad_bridge(
        model=model,
        device=device,
        sequence=sequence,
        flow=flow,
        boundary_noise_mode=boundary_noise_mode,
        seed=seed,
    )
    deterministic = bridge["deterministic"]
    missing = missing_mask(sequence.num_frames, sequence.observed_indices)
    baseline_metrics = _prediction_metrics(deterministic, sequence, missing)
    print(f"Endpoint bridge [deterministic_squad]: {baseline_metrics}")

    visual_predictions: OrderedDict[str, torch.Tensor] = OrderedDict()
    visual_predictions["deterministic_squad"] = deterministic
    all_samples: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict] = {}

    experiment_index = 0
    for sampler in samplers:
        for stochasticity in stochasticities:
            label = f"{sampler}_eta{stochasticity:.3f}"
            predictions: list[torch.Tensor] = []
            sample_metrics: list[dict] = []
            for sample_index in range(num_samples):
                prediction = sample_endpoint_bridge(
                    model=model,
                    device=device,
                    sequence=sequence,
                    flow=flow,
                    bridge_states=bridge["bridge_states"],
                    bridge_x0_hat=bridge["bridge_x0_hat"],
                    bridge_z_hat=bridge["bridge_z_hat"],
                    sampler=sampler,
                    stochasticity=stochasticity,
                    innovation_mode=innovation_mode,
                    envelope_kind=envelope_kind,
                    bridge_strength=bridge_strength,
                    bridge_power=bridge_power,
                    noise_power=noise_power,
                    bridge_blend=bridge_blend,
                    noise_refresh=noise_refresh,
                    dc_strength=dc_strength,
                    slerp_mode=slerp_mode,
                    seed=seed + 10_000 * (experiment_index + 1) + sample_index,
                    clip_x0=clip_x0,
                )
                predictions.append(prediction)
                sample_metrics.append(_prediction_metrics(prediction, sequence, missing))
            stacked = torch.stack(predictions, dim=0)
            all_samples[label] = stacked
            visual_predictions[label] = stacked[0]
            metrics[label] = {
                "aggregate": _aggregate_numbers(sample_metrics),
                "diversity": _diversity_metrics(stacked, deterministic, missing),
                "per_sample": sample_metrics,
            }
            print(f"Endpoint bridge [{label}]: {metrics[label]['aggregate']}")
            experiment_index += 1

    nearest = nearest_observed_timeline(
        sequence.observed_frames,
        sequence.observed_indices,
        sequence.num_frames,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    video_frames = make_comparison_video_frames(
        sequence.frames,
        nearest,
        visual_predictions,
        residual_scale=residual_scale,
        display_scale=display_scale,
        gap=gap,
    )
    write_video(video_frames, str(output_root / "endpoint_bridge.mp4"), fps=video_fps)

    payload = {
        "samplers": samplers,
        "stochasticities": stochasticities,
        "num_samples": num_samples,
        "innovation_mode": innovation_mode,
        "envelope": envelope_kind,
        "bridge_strength": bridge_strength,
        "bridge_power": bridge_power,
        "noise_power": noise_power,
        "bridge_blend": bridge_blend,
        "noise_refresh": noise_refresh,
        "data_consistency_strength": dc_strength,
        "slerp_mode": slerp_mode,
        "boundary_noise_mode": boundary_noise_mode,
        "clip_x0": clip_x0,
        "cadence": sequence.cadence,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "video_panel_order": [
            "ground_truth",
            "nearest_observed",
            *[
                item
                for label in visual_predictions
                for item in (label, f"{label}_absolute_residual")
            ],
        ],
        "notes": {
            "init": "One stochastic perturbation around the SQUAD terminal bridge, followed by deterministic ODE decoding.",
            "iterative": "Repeated terminal-noise guidance toward the SQUAD bridge with time-decayed stochastic residuals.",
            "video_samples": "The video shows sample 0 for each stochastic configuration.",
        },
    }
    save_json(payload, output_root / "endpoint_bridge_metrics.json")
    if save_tensors:
        torch.save(
            {
                "frames": sequence.frames,
                "observed_indices": sequence.observed_indices,
                **bridge,
                "samples": all_samples,
            },
            output_root / "endpoint_bridge_tensors.pt",
        )
        print(f"Saved endpoint-bridge tensors to {output_root / 'endpoint_bridge_tensors.pt'}")
    return payload
