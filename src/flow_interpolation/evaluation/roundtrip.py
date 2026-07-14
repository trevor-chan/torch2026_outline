from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from flow_interpolation.evaluation.common import (
    FlowSettings,
    decode_in_chunks,
    encode_in_chunks,
    image_metrics,
    integrate_flow,
    make_boundary_noise,
    perturb_to_p_eps,
    predict_clean_and_noise,
    print_noise_stats,
    save_json,
    tensor_metrics,
)
from flow_interpolation.evaluation.data import SequenceData


def _distribution_stats(samples: torch.Tensor) -> dict[str, float]:
    """Summarize signal scale globally, per sample, and across the sample batch."""
    values = samples.detach().float()
    flat = values.flatten()
    per_sample = values.flatten(start_dim=1)

    global_std = flat.std(unbiased=False)
    global_var = global_std.square()
    global_rms = flat.square().mean().sqrt()
    mean_abs = flat.abs().mean()

    per_sample_std = per_sample.std(dim=1, unbiased=False)
    per_sample_rms = per_sample.square().mean(dim=1).sqrt()

    # Variance across samples at each channel/pixel location. This is useful for
    # diagnosing image-side regions that are effectively deterministic across a batch.
    if values.shape[0] > 1:
        pixelwise_std = values.std(dim=0, unbiased=False).flatten()
    else:
        pixelwise_std = torch.zeros(values[0].numel(), dtype=values.dtype)

    return {
        "mean": flat.mean().item(),
        "mean_abs": mean_abs.item(),
        "std": global_std.item(),
        "variance": global_var.item(),
        "rms": global_rms.item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "per_sample_std_mean": per_sample_std.mean().item(),
        "per_sample_std_min": per_sample_std.min().item(),
        "per_sample_std_max": per_sample_std.max().item(),
        "per_sample_rms_mean": per_sample_rms.mean().item(),
        "pixelwise_batch_std_mean": pixelwise_std.mean().item(),
        "pixelwise_batch_std_median": pixelwise_std.median().item(),
        "pixelwise_batch_std_max": pixelwise_std.max().item(),
        "pixelwise_batch_std_below_1e-4_fraction": (pixelwise_std < 1e-4).float().mean().item(),
        "pixelwise_batch_std_below_1e-3_fraction": (pixelwise_std < 1e-3).float().mean().item(),
        "pixelwise_batch_std_below_1e-2_fraction": (pixelwise_std < 1e-2).float().mean().item(),
    }


def _normalized_error_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, object]:
    """Return ordinary errors together with scale-normalized variants.

    RMSE/std and MSE/variance answer the user's variance-scaling hypothesis.
    RMSE/RMS is also reported because it remains meaningful for non-zero-mean
    signals and is equivalent to a global relative L2 error.
    """
    prediction = prediction.detach().float()
    target = target.detach().float()
    error = prediction - target
    base = tensor_metrics(prediction, target)
    target_stats = _distribution_stats(target)
    error_stats = _distribution_stats(error)

    eps = 1e-12
    target_std = max(float(target_stats["std"]), eps)
    target_variance = max(float(target_stats["variance"]), eps)
    target_rms = max(float(target_stats["rms"]), eps)
    target_mean_abs = max(float(target_stats["mean_abs"]), eps)

    target_flat = target.flatten(start_dim=1)
    error_flat = error.flatten(start_dim=1)
    sample_rmse = error_flat.square().mean(dim=1).sqrt()
    sample_std = target_flat.std(dim=1, unbiased=False).clamp_min(eps)
    sample_rms = target_flat.square().mean(dim=1).sqrt().clamp_min(eps)
    centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    centered_norm = centered.norm(dim=1).clamp_min(eps)

    normalized = {
        "rmse_over_target_std": float(base["rmse"]) / target_std,
        "mse_over_target_variance": float(base["mse"]) / target_variance,
        "rmse_over_target_rms": float(base["rmse"]) / target_rms,
        "mae_over_target_mean_abs": float(base["mae"]) / target_mean_abs,
        "per_sample_rmse_over_std_mean": (sample_rmse / sample_std).mean().item(),
        "per_sample_rmse_over_rms_mean": (sample_rmse / sample_rms).mean().item(),
        "per_sample_centered_relative_l2_mean": (
            error_flat.norm(dim=1) / centered_norm
        ).mean().item(),
    }
    return {
        "error": base,
        "normalized_error": normalized,
        "target_scale": target_stats,
        "error_scale": error_stats,
    }


def _max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach().float() - b.detach().float()).abs().max().item()


def _batch_difference_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Difference metrics that remain interpretable for a zero-valued reference."""
    base = image_metrics(value, reference)
    reference_rms = reference.detach().float().square().mean().sqrt().item()
    return {
        **base,
        "max_abs": _max_abs_difference(value, reference),
        "rmse_over_reference_rms": float(base["rmse"]) / max(reference_rms, 1e-12),
    }


@torch.no_grad()
def _integrate_in_chunks(
    model: torch.nn.Module,
    samples: torch.Tensor,
    *,
    device: torch.device,
    t_start: float,
    t_end: float,
    num_steps: int,
    solver: str,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    chunks: Iterable[torch.Tensor] = samples.split(batch_size)
    chunks = tqdm(chunks, total=math.ceil(samples.shape[0] / batch_size), desc=desc)
    for chunk in chunks:
        outputs.append(
            integrate_flow(
                model,
                chunk.to(device),
                t_start=t_start,
                t_end=t_end,
                num_steps=num_steps,
                solver=solver,
            ).cpu()
        )
    return torch.cat(outputs, dim=0)


def _depth_target_time(flow: FlowSettings, image_depth: float) -> float:
    """Map a fraction of the noise->image path to a model time.

    depth=0 is noise_time and depth=1 is the configured image-side boundary
    data_time. Thus 0.99 stops just before the final 1% of the integration path.
    """
    if not 0.0 < image_depth <= 1.0:
        raise ValueError("All image depths must satisfy 0 < depth <= 1")
    return flow.noise_time + image_depth * (flow.data_time - flow.noise_time)


def _steps_for_depth(flow: FlowSettings, image_depth: float) -> int:
    # Scale the number of steps with path length so the nominal |dt| remains
    # approximately constant across the boundary-depth ablation.
    return max(1, int(math.ceil(flow.ode_steps * image_depth)))


@torch.no_grad()
def _run_image_boundary_sweep(
    *,
    model: torch.nn.Module,
    device: torch.device,
    initial_noise: torch.Tensor,
    flow: FlowSettings,
    image_depths: list[float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for depth in sorted(set(float(value) for value in image_depths)):
        target_time = _depth_target_time(flow, depth)
        steps = _steps_for_depth(flow, depth)
        partial_state = _integrate_in_chunks(
            model,
            initial_noise,
            device=device,
            t_start=flow.noise_time,
            t_end=target_time,
            num_steps=steps,
            solver=flow.solver,
            batch_size=flow.decode_batch_size,
            desc=f"Boundary sweep {100.0 * depth:.3f}%: noise -> partial image",
        )
        cycled_noise = _integrate_in_chunks(
            model,
            partial_state,
            device=device,
            t_start=target_time,
            t_end=flow.noise_time,
            num_steps=steps,
            solver=flow.solver,
            batch_size=flow.encode_batch_size,
            desc=f"Boundary sweep {100.0 * depth:.3f}%: partial image -> noise",
        )
        recyled_partial_state = _integrate_in_chunks(
            model,
            cycled_noise,
            device=device,
            t_start=flow.noise_time,
            t_end=target_time,
            num_steps=steps,
            solver=flow.solver,
            batch_size=flow.decode_batch_size,
            desc=f"Boundary sweep {100.0 * depth:.3f}%: noise' -> partial image'",
        )
        rows.append(
            {
                "image_depth_fraction": depth,
                "image_depth_percent": 100.0 * depth,
                "target_time": target_time,
                "distance_to_configured_data_boundary": target_time - flow.data_time,
                "steps_each_direction": steps,
                "nominal_abs_dt": abs(flow.noise_time - target_time) / steps,
                "partial_state_stats": _distribution_stats(partial_state),
                "noise_endpoint_cycle": _normalized_error_metrics(
                    cycled_noise,
                    initial_noise,
                ),
                "partial_state_cycle": _normalized_error_metrics(
                    recyled_partial_state,
                    partial_state,
                ),
            }
        )
    return {
        "definition": (
            "image_depth_fraction is the fraction of the configured noise_time -> "
            "data_time integration interval traversed toward image space. A value "
            "of 1.0 reaches data_time, not mathematical t=0 unless data_time=0."
        ),
        "rows": rows,
    }


@torch.no_grad()
def _single_batch_direction_probe(
    *,
    model: torch.nn.Module,
    device: torch.device,
    pool: torch.Tensor,
    batch_sizes: list[int],
    t_start: float,
    t_end: float,
    num_steps: int,
    solver: str,
) -> dict[str, object]:
    if pool.shape[0] == 0:
        raise ValueError("Batch-consistency pool cannot be empty")

    anchor = pool[:1].to(device)
    t_batch = anchor.new_full((1,), t_start)
    velocity_alone = model(anchor, None, t_batch).cpu()
    forward_alone = integrate_flow(
        model,
        anchor,
        t_start=t_start,
        t_end=t_end,
        num_steps=num_steps,
        solver=solver,
    ).cpu()
    cycle_alone = integrate_flow(
        model,
        forward_alone.to(device),
        t_start=t_end,
        t_end=t_start,
        num_steps=num_steps,
        solver=solver,
    ).cpu()

    rows: list[dict[str, object]] = []
    for requested_batch_size in sorted(set(int(value) for value in batch_sizes)):
        if requested_batch_size <= 0:
            raise ValueError("All batch sizes must be positive")
        effective_batch_size = min(requested_batch_size, pool.shape[0])
        batch = pool[:effective_batch_size].to(device)
        t_batch = batch.new_full((effective_batch_size,), t_start)
        velocity_batch = model(batch, None, t_batch).cpu()[:1]
        full_forward_batch = integrate_flow(
            model,
            batch,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            solver=solver,
        )
        forward_batch = full_forward_batch.cpu()[:1]
        # Keep the complete batch on the return leg so batch composition is
        # retained in both integration directions.
        full_cycle_batch = integrate_flow(
            model,
            full_forward_batch,
            t_start=t_end,
            t_end=t_start,
            num_steps=num_steps,
            solver=solver,
        ).cpu()[:1]

        rows.append(
            {
                "requested_batch_size": requested_batch_size,
                "effective_batch_size": effective_batch_size,
                "initial_velocity_difference_vs_batch1": _batch_difference_metrics(
                    velocity_batch,
                    velocity_alone,
                ),
                "forward_endpoint_difference_vs_batch1": _batch_difference_metrics(
                    forward_batch,
                    forward_alone,
                ),
                "cycle_endpoint_difference_vs_batch1": _batch_difference_metrics(
                    full_cycle_batch,
                    cycle_alone,
                ),
            }
        )

    return {
        "anchor_input_stats": _distribution_stats(anchor.cpu()),
        "batch1_cycle_error_vs_input": _normalized_error_metrics(cycle_alone, anchor.cpu()),
        "rows": rows,
    }


@torch.no_grad()
def _run_batch_consistency_test(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    batch_sizes: list[int],
    boundary_noise_mode: str,
    seed: int,
) -> dict[str, object]:
    max_requested = max(batch_sizes)
    pool_count = min(max_requested, sequence.num_frames)
    indices = torch.linspace(0, sequence.num_frames - 1, pool_count).round().long().unique()
    clean_pool = sequence.frames[indices].to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    eps_noise = make_boundary_noise(clean_pool, boundary_noise_mode, generator=generator)
    data_pool = perturb_to_p_eps(clean_pool, flow.data_time, eps_noise).cpu()
    noise_pool = torch.randn(
        (indices.numel(), *sequence.frames.shape[1:]),
        generator=generator,
        device=device,
    ).cpu()

    return {
        "definition": (
            "The first sample is evaluated alone and as the first element of larger "
            "fixed batches. Differences should be zero up to numerical/kernel effects "
            "for a model with no cross-sample operations."
        ),
        "pool_indices": indices,
        "data_to_noise_to_data": _single_batch_direction_probe(
            model=model,
            device=device,
            pool=data_pool,
            batch_sizes=batch_sizes,
            t_start=flow.data_time,
            t_end=flow.noise_time,
            num_steps=flow.ode_steps,
            solver=flow.solver,
        ),
        "noise_to_data_to_noise": _single_batch_direction_probe(
            model=model,
            device=device,
            pool=noise_pool,
            batch_sizes=batch_sizes,
            t_start=flow.noise_time,
            t_end=flow.data_time,
            num_steps=flow.ode_steps,
            solver=flow.solver,
        ),
    }


@torch.no_grad()
def _run_optional_step_sweep(
    *,
    model: torch.nn.Module,
    device: torch.device,
    initial_noise: torch.Tensor,
    encoded_data_latents: torch.Tensor,
    flow: FlowSettings,
    step_counts: list[int] | None,
) -> dict[str, object] | None:
    if not step_counts:
        return None
    rows: list[dict[str, object]] = []
    for steps in sorted(set(int(value) for value in step_counts)):
        if steps <= 0:
            raise ValueError("All step-sweep counts must be positive")
        settings = replace(flow, ode_steps=steps)

        decoded_random = decode_in_chunks(
            model,
            initial_noise,
            settings,
            device,
            desc=f"Step sweep {steps}: random noise -> data",
        )
        reencoded_random = encode_in_chunks(
            model,
            decoded_random,
            settings,
            device,
            perturb=False,
            desc=f"Step sweep {steps}: data -> random noise'",
        )

        decoded_encoded = decode_in_chunks(
            model,
            encoded_data_latents,
            settings,
            device,
            desc=f"Step sweep {steps}: encoded latent -> data",
        )
        reencoded_encoded = encode_in_chunks(
            model,
            decoded_encoded,
            settings,
            device,
            perturb=False,
            desc=f"Step sweep {steps}: data -> encoded latent'",
        )
        rows.append(
            {
                "ode_steps": steps,
                "random_gaussian_latent_cycle": _normalized_error_metrics(
                    reencoded_random,
                    initial_noise,
                ),
                "encoded_data_latent_cycle": _normalized_error_metrics(
                    reencoded_encoded,
                    encoded_data_latents,
                ),
            }
        )
    return {
        "definition": (
            "Optional fixed-step convergence sweep. For a smooth field and Heun "
            "integration, cycle error should decrease approximately quadratically "
            "with step size before conditioning or precision limits dominate."
        ),
        "rows": rows,
    }


@torch.no_grad()
def run_roundtrip_evaluation(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    num_samples: int,
    boundary_noise_mode: str,
    seed: int,
    output_json: str,
    output_tensors: str | None = None,
    image_depths: list[float] | None = None,
    batch_sizes: list[int] | None = None,
    step_counts: list[int] | None = None,
) -> dict:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    image_depths = image_depths or [0.9, 0.99, 0.999, 1.0]
    batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32]

    sample_count = min(num_samples, sequence.num_frames)
    indices = torch.linspace(0, sequence.num_frames - 1, sample_count).round().long().unique()
    clean = sequence.frames[indices].to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    eps_noise = make_boundary_noise(clean, boundary_noise_mode, generator=generator)
    x_eps = perturb_to_p_eps(clean, flow.data_time, eps_noise)

    # data -> noise -> data': encode the exact p_eps sample and invert it.
    encoded = encode_in_chunks(
        model,
        x_eps.cpu(),
        flow,
        device,
        perturb=False,
        desc="Round trip: data -> noise",
    )
    reconstructed_eps = decode_in_chunks(
        model,
        encoded,
        flow,
        device,
        desc="Round trip: noise -> data",
    )
    reconstructed_clean, _, _ = predict_clean_and_noise(
        model,
        reconstructed_eps.to(device),
        flow.data_time,
    )
    reconstructed_clean = reconstructed_clean.cpu()

    # Explicitly test a latent known to be in the numerical image of the encoder.
    reencoded_data_latent = encode_in_chunks(
        model,
        reconstructed_eps,
        flow,
        device,
        perturb=False,
        desc="Round trip: decoded encoded-latent -> encoded-latent'",
    )

    # noise -> data -> noise': do not inject a second epsilon perturbation.
    initial_noise = torch.randn(
        (sample_count, *sequence.frames.shape[1:]),
        device=device,
        generator=generator,
    )
    decoded_from_noise = decode_in_chunks(
        model,
        initial_noise.cpu(),
        flow,
        device,
        desc="Round trip: Gaussian noise -> data",
    )
    reencoded_noise = encode_in_chunks(
        model,
        decoded_from_noise,
        flow,
        device,
        perturb=False,
        desc="Round trip: data -> Gaussian noise",
    )

    boundary_sweep = _run_image_boundary_sweep(
        model=model,
        device=device,
        initial_noise=initial_noise.cpu(),
        flow=flow,
        image_depths=image_depths,
    )
    batch_consistency = _run_batch_consistency_test(
        model=model,
        device=device,
        sequence=sequence,
        flow=flow,
        batch_sizes=batch_sizes,
        boundary_noise_mode=boundary_noise_mode,
        seed=seed + 1,
    )
    step_sweep = _run_optional_step_sweep(
        model=model,
        device=device,
        initial_noise=initial_noise.cpu(),
        encoded_data_latents=encoded,
        flow=flow,
        step_counts=step_counts,
    )

    payload: dict[str, object] = {
        "sample_indices": indices,
        "boundary_noise_mode": boundary_noise_mode,
        "flow": {
            "data_time": flow.data_time,
            "noise_time": flow.noise_time,
            "ode_steps": flow.ode_steps,
            "solver": flow.solver,
        },
        "data_to_noise_to_data": {
            "cycle_at_data_eps": _normalized_error_metrics(reconstructed_eps, x_eps.cpu()),
            "clean_endpoint_estimate": _normalized_error_metrics(
                reconstructed_clean,
                clean.cpu(),
            ),
            "decoded_eps_vs_clean": _normalized_error_metrics(
                reconstructed_eps,
                clean.cpu(),
            ),
            "input_data_eps_stats": _distribution_stats(x_eps.cpu()),
            "encoded_noise_stats": print_noise_stats("Encoded data anchors", encoded),
        },
        "encoded_data_latent_to_data_to_latent": {
            "cycle": _normalized_error_metrics(reencoded_data_latent, encoded),
            "initial_encoded_latent_stats": _distribution_stats(encoded),
            "reencoded_latent_stats": _distribution_stats(reencoded_data_latent),
        },
        "noise_to_data_to_noise": {
            "cycle": _normalized_error_metrics(reencoded_noise, initial_noise.cpu()),
            "decoded_image_side_stats": _distribution_stats(decoded_from_noise),
            "initial_noise_stats": print_noise_stats(
                "Initial Gaussian anchors",
                initial_noise.cpu(),
            ),
            "reencoded_noise_stats": print_noise_stats(
                "Re-encoded Gaussian anchors",
                reencoded_noise,
            ),
        },
        "image_boundary_sweep": boundary_sweep,
        "batch_consistency": batch_consistency,
    }
    if step_sweep is not None:
        payload["solver_step_sweep"] = step_sweep

    print("Data -> noise -> data cycle:", payload["data_to_noise_to_data"])
    print("Encoded latent -> data -> encoded latent cycle:", payload["encoded_data_latent_to_data_to_latent"])
    print("Noise -> data -> noise cycle:", payload["noise_to_data_to_noise"])
    save_json(payload, output_json)

    if output_tensors is not None:
        path = Path(output_tensors)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "indices": indices,
                "clean": clean.cpu(),
                "x_eps": x_eps.cpu(),
                "encoded": encoded,
                "reconstructed_eps": reconstructed_eps,
                "reconstructed_clean": reconstructed_clean,
                "reencoded_data_latent": reencoded_data_latent,
                "initial_noise": initial_noise.cpu(),
                "decoded_from_noise": decoded_from_noise,
                "reencoded_noise": reencoded_noise,
            },
            path,
        )
        print(f"Saved round-trip tensors to {path}")
    return payload
