from __future__ import annotations

import csv
from pathlib import Path

import torch

from flow_interpolation.evaluation.common import FlowSettings, encode_in_chunks, make_boundary_noise, missing_mask, save_json
from flow_interpolation.evaluation.data import SequenceData
from flow_interpolation.evaluation.geometry import interpolate_keyframes


def _per_frame_latent_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred = prediction.float().flatten(start_dim=1)
    true = target.float().flatten(start_dim=1)
    diff = pred - true
    true_norm = true.norm(dim=1).clamp_min(1e-12)
    pred_norm = pred.norm(dim=1).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(pred, true, dim=1).clamp(-1.0, 1.0)
    return {
        "relative_l2": diff.norm(dim=1) / true_norm,
        "rmse": diff.square().mean(dim=1).sqrt(),
        "cosine_similarity": cosine,
        "angle_degrees": torch.rad2deg(torch.acos(cosine)),
        "radius_relative_error": (pred_norm - true_norm).abs() / true_norm,
    }


def _summarize(metrics: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, float]:
    selected = {key: value[mask] for key, value in metrics.items()}
    return {
        f"{key}_mean": values.mean().item()
        for key, values in selected.items()
    } | {
        f"{key}_max": values.max().item()
        for key, values in selected.items()
    }


@torch.no_grad()
def run_latent_geodesic_evaluation(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    methods: list[str],
    slerp_mode: str,
    boundary_noise_mode: str,
    seed: int,
    output_json: str,
    output_csv: str,
    output_tensors: str | None = None,
) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    frames_device = sequence.frames.to(device)
    eps_noise = make_boundary_noise(frames_device, boundary_noise_mode, generator=generator)
    ground_truth_latents = encode_in_chunks(
        model,
        sequence.frames,
        flow,
        device,
        eps_noise=eps_noise.cpu(),
        perturb=True,
        desc="Encoding full ground-truth trajectory",
    )
    keyframe_latents = ground_truth_latents[sequence.observed_indices]
    missing = missing_mask(sequence.num_frames, sequence.observed_indices)
    all_mask = torch.ones(sequence.num_frames, dtype=torch.bool)

    predictions: dict[str, torch.Tensor] = {}
    per_frame: dict[str, dict[str, torch.Tensor]] = {}
    summary: dict[str, dict] = {}
    for method in methods:
        prediction = interpolate_keyframes(
            keyframe_latents,
            sequence.cadence.endpoint_stride,
            method,
            slerp_mode=slerp_mode,
        ).cpu()
        if prediction.shape != ground_truth_latents.shape:
            raise RuntimeError(
                f"{method} produced {prediction.shape}, expected {ground_truth_latents.shape}"
            )
        metrics = _per_frame_latent_metrics(prediction, ground_truth_latents)
        predictions[method] = prediction
        per_frame[method] = metrics
        summary[method] = {
            "all_frames": _summarize(metrics, all_mask),
            "missing_frames": _summarize(metrics, missing),
        }
        print(f"Latent geodesic error [{method}]: {summary[method]}")

    payload = {
        "methods": methods,
        "slerp_mode": slerp_mode,
        "boundary_noise_mode": boundary_noise_mode,
        "cadence": sequence.cadence,
        "summary": summary,
    }
    save_json(payload, output_json)

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_index", "is_observed", "method", *next(iter(per_frame.values())).keys()]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        observed_set = set(sequence.observed_indices.tolist())
        for method, metrics in per_frame.items():
            for frame_index in range(sequence.num_frames):
                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "is_observed": frame_index in observed_set,
                        "method": method,
                        **{key: value[frame_index].item() for key, value in metrics.items()},
                    }
                )
    print(f"Saved per-frame latent errors to {csv_path}")

    if output_tensors is not None:
        tensor_path = Path(output_tensors)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ground_truth_latents": ground_truth_latents,
                "keyframe_latents": keyframe_latents,
                "predictions": predictions,
                "observed_indices": sequence.observed_indices,
            },
            tensor_path,
        )
        print(f"Saved latent tensors to {tensor_path}")
    return payload
