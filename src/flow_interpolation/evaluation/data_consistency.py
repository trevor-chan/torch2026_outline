from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm import tqdm

from flow_interpolation.evaluation.common import (
    FlowSettings,
    image_metrics,
    missing_mask,
    nearest_observed_timeline,
    predict_clean_and_noise,
    save_json,
)
from flow_interpolation.evaluation.data import SequenceData
from flow_interpolation.evaluation.geometry import global_slerp_noise
from flow_interpolation.evaluation.visualization import make_comparison_video_frames, write_video


def _sample_innovation(
    *,
    noise_control: str,
    num_frames: int,
    frame_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    slerp_mode: str,
) -> torch.Tensor:
    if noise_control == "independent":
        return torch.randn(
            (num_frames, *frame_shape),
            device=device,
            dtype=dtype,
            generator=generator,
        )
    if noise_control == "slerp":
        return global_slerp_noise(
            num_frames,
            frame_shape,
            device=device,
            dtype=dtype,
            generator=generator,
            mode=slerp_mode,
        )
    raise ValueError(f"Unknown noise control: {noise_control}")


def _project_observations(
    x0_hat: torch.Tensor,
    observed_indices: torch.Tensor,
    observed_frames: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("data consistency strength must be in [0, 1]")
    output = x0_hat.clone()
    index = observed_indices.to(x0_hat.device)
    observations = observed_frames.to(x0_hat.device, x0_hat.dtype)
    output[index] = torch.lerp(output[index], observations, strength)
    return output


@torch.no_grad()
def sample_with_data_consistency(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    noise_control: str,
    renoise_mode: str,
    eta: float,
    dc_strength: float,
    slerp_mode: str,
    seed: int,
    clip_x0: bool,
) -> torch.Tensor:
    """ISCS-inspired posterior sampling adapted to linear rectified flow.

    At every reverse step:
      1. predict clean and terminal-noise endpoints from x_t;
      2. project observed temporal frames in clean-image space;
      3. draw a fresh innovation field (iid or one global SLERP path);
      4. re-noise the projected estimate to t_next.

    ``renoise_mode='dds'`` uses
      z_mix = sqrt(1-eta^2) z_hat + eta eps,
    while ``renoise_mode='ddpm'`` discards z_hat and uses fresh eps entirely.
    This is an RF analogue, not an exact SDE identity.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    if renoise_mode not in {"dds", "ddpm"}:
        raise ValueError(f"Unknown re-noising mode: {renoise_mode}")

    num_frames = sequence.num_frames
    frame_shape = tuple(sequence.frames.shape[1:])
    generator = torch.Generator(device=device).manual_seed(seed)
    x_t = _sample_innovation(
        noise_control=noise_control,
        num_frames=num_frames,
        frame_shape=frame_shape,
        device=device,
        dtype=sequence.frames.dtype,
        generator=generator,
        slerp_mode=slerp_mode,
    )
    times = torch.linspace(
        flow.noise_time,
        flow.data_time,
        flow.ode_steps + 1,
        device=device,
        dtype=x_t.dtype,
    )

    for t_curr, t_next in tqdm(
        zip(times[:-1], times[1:]),
        total=flow.ode_steps,
        desc=f"Data-consistency sampling [{noise_control}/{renoise_mode}]",
    ):
        x0_hat, z_hat, _ = predict_clean_and_noise(model, x_t, t_curr)
        if clip_x0:
            x0_hat = x0_hat.clamp(0.0, 1.0)
        x0_dc = _project_observations(
            x0_hat,
            sequence.observed_indices,
            sequence.observed_frames,
            dc_strength,
        )
        innovation = _sample_innovation(
            noise_control=noise_control,
            num_frames=num_frames,
            frame_shape=frame_shape,
            device=device,
            dtype=x_t.dtype,
            generator=generator,
            slerp_mode=slerp_mode,
        )
        if renoise_mode == "dds":
            z_mix = math.sqrt(max(0.0, 1.0 - eta**2)) * z_hat + eta * innovation
        else:
            z_mix = innovation
        x_t = (1.0 - t_next) * x0_dc + t_next * z_mix

    x0_final, _, _ = predict_clean_and_noise(model, x_t, flow.data_time)
    x0_final = _project_observations(
        x0_final,
        sequence.observed_indices,
        sequence.observed_frames,
        dc_strength,
    )
    return x0_final.cpu().clamp(0.0, 1.0)


@torch.no_grad()
def run_data_consistency_evaluation(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sequence: SequenceData,
    flow: FlowSettings,
    noise_controls: list[str],
    renoise_mode: str,
    eta: float,
    dc_strength: float,
    slerp_mode: str,
    seed: int,
    clip_x0: bool,
    output_dir: str,
    video_fps: float,
    display_scale: int,
    gap: int,
    residual_scale: float,
    save_tensors: bool,
) -> dict:
    missing = missing_mask(sequence.num_frames, sequence.observed_indices)
    predictions: OrderedDict[str, torch.Tensor] = OrderedDict()
    metrics: dict[str, dict] = {}
    for method_index, noise_control in enumerate(noise_controls):
        prediction = sample_with_data_consistency(
            model=model,
            device=device,
            sequence=sequence,
            flow=flow,
            noise_control=noise_control,
            renoise_mode=renoise_mode,
            eta=eta,
            dc_strength=dc_strength,
            slerp_mode=slerp_mode,
            seed=seed + method_index * 10_000,
            clip_x0=clip_x0,
        )
        label = f"{noise_control}_{renoise_mode}"
        predictions[label] = prediction
        metrics[label] = {
            "all_frames": image_metrics(prediction, sequence.frames),
            "missing_frames": image_metrics(prediction[missing], sequence.frames[missing]),
            "observed_frames": image_metrics(
                prediction[sequence.observed_indices], sequence.observed_frames
            ),
        }
        print(f"Data consistency [{label}]: {metrics[label]}")

    nearest = nearest_observed_timeline(
        sequence.observed_frames,
        sequence.observed_indices,
        sequence.num_frames,
    )
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
    write_video(frames, str(output_root / "data_consistency.mp4"), fps=video_fps)

    payload = {
        "noise_controls": noise_controls,
        "renoise_mode": renoise_mode,
        "eta": eta,
        "data_consistency_strength": dc_strength,
        "slerp_mode": slerp_mode,
        "clip_x0": clip_x0,
        "cadence": sequence.cadence,
        "video_panel_order": [
            "ground_truth",
            "nearest_observed",
            *[item for label in predictions for item in (label, f"{label}_absolute_residual")],
        ],
        "metrics": metrics,
    }
    save_json(payload, output_root / "data_consistency_metrics.json")
    if save_tensors:
        torch.save(
            {
                "frames": sequence.frames,
                "observed_indices": sequence.observed_indices,
                "predictions": dict(predictions),
            },
            output_root / "data_consistency_tensors.pt",
        )
        print(f"Saved data-consistency tensors to {output_root / 'data_consistency_tensors.pt'}")
    return payload
