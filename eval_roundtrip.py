from __future__ import annotations

from pathlib import Path

import torch

from eval_common import (
    FlowSettings,
    decode_in_chunks,
    encode_in_chunks,
    image_metrics,
    make_boundary_noise,
    perturb_to_p_eps,
    predict_clean_and_noise,
    print_noise_stats,
    save_json,
    tensor_metrics,
)
from eval_data import SequenceData


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
) -> dict:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
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

    payload = {
        "sample_indices": indices,
        "boundary_noise_mode": boundary_noise_mode,
        "data_to_noise_to_data": {
            "cycle_at_data_eps": image_metrics(reconstructed_eps, x_eps.cpu()),
            "clean_endpoint_estimate": image_metrics(reconstructed_clean, clean.cpu()),
            "decoded_eps_vs_clean": image_metrics(reconstructed_eps, clean.cpu()),
            "encoded_noise_stats": print_noise_stats("Encoded data anchors", encoded),
        },
        "noise_to_data_to_noise": {
            "cycle": tensor_metrics(reencoded_noise, initial_noise.cpu()),
            "initial_noise_stats": print_noise_stats("Initial Gaussian anchors", initial_noise.cpu()),
            "reencoded_noise_stats": print_noise_stats("Re-encoded Gaussian anchors", reencoded_noise),
        },
    }
    print("Data -> noise -> data cycle:", payload["data_to_noise_to_data"])
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
                "initial_noise": initial_noise.cpu(),
                "decoded_from_noise": decoded_from_noise,
                "reencoded_noise": reencoded_noise,
            },
            path,
        )
        print(f"Saved round-trip tensors to {path}")
    return payload
