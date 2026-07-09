import argparse
import math
import os
from typing import Optional

import torch
from tqdm import tqdm

from dataset import BouncingBallVideoDataset
from transformer import TransformerDiffusionModel


def load_checkpoint(model, checkpoint_path: str, device: torch.device, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict = {}
    for key, value in checkpoint.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.removeprefix("module.").removeprefix("_orig_mod.")
        state_dict[key] = value

    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")


@torch.no_grad()
def integrate_flow(
    model,
    x: torch.Tensor,
    t_start: float,
    t_end: float,
    num_steps: int,
    solver: str = "heun",
    desc: Optional[str] = None,
) -> torch.Tensor:
    x = x.clone()
    times = torch.linspace(t_start, t_end, num_steps + 1, device=x.device)

    iterator = zip(times[:-1], times[1:])
    if desc is not None:
        iterator = tqdm(iterator, total=num_steps, desc=desc)

    for t_curr, t_next in iterator:
        dt = t_next - t_curr
        t_batch = t_curr.expand(x.shape[0])
        v_curr = model(x, None, t_batch)

        if solver == "euler":
            x = x + dt * v_curr
        elif solver == "heun":
            x_pred = x + dt * v_curr
            v_next = model(x_pred, None, t_next.expand(x.shape[0]))
            x = x + 0.5 * dt * (v_curr + v_next)
        else:
            raise ValueError(f"unknown solver: {solver}")

    return x


def slerp(a: torch.Tensor, b: torch.Tensor, weight: float, eps: float = 1e-7) -> torch.Tensor:
    if weight <= 0.0:
        return a
    if weight >= 1.0:
        return b

    original_shape = a.shape
    a_flat = a.flatten(start_dim=1)
    b_flat = b.flatten(start_dim=1)

    a_norm = a_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    b_norm = b_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    a_unit = a_flat / a_norm
    b_unit = b_flat / b_norm

    dot = (a_unit * b_unit).sum(dim=1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)

    left = torch.sin((1.0 - weight) * omega) / sin_omega
    right = torch.sin(weight * omega) / sin_omega
    direction = left * a_unit + right * b_unit
    radius = torch.lerp(a_norm, b_norm, weight)

    return (direction * radius).view(original_shape)


def decode_in_chunks(
    model,
    latents: torch.Tensor,
    ode_steps: int,
    solver: str,
    data_time: float,
    noise_time: float,
    chunk_size: int,
    device: torch.device,
    desc: str = "Decoding latents",
) -> torch.Tensor:
    chunks = []
    for chunk in tqdm(latents.split(chunk_size), desc=desc):
        chunks.append(
            integrate_flow(
                model=model,
                x=chunk.to(device),
                t_start=noise_time,
                t_end=data_time,
                num_steps=ode_steps,
                solver=solver,
            ).cpu()
        )
    return torch.cat(chunks, dim=0)


def perturb_to_p_eps(images: torch.Tensor, data_time: float, eps_noise: torch.Tensor) -> torch.Tensor:
    """Move clean images onto the marginal p_t at t=data_time.

    The encode ODE only maps to N(0, I) when started from a sample of
    (1 - t) * x0 + t * eps; starting from the clean image leaves deterministic
    pixels (e.g. the constant background) with no randomness to amplify, which
    produces spatially correlated latents instead of Gaussian noise.
    """
    return (1.0 - data_time) * images + data_time * eps_noise.to(images.device, images.dtype)


def encode_in_chunks(
    model,
    samples: torch.Tensor,
    ode_steps: int,
    solver: str,
    data_time: float,
    noise_time: float,
    chunk_size: int,
    device: torch.device,
    eps_noise: Optional[torch.Tensor] = None,
    desc: str = "Encoding frames to noise",
) -> torch.Tensor:
    chunks = []
    for chunk in tqdm(samples.split(chunk_size), desc=desc):
        chunk = chunk.to(device)
        if eps_noise is not None:
            chunk = perturb_to_p_eps(chunk, data_time, eps_noise)
        chunks.append(
            integrate_flow(
                model=model,
                x=chunk,
                t_start=data_time,
                t_end=noise_time,
                num_steps=ode_steps,
                solver=solver,
            ).cpu()
        )
    return torch.cat(chunks, dim=0)


def residual_image(prediction: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    return (prediction.clamp(0.0, 1.0) - target.clamp(0.0, 1.0)).abs().mul(scale).clamp(0.0, 1.0)


def noise_image(noise: torch.Tensor, clip: float) -> torch.Tensor:
    return noise.clamp(-clip, clip).div(2.0 * clip).add(0.5).clamp(0.0, 1.0)


def noise_residual_image(prediction_noise: torch.Tensor, target_noise: torch.Tensor, clip: float) -> torch.Tensor:
    return (prediction_noise - target_noise).abs().div(clip).clamp(0.0, 1.0)


def concat_with_gap(images, dim: int, gap: int, value: float = 0.06) -> torch.Tensor:
    if len(images) == 1 or gap <= 0:
        return torch.cat(images, dim=dim)

    output = images[0]
    for image in images[1:]:
        gap_shape = list(output.shape)
        gap_shape[dim] = gap
        gap_tensor = output.new_full(gap_shape, value)
        output = torch.cat([output, gap_tensor, image], dim=dim)
    return output


def print_noise_stats(name: str, noise: torch.Tensor):
    flat = noise.flatten(start_dim=1)
    expected_radius = math.sqrt(flat.shape[1])
    radii = flat.norm(dim=1)
    print(
        f"{name}: mean={flat.mean().item():.4f}, std={flat.std().item():.4f}, "
        f"radius={radii.mean().item():.2f} +/- {radii.std().item():.2f} "
        f"(sqrt(dim)={expected_radius:.2f})"
    )


def make_visualization_frames(
    ground_truth: torch.Tensor,
    low_rate: torch.Tensor,
    predictions: torch.Tensor,
    ground_truth_noise: torch.Tensor,
    low_rate_noise: torch.Tensor,
    prediction_noise: torch.Tensor,
    residual_scale: float,
    noise_display_clip: float,
    display_scale: int,
    row_gap: int,
) -> torch.Tensor:
    frames = []

    for target, nearest_endpoint, prediction, target_noise, nearest_noise, latent_noise in zip(
        ground_truth,
        low_rate,
        predictions,
        ground_truth_noise,
        low_rate_noise,
        prediction_noise,
    ):
        data_panels = [
            target,
            nearest_endpoint,
            prediction.clamp(0.0, 1.0),
            residual_image(prediction, target, residual_scale),
        ]
        noise_panels = [
            noise_image(target_noise, noise_display_clip),
            noise_image(nearest_noise, noise_display_clip),
            noise_image(latent_noise, noise_display_clip),
            noise_residual_image(latent_noise, target_noise, noise_display_clip),
        ]
        rows = [
            concat_with_gap(data_panels, dim=-1, gap=row_gap),
            concat_with_gap(noise_panels, dim=-1, gap=row_gap),
        ]
        panel = concat_with_gap(rows, dim=-2, gap=row_gap)

        if display_scale > 1:
            panel = panel.repeat_interleave(display_scale, dim=-2).repeat_interleave(display_scale, dim=-1)

        frame = panel.clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).round().to(torch.uint8)
        frames.append(frame)

    return torch.stack(frames, dim=0)


def make_nearest_endpoint_intervals(
    endpoints_a: torch.Tensor,
    endpoints_b: torch.Tensor,
    num_times: int,
) -> torch.Tensor:
    frames = []
    for time_idx in range(num_times):
        endpoint_weight = int(round(time_idx / max(num_times - 1, 1)))
        frames.append(endpoints_b if endpoint_weight == 1 else endpoints_a)
    return torch.stack(frames, dim=0)


def stitch_intervals(
    intervals: torch.Tensor,
    drop_duplicate_boundaries: bool,
) -> torch.Tensor:
    pieces = []
    for interval_idx in range(intervals.shape[1]):
        start_offset = 1 if interval_idx > 0 and drop_duplicate_boundaries else 0
        pieces.append(intervals[start_offset:, interval_idx])
    return torch.cat(pieces, dim=0)


def write_video(frames: torch.Tensor, path: str, fps: float):
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("Install imageio-ffmpeg to write MP4 files.") from error

    frames = frames.cpu().contiguous()
    height, width = frames.shape[1:3]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLERP interpolation in flow-model noise space.")
    parser.add_argument("--checkpoint", type=str, default="bouncing_ball_diffusion_ema.pth")
    parser.add_argument("--output", type=str, default="outputs/slerp_interpolation_eval.mp4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="compile the model before evaluation")
    parser.add_argument("--non-strict-load", action="store_true", help="allow missing/unexpected checkpoint keys")

    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--time-embed-dim", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=100.0)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=64)
    parser.add_argument(
        "--pair-stride",
        type=int,
        default=None,
        help="high-rate frames between interval starts; unset keeps intervals contiguous",
    )
    parser.add_argument("--training-frame-dt", type=float, default=0.2)
    parser.add_argument("--high-frame-dt", type=float, default=0.02)
    parser.add_argument("--training-color-walk-std", type=float, default=0.075)
    parser.add_argument("--color-walk-std", type=float, default=None)

    parser.add_argument("--ode-steps", type=int, default=128)
    parser.add_argument("--solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--data-eps", type=float, default=1e-3, help="start this far after t=0 when encoding from images")
    parser.add_argument("--t-eps", type=float, default=1e-3, help="stop this far before t=1 when encoding to noise")
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=32)

    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--display-scale", type=int, default=8)
    parser.add_argument("--row-gap", type=int, default=2, help="gap between horizontal visualization panels")
    parser.add_argument("--residual-scale", type=float, default=4.0)
    parser.add_argument("--noise-display-clip", type=float, default=3.0)

    args = parser.parse_args()

    device = torch.device(args.device)
    if args.data_eps < 0.0:
        raise ValueError("--data-eps must be non-negative")
    if not 0.0 <= args.t_eps < 1.0:
        raise ValueError("--t-eps must satisfy 0 <= t_eps < 1")
    torch.manual_seed(args.seed)
    data_time = args.data_eps
    noise_time = 1.0 - args.t_eps
    if data_time >= noise_time:
        raise ValueError("--data-eps must be smaller than 1 - --t-eps")
    endpoint_stride = max(1, int(round(args.training_frame_dt / args.high_frame_dt)))
    pair_stride = args.pair_stride if args.pair_stride is not None else endpoint_stride
    actual_endpoint_dt = endpoint_stride * args.high_frame_dt
    color_walk_std = args.color_walk_std
    if color_walk_std is None:
        color_walk_std = args.training_color_walk_std * math.sqrt(args.high_frame_dt / args.training_frame_dt)

    pair_starts = torch.arange(args.num_pairs) * pair_stride + args.start_index
    pair_ends = pair_starts + endpoint_stride
    drop_duplicate_boundaries = pair_stride == endpoint_stride
    num_samples = int(pair_ends[-1].item()) + 1

    print(
        f"Generating {num_samples} high-rate frames "
        f"(dt={args.high_frame_dt}, endpoint spacing={actual_endpoint_dt:.4f}s, "
        f"intervals={args.num_pairs}, data_time={data_time:.6f}, noise_time={noise_time:.6f})"
    )
    dataset = BouncingBallVideoDataset(
        num_samples=num_samples,
        image_size=args.image_size,
        seed=args.seed,
        frame_dt=args.high_frame_dt,
        color_walk_std=color_walk_std,
        write_video=False,
    )
    samples = dataset.samples

    endpoints_a = samples[pair_starts]
    endpoints_b = samples[pair_ends]
    ground_truth_intervals = torch.stack(
        [samples[pair_starts + offset] for offset in range(endpoint_stride + 1)],
        dim=0,
    )
    low_rate_intervals = make_nearest_endpoint_intervals(
        endpoints_a=endpoints_a,
        endpoints_b=endpoints_b,
        num_times=endpoint_stride + 1,
    )

    model = TransformerDiffusionModel(
        in_channels=3,
        dim=args.model_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        time_embed_dim=args.time_embed_dim,
        rope_theta=args.rope_theta,
    ).to(device)
    load_checkpoint(
        model,
        checkpoint_path=args.checkpoint,
        device=device,
        strict=not args.non_strict_load,
    )
    model.eval()
    if args.compile:
        model = torch.compile(model)

    # One shared eps-noise draw for every encode that starts from real data, so
    # shared boundary frames and both endpoints of a pair get consistent
    # background latents.
    encode_eps_noise = torch.randn(3, args.image_size, args.image_size)

    endpoints = torch.cat([endpoints_a, endpoints_b], dim=0).to(device)
    endpoints = perturb_to_p_eps(endpoints, data_time, encode_eps_noise)
    encoded = integrate_flow(
        model=model,
        x=endpoints,
        t_start=data_time,
        t_end=noise_time,
        num_steps=args.ode_steps,
        solver=args.solver,
        desc="Encoding endpoints to noise",
    )
    noise_a, noise_b = encoded.chunk(2, dim=0)
    encoded_cpu = encoded.detach().cpu()
    print_noise_stats(f"Encoded endpoint noise at t={noise_time:.6f}", encoded_cpu)
    print_noise_stats("Reference Gaussian", torch.randn_like(encoded_cpu))

    weights = torch.linspace(0.0, 1.0, endpoint_stride + 1)
    latent_rows = [slerp(noise_a, noise_b, float(weight)) for weight in weights]
    latent_intervals = torch.stack(latent_rows, dim=0)
    latents = latent_intervals.flatten(0, 1)

    decoded = decode_in_chunks(
        model=model,
        latents=latents,
        ode_steps=args.ode_steps,
        solver=args.solver,
        data_time=data_time,
        noise_time=noise_time,
        chunk_size=args.decode_batch_size,
        device=device,
        desc="Decoding SLERP latents",
    )
    prediction_intervals = decoded.view(endpoint_stride + 1, args.num_pairs, 3, args.image_size, args.image_size)

    ground_truth = stitch_intervals(
        ground_truth_intervals,
        drop_duplicate_boundaries=drop_duplicate_boundaries,
    )
    low_rate = stitch_intervals(
        low_rate_intervals,
        drop_duplicate_boundaries=drop_duplicate_boundaries,
    )
    predictions = stitch_intervals(
        prediction_intervals,
        drop_duplicate_boundaries=drop_duplicate_boundaries,
    )
    low_rate_noise_intervals = make_nearest_endpoint_intervals(
        endpoints_a=noise_a.cpu(),
        endpoints_b=noise_b.cpu(),
        num_times=endpoint_stride + 1,
    )
    low_rate_noise = stitch_intervals(
        low_rate_noise_intervals,
        drop_duplicate_boundaries=drop_duplicate_boundaries,
    )
    prediction_noise = stitch_intervals(
        latent_intervals.cpu(),
        drop_duplicate_boundaries=drop_duplicate_boundaries,
    )
    ground_truth_noise = encode_in_chunks(
        model=model,
        samples=ground_truth,
        ode_steps=args.ode_steps,
        solver=args.solver,
        data_time=data_time,
        noise_time=noise_time,
        chunk_size=args.encode_batch_size,
        device=device,
        eps_noise=encode_eps_noise,
        desc="Encoding ground truth frames to noise",
    )
    print_noise_stats(f"Ground-truth timeline noise at t={noise_time:.6f}", ground_truth_noise)
    print_noise_stats(f"SLERP timeline noise at t={noise_time:.6f}", prediction_noise)

    clipped_predictions = predictions.clamp(0.0, 1.0)
    mae = (clipped_predictions - ground_truth).abs().mean().item()
    mse = ((clipped_predictions - ground_truth) ** 2).mean().item()
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    print(f"MAE: {mae:.6f} | MSE: {mse:.6f} | PSNR: {psnr:.2f} dB")

    frames = make_visualization_frames(
        ground_truth=ground_truth,
        low_rate=low_rate,
        predictions=predictions,
        ground_truth_noise=ground_truth_noise,
        low_rate_noise=low_rate_noise,
        prediction_noise=prediction_noise,
        residual_scale=args.residual_scale,
        noise_display_clip=args.noise_display_clip,
        display_scale=args.display_scale,
        row_gap=args.row_gap,
    )
    write_video(frames, args.output, fps=args.video_fps)
    print(f"Saved SLERP interpolation visualization to {args.output}")


if __name__ == "__main__":
    main()
